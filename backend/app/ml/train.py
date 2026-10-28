"""Transfer learning con TensorFlow/Keras (MobileNetV3 / EfficientNetB0)."""
import json
import time
from pathlib import Path

import numpy as np

from ..config import get_settings
from ..modalities import get_modality
from .architectures import fine_tune_settings
from .data import make_dataset
from .preprocess import IMG_SIZE
from .reporter import Reporter

# Reparto del porcentaje total del trabajo
P_PREPARE_END, P_TRAIN_END, P_VALIDATE_END = 8.0, 85.0, 90.0


def configure_tf() -> None:
    import tensorflow as tf

    n = get_settings().train_threads
    if n > 0:
        try:
            tf.config.threading.set_intra_op_parallelism_threads(n)
            tf.config.threading.set_inter_op_parallelism_threads(2)
        except RuntimeError:
            pass  # ya inicializado en un trabajo anterior del mismo proceso


def build_model(base_name: str, n_classes: int, weights: str | None = "auto", size: int = IMG_SIZE):
    from tensorflow import keras

    from .architectures import build_backbone

    if weights == "auto":
        weights = "imagenet" if get_settings().pretrained_weights == "imagenet" else None
    shape = (size, size, 3)
    pre, base = build_backbone(base_name, shape, weights)
    base.trainable = False
    inputs = keras.Input(shape, name="image")  # float32 0-255; el modelo normaliza internamente
    x = inputs
    for layer in pre:
        x = layer(x)
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(n_classes, activation="softmax", name="probs")(x)
    return keras.Model(inputs, outputs, name=f"{base_name}_classifier"), base


def _augmentation():
    from tensorflow import keras

    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.08),
            keras.layers.RandomZoom(0.1),
            keras.layers.RandomContrast(0.1),
        ]
    )


def _spec_augment():
    """SpecAugment para espectrogramas: desplazamiento temporal y una banda de frecuencia y otra
    de tiempo enmascaradas (a silencio). Voltear o rotar no tiene sentido en audio."""
    import tensorflow as tf

    def mask(x, axis):
        n = tf.shape(x)[axis]
        width = tf.random.uniform([], 0, tf.maximum(n // 8, 1), dtype=tf.int32)
        start = tf.random.uniform([], 0, tf.maximum(n - width, 1), dtype=tf.int32)
        idx = tf.range(n)
        keep = tf.cast((idx < start) | (idx >= start + width), x.dtype)
        shape = [1, -1, 1, 1] if axis == 1 else [1, 1, -1, 1]
        return x * tf.reshape(keep, shape)

    def augment(x, training=True):
        n = tf.shape(x)[2]
        shift = tf.random.uniform([], -(n // 10), n // 10 + 1, dtype=tf.int32)
        x = tf.roll(x, shift, axis=2)
        return mask(mask(x, 1), 2)

    return augment


def _progress_callback(rep: Reporter, epoch_offset: int, total_epochs: int, steps: int):
    from tensorflow import keras

    class Progress(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.cur = 0

        def _percent(self, batch: int) -> tuple[float, float | None]:
            frac = min((epoch_offset + self.cur + batch / max(steps, 1)) / total_epochs, 1.0)
            pct = P_PREPARE_END + (P_TRAIN_END - P_PREPARE_END) * frac
            eta = None
            if rep.train_started_at and frac > 0.02:
                spent = time.time() - rep.train_started_at
                eta = spent / frac * (1 - frac) + 20
            return pct, eta

        def on_epoch_begin(self, epoch, logs=None):
            self.cur = epoch

        def on_train_batch_end(self, batch, logs=None):
            pct, eta = self._percent(batch + 1)
            logs = logs or {}
            rep.update(
                percent=pct, eta_seconds=eta, epoch=epoch_offset + self.cur + 1, total_epochs=total_epochs,
                metrics={"accuracy": float(logs.get("accuracy", 0)), "loss": float(logs.get("loss", 0))},
            )
            if rep.check_cancel():
                self.model.stop_training = True

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            row = {"epoch": epoch_offset + epoch + 1, **{k: round(float(v), 5) for k, v in logs.items()}}
            rep.history.append(row)
            pct, eta = self._percent(steps)
            rep.update(
                force=True, percent=pct, eta_seconds=eta, epoch=row["epoch"], total_epochs=total_epochs,
                metrics={k: v for k, v in row.items() if k != "epoch"},
            )
            rep.log(
                f"Época {row['epoch']}/{total_epochs} — loss {row.get('loss', 0):.4f}, acc {row.get('accuracy', 0):.4f}, "
                f"val_loss {row.get('val_loss', 0):.4f}, val_acc {row.get('val_accuracy', 0):.4f}"
            )

    return Progress()


def _compile(model, lr: float):
    from tensorflow import keras

    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def train(rep: Reporter, cfg: dict, base_name: str, labels: list[str], work: Path):
    """Entrena y devuelve (modelo, info). Lee los datos progresivamente desde memmaps."""
    import tensorflow as tf
    from tensorflow import keras

    epochs = int(cfg.get("epochs", 10))
    ft_epochs = int(cfg.get("fine_tune_epochs", 0))
    batch = int(cfg.get("batch_size", 32))
    size = int(cfg.get("image_size", IMG_SIZE))
    lr = float(cfg.get("learning_rate", 1e-3))
    patience = int(cfg.get("patience", 3))
    total = epochs + ft_epochs

    spec = get_modality(cfg.get("modality")).augmentation == "spec"  # espectrogramas: SpecAugment, no voltear/rotar
    aug = (_spec_augment() if spec else _augmentation()) if cfg.get("augmentation", True) else None
    train_ds, n_train, steps = make_dataset(work, "train", batch, True, aug, size)
    val_ds, n_val, _ = make_dataset(work, "val", batch, False, size=size)

    y_train = np.load(work / "train_y.npy")
    counts = np.bincount(y_train, minlength=len(labels)).astype(float)
    class_weight = {i: float(len(y_train) / (len(labels) * max(c, 1))) for i, c in enumerate(counts)}

    model, base = build_model(base_name, len(labels), size=size)
    ckpt_dir = work / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    state_path = work / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    rep.train_started_at = time.time()
    best_path = ckpt_dir / "best.weights.h5"
    phase1_path = ckpt_dir / "phase1.weights.h5"

    def callbacks(offset: int, backup: str):
        return [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(str(best_path), monitor="val_loss", save_best_only=True, save_weights_only=True),
            keras.callbacks.BackupAndRestore(str(work / backup), delete_checkpoint=True),
            _progress_callback(rep, offset, total, steps),
        ]

    epochs_run = 0
    if state.get("phase1_done") and phase1_path.exists():
        model.load_weights(str(phase1_path))
        rep.history = state["history"]
        epochs_run = state["epochs_run"]
        rep.log("Fase 1 ya completada antes del reinicio; se reanuda desde sus pesos.")
    else:
        rep.log(f"Entrenando cabeza de clasificación sobre {base_name} a {size}x{size} px ({epochs} épocas máx., batch {batch}).")
        _compile(model, lr)
        h = model.fit(
            train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks(0, "backup1"),
            class_weight=class_weight, verbose=0,
        )
        epochs_run = len(h.history["loss"])
        model.save_weights(str(phase1_path))
        state_path.write_text(json.dumps({"phase1_done": True, "history": rep.history, "epochs_run": epochs_run}))
    rep.raise_if_cancelled()

    if ft_epochs > 0:
        unfreeze, ft_factor = fine_tune_settings(base_name, cfg)
        rep.log(
            f"Ajuste fino de las últimas {unfreeze} capas del modelo base ({ft_epochs} épocas máx., "
            f"tasa {lr * ft_factor:.1e})."
        )
        base.trainable = True
        for layer in base.layers[:-unfreeze]:
            layer.trainable = False
        for layer in base.layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False
        _compile(model, lr * ft_factor)
        h = model.fit(
            train_ds, validation_data=val_ds, epochs=ft_epochs, callbacks=callbacks(epochs_run, "backup2"),
            class_weight=class_weight, verbose=0,
        )
        epochs_run += len(h.history["loss"])
    rep.raise_if_cancelled()

    if best_path.exists():  # los pesos del mejor checkpoint de validación
        try:
            model.load_weights(str(best_path))
        except Exception as exc:  # noqa: BLE001
            rep.log(f"No se pudo recargar el mejor checkpoint: {exc}", "warning")
    _compile(model, lr)
    best = min(rep.history, key=lambda r: r.get("val_loss", 1e9)) if rep.history else {}
    return model, {
        "epochs_run": epochs_run,
        "best_epoch": best.get("epoch"),
        "train_images": n_train,
        "val_images": n_val,
        "best_checkpoint": str(best_path),
        "hints": training_hints(rep.history, ft_epochs),
    }


def training_hints(history: list[dict], ft_epochs: int) -> list[str]:
    """Diagnósticos legibles a partir de la curva de entrenamiento."""
    hints: list[str] = []
    if len(history) < 2:
        return hints
    best = min(history, key=lambda r: r.get("val_loss", 1e9))
    last = history[-1]
    if best["epoch"] == last["epoch"]:
        what = "Épocas de ajuste fino" if ft_epochs > 0 else "Épocas máximas"
        hints.append(f"El modelo seguía mejorando al terminar: aumenta las «{what}» para obtener mejor exactitud.")
    gap = last.get("accuracy", 0) - last.get("val_accuracy", 0)
    if gap > 0.15:
        hints.append(
            f"Posible sobreajuste: la exactitud en entrenamiento supera a la de validación por {gap * 100:.0f} puntos. "
            "Agrega más imágenes variadas por categoría o mantén activado el data augmentation."
        )
    if ft_epochs == 0 and last.get("val_accuracy", 1) < 0.85:
        hints.append(
            "No se usó ajuste fino. Activar «Épocas de ajuste fino» (p. ej. 5) suele mejorar bastante la exactitud "
            "cuando las categorías se parecen entre sí."
        )
    return hints


def evaluate(model, work: Path, labels: list[str], batch: int, size: int = IMG_SIZE) -> dict:
    """Una sola pasada sobre el conjunto de prueba (antes eran dos: evaluate + predict)."""
    test_ds, n_test, _ = make_dataset(work, "test", batch, False, size=size)
    probs = model.predict(test_ds, verbose=0)
    y_true = np.load(work / "test_y.npy")
    y_pred = probs.argmax(axis=1)
    acc = float(np.mean(y_pred == y_true))
    loss = float(np.mean(-np.log(np.clip(probs[np.arange(len(y_true)), y_true], 1e-7, 1.0))))
    k = len(labels)
    topk = min(5, k)
    top5 = float(np.mean([t in row for t, row in zip(y_true, np.argsort(probs, axis=1)[:, -topk:])]))
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    per_class = []
    for i, name in enumerate(labels):
        tp = cm[i, i]
        precision = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        recall = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(
            {"label": name, "precision": round(float(precision), 4), "recall": round(float(recall), 4),
             "f1": round(float(f1), 4), "support": int(cm[i, :].sum())}
        )
    return {
        "test_accuracy": round(float(acc), 4),
        "test_loss": round(float(loss), 4),
        "test_images": int(n_test),
        "test_top5_accuracy": round(top5, 4),
        "top_k": topk,
        "macro_f1": round(float(np.mean([c["f1"] for c in per_class])), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }
