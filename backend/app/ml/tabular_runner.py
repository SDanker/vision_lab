"""Entrenamiento de un proyecto de datos tabulares: preparar → entrenar → evaluar → exportar → registrar.

Mismo ciclo de vida que `runner._execute` (estados, progreso, cancelación, versión del modelo y formatos validados),
pero con scikit-learn en lugar de TensorFlow: no hay épocas ni data augmentation, y un entrenamiento dura segundos.
"""
import json
import shutil
import time
from pathlib import Path

import numpy as np
from sqlalchemy import func, select

from ..config import get_settings
from ..db import SessionLocal, utcnow
from ..models import (
    J_COMPLETED,
    J_ERROR,
    J_EXPORTING,
    J_PREPARING,
    J_TRAINING,
    J_VALIDATING,
    ModelExport,
    ModelVersion,
    TrainingJob,
)
from ..modalities import get_modality
from ..services import training_queue as tq
from ..storage import data_dir
from . import export as export_mod
from . import tabular as tab
from . import tabular_features as feat
from .architectures import label_of
from .templates import README_TABULAR

P_PREPARED, P_TRAINED, P_EVALUATED = 8.0, 80.0, 90.0


def execute(rep, job_id, project_id, project_name, cfg, base_name, snap_path, snap_hash, snap_id, work: Path):
    import pandas as pd
    import sklearn

    from .runner import _set_status

    settings = get_settings()
    mod = get_modality("tabular")
    root = Path(snap_path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    labels, target, names = manifest["labels"], manifest["tabular"]["target"], manifest["tabular"]["features"]

    # 1. Preparación: etiquetas, división estratificada y conversión de cada fila en las entradas del modelo
    _set_status(rep, J_PREPARING, "Preparando datos", 1.0, total_epochs=1)
    df = pd.read_csv(root / "data.csv", dtype=str, keep_default_na=False, na_filter=False)
    index = {label: i for i, label in enumerate(labels)}
    y = np.array([index[v] for v in df[target]], dtype=np.int64)
    split = tab.split_indices(y, len(labels), labels=labels)
    tr, va, te = split["train"], split["val"], split["test"]
    features = tab.describe_features(df, names, tr)  # categorías y medianas sólo del conjunto de entrenamiento
    X = tab.build_frame(df, features)
    rep.log(
        f"Dataset: {len(tr)} entrenamiento / {len(va)} validación / {len(te)} prueba (filas), {len(labels)} categorías, "
        f"{len(features)} columnas de entrada. Columna a predecir: {target}."
    )
    rep.update(force=True, percent=P_PREPARED)
    rep.raise_if_cancelled()

    # 2. Entrenamiento
    _set_status(rep, J_TRAINING, "Entrenando", P_PREPARED)
    rep.train_started_at = time.time()
    pipe = tab.build_pipeline(base_name, features, n_jobs=settings.train_threads if settings.train_threads > 0 else -1)
    rep.log(f"Entrenando {label_of(base_name)}.")
    tab.fit(pipe, X.iloc[tr], y[tr])
    rep.raise_if_cancelled()
    train_m = tab.metrics_from_probs(pipe.predict_proba(X.iloc[tr]), y[tr], labels)
    val_m = tab.metrics_from_probs(pipe.predict_proba(X.iloc[va]), y[va], labels)
    row = {"epoch": 1, "loss": train_m["test_loss"], "accuracy": train_m["test_accuracy"],
           "val_loss": val_m["test_loss"], "val_accuracy": val_m["test_accuracy"]}
    rep.history = [row]
    rep.update(force=True, percent=P_TRAINED, epoch=1, total_epochs=1, eta_seconds=None,
               metrics={k: v for k, v in row.items() if k != "epoch"})
    rep.log(f"Entrenado — exactitud {row['accuracy']:.4f}, validación {row['val_accuracy']:.4f} (loss {row['val_loss']:.4f}).")

    # 3. Evaluación en el conjunto de prueba
    _set_status(rep, J_VALIDATING, "Evaluando en el conjunto de prueba", P_TRAINED, eta_seconds=None)
    metrics = tab.metrics_from_probs(pipe.predict_proba(X.iloc[te]), y[te], labels)
    majority = float(np.bincount(y[te], minlength=len(labels)).max() / len(te))
    metrics.update(
        epochs_run=1, best_epoch=1, train_images=len(tr), val_images=len(va), test_images=len(te),
        val_accuracy=row["val_accuracy"], majority_baseline=round(majority, 4), unit="filas",
        feature_importance=tab.feature_importance(pipe, X.iloc[va], y[va], features),
        hints=tab.training_hints(rep.history, metrics["test_accuracy"], majority, len(tr)),
    )
    rep.log(f"Exactitud en prueba: {metrics['test_accuracy']:.4f} · F1 macro: {metrics['macro_f1']:.4f} (loss {metrics['test_loss']:.4f}).")
    for hint in metrics["hints"]:
        rep.log(f"Sugerencia: {hint}", "warning")
    rep.raise_if_cancelled()

    # 4. Exportación y validación de los formatos
    _set_status(rep, J_EXPORTING, "Exportando y validando formatos", P_EVALUATED,
                metrics={**{k: v for k, v in row.items() if k != "epoch"}, "test_accuracy": metrics["test_accuracy"]})
    with SessionLocal() as db:
        version = (db.scalar(select(func.max(ModelVersion.version)).where(ModelVersion.project_id == project_id)) or 0) + 1
    out = data_dir() / "models" / str(project_id) / f"v{version}"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    exports = tab.export_all(pipe, features, labels, out, X.iloc[te].head(50), lambda m, level="info": rep.log(m, level))
    rep.update(force=True, percent=97.0)

    created = utcnow()
    meta = {
        "project": project_name, "version": version, "created_at": created.isoformat(),
        "base_model": base_name, "base_model_label": label_of(base_name), "framework": f"scikit-learn {sklearn.__version__}",
        "input": {"names": [f"c{i}" for i in range(len(features))], "layout": "una entrada (N, 1) por columna",
                  "dtype": "float32 (numéricas) o texto (categóricas)", "normalization": "incluida en el modelo"},
        **mod.metadata(cfg),
        "tabular": {"target": target, "features": features, "missing_token": feat.MISSING, "other_token": feat.OTHER,
                    "sklearn": sklearn.__version__},
        "output": {"name": "probabilities", "dtype": "float32", "activation": "probabilidades", "classes": len(labels)},
        "labels": labels,
        "dataset": {"hash": snap_hash, "snapshot_id": snap_id, "train": len(tr), "val": len(va), "test": len(te)},
        "training_config": cfg,
        "metrics": {k: metrics[k] for k in ("test_accuracy", "test_top5_accuracy", "test_loss", "macro_f1", "val_accuracy")},
        "exports": [{k: e[k] for k in ("fmt", "status", "filename", "size_bytes", "validation")} for e in exports],
        "low_confidence_threshold": settings.low_confidence_threshold,
    }
    (out / "labels.json").write_text(json.dumps({"labels": labels}, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps({**metrics, "history": rep.history}, ensure_ascii=False, indent=2), encoding="utf-8")
    mod.export_assets(out)
    (out / "README.md").write_text(
        README_TABULAR.format(
            project=project_name, version=version, created=created.strftime("%Y-%m-%d %H:%M UTC"), base_model=label_of(base_name),
            target=target, labels=", ".join(labels), features=", ".join(f["name"] for f in features), accuracy=metrics["test_accuracy"],
        ),
        encoding="utf-8",
    )
    export_mod.build_bundle(out)

    all_ok = all(e["status"] == "ok" for e in exports)
    with SessionLocal() as db:
        mv = ModelVersion(
            project_id=project_id, job_id=job_id, snapshot_id=snap_id, version=version, base_model=base_name, labels=labels,
            metrics={k: v for k, v in metrics.items() if k != "history"}, config=cfg, dataset_hash=snap_hash, dir_path=str(out),
            published=any(e["status"] == "ok" for e in exports),
        )
        for e in exports:
            mv.exports.append(ModelExport(**e))
        db.add(mv)
        db.flush()
        job = db.get(TrainingJob, job_id)
        job.model_version_id = mv.id
        tq.add_log(db, job_id, f"Modelo v{version} registrado ({sum(e['status'] == 'ok' for e in exports)}/{len(exports)} formatos validados).")
        db.commit()
        model_version_id = mv.id
    shutil.rmtree(work, ignore_errors=True)
    final = {"test_accuracy": metrics["test_accuracy"], "test_loss": metrics["test_loss"],
             **{k: v for k, v in row.items() if k != "epoch"}}
    if all_ok:
        rep.update(force=True, status=J_COMPLETED, phase="Completado", percent=100.0, finished_at=utcnow(), eta_seconds=0,
                   metrics=final, model_version_id=model_version_id)
    else:
        failed = ", ".join(e["fmt"] for e in exports if e["status"] != "ok")
        rep.update(force=True, status=J_ERROR, phase="Exportación incompleta", percent=100.0, finished_at=utcnow(), eta_seconds=0,
                   error=f"Formatos que no pasaron la validación: {failed}. El modelo v{version} quedó disponible en los formatos válidos.",
                   model_version_id=model_version_id)
