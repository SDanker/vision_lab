"""Catálogo de modelos base (Keras Applications, pesos ImageNet gratuitos).

Todos usan entrada float32 0-255 de NxNx3 (N configurable por proyecto): la normalización propia de cada
arquitectura va DENTRO del modelo, así la inferencia y los 5 formatos exportados
reciben siempre la misma entrada. Este módulo no importa TensorFlow al cargarse.
"""

# preprocessing: "builtin" = la arquitectura ya normaliza internamente;
#                "tf" = escala a [-1, 1]; "torch" = /255 y normalización media/desv. de ImageNet.
ARCHITECTURES: dict[str, dict] = {
    "mobilenetv3small": {
        "label": "MobileNetV3 Small", "keras": "MobileNetV3Small", "preprocessing": "builtin",
        "kwargs": {"include_preprocessing": True}, "size_mb": 4, "speed": "muy rápido",
        "description": "El más liviano. Ideal para celulares o equipos modestos; menor exactitud.",
    },
    "mobilenetv3": {
        "label": "MobileNetV3 Large", "keras": "MobileNetV3Large", "preprocessing": "builtin",
        "kwargs": {"include_preprocessing": True}, "size_mb": 12, "speed": "rápido",
        "description": "Buen equilibrio entre velocidad y exactitud. Recomendado para empezar.",
    },
    "mobilenetv2": {
        "label": "MobileNetV2", "keras": "MobileNetV2", "preprocessing": "tf",
        "kwargs": {}, "size_mb": 9, "speed": "rápido",
        "description": "Clásico, liviano y muy compatible con dispositivos móviles.",
    },
    "efficientnetb0": {
        "label": "EfficientNetB0", "keras": "EfficientNetB0", "preprocessing": "builtin",
        "kwargs": {}, "size_mb": 17, "speed": "medio",
        "description": "Más exacto que MobileNet con un costo moderado.",
    },
    "efficientnetv2b0": {
        "label": "EfficientNetV2 B0", "keras": "EfficientNetV2B0", "preprocessing": "builtin",
        "kwargs": {"include_preprocessing": True}, "size_mb": 25, "speed": "medio",
        "description": "Versión mejorada de EfficientNet: entrena más rápido y suele ser más exacta.",
    },
    "efficientnetb3": {
        "label": "EfficientNetB3", "keras": "EfficientNetB3", "preprocessing": "builtin",
        "kwargs": {}, "size_mb": 45, "speed": "lento",
        "description": "Alta exactitud para categorías muy parecidas entre sí; más lento en CPU.",
    },
    "resnet50v2": {
        "label": "ResNet50V2", "keras": "ResNet50V2", "preprocessing": "tf",
        "kwargs": {}, "size_mb": 95, "speed": "lento",
        "description": "Arquitectura robusta y muy probada; modelo grande.",
    },
    "densenet121": {
        "label": "DenseNet121", "keras": "DenseNet121", "preprocessing": "torch",
        "kwargs": {}, "size_mb": 30, "speed": "lento",
        "description": "Buena exactitud con pocos datos; inferencia más lenta.",
    },
    "efficientnetv2s": {
        "label": "EfficientNetV2 S", "keras": "EfficientNetV2S", "preprocessing": "builtin",
        "kwargs": {"include_preprocessing": True}, "size_mb": 82, "speed": "lento",
        "description": "La mejor relación exactitud/costo del catálogo; recomendado para categorías parecidas entre sí.",
    },
    "convnext_tiny": {
        "label": "ConvNeXt Tiny", "keras": "ConvNeXtTiny", "preprocessing": "builtin",
        "kwargs": {"include_preprocessing": True}, "depthwise": True, "size_mb": 110, "speed": "muy lento",
        "description": "Arquitectura moderna y muy exacta; pesada en CPU.",
    },
    "convnext_small": {
        "label": "ConvNeXt Small", "keras": "ConvNeXtSmall", "preprocessing": "builtin",
        "kwargs": {"include_preprocessing": True}, "depthwise": True, "size_mb": 190, "speed": "muy lento",
        "description": "La más exacta del catálogo; entrenamientos de muchas horas en CPU.",
    },
}

# Ajuste recomendado por modelo: alimenta el botón «Usar ajuste recomendado» de la configuración avanzada
# (se guarda como `model_overrides`, así que el usuario puede cambiarlo). Mismos campos que `ModelOverride`.
# Criterio: la cabeza converge en pocas épocas y el ajuste fino es lo que más mejora, pero en CPU cuesta mucho
# más por época, así que cuanto más lento el modelo, menos épocas de ajuste fino y menos paciencia.
# Medido en este servidor: MobileNetV3 (cabeza estable hacia la época 10; el ajuste fino seguía mejorando a las 15),
# 224 px superó a 260 px con la mitad del tiempo, y ConvNeXt Small igualó a MobileNetV3 en 26 h contra 1 h.
# Sin medición propia todavía (criterio general): el resto de los modelos.
# `unfreeze_layers` / `fine_tune_lr_factor` (capas finales que se descongelan en el ajuste fino y tasa = learning_rate
# × factor): antes eran 30 y 0,01 para todos, pero «30 capas» es el 9 % de los pesos en DenseNet121 y EfficientNetV2-S
# y el 69 % en MobileNetV2. Se fijaron para descongelar ≈ 30 % de los pesos (lo que MobileNetV3 Large, el único
# medido, ya hacía con 30 capas) y algo menos en los lentos, para que el ajuste fino no domine el tiempo. ConvNeXt
# (LayerNorm, entrenado con AdamW) lleva una tasa algo mayor porque con 1e-5 mejoraba muy lento (medido).
# EfficientNetB3 / V2-S se diseñaron para ≥300 px y son los que más pierden por debajo; 260 px es el
# compromiso entre exactitud y tiempo. Los demás van a 224 px (su resolución nativa).
TUNING: dict[str, dict] = {
    "mobilenetv3small": {"epochs": 10, "fine_tune_epochs": 12, "patience": 4, "batch_size": 32, "learning_rate": 1e-3, "image_size": 224, "unfreeze_layers": 25, "fine_tune_lr_factor": 0.01},
    "mobilenetv3": {"epochs": 10, "fine_tune_epochs": 15, "patience": 4, "batch_size": 32, "learning_rate": 1e-3, "image_size": 224, "unfreeze_layers": 30, "fine_tune_lr_factor": 0.01},
    "mobilenetv2": {"epochs": 10, "fine_tune_epochs": 12, "patience": 4, "batch_size": 32, "learning_rate": 1e-3, "image_size": 224, "unfreeze_layers": 10, "fine_tune_lr_factor": 0.01},
    "efficientnetb0": {"epochs": 8, "fine_tune_epochs": 10, "patience": 3, "batch_size": 32, "learning_rate": 1e-3, "image_size": 224, "unfreeze_layers": 20, "fine_tune_lr_factor": 0.01},
    "efficientnetv2b0": {"epochs": 8, "fine_tune_epochs": 10, "patience": 3, "batch_size": 32, "learning_rate": 1e-3, "image_size": 224, "unfreeze_layers": 40, "fine_tune_lr_factor": 0.01},
    "efficientnetb3": {"epochs": 6, "fine_tune_epochs": 8, "patience": 3, "batch_size": 32, "learning_rate": 1e-3, "image_size": 260, "unfreeze_layers": 20, "fine_tune_lr_factor": 0.01},
    "efficientnetv2s": {"epochs": 6, "fine_tune_epochs": 8, "patience": 3, "batch_size": 32, "learning_rate": 1e-3, "image_size": 260, "unfreeze_layers": 70, "fine_tune_lr_factor": 0.01},
    "resnet50v2": {"epochs": 8, "fine_tune_epochs": 8, "patience": 3, "batch_size": 32, "learning_rate": 1e-3, "image_size": 224, "unfreeze_layers": 20, "fine_tune_lr_factor": 0.01},
    "densenet121": {"epochs": 8, "fine_tune_epochs": 8, "patience": 3, "batch_size": 32, "learning_rate": 1e-3, "image_size": 224, "unfreeze_layers": 70, "fine_tune_lr_factor": 0.01},
    "convnext_tiny": {"epochs": 10, "fine_tune_epochs": 6, "patience": 3, "batch_size": 32, "learning_rate": 1e-3, "image_size": 224, "unfreeze_layers": 15, "fine_tune_lr_factor": 0.03},
    "convnext_small": {"epochs": 10, "fine_tune_epochs": 5, "patience": 3, "batch_size": 32, "learning_rate": 1e-3, "image_size": 224, "unfreeze_layers": 20, "fine_tune_lr_factor": 0.03},
}

DEFAULT_FINE_TUNE = {"unfreeze_layers": 30, "fine_tune_lr_factor": 0.01}


def fine_tune_settings(name: str, cfg: dict | None = None) -> tuple[int, float]:
    """(capas finales a descongelar, factor de la tasa del ajuste fino). Lo indicado en `cfg` manda;
    si no, el valor propio de la arquitectura (`TUNING`) y, para un modelo desconocido, el de siempre."""
    base = {**DEFAULT_FINE_TUNE, **{k: v for k, v in TUNING.get(name, {}).items() if k in DEFAULT_FINE_TUNE}}
    cfg = cfg or {}
    return (
        int(cfg.get("unfreeze_layers") or base["unfreeze_layers"]),
        float(cfg.get("fine_tune_lr_factor") or base["fine_tune_lr_factor"]),
    )

DEFAULT_ARCH = "mobilenetv3"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def tabular_models() -> dict[str, dict]:
    """Modelos de scikit-learn para datos tabulares (import diferido: este módulo no debe depender de otros del paquete)."""
    from .tabular import TABULAR_MODELS

    return TABULAR_MODELS


def all_model_ids() -> list[str]:
    return [*ARCHITECTURES, *tabular_models()]


def is_tabular_model(name: str) -> bool:
    return name in tabular_models()


def label_of(name: str) -> str:
    if name in ARCHITECTURES:
        return ARCHITECTURES[name]["label"]
    return tabular_models().get(name, {}).get("label", name)


def public_catalog() -> list[dict]:
    from .tabular import public_catalog as tabular_catalog

    return [
        {"id": k, **{f: v[f] for f in ("label", "description", "size_mb", "speed")}, "modality": "image", "tuning": TUNING[k]}
        for k, v in ARCHITECTURES.items()
    ] + tabular_catalog()


def build_backbone(name: str, shape: tuple[int, int, int], weights: str | None):
    """Devuelve (capas_de_preprocesado, modelo_base)."""
    from tensorflow import keras

    if name not in ARCHITECTURES:
        raise ValueError(f"Modelo base no soportado: {name}")
    spec = ARCHITECTURES[name]
    fn = getattr(keras.applications, spec["keras"])
    base = fn(input_shape=shape, include_top=False, weights=weights, **spec["kwargs"])
    if spec.get("depthwise"):
        base = grouped_to_depthwise(base)
    pre = []
    if spec["preprocessing"] == "tf":
        pre = [keras.layers.Rescaling(1 / 127.5, offset=-1.0, name="preprocess_tf")]
    elif spec["preprocessing"] == "torch":
        pre = [
            keras.layers.Rescaling(1 / 255.0, name="preprocess_scale"),
            keras.layers.Normalization(
                mean=list(IMAGENET_MEAN), variance=[s * s for s in IMAGENET_STD], name="preprocess_norm"
            ),
        ]
    return pre, base


def custom_objects() -> dict:
    """Capas propias de ConvNeXt, necesarias para recargar modelos .h5/.keras."""
    from keras.src.applications.convnext import LayerScale, StochasticDepth

    return {"LayerScale": LayerScale, "StochasticDepth": StochasticDepth}


def grouped_to_depthwise(model):
    """ConvNeXt usa Conv2D con groups=canales, que Keras compila en un bloque que tf2onnx no
    puede convertir. Se reemplazan por DepthwiseConv2D, matemáticamente equivalente, con los
    mismos pesos: así se exporta a ONNX/TFLite sin cambiar los resultados."""
    from tensorflow import keras

    def swap(layer):
        cfg = layer.get_config()
        if (
            isinstance(layer, keras.layers.Conv2D)
            and not isinstance(layer, keras.layers.DepthwiseConv2D)
            and cfg.get("groups", 1) > 1
            and cfg["groups"] == cfg["filters"]
        ):
            return keras.layers.DepthwiseConv2D(
                kernel_size=cfg["kernel_size"], strides=cfg["strides"], padding=cfg["padding"],
                use_bias=cfg["use_bias"], name=cfg["name"],
            )
        return layer.__class__.from_config(cfg)

    new = keras.models.clone_model(model, clone_function=swap)
    for src in model.layers:
        dst = new.get_layer(src.name)
        weights = src.get_weights()
        if isinstance(dst, keras.layers.DepthwiseConv2D) and not isinstance(src, keras.layers.DepthwiseConv2D):
            k = weights[0]  # (kh, kw, 1, C) -> (kh, kw, C, 1): mismo orden en memoria
            weights[0] = k.reshape(k.shape[0], k.shape[1], k.shape[3], 1)
        dst.set_weights(weights)
    return new
