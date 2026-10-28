"""Inferencia en la API: ONNX Runtime (ligero) con respaldo a Keras."""
import os
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

from ..config import get_settings
from ..ml.preprocess import load_array, to_model_input

_cache: "OrderedDict[str, object]" = OrderedDict()
_lock = threading.Lock()
MAX_CACHED = 4  # modelo + embedder de dos modelos distintos


def _session(model_dir: Path, fmt: str):
    path = model_dir / {"onnx": "model.onnx", "embedder": "embedder.onnx"}.get(fmt, "model.keras")
    key = f"{path}:{path.stat().st_mtime_ns}"
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    if fmt in ("onnx", "embedder"):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        if fmt == "embedder":  # se usa para calcular miles de vectores en un worker de 1 GB: sin la reserva de memoria «por si acaso»
            so.enable_cpu_mem_arena = False
            so.enable_mem_pattern = False
        obj = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
    else:
        from tensorflow import keras

        from ..ml.architectures import custom_objects

        obj = keras.models.load_model(path, compile=False, custom_objects=custom_objects())
    with _lock:
        _cache[key] = obj
        while len(_cache) > MAX_CACHED:
            _cache.popitem(last=False)
    return obj


def evict(model_dir: str) -> None:
    """Libera de la caché las sesiones de un modelo que se acaba de borrar."""
    prefix = str(Path(model_dir)) + os.sep  # con el separador: «v1» no debe tocar «v10»
    with _lock:
        for key in [k for k in _cache if k.startswith(prefix)]:
            del _cache[key]


MAX_AUDIO_CLIPS = 60 # un audio de prueba se analiza como máximo en 60 segmentos


def _run(model_dir: str, x: np.ndarray, onnx_ok: bool, keras_ok: bool) -> tuple[np.ndarray, str]:
    """Probabilidades para un lote (N, S, S, 3)."""
    d = Path(model_dir)
    if onnx_ok:
        sess = _session(d, "onnx")
        return sess.run(None, {sess.get_inputs()[0].name: x})[0], "onnxruntime"
    if keras_ok:
        return _session(d, "keras").predict(x, verbose=0), "keras"
    raise RuntimeError("El modelo no tiene un formato ejecutable válido (ONNX o Keras).")


def _result(labels: list[str], probs: np.ndarray, engine: str) -> dict:
    probs = np.asarray(probs, dtype=np.float64)
    order = np.argsort(probs)[::-1]
    scores = [{"label": labels[i], "score": float(probs[i])} for i in order]
    top, second = scores[0]["score"], scores[1]["score"] if len(scores) > 1 else 0.0
    threshold = get_settings().low_confidence_threshold
    low = top < threshold
    return {
        "prediction": scores[0]["label"],
        "confidence": top,
        "scores": scores,
        "low_confidence": low,
        "warning": (
            f"Puntuación insuficiente ({top * 100:.1f}% < {threshold * 100:.0f}%): la clasificación no es confiable."
            if low
            else None
        ),
        "margin": top - second,
        "engine": engine,
    }


def predict(model_dir: str, labels: list[str], image_bytes: bytes, onnx_ok: bool, keras_ok: bool, size: int = 224) -> dict:
    probs, engine = _run(model_dir, to_model_input(load_array(image_bytes, size)), onnx_ok, keras_ok)
    return _result(labels, probs[0], engine)


def predict_audio(
    model_dir: str, labels: list[str], audio_bytes: bytes, onnx_ok: bool, keras_ok: bool,
    size: int = 224, clip_seconds: float = 5.0,
) -> dict:
    """Corta el audio en clips, clasifica cada uno y promedia; también devuelve cada segmento."""
    from ..ml import audio_features as af

    signal = af.load_audio(audio_bytes, max_seconds=MAX_AUDIO_CLIPS * clip_seconds)
    clips = af.split_clips(signal, clip_seconds, max_clips=MAX_AUDIO_CLIPS)
    x = np.stack([af.spectrogram_image(c, size) for c in clips]).astype(np.float32)
    probs, engine = _run(model_dir, x, onnx_ok, keras_ok)
    out = _result(labels, probs.mean(axis=0), engine)
    out["duration_s"] = round(len(signal) / af.SAMPLE_RATE, 2)
    out["segments"] = [
        {"start": round(k * clip_seconds, 2), "end": round(min((k + 1) * clip_seconds, len(signal) / af.SAMPLE_RATE), 2),
         "label": labels[int(p.argmax())], "confidence": float(p.max())}
        for k, p in enumerate(probs)
    ]
    return out
