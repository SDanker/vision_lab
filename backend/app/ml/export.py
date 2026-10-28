"""Exportación a 5 formatos con validación obligatoria (se compara contra el modelo Keras)."""
import shutil
import zipfile
from pathlib import Path

import numpy as np

from .architectures import custom_objects

TOLERANCE = 0.02  # diferencia máxima admitida entre probabilidades


def _check(ref: np.ndarray, out: np.ndarray, name: str) -> dict:
    out = np.asarray(out).reshape(ref.shape)
    diff = float(np.max(np.abs(ref - out)))
    same = bool(np.all(ref.argmax(1) == out.argmax(1)))
    if not same or diff > TOLERANCE:
        raise ValueError(f"La salida de {name} difiere del modelo original (diff máx {diff:.4f}, argmax igual: {same}).")
    return {"samples": int(ref.shape[0]), "max_abs_diff": round(diff, 6), "argmax_match": same}


def _export_keras(model, out: Path, ref, samples):
    from tensorflow import keras

    path = out / "model.keras"
    model.save(path)
    loaded = keras.models.load_model(path, compile=False, custom_objects=custom_objects())
    return path, _check(ref, loaded.predict(samples, verbose=0), "Keras")


def _export_h5(model, out: Path, ref, samples):
    from tensorflow import keras

    path = out / "model.h5"
    model.save(path, save_format="h5")
    loaded = keras.models.load_model(path, compile=False, custom_objects=custom_objects())
    return path, _check(ref, loaded.predict(samples, verbose=0), "HDF5")


def _export_savedmodel(model, out: Path, ref, samples):
    import tensorflow as tf

    tmp = out / "_saved_model"
    shutil.rmtree(tmp, ignore_errors=True)
    tf.saved_model.save(model, str(tmp))
    zpath = out / "saved_model.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(tmp.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(tmp).as_posix())
    # validar desde el ZIP, que es lo que se entrega
    check_dir = out / "_saved_model_check"
    shutil.rmtree(check_dir, ignore_errors=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(check_dir)
    loaded = tf.saved_model.load(str(check_dir))
    infer = loaded.signatures["serving_default"]
    res = infer(tf.constant(samples))
    detail = _check(ref, list(res.values())[0].numpy(), "SavedModel")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(check_dir, ignore_errors=True)
    return zpath, detail


def _export_tflite(model, out: Path, ref, samples):
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    data = converter.convert()
    path = out / "model.tflite"
    path.write_bytes(data)
    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    inp, outp = interp.get_input_details()[0], interp.get_output_details()[0]
    results = []
    for i in range(samples.shape[0]):
        interp.set_tensor(inp["index"], samples[i : i + 1])
        interp.invoke()
        results.append(interp.get_tensor(outp["index"])[0])
    return path, _check(ref, np.stack(results), "TFLite")


def _export_onnx(model, out: Path, ref, samples):
    import onnx
    import onnxruntime as ort
    import tensorflow as tf
    import tf2onnx

    path = out / "model.onnx"
    size = samples.shape[1]
    spec = (tf.TensorSpec((None, size, size, 3), tf.float32, name="image"),)
    tf2onnx.convert.from_keras(model, input_signature=spec, opset=15, output_path=str(path))
    onnx.checker.check_model(onnx.load(str(path)))
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    res = sess.run(None, {sess.get_inputs()[0].name: samples})[0]
    return path, _check(ref, res, "ONNX")


EXPORTERS = {
    "keras": _export_keras,
    "h5": _export_h5,
    "savedmodel": _export_savedmodel,
    "tflite": _export_tflite,
    "onnx": _export_onnx,
}


def export_all(model, out: Path, samples: np.ndarray, log=lambda m, level="info": None) -> list[dict]:
    """Devuelve una entrada por formato. Un formato sólo es 'ok' si su validación pasó."""
    out.mkdir(parents=True, exist_ok=True)
    ref = model.predict(samples, verbose=0)
    results = []
    for fmt, fn in EXPORTERS.items():
        log(f"Exportando {fmt}...")
        try:
            path, validation = fn(model, out, ref, samples)
            results.append(
                {"fmt": fmt, "status": "ok", "filename": path.name, "size_bytes": path.stat().st_size,
                 "validation": validation, "error": None}
            )
            log(f"Formato {fmt} validado (diff máx {validation['max_abs_diff']}).")
        except Exception as exc:  # noqa: BLE001 - un formato fallido no tumba a los demás
            results.append(
                {"fmt": fmt, "status": "failed", "filename": None, "size_bytes": None,
                 "validation": None, "error": f"{type(exc).__name__}: {exc}"[:1500]}
            )
            log(f"Falló la exportación {fmt}: {exc}", "error")
    return results


def build_bundle(out: Path) -> Path:
    """ZIP completo con los archivos ya validados y los metadatos."""
    bundle = out / "bundle.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.iterdir()):
            if f.is_file() and f.name != "bundle.zip":
                z.write(f, f.name)
    return bundle
