INFERENCE_PY = '''#!/usr/bin/env python3
"""Inferencia con el modelo exportado por Vision Lab AI.

Uso:
    pip install pillow numpy onnxruntime          # para ONNX
    python inference.py imagen.jpg                # usa model.onnx
    python inference.py imagen.jpg --format tflite    # requiere tflite-runtime o tensorflow
    python inference.py imagen.jpg --format keras     # requiere tensorflow

Entrada del modelo: float32 NHWC (1, N, N, 3) con valores 0-255; N está en metadata.json.
La normalización está incluida dentro del modelo.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
try:
    SIZE = json.loads((HERE / "metadata.json").read_text(encoding="utf-8"))["input"]["shape"][1]
except (OSError, KeyError, ValueError):
    SIZE = 224


def load_image(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
        im = im.resize((SIZE, SIZE), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32)[None, ...]


def predict_onnx(x):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(HERE / "model.onnx"), providers=["CPUExecutionProvider"])
    return sess.run(None, {sess.get_inputs()[0].name: x})[0][0]


def predict_tflite(x):
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter
    it = Interpreter(model_path=str(HERE / "model.tflite"))
    it.allocate_tensors()
    it.set_tensor(it.get_input_details()[0]["index"], x)
    it.invoke()
    return it.get_tensor(it.get_output_details()[0]["index"])[0]


def predict_keras(x):
    from tensorflow import keras

    return keras.models.load_model(str(HERE / "model.keras"), compile=False).predict(x, verbose=0)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--format", choices=["onnx", "tflite", "keras"], default="onnx")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()
    labels = json.loads((HERE / "labels.json").read_text(encoding="utf-8"))["labels"]
    probs = {"onnx": predict_onnx, "tflite": predict_tflite, "keras": predict_keras}[args.format](load_image(args.image))
    order = np.argsort(probs)[::-1][: args.top]
    for i in order:
        print(f"{labels[i]:<30} {probs[i] * 100:6.2f}%")


if __name__ == "__main__":
    main()
'''

README = """# {project} — modelo v{version}

Generado por Vision Lab AI el {created}.

| Dato | Valor |
|---|---|
| Modelo base | {base_model} |
| Categorías | {labels} |
| Entrada | float32 NHWC `(1, {size}, {size}, 3)`, valores 0-255 (normalización incluida) |
| Salida | probabilidades softmax por categoría, en el orden de `labels.json` |
| Exactitud en test | {accuracy} |

## Archivos
- `model.keras`, `model.h5`, `saved_model.zip`, `model.tflite`, `model.onnx`: el mismo modelo en 5 formatos.
- `labels.json`: orden de las categorías. `metadata.json`: especificaciones. `metrics.json`: métricas de evaluación.
- `inference.py`: ejemplo de inferencia (`python inference.py foto.jpg`).

## Cargar en Keras
`keras.models.load_model("model.keras", compile=False)`. Para modelos ConvNeXt con `.h5`, pasa
`custom_objects={{"LayerScale": ..., "StochasticDepth": ...}}` de `keras.src.applications.convnext`
(o usa `.keras`, ONNX o TFLite, que no lo requieren).

## Preprocesado
Convertir a RGB, aplicar la orientación EXIF, redimensionar a {size}x{size} (bilineal) y enviar como float32 0-255.
"""

INFERENCE_AUDIO_PY = '''#!/usr/bin/env python3
"""Inferencia de audio con el modelo exportado por Vision Lab AI.

Uso:
    pip install numpy pillow onnxruntime         # y ffmpeg instalado para MP3/OGG/M4A... (WAV no lo requiere)
    python inference.py grabacion.wav
    python inference.py grabacion.mp3 --format tflite

El audio se convierte a mono 16 kHz, se corta en clips (ver metadata.json), cada clip se transforma
en un espectrograma log-mel con audio_features.py (el mismo código usado al entrenar) y la
predicción final es el promedio de las probabilidades de todos los clips.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import audio_features as af

HERE = Path(__file__).resolve().parent
META = json.loads((HERE / "metadata.json").read_text(encoding="utf-8"))
SIZE = META["input"]["shape"][1]
CLIP = META["audio"]["clip_seconds"]


def batches(path, max_clips):
    signal = af.load_audio(path)
    clips = af.split_clips(signal, CLIP, max_clips=max_clips)
    return np.stack([af.spectrogram_image(c, SIZE) for c in clips]).astype(np.float32)


def run_onnx(x):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(HERE / "model.onnx"), providers=["CPUExecutionProvider"])
    return sess.run(None, {sess.get_inputs()[0].name: x})[0]


def run_tflite(x):
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter
    it = Interpreter(model_path=str(HERE / "model.tflite"))
    it.allocate_tensors()
    out = []
    for i in range(len(x)):
        it.set_tensor(it.get_input_details()[0]["index"], x[i : i + 1])
        it.invoke()
        out.append(it.get_tensor(it.get_output_details()[0]["index"])[0])
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--format", choices=["onnx", "tflite"], default="onnx")
    ap.add_argument("--max-clips", type=int, default=60)
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()
    labels = json.loads((HERE / "labels.json").read_text(encoding="utf-8"))["labels"]
    x = batches(args.audio, args.max_clips)
    probs = {"onnx": run_onnx, "tflite": run_tflite}[args.format](x)
    mean = probs.mean(axis=0)
    for i in np.argsort(mean)[::-1][: args.top]:
        print(f"{labels[i]:<30} {mean[i] * 100:6.2f}%")
    if len(probs) > 1:
        print()
        print("Por segmento:")
        for k, p in enumerate(probs):
            j = int(p.argmax())
            print(f"  {k * CLIP:7.1f}-{(k + 1) * CLIP:7.1f} s  {labels[j]}  ({p[j] * 100:.1f}%)")


if __name__ == "__main__":
    main()
'''

INFERENCE_TABULAR_PY = '''#!/usr/bin/env python3
"""Inferencia con un modelo de datos tabulares exportado por Vision Lab AI.

Uso:
    pip install numpy onnxruntime                 # para ONNX (recomendado)
    python inference.py fila.json                 # una fila: {"columna": valor, ...}
    python inference.py datos.csv                 # varias filas: CSV con los nombres de columna
    python inference.py datos.csv --format joblib # requiere: pip install scikit-learn joblib pandas

Las columnas que el modelo usa están en metadata.json (tabular.features). Un valor vacío se reemplaza por la mediana
(columnas numéricas) o por «sin dato» (categóricas); una categoría que el modelo no vio al entrenar cuenta como «otra».
La conversión de cada fila es la de tabular_features.py, el mismo código usado al entrenar.
"""
import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np

import tabular_features as tf

HERE = Path(__file__).resolve().parent
META = json.loads((HERE / "metadata.json").read_text(encoding="utf-8"))
FEATURES = META["tabular"]["features"]
LABELS = json.loads((HERE / "labels.json").read_text(encoding="utf-8"))["labels"]


def load_rows(path):
    p = Path(path)
    text = p.read_text(encoding="utf-8-sig")
    if p.suffix.lower() == ".json":
        data = json.loads(text)
        return [data] if isinstance(data, dict) else list(data)
    first = text.splitlines()[0] if text.strip() else ""
    delimiter = max(",;\t|", key=first.count)
    return list(csv.DictReader(io.StringIO(text), delimiter=delimiter))


def run_onnx(inputs):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(HERE / "model.onnx"), providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    out = sess.run(None, inputs)
    return out[names.index("probabilities")] if "probabilities" in names else out[1]


def run_joblib(inputs):
    import joblib
    import pandas as pd

    pipeline = joblib.load(HERE / "model.joblib")["pipeline"]
    return pipeline.predict_proba(pd.DataFrame({k: v.reshape(-1) for k, v in inputs.items()}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="archivo .json (una fila u objetos) o .csv")
    ap.add_argument("--format", choices=["onnx", "joblib"], default="onnx")
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()
    rows = load_rows(args.input)
    probs = {"onnx": run_onnx, "joblib": run_joblib}[args.format](tf.build_inputs(rows, FEATURES))
    for k, p in enumerate(probs, 1):
        if len(probs) > 1:
            print(f"Fila {k}:")
        for i in np.argsort(p)[::-1][: args.top]:
            print(f"{LABELS[i]:<30} {p[i] * 100:6.2f}%")


if __name__ == "__main__":
    main()
'''

README_TABULAR = """# {project} — modelo v{version}

Generado por Vision Lab AI el {created}.

| Dato | Valor |
|---|---|
| Modelo base | {base_model} |
| Columna a predecir | {target} |
| Categorías | {labels} |
| Columnas de entrada | {features} |
| Salida | probabilidades por categoría, en el orden de `labels.json` |
| Exactitud en prueba | {accuracy} |

## Archivos
- `model.onnx`: el modelo completo (preprocesado incluido) para ONNX Runtime. `model.joblib`: el mismo modelo de scikit-learn.
- `labels.json`: orden de las categorías. `metadata.json`: columnas de entrada (tipo, categorías conocidas, mediana). `metrics.json`: métricas.
- `inference.py` y `tabular_features.py`: ejemplo de uso (`python inference.py fila.json`).

## Entradas del modelo
Cada columna `i` de `metadata.json > tabular.features` es una entrada `c{{i}}` de forma `(N, 1)`: float32 (numéricas) o texto (categóricas).
`tabular_features.build_inputs(filas, features)` las construye a partir de filas `{{columna: valor}}`: los vacíos de una columna
numérica se reemplazan por su mediana y los de una categórica por «__missing__»; una categoría desconocida es «__other__».

## joblib
`joblib.load("model.joblib")` devuelve `{{"pipeline", "labels", "features"}}`. Requiere la misma versión mayor de scikit-learn con la que se
entrenó (ver `metadata.json > tabular.sklearn`). Sólo cargues archivos joblib de origen confiable.
"""
