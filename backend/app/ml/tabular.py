"""Datos tabulares: lectura de CSV, perfil de columnas, modelos de scikit-learn, métricas y exportación.

No importa TensorFlow ni la base de datos: lo usan la API (perfil y predicción) y el worker de entrenamiento.
La conversión de filas en entradas del modelo vive en `tabular_features.py` (se entrega junto al modelo).
"""
import io
import json
import threading
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

from . import tabular_features as feat

MAX_CLASSES = 100  # columnas objetivo con más valores distintos no sirven como categorías
MAX_CATEGORIES = 50  # valores distintos que se conservan por columna categórica; el resto pasa a «__other__»
MAX_NAME_LEN = 100
NUMERIC_SHARE = 0.95  # fracción de valores que deben ser números para tratar la columna como numérica
TOLERANCE = 0.02  # diferencia máxima admitida entre las probabilidades de sklearn y las del formato exportado

# Modelos disponibles para datos tabulares (scikit-learn). Aparte de `ARCHITECTURES`, que son redes de visión.
TABULAR_MODELS: dict[str, dict] = {
    "random_forest": {
        "label": "Random Forest", "size_mb": 5, "speed": "rápido",
        "description": "Muchos árboles de decisión que votan. Robusto y casi sin ajustes: buen punto de partida.",
    },
    "gradient_boosting": {
        "label": "Gradient Boosting", "size_mb": 2, "speed": "medio",
        "description": "Árboles que corrigen los errores de los anteriores. Suele ser el más exacto en tablas.",
    },
    "logistic_regression": {
        "label": "Regresión logística", "size_mb": 1, "speed": "muy rápido",
        "description": "Modelo lineal simple. Sirve de referencia y se explica fácilmente.",
    },
    "mlp": {
        "label": "Red neuronal (MLP)", "size_mb": 1, "speed": "medio",
        "description": "Red neuronal pequeña. Puede captar relaciones complejas si hay muchas filas.",
    },
}
DEFAULT_TABULAR_MODEL = "random_forest"
_SCALED = ("logistic_regression", "mlp")  # los modelos lineales y las redes necesitan las columnas numéricas escaladas

# segundos por fila y característica (referencia del tiempo de ajuste; el resto es costo fijo de exportar)
_FIT_RATE = {"random_forest": 4e-5, "gradient_boosting": 3e-5, "logistic_regression": 1e-5, "mlp": 1.2e-4}


def public_catalog() -> list[dict]:
    return [
        {"id": k, **{f: v[f] for f in ("label", "description", "size_mb", "speed")}, "modality": "tabular", "tuning": {}}
        for k, v in TABULAR_MODELS.items()
    ]


def estimate_seconds(model_id: str, n_rows: int, n_features: int) -> float:
    return 30.0 + _FIT_RATE.get(model_id, 5e-5) * max(n_rows, 1) * max(n_features, 1) * 20


class InvalidTable(Exception):
    """El archivo o la entrada no es una tabla utilizable; el mensaje se muestra tal cual a la persona."""


# ------------------------------------------------------------------------------------------- lectura
def _decode(data: bytes) -> str:
    if b"\x00" in data[:65536]:
        raise InvalidTable("El archivo no es de texto (parece binario). Sube un CSV.")
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _delimiter(text: str) -> str:
    lines = text[:16384].splitlines()
    first = lines[0] if lines else ""
    counts = {d: first.count(d) for d in (",", ";", "\t", "|")}
    best = max(counts, key=counts.get)
    return best if counts[best] else ","


def read_table(data: bytes, max_rows: int, max_cols: int):
    """CSV/TSV -> DataFrame de textos (la primera línea son los nombres de columna)."""
    import pandas as pd

    if not data:
        raise InvalidTable("Archivo vacío.")
    text = _decode(data)
    if not text.strip():
        raise InvalidTable("Archivo vacío.")
    try:
        raw = pd.read_csv(
            io.StringIO(text), sep=_delimiter(text), header=None, dtype=str, keep_default_na=False, na_filter=False,
            nrows=max_rows + 2, skip_blank_lines=True,
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
        raise InvalidTable(f"No se pudo leer como tabla CSV: {str(exc)[:160]}") from None
    header = [str(h).strip() for h in raw.iloc[0]]
    if len(header) < 2:
        raise InvalidTable("La tabla necesita al menos 2 columnas (¿el separador es coma, punto y coma o tabulación?).")
    if len(header) > max_cols:
        raise InvalidTable(f"La tabla tiene {len(header)} columnas; el máximo es {max_cols}.")
    header = [h or f"columna_{i + 1}" for i, h in enumerate(header)]
    if any(len(h) > MAX_NAME_LEN for h in header):
        raise InvalidTable(f"Hay un nombre de columna de más de {MAX_NAME_LEN} caracteres.")
    dup = sorted(h for h, c in Counter(header).items() if c > 1)
    if dup:
        raise InvalidTable(f"Hay columnas repetidas en el encabezado: {', '.join(dup[:5])}.")
    body = raw.iloc[1:].reset_index(drop=True)
    body.columns = header
    body = body[(body != "").any(axis=1)].reset_index(drop=True)  # líneas completamente vacías
    if len(body) == 0:
        raise InvalidTable("La tabla no tiene filas de datos (sólo el encabezado).")
    if len(body) > max_rows:
        raise InvalidTable(f"La tabla tiene demasiadas filas (máximo {max_rows:,}).".replace(",", "."))
    return body


# --------------------------------------------------------------------------------------------- perfil
def _split_column(series):
    """(valores no vacíos, serie numérica de ellos, ¿es numérica?) de una columna de textos."""
    import pandas as pd

    s = series.str.strip()
    nn = s[~s.str.lower().isin(feat._NULLS)]
    if len(nn) == 0:
        return nn, pd.Series([], dtype="float64"), False
    num = pd.to_numeric(nn.str.replace(",", ".", regex=False), errors="coerce")
    num = num[np.isfinite(num)]  # lo que no es número (NaN) o es infinito no cuenta
    return nn, num, bool(len(num) / len(nn) >= NUMERIC_SHARE)


def _counts(nn) -> Counter:
    out: Counter = Counter()
    for value, count in nn.value_counts().items():
        out[feat.normalize_label(value)] += int(count)
    return out


def profile(df) -> dict:
    """Resumen de la tabla para mostrar en la interfaz y decidir columnas."""
    n = len(df)
    columns = []
    for name in df.columns:
        nn, num, numeric = _split_column(df[name])
        counts = _counts(nn)
        unique = len(counts)
        col = {
            "name": name, "kind": "numeric" if numeric else "categorical", "unique": unique,
            "missing": int(n - len(nn)), "missing_pct": round((n - len(nn)) / max(n, 1), 4),
            "examples": [v[:40] for v in nn.drop_duplicates().head(3).tolist()],
            "min": None, "max": None, "mean": None,
            "values": [{"value": k, "count": c} for k, c in counts.most_common(MAX_CLASSES)] if unique <= MAX_CLASSES else None,
            "constant": unique <= 1,
        }
        if numeric and len(num):
            col.update(min=round(float(num.min()), 6), max=round(float(num.max()), 6), mean=round(float(num.mean()), 6))
        # una columna casi sin repeticiones (nombres, identificadores) no sirve para predecir
        ids = n >= 20 and len(nn) > 0 and unique / len(nn) >= 0.9
        if ids and numeric:
            ids = bool((num % 1 == 0).all() and num.is_monotonic_increasing)
        col["id_like"] = bool(ids)
        col["target_ok"] = bool(2 <= unique <= MAX_CLASSES and unique < 0.5 * max(len(nn), 1))
        columns.append(col)
    preview = df.head(10)
    return {
        "n_rows": n,
        "columns": columns,
        "preview": {"columns": list(df.columns), "rows": [[str(v)[:60] for v in row] for row in preview.itertuples(index=False)]},
    }


def suggest_features(columns: list[dict], target: str | None) -> list[str]:
    return [
        c["name"] for c in columns
        if c["name"] != target and not c["id_like"] and not c["constant"] and c["missing_pct"] < 0.6
    ]


def describe_features(df, names: list[str], rows=None) -> list[dict]:
    """Especificación de cada característica: tipo, categorías conocidas (o estadísticas) y mediana.

    Se calcula con las filas de entrenamiento (`rows`); es lo que viaja en metadata.json y reproduce la
    conversión de una fila en las entradas del modelo.
    """
    out = []
    for name in names:
        series = df[name]
        _, _, numeric = _split_column(series)  # el tipo se decide con toda la tabla, como en el perfil
        part = series if rows is None else series.iloc[rows]
        nn, num, _ = _split_column(part)
        if numeric:
            median = float(num.median()) if len(num) else 0.0
            out.append({
                "name": name, "kind": "numeric", "median": round(median, 6),
                "min": round(float(num.min()), 6) if len(num) else 0.0, "max": round(float(num.max()), 6) if len(num) else 0.0,
            })
        else:
            top = [k for k, _ in _counts(nn).most_common(MAX_CATEGORIES)]
            out.append({"name": name, "kind": "categorical", "categories": top, "default": top[0] if top else feat.MISSING})
    return out


# ------------------------------------------------------------------------------------ entrenamiento
def build_frame(df, features: list[dict]):
    """DataFrame con las columnas internas c0..cN del modelo a partir de la tabla de textos."""
    import pandas as pd

    return pd.DataFrame({f"c{i}": feat.column_array(df[f["name"]].tolist(), f) for i, f in enumerate(features)})


def frame_from_inputs(inputs: dict[str, np.ndarray]):
    import pandas as pd

    return pd.DataFrame({k: v.reshape(-1) for k, v in inputs.items()})


def split_indices(y: np.ndarray, n_classes: int, seed: int = 42, val_frac: float = 0.15, test_frac: float = 0.15, labels=None):
    """División estratificada train/val/test por clase, determinista por semilla."""
    rng = np.random.default_rng(seed)
    out = {"train": [], "val": [], "test": []}
    for k in range(n_classes):
        idx = np.flatnonzero(y == k)
        idx = idx[rng.permutation(len(idx))]
        n = len(idx)
        n_test, n_val = max(1, round(n * test_frac)), max(1, round(n * val_frac))
        n_train = n - n_val - n_test
        if n_train < 1:
            name = labels[k] if labels else k
            raise ValueError(f"La clase '{name}' tiene muy pocas filas ({n}) para dividir el dataset.")
        out["train"].append(idx[:n_train])
        out["val"].append(idx[n_train : n_train + n_val])
        out["test"].append(idx[n_train + n_val :])
    return {k: np.sort(np.concatenate(v)) for k, v in out.items()}


def build_pipeline(model_id: str, features: list[dict], n_jobs: int = 1):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num = [f"c{i}" for i, f in enumerate(features) if f["kind"] == "numeric"]
    cat = [(f"c{i}", list(f["categories"]) + [feat.MISSING, feat.OTHER]) for i, f in enumerate(features) if f["kind"] == "categorical"]
    parts = []
    if num:
        parts.append(("num", StandardScaler() if model_id in _SCALED else "passthrough", num))
    if cat:
        parts.append((
            "cat", OneHotEncoder(categories=[c for _, c in cat], handle_unknown="ignore", sparse_output=False), [n for n, _ in cat],
        ))
    pre = ColumnTransformer(parts, remainder="drop", sparse_threshold=0.0)
    if model_id == "random_forest":
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", n_jobs=n_jobs, random_state=42)
    elif model_id == "gradient_boosting":
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, early_stopping=True, n_iter_no_change=20, class_weight="balanced", random_state=42
        )
    elif model_id == "logistic_regression":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    elif model_id == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True, n_iter_no_change=15, max_iter=500, random_state=42)
    else:
        raise ValueError(f"Modelo base no soportado para datos tabulares: {model_id}")
    return Pipeline([("pre", pre), ("clf", clf)])


def fit(pipe, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # avisos de convergencia: se informan con las métricas, no como ruido
        pipe.fit(X, y)
    return pipe


def metrics_from_probs(probs: np.ndarray, y_true: np.ndarray, labels: list[str]) -> dict:
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
        per_class.append({"label": name, "precision": round(float(precision), 4), "recall": round(float(recall), 4),
                          "f1": round(float(f1), 4), "support": int(cm[i, :].sum())})
    return {
        "test_accuracy": round(acc, 4), "test_loss": round(loss, 4), "test_top5_accuracy": round(top5, 4), "top_k": topk,
        "macro_f1": round(float(np.mean([c["f1"] for c in per_class])), 4),
        "per_class": per_class, "confusion_matrix": cm.tolist(),
    }


def feature_importance(pipe, X, y, features: list[dict], top: int = 15) -> list[dict]:
    """Cuánto empeora la exactitud al desordenar cada columna (importancia por permutación, sobre validación)."""
    from sklearn.inspection import permutation_importance

    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(len(y), size=min(len(y), 1500), replace=False))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = permutation_importance(pipe, X.iloc[idx], y[idx], n_repeats=3, random_state=42, n_jobs=1, scoring="accuracy")
    order = np.argsort(-r.importances_mean)[:top]
    return [{"feature": features[int(i)]["name"], "importance": round(float(max(r.importances_mean[i], 0.0)), 4)} for i in order
            if r.importances_mean[i] > 0]


def training_hints(history: list[dict], test_acc: float, majority: float, n_train: int) -> list[str]:
    hints: list[str] = []
    row = history[-1] if history else {}
    gap = row.get("accuracy", 0) - row.get("val_accuracy", 0)
    if gap > 0.15:
        hints.append(
            f"Posible sobreajuste: la exactitud en entrenamiento supera a la de validación por {gap * 100:.0f} puntos. "
            "Prueba con más filas, menos columnas o un modelo más simple (regresión logística)."
        )
    if test_acc <= majority + 0.02:
        hints.append(
            f"El modelo no supera a responder siempre la clase más común ({majority * 100:.0f}%). Revisa que las columnas "
            "elegidas tengan relación con la columna objetivo y que haya suficientes filas por clase."
        )
    if n_train < 200:
        hints.append(f"Hay pocas filas de entrenamiento ({n_train}): las métricas pueden variar mucho. Agrega más datos si puedes.")
    return hints


# ---------------------------------------------------------------------------------------- exportación
def _agree(ref: np.ndarray, out: np.ndarray, name: str) -> dict:
    out = np.asarray(out, dtype=np.float64).reshape(ref.shape)
    diff = float(np.max(np.abs(ref - out)))
    srt = np.sort(ref, axis=1)
    margin = srt[:, -1] - srt[:, -2] if ref.shape[1] > 1 else np.ones(len(ref))
    same = bool(np.all((ref.argmax(1) == out.argmax(1)) | (margin < 1e-3)))  # un empate real no cuenta como diferencia
    if not same or diff > TOLERANCE:
        raise ValueError(f"La salida de {name} difiere del modelo original (diff máx {diff:.4f}, argmax igual: {same}).")
    return {"samples": int(ref.shape[0]), "max_abs_diff": round(diff, 6), "argmax_match": same}


def _export_joblib(pipe, features, labels, out: Path, ref, sample):
    import joblib

    path = out / "model.joblib"
    joblib.dump({"pipeline": pipe, "labels": labels, "features": features}, path, compress=3)
    loaded = joblib.load(path)["pipeline"]
    return path, _agree(ref, loaded.predict_proba(sample), "joblib")


def _export_onnx(pipe, features, labels, out: Path, ref, sample):
    import onnxruntime as ort
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType, StringTensorType

    initial = [
        (f"c{i}", FloatTensorType([None, 1]) if f["kind"] == "numeric" else StringTensorType([None, 1]))
        for i, f in enumerate(features)
    ]
    clf = pipe.steps[-1][1]
    onx = convert_sklearn(pipe, initial_types=initial, options={id(clf): {"zipmap": False}}, target_opset={"": 15, "ai.onnx.ml": 3})
    path = out / "model.onnx"
    path.write_bytes(onx.SerializeToString())
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = {f"c{i}": sample[f"c{i}"].to_numpy().reshape(-1, 1) for i in range(len(features))}
    names = [o.name for o in sess.get_outputs()]
    res = sess.run(None, inputs)
    probs = res[names.index("probabilities")] if "probabilities" in names else res[1]
    return path, _agree(ref, probs, "ONNX")


def export_all(pipe, features, labels, out: Path, sample, log=lambda m, level="info": None) -> list[dict]:
    """Una entrada por formato; sólo es 'ok' si su validación contra el modelo original pasó."""
    out.mkdir(parents=True, exist_ok=True)
    ref = np.asarray(pipe.predict_proba(sample), dtype=np.float64)
    results = []
    for fmt, fn in (("onnx", _export_onnx), ("joblib", _export_joblib)):
        log(f"Exportando {fmt}...")
        try:
            path, validation = fn(pipe, features, labels, out, ref, sample)
            results.append({"fmt": fmt, "status": "ok", "filename": path.name, "size_bytes": path.stat().st_size,
                            "validation": validation, "error": None})
            log(f"Formato {fmt} validado (diff máx {validation['max_abs_diff']}).")
        except Exception as exc:  # noqa: BLE001 - un formato fallido no tumba al otro
            results.append({"fmt": fmt, "status": "failed", "filename": None, "size_bytes": None, "validation": None,
                            "error": f"{type(exc).__name__}: {exc}"[:1500]})
            log(f"Falló la exportación {fmt}: {exc}", "error")
    return results


# -------------------------------------------------------------------------------------------- uso
_joblib_cache: dict[str, object] = {}
_joblib_lock = threading.Lock()


def _load_joblib(path: Path):
    key = f"{path}:{path.stat().st_mtime_ns}"
    with _joblib_lock:
        if key in _joblib_cache:
            return _joblib_cache[key]
    import joblib

    obj = joblib.load(path)
    with _joblib_lock:
        _joblib_cache[key] = obj
        while len(_joblib_cache) > 2:
            _joblib_cache.pop(next(iter(_joblib_cache)))
    return obj


def predict_proba(model_dir: str, features: list[dict], rows: list[dict], onnx_ok: bool, joblib_ok: bool):
    """(probabilidades (N, K), motor). Prefiere ONNX Runtime; si no, el modelo scikit-learn guardado."""
    inputs = feat.build_inputs(rows, features)
    d = Path(model_dir)
    if onnx_ok:
        from ..services import inference

        sess = inference._session(d, "onnx")
        names = [o.name for o in sess.get_outputs()]
        res = sess.run(None, inputs)
        probs = res[names.index("probabilities")] if "probabilities" in names else res[1]
        return np.asarray(probs, dtype=np.float64), "onnxruntime"
    if joblib_ok:
        bundle = _load_joblib(d / "model.joblib")
        return np.asarray(bundle["pipeline"].predict_proba(frame_from_inputs(inputs)), dtype=np.float64), "scikit-learn"
    raise RuntimeError("El modelo no tiene un formato ejecutable válido (ONNX o joblib).")


def parse_input(data: bytes, filename: str, max_rows: int) -> list[dict]:
    """Filas a clasificar: un objeto JSON, una lista de objetos o un CSV con encabezado."""
    if not data.strip():
        raise InvalidTable("Archivo vacío.")
    name = (filename or "").lower()
    if name.endswith(".json") or data.lstrip()[:1] in (b"{", b"["):
        try:
            obj = json.loads(data.decode("utf-8-sig"))
        except (ValueError, UnicodeDecodeError):
            raise InvalidTable("El JSON no es válido.") from None
        rows = [obj] if isinstance(obj, dict) else obj
        if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
            raise InvalidTable("El JSON debe ser un objeto {columna: valor} o una lista de objetos.")
    else:
        df = read_table(data, max_rows, 1000)
        rows = df.to_dict("records")
    if len(rows) > max_rows:
        raise InvalidTable(f"Máximo {max_rows} filas por consulta.")
    return rows
