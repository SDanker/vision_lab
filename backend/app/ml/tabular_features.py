"""Conversión de filas de una tabla en las entradas del modelo (datos tabulares).

Este mismo archivo se usa al entrenar, en la API y, copiado junto al modelo exportado, en el `inference.py`
que se entrega: así una fila produce exactamente las mismas entradas en los tres sitios. Sólo depende de NumPy.

Cada característica `i` es una entrada `c{i}` del modelo:
  · numérica  -> float32 (N, 1); un valor vacío o ilegible se reemplaza por la mediana del entrenamiento
  · categórica -> texto (N, 1); vacío = «__missing__» y un valor que no se vio al entrenar = «__other__»
"""
import math

import numpy as np

MISSING = "__missing__"
OTHER = "__other__"
_NULLS = {"", "nan", "null", "n/a"}  # «NA» no: es un valor real (p. ej. el código de Namibia)


def is_null(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return isinstance(v, str) and v.strip().lower() in _NULLS


def to_float(v) -> float:
    """Número de un valor de la tabla; NaN si está vacío o no es un número."""
    if is_null(v):
        return float("nan")
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        x = float(v)
    else:
        s = str(v).strip()
        if "," in s and "." in s:
            s = s.replace(",", "")  # 1,234.5
        elif "," in s:
            s = s.replace(",", ".")  # 3,14
        try:
            x = float(s)
        except ValueError:
            return float("nan")
    return x if math.isfinite(x) else float("nan")


def normalize_label(v) -> str:
    """Texto canónico de un valor (también el de las clases): `1.0` y `1` son la misma clase."""
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    s = str(v).strip()
    head, dot, tail = s.partition(".")
    if dot and tail and set(tail) == {"0"} and head.lstrip("-").isdigit():
        return head
    return s


def to_text(v, allowed: set | None = None) -> str:
    if is_null(v):
        return MISSING
    s = normalize_label(v)
    return s if allowed is None or s in allowed else OTHER


def column_array(values, feature: dict) -> np.ndarray:
    """Vector 1-D con los valores de una columna, listo para el modelo."""
    if feature["kind"] == "numeric":
        arr = np.array([to_float(v) for v in values], dtype=np.float32)
        bad = ~np.isfinite(arr)
        if bad.any():
            arr[bad] = np.float32(feature.get("median", 0.0))
        return arr
    allowed = set(feature.get("categories") or ()) or None
    return np.array([to_text(v, allowed) for v in values], dtype=object)


def build_inputs(rows: list[dict], features: list[dict]) -> dict[str, np.ndarray]:
    """Entradas del modelo (un arreglo (N, 1) por característica) a partir de filas {columna: valor}."""
    return {
        f"c{i}": column_array([r.get(f["name"]) for r in rows], f).reshape(-1, 1)
        for i, f in enumerate(features)
    }
