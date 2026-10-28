"""De dónde sale la categoría de cada imagen cuando el dataset no la trae lista.

Todo es puro (sin red ni base de datos) y se usa igual en la vista previa y en la importación, así que lo que el
usuario ve en pantalla es exactamente lo que después se importa.

Las reglas de texto NO usan expresiones regulares: una regla escrita por un usuario podría bloquear el worker
(retroceso catastrófico). Se buscan frases con `str.find` y se cortan por palabras o por un delimitador: tiempo
lineal, sin sorpresas.
"""
import math
import re
from collections import Counter

MAX_TEXT_CHARS = 4000  # de cada texto sólo se mira el principio
MAX_LABEL_CHARS = 80
SEPARATORS = " \t\r\n:=-–—"  # lo que puede haber entre la frase y el valor
EDGE_PUNCT = ".,;:!?\"'()[]{}"


def clean_label(value) -> str | None:
    if value is None:
        return None
    s = re.sub(r"\s+", " ", str(value)).strip()[:MAX_LABEL_CHARS]
    return s or None


# ------------------------------------------------------------------------------ extraer de un texto
def _find_phrase(text_l: str, phrase_l: str) -> int:
    """Posición donde termina la primera aparición de la frase como palabras completas (-1 si no está)."""
    start = 0
    while True:
        i = text_l.find(phrase_l, start)
        if i < 0:
            return -1
        j = i + len(phrase_l)
        if (i == 0 or not text_l[i - 1].isalnum()) and (j >= len(text_l) or not text_l[j].isalnum()):
            return j
        start = i + 1


def extract_phrase(text, after: str, words: int | None = 1, until: str | None = None) -> str | None:
    """Valor que sigue a la frase `after`: las N palabras siguientes o, con `until`, todo hasta ese delimitador.

    Sin coincidencia devuelve None (la imagen queda «sin categoría»). Mayúsculas y minúsculas dan igual al buscar.
    """
    if not isinstance(text, str) or not after or not after.strip():
        return None
    text = text[:MAX_TEXT_CHARS]
    text_l = text.lower()
    if len(text_l) != len(text):  # unos pocos caracteres cambian de largo al pasar a minúsculas: se trabaja en minúsculas
        text = text_l
    end = _find_phrase(text_l, after.lower().strip())
    if end < 0:
        return None
    rest = text[end:].lstrip(SEPARATORS)
    if until:
        rest_l = rest.lower()
        i = rest_l.find(until.lower())
        if i < 0:
            return None
        rest = rest[:i] if len(rest_l) == len(rest) else rest_l[:i]
        if words:
            rest = " ".join(rest.split()[:words])
    else:
        rest = " ".join(rest.split()[: max(1, words or 1)])
    return clean_label(rest.strip(EDGE_PUNCT + " "))


# ------------------------------------------------------------------------------ contar valores
def count_labels(labels) -> list[tuple[str, int]]:
    """(valor, cantidad) de mayor a menor; «Grass» y «grass» cuentan como el mismo valor (se muestra el primero visto)."""
    shown: dict[str, str] = {}
    counts: Counter = Counter()
    for lab in labels:
        if lab is None:
            continue
        key = lab.lower()
        shown.setdefault(key, lab)
        counts[key] += 1
    return [(shown[k], n) for k, n in counts.most_common()]


def column_stats(sample: dict[str, list]) -> dict[str, dict]:
    """Por columna de la muestra: filas con dato, valores distintos, largo medio y los más frecuentes."""
    out: dict[str, dict] = {}
    for col, values in sample.items():
        vals = [v for v in values if v is not None and v != ""]
        top = count_labels(clean_label(v) for v in vals)
        lens = [len(v) for v in vals if isinstance(v, str)]
        out[col] = {
            "nonnull": len(vals), "distinct": len(top), "top": top[:12],
            "avg_len": (sum(lens) / len(lens)) if lens else 0.0, "is_text": bool(lens),
        }
    return out


def is_categorical(stat: dict, rows: int) -> bool:
    """¿Parece una columna de categorías? Pocos valores distintos y repetidos (no un id, no un texto largo)."""
    d, n = stat["distinct"], stat["nonnull"]
    return n > 0 and 2 <= d <= 300 and d <= 0.5 * n and stat["avg_len"] <= 60


# ------------------------------------------------------------------------------ sugerencias automáticas
def _norm(tok: str) -> str:
    return tok.strip(EDGE_PUNCT).lower()


def suggest_phrases(texts, max_suggestions: int = 6, min_coverage: float = 0.8, max_distinct: int = 30) -> list[dict]:
    """Busca en una muestra de textos frases que se repiten seguidas de pocos valores distintos.

    Ejemplo: «… of type Grass …», «… of type Fire …» -> frase «of type» con valores Grass, Fire, …
    Cada candidata se comprueba con `extract_phrase` (la misma función de la importación), así que las cifras que
    se muestran son las reales sobre la muestra.
    """
    rows = [t[:600] for t in texts if isinstance(t, str) and t.strip()]
    n = len(rows)
    if n < 10:
        return []
    seen_rows: dict[tuple[str, str], int] = {}  # frase de 2 palabras -> en cuántos textos aparece
    for text in rows:
        toks = text.split()
        anchors = set()
        for i in range(len(toks) - 2):
            a = (_norm(toks[i]), _norm(toks[i + 1]))
            if a[0] and a[1] and _norm(toks[i + 2]):
                anchors.add(a)
        for a in anchors:
            seen_rows[a] = seen_rows.get(a, 0) + 1
    candidates = [a for a, c in seen_rows.items() if c >= min_coverage * n]
    candidates.sort(key=lambda a: -seen_rows[a])
    found, signatures = [], set()
    for a in candidates[:60]:
        phrase = " ".join(a)
        labels = [extract_phrase(t, phrase, 1) for t in rows]
        counted = count_labels(labels)
        matched = sum(c for _, c in counted)
        distinct = len(counted)
        if matched < min_coverage * n or not 2 <= distinct <= max_distinct:
            continue
        if counted[0][1] / matched > 0.95:  # casi todo igual: no separa nada
            continue
        if matched / distinct < 3:  # valores casi únicos (nombres, números de serie): no son categorías
            continue
        sig = tuple(lab.lower() if lab else None for lab in labels)
        if sig in signatures:  # otra frase que da exactamente el mismo reparto
            continue
        signatures.add(sig)
        probs = [c / matched for _, c in counted]
        balance = -sum(p * math.log(p) for p in probs) / math.log(distinct)
        found.append({
            "after": phrase, "words": 1, "coverage": round(matched / n, 3), "distinct": distinct,
            "values": [{"value": v, "count": c} for v, c in counted[:8]], "_score": 0.5 * (matched / n) + 0.5 * balance,
        })
    found.sort(key=lambda s: -s["_score"])
    for s in found:
        s.pop("_score")
    return found[:max_suggestions]
