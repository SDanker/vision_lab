"""Reglas para sacar la categoría de una columna o de un texto (puras: sin red ni base de datos)."""
import time

import pytest

from app.services import hf_labels as L

CARD = "A Stage 1 Pokemon Card of type Colorless with the title Aerodactyl and 70 HP of rarity Rare Holo from the set Legend Maker."


# ------------------------------------------------------------------------------ extract_phrase
@pytest.mark.parametrize("text,after,words,until,expected", [
    (CARD, "of type", 1, None, "Colorless"),
    (CARD, "OF TYPE", 1, None, "Colorless"),                       # mayúsculas dan igual al buscar
    (CARD, "of rarity", 2, None, "Rare Holo"),                      # N palabras
    (CARD, "of rarity", 1, None, "Rare"),
    (CARD, "the title", None, " and ", "Aerodactyl"),               # hasta un delimitador
    (CARD, "from the set", 6, None, "Legend Maker"),                # puntuación final fuera
    (CARD, "A", 2, " Pokemon", "Stage 1"),
    ("type: Grass, size: big", "type", 1, None, "Grass"),           # separadores entre frase y valor
    ("type = Fuego.", "type", 1, None, "Fuego"),
    ("kind of typewriter of type Fire", "of type", 1, None, "Fire"),  # «of typewriter» no cuenta: palabra completa
    ("a kind of typewriter", "of type", 1, None, None),
    (CARD, "of nothing", 1, None, None),                            # sin la frase
    (CARD, "the title", None, "###", None),                         # sin el delimitador
    (CARD, "", 1, None, None),
    (None, "of type", 1, None, None),
    (12, "of type", 1, None, None),
    ("of type", "of type", 1, None, None),                          # la frase sin nada detrás
])
def test_extract_phrase(text, after, words, until, expected):
    assert L.extract_phrase(text, after, words, until) == expected


def test_extract_phrase_limits_label_length_and_survives_odd_unicode():
    assert len(L.extract_phrase("x " + "palabra " * 50, "x", 6)) <= L.MAX_LABEL_CHARS
    # «İ» cambia de largo al pasar a minúsculas: se trabaja en minúsculas, pero encuentra el valor y no corta mal
    assert L.extract_phrase("TİPO fuego of type Agua", "of type", 1).lower() == "agua"
    assert L.extract_phrase("É of type Èau", "of type", 1).lower() == "èau"


def test_extract_phrase_is_linear_on_hostile_input():
    """Una frase escrita por un usuario no puede colgar el worker (con una regex esto sería retroceso catastrófico)."""
    text = "a" * 4000
    start = time.perf_counter()
    for _ in range(200):
        assert L.extract_phrase(text, "a" * 40 + " b", 3) is None
        assert L.extract_phrase(text, "aaaa", 3, until="b" * 20) is None
    assert time.perf_counter() - start < 2.0


# ------------------------------------------------------------------------------ contar
def test_count_labels_merges_case_and_ignores_missing():
    got = L.count_labels(["Grass", "grass", "Fire", None, "GRASS", "Fire", "Water"])
    assert got == [("Grass", 3), ("Fire", 2), ("Water", 1)]


def test_column_stats_and_categorical_detection():
    rows = 40
    sample = {
        "id": [f"id{i}" for i in range(rows)],                       # único: un id, no una categoría
        "type": ["Grass", "Fire", "Water", "Fire"] * 10,              # categorías
        "caption": [f"A very long description of the card number {i} with many words" for i in range(rows)],
        "hp": list(range(rows)),
        "flag": [True, False] * 20,
        "mostly_empty": [None] * 39 + ["x"],
    }
    st = L.column_stats(sample)
    assert st["type"]["distinct"] == 3 and st["type"]["top"][0] == ("Fire", 20)
    assert L.is_categorical(st["type"], rows) and L.is_categorical(st["flag"], rows)
    assert not L.is_categorical(st["id"], rows) and not L.is_categorical(st["caption"], rows)
    assert not L.is_categorical(st["hp"], rows) and not L.is_categorical(st["mostly_empty"], rows)


# ------------------------------------------------------------------------------ sugerencias
def _cards(n=60):
    types = ["Grass"] * 30 + ["Fire"] * 18 + ["Water"] * 12
    out = []
    for i in range(n):
        out.append(
            f"A {['Basic', 'Stage 1'][i % 2]} Pokemon Card of type {types[i % len(types)]} with the title Card{i} "
            f"and {50 + i} HP of rarity {['Common', 'Rare', 'Uncommon'][i % 3]} from the set Set{i % 7}."
        )
    return out


def test_suggestions_find_the_repeated_phrases_with_few_values():
    sug = {s["after"]: s for s in L.suggest_phrases(_cards())}
    assert "of type" in sug and "of rarity" in sug
    t = sug["of type"]
    assert t["distinct"] == 3 and t["coverage"] == 1.0 and t["words"] == 1
    assert [(v["value"], v["count"]) for v in t["values"]] == [("Grass", 30), ("Fire", 18), ("Water", 12)]
    assert sug["of rarity"]["distinct"] == 3


def test_suggestions_skip_constants_unique_values_and_tiny_samples():
    texts = _cards()
    afters = {s["after"] for s in L.suggest_phrases(texts)}
    assert "with the" not in afters and "pokemon card" not in afters   # siempre la misma palabra detrás
    assert "the title" not in afters and "from the" not in afters      # nombres casi únicos / constantes
    assert L.suggest_phrases(texts[:5]) == []                          # muestra demasiado chica
    assert L.suggest_phrases([None, "", 3] * 20) == []


def test_every_suggestion_reproduces_with_extract_phrase():
    """Lo que muestra la vista previa es lo que se importa: la sugerencia se calcula con la misma función."""
    texts = _cards()
    for s in L.suggest_phrases(texts):
        counted = L.count_labels(L.extract_phrase(t, s["after"], s["words"]) for t in texts)
        assert [(v["value"], v["count"]) for v in s["values"]] == counted[: len(s["values"])]
