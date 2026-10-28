"""Importación desde Hugging Face: categoría desde columnas y textos, vista previa, filtros y máximos.

Reutiliza el Hub simulado de test_huggingface.py (sin red).
"""
import io

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from app import models
from app.config import get_settings
from app.db import SessionLocal
from tests.conftest import create_project, logged_in, png_bytes
from tests.test_huggingface import cats_dataset, hub, parquet_bytes, project_state, start  # noqa: F401  (hub es una fixture)

TYPES = ["Grass"] * 10 + ["Fire"] * 6 + ["Water"] * 4  # reparto desigual a propósito
IMG_T = pa.struct([("bytes", pa.binary()), ("path", pa.string())])


def cards_parquet(n=20, seed0=0):
    """Tabla como la de las cartas Pokémon: imagen + texto descriptivo + id + una columna de rareza, sin etiqueta."""
    kinds = {"Grass": "green", "Fire": "red", "Water": "blue"}
    rows = [(i, TYPES[i % len(TYPES)]) for i in range(n)]
    table = pa.table({
        "id": pa.array([f"set-{i}" for i, _ in rows]),
        "caption": pa.array([
            f"A {['Basic', 'Stage 1'][i % 2]} Pokemon Card of type {t} with the title Card{i} and {50 + i} HP "
            f"of rarity {['Common', 'Rare', 'Uncommon'][i % 3]} from the set Set{i % 7}." for i, t in rows
        ]),
        "rarity": pa.array([["Common", "Rare", "Uncommon"][i % 3] for i, _ in rows]),
        "image": pa.array([{"bytes": png_bytes(kinds[t], seed0 + i), "path": f"{i}.png"} for i, t in rows], IMG_T),
    })
    buf = io.BytesIO()
    pq.write_table(table, buf, row_group_size=7)
    return buf.getvalue()


def phrase(after="of type", column="caption", words=1, until=None):
    return {"label_source": "phrase", "label_phrase": {"column": column, "after": after, "words": words, "until": until}}


def inspect(client, pid, source="ana/cartas", **extra):
    return client.post(f"/api/projects/{pid}/hf/inspect", json={"source": source, **extra})


def last_import(client, pid):
    return client.get(f"/api/projects/{pid}/hf/imports").json()[0]


@pytest.fixture
def cards(hub):  # noqa: F811
    ana = logged_in("ana")
    p = create_project(ana, categories=())
    hub.add("ana/cartas", {"data/train-0.parquet": cards_parquet(20)})
    return ana, p["id"], hub


# ------------------------------------------------------------------------------ revisar
def test_dataset_without_labels_offers_columns_texts_and_suggestions(cards):
    ana, pid, _ = cards
    b = inspect(ana, pid).json()
    assert b["label_source"] == "none" and b["label_column"] is None and not b["can_import"]
    assert "categoría" in b["blockers"][0]
    assert b["text_columns"] == ["caption"]
    assert b["label_columns"] == ["rarity"]  # no el id (único) ni el texto largo
    sug = {s["after"]: s for s in b["suggestions"]}
    assert sug["of type"]["column"] == "caption" and sug["of type"]["distinct"] == 3
    assert [(v["value"], v["count"]) for v in sug["of type"]["values"]] == [("Grass", 10), ("Fire", 6), ("Water", 4)]
    assert b["label_preview"] is None


def test_label_from_a_text_phrase_previews_the_distribution(cards):
    ana, pid, _ = cards
    b = inspect(ana, pid, **phrase()).json()
    assert b["can_import"] and b["label_source"] == "phrase" and b["label_column"] is None
    pv = b["label_preview"]
    assert pv["exact"] is False and pv["sample_rows"] == 20 and pv["matched"] == 20 and pv["distinct"] == 3
    assert [(v["value"], v["count"]) for v in pv["values"]] == [("Grass", 10), ("Fire", 6), ("Water", 4)]
    b = inspect(ana, pid, **phrase("A", words=None, until=" Pokemon")).json()  # hasta un delimitador
    assert {v["value"] for v in b["label_preview"]["values"]} == {"Basic", "Stage 1"}


def test_label_from_any_categorical_column(cards):
    ana, pid, _ = cards
    b = inspect(ana, pid, label_source="column", label_column="rarity").json()
    assert b["can_import"] and b["label_source"] == "column" and b["label_column"] == "rarity"
    assert {v["value"]: v["count"] for v in b["label_preview"]["values"]} == {"Common": 7, "Rare": 7, "Uncommon": 6}


def test_label_source_none_ignores_the_datasets_own_labels(hub):  # noqa: F811
    ana = logged_in("ana")
    p = create_project(ana, categories=())
    hub.add("ana/tabla", {"d.parquet": parquet_bytes([(png_bytes("red", i), i % 2) for i in range(6)])})
    auto = inspect(ana, p["id"], source="ana/tabla").json()
    assert auto["label_source"] == "dataset" and auto["label_column"] == "label" and auto["can_import"]
    none = inspect(ana, p["id"], source="ana/tabla", label_source="none").json()
    assert none["label_source"] == "none" and none["label_column"] is None and not none["can_import"]
    assert inspect(ana, p["id"], source="ana/tabla", label_source="none", default_category="todo").json()["can_import"]


def test_bad_rules_are_refused_with_a_clear_message(cards):
    ana, pid, hub = cards
    b = inspect(ana, pid, **phrase("of colour")).json()  # la frase no existe en ningún texto
    assert not b["can_import"] and "no aparece" in b["blockers"][0]
    assert inspect(ana, pid, **phrase(column="nada")).json()["code"] == "HF_BAD_LABEL"
    assert inspect(ana, pid, label_source="column", label_column="nada").json()["code"] == "HF_BAD_LABEL"
    assert inspect(ana, pid, label_source="phrase").status_code == 422  # validación de la petición
    assert inspect(ana, pid, label_source="column").status_code == 422
    assert inspect(ana, pid, **phrase(words=9)).status_code == 422
    assert inspect(ana, pid, **phrase(after="   ")).status_code == 422
    assert inspect(ana, pid, include=[""]).status_code == 422
    assert inspect(ana, pid, max_items=0).status_code == 422
    hub.add("ana/uno", {"d.parquet": parquet_bytes([(png_bytes("red", i), 0) for i in range(8)])})
    one = inspect(ana, pid, source="ana/uno").json()  # una sola categoría: aviso, no bloqueo
    assert one["can_import"] and any("una sola categoría" in w for w in one["warnings"])


def test_rules_that_only_make_sense_for_tables_are_refused_for_folders(hub):  # noqa: F811
    ana = logged_in("ana")
    p = create_project(ana, categories=())
    hub.add("ana/gatos", cats_dataset(3))
    r = inspect(ana, p["id"], source="ana/gatos", **phrase())
    assert r.status_code == 422 and r.json()["code"] == "HF_BAD_LABEL" and "parquet" in r.json()["detail"]
    assert inspect(ana, p["id"], source="ana/gatos").json()["label_preview"]["exact"] is True  # carpetas: reparto exacto


# ------------------------------------------------------------------------------ importar
def test_import_with_a_text_rule_creates_the_categories_it_previewed(cards):
    ana, pid, _ = cards
    assert start(ana, pid, source="ana/cartas", **phrase()).status_code == 201
    imp = last_import(ana, pid)
    assert imp["status"] == "completed" and imp["imported"] == 20 and imp["unlabeled"] == 0 and imp["skipped"] == 0
    assert project_state(ana, pid) == ({"Grass": 10, "Fire": 6, "Water": 4}, 20)
    with SessionLocal() as db:
        opts = db.query(models.HfImport).one().options
        assert opts["label_phrase"]["after"] == "of type" and opts["label_column"] is None
        assert db.query(models.AuditLog).filter_by(action="hf.import_started").one().detail["label_source"] == "phrase"


def test_import_with_a_column_rule(cards):
    ana, pid, _ = cards
    start(ana, pid, source="ana/cartas", label_source="column", label_column="rarity")
    assert project_state(ana, pid) == ({"Common": 7, "Rare": 7, "Uncommon": 6}, 20)


def test_rows_the_rule_cannot_label_are_skipped_or_sent_to_the_default_category(hub):  # noqa: F811
    ana = logged_in("ana")
    p = create_project(ana, categories=())
    caps = [f"Card of type {'Fire' if i % 2 else 'Grass'}" if i % 4 else f"sin tipo {i}" for i in range(20)]
    t = pa.table({"caption": pa.array(caps), "image": pa.array([{"bytes": png_bytes("red", i), "path": None} for i in range(20)], IMG_T)})
    buf = io.BytesIO()
    pq.write_table(t, buf)
    hub.add("ana/mixto", {"d.parquet": buf.getvalue()})
    b = inspect(ana, p["id"], source="ana/mixto", **phrase("of type")).json()
    assert b["label_preview"]["unmatched"] == 5 and any("se omitirán" in w for w in b["warnings"])
    start(ana, p["id"], source="ana/mixto", **phrase("of type"))
    imp = last_import(ana, p["id"])
    assert imp["imported"] == 15 and imp["unlabeled"] == 5
    start(ana, p["id"], source="ana/mixto", default_category="sin tipo", **phrase("of type"))
    assert project_state(ana, p["id"])[0]["sin tipo"] == 5  # ahora las 5 sin categoría van al destino; las otras 15 ya estaban


# ------------------------------------------------------------------------------ imágenes problemáticas
def test_images_with_corrupt_exif_are_imported_and_unsavable_ones_do_not_abort_the_import(hub, monkeypatch):  # noqa: F811
    """Caso real: una imagen con EXIF dañado abortó una importación completa («Error inesperado (SyntaxError)»)."""
    from PIL import Image

    from app.services import hf

    ana = logged_in("ana")
    p = create_project(ana, categories=())
    files = cats_dataset(4)
    buf = io.BytesIO()
    Image.new("RGB", (40, 40), "green").save(buf, "PNG", exif=b"esto no es una cabecera TIFF")
    files["train/gato/exif_roto.png"] = buf.getvalue()
    hub.add("ana/exif", files)
    start(ana, p["id"], source="ana/exif")
    imp = last_import(ana, p["id"])
    assert imp["status"] == "completed" and imp["imported"] == 9 and imp["rejected"] == 0  # entra con su miniatura

    real = hf.get_modality("image").save  # y si de todos modos una imagen no se pudiera guardar, se rechaza sólo esa
    calls = {"n": 0}

    def flaky(project_id, v):
        calls["n"] += 1
        if calls["n"] == 3:
            raise SyntaxError("not a TIFF file")
        return real(project_id, v)

    monkeypatch.setattr(hf.get_modality("image"), "save", flaky)
    pid2 = create_project(ana, name="Otro", categories=())["id"]
    start(ana, pid2, source="ana/exif")
    imp2 = last_import(ana, pid2)
    assert imp2["status"] == "completed" and imp2["imported"] == 8 and imp2["rejected"] == 1
    assert "SyntaxError" in imp2["rejected_sample"][0]["reason"]


def test_a_full_disk_still_stops_the_import(hub, monkeypatch):  # noqa: F811
    import errno

    from app.services import hf

    ana = logged_in("ana")
    p = create_project(ana, categories=())
    hub.add("ana/gatos", cats_dataset(3))

    def full(project_id, v):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(hf.get_modality("image"), "save", full)
    start(ana, p["id"], source="ana/gatos")
    imp = last_import(ana, p["id"])
    assert imp["status"] == "error" and "espacio" in imp["error"] and imp["imported"] == 0


# ------------------------------------------------------------------------------ filtros y máximos
def test_only_the_chosen_categories_are_imported(cards):
    ana, pid, _ = cards
    start(ana, pid, source="ana/cartas", include=["Fire", "water"], **phrase())  # mayúsculas dan igual
    imp = last_import(ana, pid)
    assert imp["status"] == "completed" and imp["imported"] == 10 and imp["skipped"] == 10
    assert project_state(ana, pid) == ({"Fire": 6, "Water": 4}, 10)


def test_the_rest_can_be_grouped_into_another_category(cards):
    ana, pid, _ = cards
    start(ana, pid, source="ana/cartas", include=["Fire"], rest="other", other_label="resto", **phrase())
    assert project_state(ana, pid) == ({"Fire": 6, "resto": 14}, 20)
    assert last_import(ana, pid)["skipped"] == 0


def test_max_per_category_caps_each_category_and_stops_when_the_chosen_ones_are_full(cards):
    ana, pid, _ = cards
    start(ana, pid, source="ana/cartas", max_per_category=3, **phrase())
    assert project_state(ana, pid) == ({"Grass": 3, "Fire": 3, "Water": 3}, 9)
    imp = last_import(ana, pid)
    assert imp["imported"] == 9 and imp["skipped"] == 11 and imp["status"] == "completed"
    pid2 = create_project(ana, name="Otro", categories=())["id"]  # con las elegidas completas no se sigue leyendo
    start(ana, pid2, source="ana/cartas", max_per_category=2, include=["Grass", "Fire"], **phrase())
    imp2 = last_import(ana, pid2)
    assert project_state(ana, pid2) == ({"Grass": 2, "Fire": 2}, 4)
    assert imp2["processed"] < imp2["total"] and "máximo" in imp2["phase"]


def test_max_items_stops_the_import_as_a_normal_completion(cards):
    ana, pid, _ = cards
    start(ana, pid, source="ana/cartas", max_items=7, **phrase())
    imp = last_import(ana, pid)
    assert imp["status"] == "completed" and imp["imported"] == 7 and "máximo" in imp["phase"] and imp["error"] is None
    assert project_state(ana, pid)[1] == 7
    with SessionLocal() as db:
        assert db.query(models.Notification).filter_by(kind="hf.import_done").count() == 1


def test_max_items_shrinks_the_size_that_counts_against_the_limit(cards, monkeypatch):
    ana, pid, hub = cards
    shard = len(hub.repos["ana/cartas"]["files"]["data/train-0.parquet"])
    monkeypatch.setattr(get_settings(), "hf_import_max_gb", shard / 2 / 1024**3)  # el dataset entero no cabe
    big = inspect(ana, pid, **phrase()).json()
    assert not big["can_import"] and "máximo por importación" in big["blockers"][0] and "máximo de imágenes" in big["blockers"][0]
    small = inspect(ana, pid, max_items=5, **phrase()).json()  # 5 de 20 filas ≈ un cuarto del tamaño
    assert small["can_import"] and small["effective_bytes"] < shard / 2 and small["total_bytes"] == shard
    assert start(ana, pid, source="ana/cartas", max_items=5, **phrase()).status_code == 201


def test_real_size_above_the_limit_stops_the_import(cards, monkeypatch):
    """El dataset informa menos de lo que pesa: el tope real se hace cumplir mientras se guarda."""
    ana, pid, hub = cards
    hub.reported_size = 1
    monkeypatch.setattr(get_settings(), "hf_import_max_gb", 60_000 / 1024**3)  # ~60 KB: caben unas 2 imágenes
    assert start(ana, pid, source="ana/cartas", **phrase()).status_code == 201
    imp = last_import(ana, pid)
    assert imp["status"] == "error" and "máximo" in imp["error"] and 0 < imp["imported"] < 20
    assert project_state(ana, pid)[1] == imp["imported"]


def test_folders_honour_filters_and_never_download_what_is_left_out(hub):  # noqa: F811
    ana = logged_in("ana")
    p = create_project(ana, categories=())
    hub.add("ana/gatos", cats_dataset(8, kinds=(("gato", "red"), ("perro", "blue"), ("pajaro", "green"))))
    downloads = []
    hub.on_download = downloads.append
    start(ana, p["id"], source="ana/gatos", include=["perro"])
    assert project_state(ana, p["id"]) == ({"perro": 8}, 8)
    assert len(downloads) == 8 and all("/perro/" in d for d in downloads)  # lo filtrado no se descarga
    imp = last_import(ana, p["id"])
    assert imp["skipped"] == 16 and imp["processed"] == 24 == imp["total"]
    start(ana, p["id"], source="ana/gatos", max_per_category=2)
    cats = project_state(ana, p["id"])[0]
    assert cats["gato"] == 2 and cats["pajaro"] == 2 and cats["perro"] == 8


def test_filter_that_matches_nothing_is_flagged_before_importing(hub):  # noqa: F811
    ana = logged_in("ana")
    p = create_project(ana, categories=())
    hub.add("ana/gatos", cats_dataset(3))
    b = inspect(ana, p["id"], source="ana/gatos", include=["elefante"]).json()
    assert not b["can_import"] and "Ninguna de las categorías marcadas" in b["blockers"][0]
    assert not inspect(ana, p["id"], source="ana/gatos", include=[]).json()["can_import"]  # nada marcado


def test_requests_without_the_new_fields_behave_as_before(hub):  # noqa: F811
    ana = logged_in("ana")
    p = create_project(ana, categories=())
    hub.add("ana/gatos", cats_dataset(4))
    assert start(ana, p["id"], source="ana/gatos").status_code == 201
    imp = last_import(ana, p["id"])
    assert imp["imported"] == 8 and imp["skipped"] == 0
