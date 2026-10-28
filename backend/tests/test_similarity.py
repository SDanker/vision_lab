import shutil

import numpy as np
import pytest

from app import models
from app.db import SessionLocal
from app.services import similarity
from app.storage import data_dir
from tests.conftest import create_project, logged_in, png_bytes, upload

SIZE = 32


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    """Modelo Keras diminuto (convolución → pooling → densa) exportado a ONNX, y el submodelo del pooling."""
    import tensorflow as tf
    import tf2onnx
    from tensorflow import keras

    keras.utils.set_random_seed(0)
    inp = keras.Input((SIZE, SIZE, 3), name="image")
    x = keras.layers.Rescaling(1 / 127.5, offset=-1)(inp)
    x = keras.layers.Conv2D(8, 3, strides=2)(x)  # lineal: dos colores distintos dan vectores claramente distintos
    pool_out = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(pool_out)
    out = keras.layers.Dense(2, activation="softmax", name="probs")(x)
    model, pool = keras.Model(inp, out), keras.Model(inp, pool_out)
    d = tmp_path_factory.mktemp("tiny")
    spec = (tf.TensorSpec((None, SIZE, SIZE, 3), tf.float32, name="image"),)
    tf2onnx.convert.from_keras(model, input_signature=spec, opset=15, output_path=str(d / "model.onnx"))
    return d, pool


@pytest.fixture
def model_dir(tiny_model, tmp_path):
    shutil.copy2(tiny_model[0] / "model.onnx", tmp_path / "model.onnx")  # copia nueva: sin embedder ni índice
    return tmp_path


def seed(model_dir, reds=4, blues=4):
    ana = logged_in("ana")
    p = create_project(ana, name="Parecidos")
    cats = {c["name"]: c["id"] for c in p["categories"]}
    for kind, name, n in (("red", "rojo", reds), ("blue", "azul", blues)):
        files = [(f"{kind}{i}.png", png_bytes(kind, i), "image/png") for i in range(n)]
        assert upload(ana, p["id"], cats[name], files).status_code == 201
    with SessionLocal() as db:
        mv = models.ModelVersion(
            project_id=p["id"], version=1, base_model="mobilenetv3", labels=["azul", "rojo"], metrics={"test_accuracy": 0.9},
            config={"image_size": SIZE}, dataset_hash="x" * 64, dir_path=str(model_dir), published=True,
        )
        mv.exports = [models.ModelExport(fmt="onnx", status="ok", filename="model.onnx", size_bytes=1)]
        db.add(mv)
        db.commit()
        return ana, p, mv.id


def predict(client, mid, data, similar=3, name="q.png"):
    return client.post(f"/api/models/{mid}/predict?similar={similar}", files={"file": (name, data, "image/png")})


# --------------------------------------------------------------------------- embedder
def test_embedder_is_the_pooling_output_of_the_model(tiny_model, model_dir):
    import onnxruntime as ort

    pool = tiny_model[1]
    path = similarity.ensure_embedder(str(model_dir), SIZE)
    x = np.random.default_rng(1).uniform(0, 255, (3, SIZE, SIZE, 3)).astype(np.float32)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    assert got.shape == (3, 8) and np.allclose(got, pool.predict(x, verbose=0), atol=1e-4)
    mtime = path.stat().st_mtime_ns
    assert similarity.ensure_embedder(str(model_dir), SIZE) == path and path.stat().st_mtime_ns == mtime  # se reutiliza


def test_embedder_needs_a_valid_onnx(tmp_path):
    with pytest.raises(similarity.EmbedderUnavailable):
        similarity.ensure_embedder(str(tmp_path), SIZE)


# ------------------------------------------------------------------------------- índice
def test_search_returns_the_closest_project_items_and_marks_readiness(model_dir):
    ana, p, mid = seed(model_dir)
    r = predict(ana, mid, png_bytes("red", 99))  # aún no hay índice
    assert r.status_code == 200 and r.json()["similar"] == [] and r.json()["similar_status"] == "building"
    assert ana.get(f"/api/models/{mid}/similar-status").json()["state"] == "missing"

    assert similarity.build_index(mid) == 8
    assert ana.get(f"/api/models/{mid}/similar-status").json() == {"state": "ready", "up_to_date": True, "done": None, "total": None, "error": None}
    body = predict(ana, mid, png_bytes("red", 99)).json()
    assert body["similar_status"] == "ready" and body["prediction"] in ("azul", "rojo")
    assert [s["category"] for s in body["similar"]] == ["rojo"] * 3  # un rojo nuevo se parece a los rojos
    sims = [s["similarity"] for s in body["similar"]]
    assert sims == sorted(sims, reverse=True) and sims[0] <= 1.0 and body["similar"][0]["kind"] == "image"

    # una imagen que ya está en el proyecto: las 4 azules (casi idénticas entre sí) son las más cercanas, con ~100 %
    exact = predict(ana, mid, png_bytes("blue", 2), similar=4).json()["similar"]
    assert {s["name"] for s in exact} == {f"blue{i}.png" for i in range(4)} and all(s["similarity"] > 0.999 for s in exact)


def test_index_updates_incrementally_and_never_returns_deleted_items(model_dir):
    ana, p, mid = seed(model_dir)
    assert similarity.build_index(mid) == 8
    cats = {c["name"]: c["id"] for c in ana.get(f"/api/projects/{p['id']}").json()["categories"]}
    new = upload(ana, p["id"], cats["azul"], [("nuevo.png", png_bytes("blue", 50), "image/png")]).json()["uploaded"][0]
    with SessionLocal() as db:
        mv = db.get(models.ModelVersion, mid)
        assert similarity.index_state(db, mv) == "stale"
    stale = predict(ana, mid, png_bytes("blue", 50)).json()
    assert stale["similar_status"] == "updating" and "nuevo.png" not in [s["name"] for s in stale["similar"]]
    assert similarity.build_index(mid) == 1  # sólo calcula lo que falta
    names = [s["name"] for s in predict(ana, mid, png_bytes("blue", 50), similar=5).json()["similar"]]
    assert "nuevo.png" in names  # ya forma parte del índice (los azules son casi idénticos entre sí)

    assert ana.delete(f"/api/images/{new['id']}").status_code == 204
    after = predict(ana, mid, png_bytes("blue", 50)).json()  # el índice aún lo tiene, pero no se devuelve
    assert new["id"] not in [s["image_id"] for s in after["similar"]] and after["similar_status"] == "updating"
    assert similarity.build_index(mid) == 0
    ids, _ = similarity.load_index(str(model_dir))
    assert new["id"] not in ids and len(ids) == 8


def test_permissions_and_nothing_is_stored(model_dir):
    ana, p, mid = seed(model_dir)
    similarity.build_index(mid)
    bob, carl = logged_in("bob"), logged_in("carl")
    ana.post(f"/api/projects/{p['id']}/members", json={"identifier": "bob", "role": "viewer"})
    assert predict(carl, mid, png_bytes("red", 1)).status_code == 404  # sin acceso: como si no existiera
    assert carl.get(f"/api/models/{mid}/similar-status").status_code == 404
    before = sorted(x.name for x in (data_dir() / "images").rglob("*") if x.is_file())
    ok = predict(bob, mid, png_bytes("red", 1))  # un lector sí puede probar y ver los parecidos
    assert ok.status_code == 200 and len(ok.json()["similar"]) == 3
    after = sorted(x.name for x in (data_dir() / "images").rglob("*") if x.is_file())
    assert before == after  # lo capturado no se guarda en ningún sitio
    assert predict(ana, mid, png_bytes("red", 1), similar=9).status_code == 422


def test_a_failure_in_similarity_never_blocks_the_prediction(model_dir, monkeypatch):
    ana, _, mid = seed(model_dir)
    similarity.build_index(mid)

    def boom(*a, **k):
        raise RuntimeError("sin memoria")

    monkeypatch.setattr(similarity, "embed_mean", boom)
    r = predict(ana, mid, png_bytes("red", 7))
    assert r.status_code == 200 and r.json()["similar"] == [] and r.json()["similar_status"] == "unavailable"
    assert "prediction" in r.json()
    assert "similar" not in ana.post(f"/api/models/{mid}/predict", files={"file": ("q.png", png_bytes("red", 7), "image/png")}).json()


def test_index_files_live_next_to_the_model(model_dir):
    ana, _, mid = seed(model_dir)
    similarity.build_index(mid)
    assert (model_dir / "similar.npz").exists() and (model_dir / "embedder.onnx").exists()
    assert similarity.load_index(str(model_dir)) is not None


def test_only_one_index_builds_at_a_time():
    assert similarity.acquire_build_lock() is True
    assert similarity.acquire_build_lock() is False  # otro cálculo debe esperar su turno
    similarity.release_build_lock()
    assert similarity.acquire_build_lock() is True
    similarity.release_build_lock()
