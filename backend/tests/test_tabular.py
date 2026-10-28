"""Proyectos de datos tabulares (CSV): carga, perfil de columnas, entrenamiento real con scikit-learn, exportación
(ONNX y joblib), predicción y paquete entregable. Los datos son sintéticos y sólo sirven de fixture de prueba."""
import io
import json
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from app import calendar_rules as cal
from app import models
from app.db import SessionLocal
from app.ml import tabular as tab
from app.ml import tabular_features as feat
from app.ml.runner import claim_next_job, run_job
from app.services import training_queue as tq
from app.storage import image_dir
from app.tasks import ensure_weekly_run
from tests.conftest import TZ, create_project, logged_in, png_bytes, upload


def csv_bytes(n: int = 120, seed: int = 0, sep: str = ",", with_target: bool = True) -> bytes:
    """Tabla con señal real: la clase depende de x1 y del color. `id` y `nombre` son identificadores sin relación."""
    rng = np.random.default_rng(seed)
    colors = np.array(["rojo", "azul", "verde"])
    x1, x2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    color = colors[rng.integers(0, 3, n)]
    y = np.where(x1 + (color == "rojo") * 2.5 > 1.0, "alto", "bajo")
    lines = [sep.join(["id", "nombre", "x1", "x2", "color", "objetivo"] if with_target else ["id", "nombre", "x1", "x2", "color"])]
    for i in range(n):
        x2_txt = "" if i % 17 == 0 else f"{x2[i]:.4f}"  # algunos vacíos
        row = [str(i + 1), f"item-{seed}-{i}", f"{x1[i]:.4f}", x2_txt, color[i]] + ([y[i]] if with_target else [])
        lines.append(sep.join(row))
    return "\n".join(lines).encode("utf-8")


def csv_file(name="datos.csv", **kw):
    return (name, csv_bytes(**kw), "text/csv")


def new_project(c, name="Tabla", **extra):
    p = create_project(c, name=name, modality="tabular", **extra)
    return p


def prepared_project(c, name="Tabla", **extra):
    """Proyecto con un CSV cargado y la columna objetivo elegida."""
    p = new_project(c, name, **extra)
    r = upload_no_category(c, p["id"], [csv_file()])
    assert r.status_code == 201 and len(r.json()["uploaded"]) == 1, r.text
    res = c.put(f"/api/projects/{p['id']}/tabular/columns", json={"target": "objetivo"})
    assert res.status_code == 200, res.text
    return p


def upload_no_category(c, project_id, files):
    return c.post(f"/api/projects/{project_id}/images", files=[("files", f) for f in files])


# ----------------------------------------------------------------------------- funciones puras
def test_features_row_conversion():
    assert feat.to_float("3,14") == pytest.approx(3.14) and feat.to_float("1,234.5") == pytest.approx(1234.5)
    assert np.isnan(feat.to_float("")) and np.isnan(feat.to_float("abc")) and np.isnan(feat.to_float(float("inf")))
    assert feat.normalize_label("1.0") == "1" and feat.normalize_label(2.0) == "2" and feat.normalize_label(True) == "True"
    assert feat.normalize_label(" a ") == "a" and feat.normalize_label("1.50") == "1.50"
    assert feat.to_text(None) == feat.MISSING and feat.to_text("n/a") == feat.MISSING and feat.to_text("NA") == "NA"
    assert feat.to_text("x", {"y"}) == feat.OTHER and feat.to_text("y", {"y"}) == "y"
    f = {"name": "a", "kind": "numeric", "median": 7.0}
    assert feat.column_array(["1", "", "x"], f).tolist() == [1.0, 7.0, 7.0]  # lo vacío o ilegible = mediana
    out = feat.build_inputs([{"a": "2", "b": "u"}, {"b": "zz"}], [f, {"name": "b", "kind": "categorical", "categories": ["u"]}])
    assert out["c0"].shape == (2, 1) and out["c1"].reshape(-1).tolist() == ["u", feat.OTHER]


def test_read_table_validations():
    ok = tab.read_table(csv_bytes(10, sep=";"), 100, 10)  # separador detectado
    assert ok.shape == (10, 6) and list(ok.columns)[:2] == ["id", "nombre"]
    assert tab.read_table("a\tb\n1\t2\n3\t4".encode(), 10, 5).shape == (2, 2)
    assert tab.read_table("a,b\ncafé,ñu".encode("cp1252"), 10, 5).iloc[0, 0] == "café"  # no UTF-8
    for bad, text in [
        (b"", "vacío"), (b"a\n1\n2", "al menos 2 columnas"), (b"a,a\n1,2", "repetidas"), (b"a,b\n", "no tiene filas"),
        (b"\x00\x01\x02\x03" * 50, "binario"), (b"a,b\n1,2\n3,4\n5,6", "demasiadas filas"), (b"a,b,c\n1,2,3", "columnas"),
    ]:
        limit_rows, limit_cols = (2, 10) if "filas" in text else (100, 2 if text == "columnas" else 10)
        with pytest.raises(tab.InvalidTable, match=text):
            tab.read_table(bad, limit_rows, limit_cols)


def test_profile_detects_kinds_ids_and_targets():
    prof = tab.profile(tab.read_table(csv_bytes(120), 1000, 50))
    cols = {c["name"]: c for c in prof["columns"]}
    assert cols["x1"]["kind"] == "numeric" and cols["color"]["kind"] == "categorical" and cols["objetivo"]["kind"] == "categorical"
    assert cols["id"]["id_like"] and cols["nombre"]["id_like"] and not cols["x1"]["id_like"] and not cols["color"]["id_like"]
    assert cols["x2"]["missing"] == 8 and cols["objetivo"]["target_ok"] and not cols["x1"]["target_ok"]
    assert {v["value"] for v in cols["color"]["values"]} == {"rojo", "azul", "verde"}
    assert prof["n_rows"] == 120 and len(prof["preview"]["rows"]) == 10
    assert tab.suggest_features(prof["columns"], "objetivo") == ["x1", "x2", "color"]


# ----------------------------------------------------------------------------------------- carga
def test_tabular_project_upload_and_columns():
    c = logged_in("ana")
    p = new_project(c, categories=("ignorada",))
    assert p["modality"] == "tabular" and p["base_models"] == ["random_forest"]
    assert [x["name"] for x in p["categories"]] == ["Datos"]  # categoría interna; las pedidas se ignoran
    info = {m["id"]: m for m in c.get("/api/modalities").json()}
    assert info["tabular"]["tabular"] is True and info["image"]["tabular"] is False

    s = c.get(f"/api/projects/{p['id']}/tabular").json()
    assert s["ready"] is False and "al menos un archivo" in s["problems"][0]

    files = [
        csv_file("a.csv", seed=1), csv_file("copia.csv", seed=1),                # duplicado
        ("roto.csv", b"solo una columna\n1\n2", "text/csv"),                     # <2 columnas
        ("binario.csv", bytes(range(256)) * 4, "text/csv"),
        ("foto.png", png_bytes("red", 1), "image/png"),                           # imagen en proyecto tabular
        ("notas.txt", b"a,b\n1,2", "text/plain"),                                 # extensión no permitida
    ]
    r = upload_no_category(c, p["id"], files).json()
    assert len(r["uploaded"]) == 1 and len(r["duplicates"]) == 1 and len(r["rejected"]) == 4
    up = r["uploaded"][0]
    assert up["format"] == "csv" and up["width"] == 6 and up["height"] == 120  # columnas y filas del archivo
    assert c.get(f"/api/images/{up['id']}/file").headers["content-type"].startswith("text/csv")
    assert len(list(image_dir(p["id"]).iterdir())) == 1

    s = c.get(f"/api/projects/{p['id']}/tabular").json()
    assert s["n_rows"] == 120 and s["n_columns"] == 6 and s["target"] is None and "Elige la columna" in s["problems"][0]
    assert s["suggested_features"] == ["x1", "x2", "color", "objetivo"]  # sin objetivo elegido todavía

    for body, code in [({"target": "no_existe"}, 422), ({"target": "objetivo", "features": ["objetivo"]}, 422),
                       ({"target": "objetivo", "features": ["zzz"]}, 422), ({"target": "objetivo", "features": []}, 422)]:
        assert c.put(f"/api/projects/{p['id']}/tabular/columns", json=body).status_code == code, body
    s = c.put(f"/api/projects/{p['id']}/tabular/columns", json={"target": "objetivo"}).json()
    assert s["ready"] is True and s["features"] == ["x1", "x2", "color"] and s["target"] == "objetivo"
    assert sorted((k["label"], k["used"]) for k in s["classes"]) == [("alto", True), ("bajo", True)]
    s = c.put(f"/api/projects/{p['id']}/tabular/columns", json={"target": "objetivo", "features": ["x1"]}).json()
    assert s["features"] == ["x1"]
    # elegir un objetivo con demasiados valores distintos explica por qué no sirve
    s = c.put(f"/api/projects/{p['id']}/tabular/columns", json={"target": "x1"}).json()
    assert s["ready"] is False and "valores distintos" in " ".join(s["problems"])

    # guardar el resto de la configuración no borra las columnas elegidas
    cfg = c.get(f"/api/projects/{p['id']}").json()["config"]
    saved = c.patch(f"/api/projects/{p['id']}", json={"config": {k: v for k, v in cfg.items() if not k.startswith("tabular_")}}).json()
    assert saved["config"]["tabular_target"] == "x1"

    # las categorías no se editan a mano en una tabla
    cat_id = p["categories"][0]["id"]
    assert c.post(f"/api/projects/{p['id']}/categories", json={"name": "otra"}).status_code == 409
    assert c.patch(f"/api/projects/{p['id']}/categories/{cat_id}", json={"name": "otra"}).status_code == 409
    assert c.delete(f"/api/projects/{p['id']}/categories/{cat_id}").status_code == 409
    assert c.patch(f"/api/images/{up['id']}", json={"category_id": cat_id}).status_code == 409

    # un proyecto de imágenes no acepta CSV ni tiene columnas
    pi = create_project(c, name="Fotos")
    r = upload(c, pi["id"], pi["categories"][0]["id"], [csv_file()]).json()
    assert len(r["rejected"]) == 1 and not r["uploaded"]
    assert c.get(f"/api/projects/{pi['id']}/tabular").status_code == 409
    assert c.post(f"/api/projects/{pi['id']}/images", files=[("files", csv_file())]).status_code == 404  # sin categoría
    assert logged_in("bob").get(f"/api/projects/{p['id']}/tabular").status_code == 404

    # eliminar el archivo lo quita de la tabla
    assert c.delete(f"/api/images/{up['id']}").status_code == 204
    assert c.get(f"/api/projects/{p['id']}/tabular").json()["n_files"] == 0 and not list(image_dir(p["id"]).iterdir())


def test_model_and_modality_must_match():
    c = logged_in("ana")
    r = c.post("/api/projects", json={"name": "X", "modality": "tabular", "base_models": ["mobilenetv3"]})
    assert r.status_code == 422 and "no sirven" in r.text
    r = c.post("/api/projects", json={"name": "Y", "modality": "image", "base_models": ["random_forest"]})
    assert r.status_code == 422
    p = new_project(c, base_models=["gradient_boosting", "mlp"])
    assert p["base_models"] == ["gradient_boosting", "mlp"]
    assert c.patch(f"/api/projects/{p['id']}", json={"base_models": ["mobilenetv3"]}).status_code == 422
    assert c.patch(f"/api/projects/{p['id']}", json={"base_models": ["logistic_regression"]}).json()["base_models"] == ["logistic_regression"]
    ids = {m["id"]: m["modality"] for m in c.get("/api/architectures").json()["items"]}
    assert ids["mobilenetv3"] == "image" and ids["random_forest"] == "tabular" and len(ids) == 15


# --------------------------------------------------------------------------- entrenamiento real
@pytest.mark.parametrize("model", ["random_forest", "gradient_boosting", "logistic_regression", "mlp"])
def test_tabular_training_export_and_prediction(model, tmp_path):
    c = logged_in("ana")
    p = prepared_project(c, name=f"T-{model}", base_models=[model])

    cal.set_clock_override(datetime(2026, 10, 24, 0, 5, tzinfo=TZ))
    assert ensure_weekly_run() is True
    with SessionLocal() as db:
        job = db.query(models.TrainingJob).one()
        snap = db.get(models.DatasetSnapshot, job.snapshot_id)
        assert job.config["modality"] == "tabular" and job.config["epochs"] == 1 and "image_size" not in job.config
        assert snap.total_images == 120 and snap.class_counts == {"alto": snap.class_counts["alto"], "bajo": snap.class_counts["bajo"]}
        assert (Path(snap.path) / "data.csv").exists() and 0 < job.estimated_seconds < 600
        manifest = json.loads((Path(snap.path) / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["tabular"]["features"] == ["x1", "x2", "color"] and manifest["labels"] == ["alto", "bajo"]
        first_hash = snap.dataset_hash
    job_id = claim_next_job()
    run_job(job_id)

    with SessionLocal() as db:
        job = db.get(models.TrainingJob, job_id)
        assert job.status == "completed", job.error
        assert job.percent == 100 and job.history[0]["epoch"] == 1 and {"loss", "accuracy", "val_loss", "val_accuracy"} <= set(job.history[0])
        mv = db.query(models.ModelVersion).one()
        assert mv.base_model == model and mv.labels == ["alto", "bajo"] and mv.published
        assert {e.fmt: e.status for e in mv.exports} == {"onnx": "ok", "joblib": "ok"}
        assert all(e.validation["argmax_match"] and e.validation["max_abs_diff"] < 0.02 for e in mv.exports)
        assert mv.metrics["test_accuracy"] >= 0.75, mv.metrics["test_accuracy"]  # hay señal clara: debe aprenderla
        assert mv.metrics["train_images"] + mv.metrics["val_images"] + mv.metrics["test_images"] == 120 and mv.metrics["unit"] == "filas"
        assert len(mv.metrics["confusion_matrix"]) == 2 and mv.metrics["per_class"][0]["support"] > 0
        out, model_id = Path(mv.dir_path), mv.id
        logs = " ".join(l.message for l in db.query(models.TrainingLog).filter_by(job_id=job_id))
        assert "Columna a predecir: objetivo" in logs
    for name in ("model.onnx", "model.joblib", "labels.json", "metadata.json", "metrics.json", "inference.py", "tabular_features.py",
                 "README.md", "bundle.zip"):
        assert (out / name).stat().st_size > 0, name
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert meta["modality"] == "tabular" and meta["tabular"]["target"] == "objetivo"
    assert [f["name"] for f in meta["tabular"]["features"]] == ["x1", "x2", "color"]
    assert sorted(meta["tabular"]["features"][2]["categories"]) == ["azul", "rojo", "verde"]
    if model == "random_forest":
        assert meta["tabular"]["features"][0]["median"] != 0 and "feature_importance" in json.loads((out / "metrics.json").read_text(encoding="utf-8"))

    # el calendario se reabre; la API expone el modelo con sus columnas de entrada y los dos formatos
    cal.set_clock_override(datetime(2026, 10, 26, 9, 0, tzinfo=TZ))
    detail = c.get(f"/api/models/{model_id}").json()
    assert detail["modality"] == "tabular" and [f["name"] for f in detail["input_spec"]["features"]] == ["x1", "x2", "color"]
    assert [e["format"] for e in detail["exports"]] == ["onnx", "joblib"]
    for fmt in ("onnx", "joblib", "labels", "metadata", "inference", "all"):
        assert c.get(f"/api/models/{model_id}/download/{fmt}").status_code == 200, fmt
    assert c.get(f"/api/models/{model_id}/download/tflite").status_code == 409  # no existe para tablas

    # predicción: una fila como JSON, varias como CSV, columnas que faltan y categorías nuevas
    def predict(name, data, mime="application/json"):
        return c.post(f"/api/models/{model_id}/predict", files={"file": (name, data, mime)})

    r = predict("fila.json", json.dumps({"x1": 2.5, "x2": 0.1, "color": "rojo"}).encode())
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["modality"] == "tabular" and res["prediction"] == "alto" and res["n_rows"] == 1 and res["missing_fields"] == []
    assert abs(sum(s["score"] for s in res["scores"]) - 1) < 1e-3 and res["inputs"]["color"] == "rojo" and res["engine"] == "onnxruntime"
    assert predict("f.json", json.dumps({"x1": -2.0, "x2": 0, "color": "azul"}).encode()).json()["prediction"] == "bajo"
    part = predict("f.json", json.dumps({"x1": 2.5}).encode()).json()
    assert part["missing_fields"] == ["color", "x2"] and part["prediction"] in ("alto", "bajo")  # se completan solos
    assert predict("f.json", json.dumps({"x1": 0.3, "x2": 0, "color": "violeta"}).encode()).status_code == 200  # categoría nueva
    batch = predict("lote.csv", csv_bytes(30, seed=99, with_target=False), "text/csv").json()
    assert batch["n_rows"] == 30 and len(batch["rows"]) == 30 and batch["rows"][0]["row"] == 1 and len(batch["rows"][0]["top"]) == 2
    truth = [r["objetivo"] for r in tab.read_table(csv_bytes(30, seed=99), 100, 10).to_dict("records")]
    assert sum(r["prediction"] == t for r, t in zip(batch["rows"], truth)) >= 23  # >75 % de aciertos en filas nuevas
    for bad in (("x.json", b"nope"), ("x.json", b"[]"), ("x.json", b'{"otra": 1}'), ("x.csv", b"a,b\n1,2"), ("x.json", b"")):
        assert predict(*bad).status_code == 422, bad
    assert c.post(f"/api/models/{model_id}/predict", files={"file": ("x.png", png_bytes("red", 1), "image/png")}).status_code == 422
    assert predict("lote.json", json.dumps([{"x1": 1, "x2": 1, "color": "rojo"}] * 1001).encode()).status_code == 422  # límite de filas
    # ONNX no disponible -> el respaldo joblib da la misma respuesta
    with SessionLocal() as db:
        for e in db.query(models.ModelExport).filter_by(model_id=model_id, fmt="onnx"):
            e.status = "failed"
        db.commit()
    again = predict("fila.json", json.dumps({"x1": 2.5, "x2": 0.1, "color": "rojo"}).encode()).json()
    assert again["engine"] == "scikit-learn" and again["prediction"] == "alto"
    assert abs(again["confidence"] - res["confidence"]) < 0.02

    # el paquete descargado funciona fuera de la plataforma
    bundle = tmp_path / "bundle"
    with zipfile.ZipFile(out / "bundle.zip") as z:
        z.extractall(bundle)
        assert {"model.onnx", "model.joblib", "inference.py", "tabular_features.py"} <= set(z.namelist())
    sample = tmp_path / "fila.json"
    sample.write_text(json.dumps({"x1": 2.5, "x2": 0.1, "color": "rojo"}))
    proc = subprocess.run([sys.executable, "inference.py", str(sample)], cwd=bundle, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[0].split()[0] == res["prediction"]
    proc = subprocess.run([sys.executable, "inference.py", str(sample), "--format", "joblib"], cwd=bundle, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[0].split()[0] == res["prediction"]

    # sin cambios en los datos ni en las columnas no se reentrena; al cambiar las columnas, sí
    with SessionLocal() as db:
        project = db.get(models.Project, p["id"])
        assert tq.dataset_digest(db, project)[0] == first_hash
        assert tq.models_needing_training(db, project, first_hash) == []
    assert c.put(f"/api/projects/{p['id']}/tabular/columns", json={"target": "objetivo", "features": ["x1", "color"]}).status_code == 200
    with SessionLocal() as db:
        project = db.get(models.Project, p["id"])
        digest = tq.dataset_digest(db, project)
        assert digest[0] != first_hash and digest[1] == 120
        assert [m for m, _ in tq.models_needing_training(db, project, digest[0])] == [model]
    # y borrar el modelo funciona como en los demás tipos
    assert c.delete(f"/api/models/{model_id}").status_code == 204 and not out.exists()


def test_classes_below_minimum_are_excluded_and_unready_projects_are_skipped():
    c = logged_in("ana")
    p = new_project(c, name="Desbalanceada")
    rows = ["x,clase"] + [f"{i},grande" for i in range(40)] + [f"{i},mediana" for i in range(12)] + [f"{i},chica" for i in range(3)]
    assert upload_no_category(c, p["id"], [("d.csv", "\n".join(rows).encode(), "text/csv")]).status_code == 201
    s = c.put(f"/api/projects/{p['id']}/tabular/columns", json={"target": "clase"}).json()
    assert {k["label"]: k["used"] for k in s["classes"]} == {"grande": True, "mediana": True, "chica": False}  # mínimo 6 en las pruebas
    with SessionLocal() as db:
        snap, reason = tq.create_snapshot(db, db.get(models.Project, p["id"]), None)
        assert snap is not None and snap.class_counts == {"grande": 40, "mediana": 12} and snap.excluded == {"chica": 3}
        assert snap.total_images == 52
        db.rollback()
    # con un solo valor utilizable no se puede entrenar y se explica el motivo
    q = new_project(c, name="Una sola")
    rows = ["x,clase"] + [f"{i},a" for i in range(30)] + [f"{i},b" for i in range(2)]
    upload_no_category(c, q["id"], [("d.csv", "\n".join(rows).encode(), "text/csv")])
    c.put(f"/api/projects/{q['id']}/tabular/columns", json={"target": "clase"})
    with SessionLocal() as db:
        digest, n, reason = tq.dataset_digest(db, db.get(models.Project, q["id"]))
        assert digest is None and "al menos 2 valores" in reason
