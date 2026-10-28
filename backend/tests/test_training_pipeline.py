"""Prueba de extremo a extremo con el motor real: TensorFlow, transfer learning, exportación
a 5 formatos con validación e inferencia. Las imágenes sintéticas son sólo fixtures de prueba."""
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from app import calendar_rules as cal
from app import models
from app.config import get_settings
from app.db import SessionLocal
from app.ml.runner import claim_next_job, run_job
from app.services import training_queue as tq
from app.tasks import ensure_weekly_run
from tests.conftest import (
    TZ, create_project, logged_in, make_user, new_client, login, png_bytes, upload,
)
from datetime import datetime


@pytest.mark.parametrize("base", ["mobilenetv3", "efficientnetb0"])
def test_full_pipeline(base, tmp_path, monkeypatch):
    c = logged_in("ana")
    p = create_project(
        c, name=f"Colores-{base}", categories=("rojo", "azul"), base_model=base,
        config={"epochs": 3, "fine_tune_epochs": 1, "batch_size": 8, "patience": 3,
                "image_size": 160 if base == "mobilenetv3" else 224},
    )
    cats = {x["name"]: x["id"] for x in p["categories"]}
    for name, kind in (("rojo", "red"), ("azul", "blue")):
        files = [(f"{kind}{i}.png", png_bytes(kind, i), "image/png") for i in range(14)]
        assert upload(c, p["id"], cats[name], files).status_code == 201

    # Sábado 00:05: snapshot + cola semanal
    cal.set_clock_override(datetime(2026, 10, 24, 0, 5, tzinfo=TZ))
    assert ensure_weekly_run() is True
    with SessionLocal() as db:
        job = db.query(models.TrainingJob).one()
        snap = db.get(models.DatasetSnapshot, job.snapshot_id)
        assert job.status == "pending" and snap.total_images == 28 and snap.class_counts == {"azul": 14, "rojo": 14}
        assert (Path(snap.path) / "manifest.json").exists()
        assert len(list((Path(snap.path) / "images").iterdir())) == 28
    assert c.get("/api/training/jobs").json()[0]["queue_position"] == 1

    # Un único entrenamiento: se reclama y ejecuta el trabajo
    job_id = claim_next_job()
    assert job_id is not None and claim_next_job() is None
    run_job(job_id)

    with SessionLocal() as db:
        job = db.get(models.TrainingJob, job_id)
        assert job.status == "completed", job.error
        assert job.percent == 100 and job.epoch >= 1 and job.history
        assert {"loss", "accuracy", "val_loss", "val_accuracy"} <= set(job.history[0])
        assert job.cpu_percent is not None and job.ram_mb and job.ram_mb > 0
        run = db.query(models.WeeklyRun).one()
        assert run.status == "published"
        mv = db.query(models.ModelVersion).one()
        assert mv.version == 1 and mv.labels == ["azul", "rojo"]
        assert [e.status for e in mv.exports] == ["ok"] * 5
        assert {e.fmt for e in mv.exports} == set(models.EXPORT_FORMATS)
        assert all(e.validation["argmax_match"] and e.validation["max_abs_diff"] < 0.02 for e in mv.exports)
        out, model_id = Path(mv.dir_path), mv.id
        assert mv.metrics["test_accuracy"] >= 0.0 and len(mv.metrics["confusion_matrix"]) == 2
    for name in ("model.keras", "model.h5", "saved_model.zip", "model.tflite", "model.onnx", "labels.json",
                 "metadata.json", "metrics.json", "inference.py", "bundle.zip"):
        assert (out / name).stat().st_size > 0, name
    with zipfile.ZipFile(out / "saved_model.zip") as z:
        assert "saved_model.pb" in z.namelist()
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    size = 160 if base == "mobilenetv3" else 224
    assert meta["version"] == 1 and meta["input"]["shape"] == [1, size, size, 3] and meta["labels"] == ["azul", "rojo"]
    import onnxruntime as ort

    assert ort.InferenceSession(str(out / "model.onnx")).get_inputs()[0].shape[1:] == [size, size, 3]
    with SessionLocal() as db:
        assert db.query(models.TrainingLog).filter_by(job_id=job_id).count() >= 5

    # Lunes: cargas reabiertas tras publicar
    cal.set_clock_override(datetime(2026, 10, 26, 0, 0, tzinfo=TZ))
    assert c.get("/api/schedule").json()["open"] is True

    # Descargas individuales y ZIP completo (con permisos)
    for fmt in ("keras", "h5", "savedmodel", "tflite", "onnx", "labels", "metadata", "metrics", "inference", "all"):
        r = c.get(f"/api/models/{model_id}/download/{fmt}")
        assert r.status_code == 200 and len(r.content) > 10, fmt
    with zipfile.ZipFile(__import__("io").BytesIO(c.get(f"/api/models/{model_id}/download/all").content)) as z:
        assert {"model.keras", "model.h5", "saved_model.zip", "model.tflite", "model.onnx", "labels.json"} <= set(z.namelist())
    bob = logged_in("bob")
    assert bob.get(f"/api/models/{model_id}").status_code == 404
    assert bob.get(f"/api/models/{model_id}/download/onnx").status_code == 404
    pending = new_client()
    make_user("pen", status=models.PENDING)
    login(pending, "pen")
    assert pending.get(f"/api/models/{model_id}/download/onnx").status_code == 403

    # Inferencia: puntuaciones de todas las categorías y versión del modelo
    r = c.post(f"/api/models/{model_id}/predict", files={"file": ("t.png", png_bytes("red", 500), "image/png")})
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["model_version"] == 1 and len(res["scores"]) == 2
    assert abs(sum(s["score"] for s in res["scores"]) - 1) < 1e-3
    # modelo de prueba diminuto: se exige mayoría de aciertos en imágenes nuevas, no una predicción puntual
    hits = 0
    for k, kind in enumerate(["red", "blue"] * 4):
        pr = c.post(f"/api/models/{model_id}/predict", files={"file": ("t.png", png_bytes(kind, 600 + k), "image/png")}).json()
        hits += pr["prediction"] == {"red": "rojo", "blue": "azul"}[kind]
    assert hits >= 6, f"sólo {hits}/8 aciertos"
    monkeypatch.setattr(get_settings(), "low_confidence_threshold", 1.01)
    warn = c.post(f"/api/models/{model_id}/predict", files={"file": ("t.png", png_bytes("blue", 501), "image/png")}).json()
    assert warn["low_confidence"] is True and "insuficiente" in warn["warning"]
    assert c.post(f"/api/models/{model_id}/predict", files={"file": ("x.png", b"nope", "image/png")}).status_code == 422

    # «lo más parecido» sobre el ONNX realmente exportado de esta arquitectura (cada una genera un grafo distinto)
    from app.services import similarity

    assert (out / "embedder.onnx").exists()  # lo deja listo el worker de entrenamiento (tiene memoria de sobra)
    assert similarity.build_index(model_id) > 0
    red = c.post(f"/api/models/{model_id}/predict?similar=3", files={"file": ("n.png", png_bytes("red", 321), "image/png")}).json()
    assert red["similar_status"] == "ready" and [s["category"] for s in red["similar"]] == ["rojo"] * 3

    # El script de inferencia entregado funciona de forma independiente (ONNX Runtime)
    sample = tmp_path / "blue.png"
    sample.write_bytes(png_bytes("blue", 777))
    bundle_dir = tmp_path / "bundle"
    with zipfile.ZipFile(out / "bundle.zip") as z:
        z.extractall(bundle_dir)
    proc = subprocess.run([sys.executable, str(bundle_dir / "inference.py"), str(sample)], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    api_pred = c.post(f"/api/models/{model_id}/predict", files={"file": ("s.png", sample.read_bytes(), "image/png")}).json()["prediction"]
    assert proc.stdout.strip().splitlines()[0].split()[0] == api_pred  # el script entregado coincide con la plataforma

    # Terminada la semana el proyecto quedó desmarcado: la semana siguiente no entra sola...
    assert c.get(f"/api/projects/{p['id']}").json()["auto_train"] is False
    with SessionLocal() as db:
        assert tq.start_run(db, "2026-10-30", "schedule").summary["project_ids"] == []
    # ...hasta que el lunes la persona lo vuelve a marcar y guarda (y se le avisa la fecha)
    cal.set_clock_override(datetime(2026, 10, 26, 9, 0, tzinfo=TZ))
    saved = c.patch(f"/api/projects/{p['id']}", json={"auto_train": True}).json()
    assert saved["auto_train"] is True and saved["next_training_start"].startswith("2026-10-31T00:05")

    # Sin cambios en el dataset la semana siguiente: no se reentrena
    with SessionLocal() as db:
        run = tq.start_run(db, "2026-10-31", "schedule")
        assert run.status == "published" and "Sin cambios" in run.summary["skipped"][0]["reason"]
        assert db.query(models.TrainingJob).count() == 1


def test_cancel_pending_job_and_unique_job_claiming():
    c = logged_in("ana")
    admin = logged_in("root", role=models.ROLE_ADMIN)
    p = create_project(c)
    cats = {x["name"]: x["id"] for x in p["categories"]}
    for name, kind in (("rojo", "red"), ("azul", "blue")):
        upload(c, p["id"], cats[name], [(f"{kind}{i}.png", png_bytes(kind, i), "image/png") for i in range(8)])
    assert admin.post("/api/admin/training/enqueue", json={"project_ids": [p["id"]]}).status_code == 200
    jobs = admin.get("/api/admin/queue").json()["jobs"]
    assert len(jobs) == 1 and jobs[0]["status"] == "pending"
    jid = jobs[0]["id"]
    assert c.post(f"/api/admin/training/{jid}/cancel").status_code == 403  # sólo admin
    assert admin.post(f"/api/admin/training/{jid}/cancel").json()["status"] == "cancelled"
    assert claim_next_job() is None
    with SessionLocal() as db:
        tq.finalize_runs(db)
    assert admin.post(f"/api/admin/training/{jid}/retry").json()["status"] == "pending"
    assert claim_next_job() == jid


def test_cancel_running_job_stops_training_and_publishes_run():
    """Cancelación de un trabajo en proceso: el worker se detiene y se libera la cola."""
    c = logged_in("ana")
    admin = logged_in("root", role=models.ROLE_ADMIN)
    p = create_project(c)
    cats = {x["name"]: x["id"] for x in p["categories"]}
    for name, kind in (("rojo", "red"), ("azul", "blue")):
        upload(c, p["id"], cats[name], [(f"{kind}{i}.png", png_bytes(kind, i), "image/png") for i in range(8)])
    admin.post("/api/admin/training/enqueue", json={"project_ids": [p["id"]]})
    jid = claim_next_job()  # queda "preparing" (en proceso)
    r = admin.post(f"/api/admin/training/{jid}/cancel")
    assert r.status_code == 200 and r.json()["cancel_requested"] is True
    assert c.post(f"/api/admin/training/{jid}/cancel").status_code == 403  # usuarios normales no
    run_job(jid)
    with SessionLocal() as db:
        job = db.get(models.TrainingJob, jid)
        assert job.status == "cancelled" and job.finished_at is not None
        assert db.query(models.ModelVersion).count() == 0
        assert db.query(models.WeeklyRun).one().status == "published"
    assert c.get("/api/schedule").json()["open"] is True  # las cargas se reabren
