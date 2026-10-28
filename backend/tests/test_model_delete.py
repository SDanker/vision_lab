from pathlib import Path

from fastapi.testclient import TestClient

from app import models
from app.db import SessionLocal
from app.main import app
from app.storage import data_dir, user_model_bytes
from tests.conftest import create_project, logged_in


def seed_model(project_id: int, version: int = 1, base: str = "mobilenetv3", dir_path: Path | None = None):
    """Un modelo ya entrenado con su carpeta, sus exportaciones y el trabajo que lo generó."""
    d = dir_path or data_dir() / "models" / str(project_id) / f"v{version}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.onnx").write_bytes(b"x" * 10)
    with SessionLocal() as db:
        job = models.TrainingJob(project_id=project_id, status="completed", base_model=base, config={})
        db.add(job)
        db.flush()
        mv = models.ModelVersion(
            project_id=project_id, job_id=job.id, version=version, base_model=base, labels=["a", "b"],
            metrics={"test_accuracy": 0.9}, config={}, dataset_hash="h" * 64, dir_path=str(d), published=True,
        )
        mv.exports = [models.ModelExport(fmt="onnx", status="ok", filename="model.onnx", size_bytes=2048)]
        db.add(mv)
        db.flush()
        job.model_version_id = mv.id
        db.commit()
        return mv.id, job.id, d


def setup(role="editor"):
    ana, bob = logged_in("ana"), logged_in("bob")
    p = create_project(ana, name="Con modelos")
    ana.post(f"/api/projects/{p['id']}/members", json={"identifier": "bob", "role": role})
    return ana, bob, p


def test_owner_deletes_model_files_exports_and_quota_but_keeps_training_history():
    ana, _, p = setup()
    mid, jid, d = seed_model(p["id"])
    other, _, odir = seed_model(p["id"], version=2)
    with SessionLocal() as db:
        owner_id = db.query(models.User).filter_by(username="ana").one().id
        assert user_model_bytes(db, owner_id) == 4096
    assert ana.delete(f"/api/models/{mid}").status_code == 204
    assert ana.get(f"/api/models/{mid}").status_code == 404
    assert not d.exists() and odir.exists()  # sólo se borra la carpeta de ese modelo
    assert [m["id"] for m in ana.get(f"/api/projects/{p['id']}/models").json()] == [other]
    with SessionLocal() as db:
        assert db.query(models.ModelExport).count() == 1  # las exportaciones se van con el modelo
        assert user_model_bytes(db, owner_id) == 2048  # y dejan de contar en el almacenamiento
        job = db.get(models.TrainingJob, jid)
        assert job is not None and job.status == "completed" and job.model_version_id is None
        row = db.query(models.AuditLog).filter_by(action="model.deleted").one()
        assert row.actor_username == "ana" and row.detail["version"] == 1 and row.detail["project"] == "Con modelos"
    assert ana.delete(f"/api/models/{mid}").status_code == 404  # ya no existe


def test_only_owner_or_admin_can_delete():
    ana, bob, p = setup("editor")
    carl = logged_in("carl")
    admin = logged_in("root", role=models.ROLE_ADMIN)
    mid, _, d = seed_model(p["id"])
    r = bob.delete(f"/api/models/{mid}")  # editor del proyecto
    assert r.status_code == 403 and r.json()["code"] == "PROJECT_FORBIDDEN"
    assert carl.delete(f"/api/models/{mid}").status_code == 404  # sin acceso: como si no existiera
    assert d.exists() and ana.get(f"/api/models/{mid}").status_code == 200
    members = ana.get(f"/api/projects/{p['id']}/members").json()["members"]
    ana.patch(f"/api/projects/{p['id']}/members/{members[0]['user_id']}", json={"role": "viewer"})
    assert bob.delete(f"/api/models/{mid}").status_code == 403  # lector tampoco
    assert admin.delete(f"/api/models/{mid}").status_code == 204  # el administrador sí
    assert not d.exists()


def test_listings_tell_each_user_whether_they_can_delete():
    ana, bob, p = setup("editor")
    mid, _, _ = seed_model(p["id"])
    mine = ana.get("/api/models").json()[0]
    assert mine["can_delete"] is True and mine["my_role"] == "owner" and mine["project_name"] == "Con modelos"
    theirs = bob.get("/api/models").json()[0]
    assert theirs["can_delete"] is False and theirs["my_role"] == "editor" and theirs["project_owner"] == "ana"
    assert bob.get(f"/api/models/{mid}").json()["can_delete"] is False
    assert ana.get(f"/api/models/{mid}").json()["can_delete"] is True
    assert ana.get(f"/api/projects/{p['id']}/models").json()[0]["can_delete"] is True
    assert bob.get(f"/api/projects/{p['id']}/models").json()[0]["can_delete"] is False


def test_files_outside_the_models_folder_are_never_removed():
    ana, _, p = setup()
    elsewhere = data_dir() / "images" / "no-borrar"
    mid, _, d = seed_model(p["id"], dir_path=elsewhere)
    assert ana.delete(f"/api/models/{mid}").status_code == 204
    assert d.exists() and (d / "model.onnx").exists()  # el registro se borra, la ruta ajena no se toca
    # y una ruta que no existe no rompe nada
    mid2, _, d2 = seed_model(p["id"], version=2)
    with SessionLocal() as db:
        db.get(models.ModelVersion, mid2).dir_path = "/nowhere"
        db.commit()
    assert ana.delete(f"/api/models/{mid2}").status_code == 204


def test_api_tokens_delete_models_only_with_write_scope_and_ownership():
    ana, bob, p = setup("editor")
    admin = logged_in("root", role=models.ROLE_ADMIN)
    mid, _, d = seed_model(p["id"])
    read = ana.post("/api/tokens", json={"name": "lee"}).json()["token"]
    r = TestClient(app, headers={"Authorization": f"Bearer {read}"}).delete(f"/api/models/{mid}")
    assert r.status_code == 403 and r.json()["code"] == "TOKEN_READ_ONLY" and d.exists()
    rid = ana.post("/api/tokens/requests", json={"name": "esc", "reason": "limpiar modelos viejos"}).json()["id"]
    admin.post(f"/api/admin/token-requests/{rid}/approve", json={})
    write = ana.post(f"/api/tokens/requests/{rid}/generate").json()["token"]
    api = TestClient(app, headers={"Authorization": f"Bearer {write}"})
    assert api.delete(f"/api/models/{mid}").status_code == 204 and not d.exists()
    # un token de editor no gana el permiso de borrar
    mid2, _, d2 = seed_model(p["id"], version=2)
    rid = bob.post("/api/tokens/requests", json={"name": "esc", "reason": "automatizar cargas"}).json()["id"]
    admin.post(f"/api/admin/token-requests/{rid}/approve", json={})
    bob_token = bob.post(f"/api/tokens/requests/{rid}/generate").json()["token"]
    bw = TestClient(app, headers={"Authorization": f"Bearer {bob_token}"})
    assert bw.delete(f"/api/models/{mid2}").status_code == 403 and d2.exists()


def test_deleting_the_latest_model_lets_the_weekly_run_train_it_again():
    """Sin versión previa del modelo base, el dataset sin cambios vuelve a entrenarse."""
    from app.services import training_queue as tq
    from tests.test_sharing import seed_images

    ana = logged_in("ana")
    p = create_project(ana, name="Reentrenar", base_models=["mobilenetv3"])
    cats = {c["name"]: c["id"] for c in p["categories"]}
    seed_images(ana, p, cats)
    with SessionLocal() as db:
        run = tq.start_run(db, "k1", "manual", project_ids=[p["id"]], force=True)
        job = db.query(models.TrainingJob).filter_by(run_id=run.id).one()
        snap = db.get(models.DatasetSnapshot, job.snapshot_id)
        mv = models.ModelVersion(
            project_id=p["id"], job_id=job.id, version=1, base_model="mobilenetv3", labels=["a", "b"], metrics={},
            config=job.config, dataset_hash=snap.dataset_hash, dir_path=str(data_dir() / "models" / str(p["id"]) / "v1"),
        )
        db.add(mv)
        job.status = "completed"
        run.status = "published"
        db.commit()
        mid = mv.id
    with SessionLocal() as db:
        again = tq.start_run(db, "k2", "manual", project_ids=[p["id"]])
        assert not db.query(models.TrainingJob).filter_by(run_id=again.id).count()  # sin cambios: no se reentrena
        again.status = "published"
        db.commit()
    assert ana.delete(f"/api/models/{mid}").status_code == 204
    with SessionLocal() as db:
        third = tq.start_run(db, "k3", "manual", project_ids=[p["id"]])
        assert db.query(models.TrainingJob).filter_by(run_id=third.id).count() == 1
