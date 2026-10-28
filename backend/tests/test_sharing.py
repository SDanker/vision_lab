from app import models
from app.db import SessionLocal
from app.services import training_queue as tq
from tests.conftest import create_project, logged_in, make_user, png_bytes, upload


def setup_shared(role="editor", owner="ana", guest="bob"):
    ana = logged_in(owner)
    bob = logged_in(guest)
    p = create_project(ana, name="Compartido")
    cats = {c["name"]: c["id"] for c in p["categories"]}
    r = ana.post(f"/api/projects/{p['id']}/members", json={"identifier": guest, "role": role})
    assert r.status_code == 201, r.text
    return ana, bob, p, cats


def png(i, kind="red"):
    return (f"{kind}{i}.png", png_bytes(kind, i), "image/png")


# ------------------------------------------------------------------ acceso
def test_shared_project_is_listed_and_readable_by_member_only():
    ana, bob, p, _ = setup_shared()
    carl = logged_in("carl")
    mine = bob.get("/api/projects").json()
    assert [x["id"] for x in mine] == [p["id"]]
    assert mine[0]["my_role"] == "editor" and mine[0]["owner"] == "ana"
    assert [m["username"] for m in mine[0]["members"]] == ["bob"]
    assert ana.get(f"/api/projects/{p['id']}").json()["my_role"] == "owner"
    assert bob.get(f"/api/projects/{p['id']}").status_code == 200
    assert carl.get(f"/api/projects/{p['id']}").status_code == 404  # sigue siendo privado para el resto
    assert carl.get("/api/projects").json() == []


def test_member_gets_notified_when_project_is_shared():
    _, _, p, _ = setup_shared()
    with SessionLocal() as db:
        n = db.query(models.Notification).filter_by(kind="project.shared").one()
        assert n.user_id == db.query(models.User).filter_by(username="bob").one().id
        assert "Compartido" in n.title


def test_editor_can_upload_add_categories_and_adjust_parameters():
    ana, bob, p, cats = setup_shared("editor")
    r = upload(bob, p["id"], cats["rojo"], [png(1)])
    assert r.status_code == 201 and len(r.json()["uploaded"]) == 1
    assert bob.post(f"/api/projects/{p['id']}/categories", json={"name": "verde"}).status_code == 201
    r = bob.patch(f"/api/projects/{p['id']}", json={"config": {"epochs": 4, "learning_rate": 0.0005}, "auto_train": False})
    assert r.status_code == 200
    seen_by_owner = ana.get(f"/api/projects/{p['id']}").json()
    assert seen_by_owner["config"]["epochs"] == 4 and seen_by_owner["auto_train"] is False
    assert {c["name"] for c in seen_by_owner["categories"]} == {"rojo", "azul", "verde"}
    img = r_img = upload(bob, p["id"], cats["azul"], [png(2, "blue")]).json()["uploaded"][0]
    assert bob.patch(f"/api/images/{img['id']}", json={"category_id": cats["rojo"]}).status_code == 200
    assert bob.delete(f"/api/images/{r_img['id']}").status_code == 204


def test_editor_cannot_do_owner_only_actions():
    ana, bob, p, cats = setup_shared("editor")
    pid = p["id"]
    assert bob.delete(f"/api/projects/{pid}").status_code == 403
    assert bob.delete(f"/api/projects/{pid}/categories/{cats['rojo']}").status_code == 403
    assert bob.patch(f"/api/projects/{pid}", json={"name": "Otro"}).status_code == 403
    assert bob.patch(f"/api/projects/{pid}", json={"description": "x"}).status_code == 403
    assert bob.post(f"/api/projects/{pid}/members", json={"identifier": "ana"}).status_code == 403
    logged_in("carl")
    assert bob.post(f"/api/projects/{pid}/members", json={"identifier": "carl"}).json()["code"] == "PROJECT_FORBIDDEN"
    assert ana.get(f"/api/projects/{pid}").status_code == 200  # nada se rompió


def test_viewer_is_read_only():
    ana, bob, p, cats = setup_shared("viewer")
    pid = p["id"]
    img = upload(ana, pid, cats["rojo"], [png(1)]).json()["uploaded"][0]
    assert bob.get(f"/api/projects/{pid}/images").json()["total"] == 1
    assert bob.get(f"/api/images/{img['id']}/file").status_code == 200
    assert bob.get(f"/api/projects/{pid}/models").status_code == 200
    r = upload(bob, pid, cats["rojo"], [png(2)])
    assert r.status_code == 403 and r.json()["code"] == "PROJECT_FORBIDDEN"
    assert bob.post(f"/api/projects/{pid}/categories", json={"name": "nueva"}).status_code == 403
    assert bob.patch(f"/api/projects/{pid}", json={"config": {"epochs": 2}}).status_code == 403
    assert bob.patch(f"/api/images/{img['id']}", json={"category_id": cats["azul"]}).status_code == 403
    assert bob.delete(f"/api/images/{img['id']}").status_code == 403
    assert ana.get(f"/api/projects/{pid}").json()["config"]["epochs"] == 15


def test_calendar_lock_still_applies_to_editors():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from app import calendar_rules

    _, bob, p, cats = setup_shared("editor")
    calendar_rules.set_clock_override(datetime(2026, 10, 24, 12, 0, tzinfo=ZoneInfo("America/Santiago")))  # sábado
    assert upload(bob, p["id"], cats["rojo"], [png(1)]).status_code == 423


# ------------------------------------------------------------ gestión de miembros
def test_member_management_rules():
    ana, bob, p, _ = setup_shared("viewer")
    pid = p["id"]
    carl = logged_in("carl")
    make_user("pendiente", status=models.PENDING)
    assert ana.post(f"/api/projects/{pid}/members", json={"identifier": "nadie"}).json()["code"] == "USER_NOT_FOUND"
    assert ana.post(f"/api/projects/{pid}/members", json={"identifier": "pendiente"}).status_code == 404
    assert ana.post(f"/api/projects/{pid}/members", json={"identifier": "ana"}).json()["code"] == "ALREADY_OWNER"
    assert ana.post(f"/api/projects/{pid}/members", json={"identifier": "BOB"}).json()["code"] == "ALREADY_MEMBER"
    assert ana.post(f"/api/projects/{pid}/members", json={"identifier": "carl@example.com", "role": "editor"}).status_code == 201
    assert ana.post(f"/api/projects/{pid}/members", json={"identifier": "carl", "role": "admin"}).status_code in (409, 422)
    listing = ana.get(f"/api/projects/{pid}/members").json()
    assert listing["owner"]["username"] == "ana" and {m["username"] for m in listing["members"]} == {"bob", "carl"}
    bob_id = next(m["user_id"] for m in listing["members"] if m["username"] == "bob")
    # el propietario cambia el rol y bob pasa de lector a editor
    assert ana.patch(f"/api/projects/{pid}/members/{bob_id}", json={"role": "editor"}).json()["role"] == "editor"
    assert bob.post(f"/api/projects/{pid}/categories", json={"name": "extra"}).status_code == 201
    # un miembro no puede cambiar roles ni expulsar a otros
    assert bob.patch(f"/api/projects/{pid}/members/{bob_id}", json={"role": "viewer"}).status_code == 403
    carl_id = next(m["user_id"] for m in listing["members"] if m["username"] == "carl")
    assert bob.delete(f"/api/projects/{pid}/members/{carl_id}").status_code == 403
    # el propietario quita a carl: pierde el acceso
    assert ana.delete(f"/api/projects/{pid}/members/{carl_id}").status_code == 204
    assert carl.get(f"/api/projects/{pid}").status_code == 404
    # bob abandona el proyecto por sí mismo
    assert bob.delete(f"/api/projects/{pid}/members/{bob_id}").status_code == 204
    assert bob.get(f"/api/projects/{pid}").status_code == 404
    assert bob.get("/api/projects").json() == []


def test_sharing_is_audited():
    ana, _, p, _ = setup_shared("editor")
    with SessionLocal() as db:
        row = db.query(models.AuditLog).filter_by(action="project.shared").one()
        assert row.actor_username == "ana" and row.detail == {"with": "bob", "role": "editor"}


def test_deleting_member_account_or_project_cleans_up():
    ana, bob, p, _ = setup_shared("editor")
    admin = logged_in("root", role=models.ROLE_ADMIN)
    with SessionLocal() as db:
        bob_id = db.query(models.User).filter_by(username="bob").one().id
    assert admin.delete(f"/api/admin/users/{bob_id}").status_code == 204
    assert ana.get(f"/api/projects/{p['id']}").json()["members"] == []
    with SessionLocal() as db:
        assert db.query(models.ProjectMember).count() == 0
    logged_in("dan")
    assert ana.post(f"/api/projects/{p['id']}/members", json={"identifier": "dan"}).status_code == 201
    assert ana.delete(f"/api/projects/{p['id']}").status_code == 204
    with SessionLocal() as db:
        assert db.query(models.ProjectMember).count() == 0


def test_shared_models_and_jobs_are_visible_to_members():
    ana, bob, p, _ = setup_shared("viewer")
    carl = logged_in("carl")
    with SessionLocal() as db:
        db.add(models.TrainingJob(project_id=p["id"], status="completed", base_model="mobilenetv3", config={}))
        db.add(models.ModelVersion(
            project_id=p["id"], version=1, base_model="mobilenetv3", labels=["a", "b"],
            metrics={"test_accuracy": 0.9}, config={}, dataset_hash="x" * 64, dir_path="/nowhere",
        ))
        db.commit()
    assert len(bob.get("/api/training/jobs").json()) == 1
    assert len(bob.get(f"/api/training/jobs?project_id={p['id']}").json()) == 1
    assert [m["project_id"] for m in bob.get("/api/models").json()] == [p["id"]]
    assert carl.get("/api/training/jobs").json() == [] and carl.get("/api/models").json() == []
    assert carl.get(f"/api/training/jobs?project_id={p['id']}").status_code == 404


def test_training_events_are_routed_to_members():
    _, _, p, _ = setup_shared("viewer")
    published = []
    from app import events

    real = events.publish
    events.publish = published.append
    try:
        with SessionLocal() as db:
            job = models.TrainingJob(project_id=p["id"], status="pending", base_model="mobilenetv3", config={})
            db.add(job)
            db.commit()
            tq.publish_job_event(job)
            bob_id = db.query(models.User).filter_by(username="bob").one().id
    finally:
        events.publish = real
    assert published[0]["member_ids"] == [bob_id]


# --------------------------------------------- configuración avanzada por modelo
def seed_images(client, p, cats):
    for name, kind in (("rojo", "red"), ("azul", "blue")):
        upload(client, p["id"], cats[name], [png(i, kind) for i in range(7)])


def test_per_model_overrides_are_saved_and_applied_to_each_job():
    c = logged_in("ana")
    p = create_project(c, name="Avanzada", base_models=["mobilenetv3", "efficientnetv2b0"], config={"epochs": 6})
    cats = {x["name"]: x["id"] for x in p["categories"]}
    seed_images(c, p, cats)
    r = c.patch(f"/api/projects/{p['id']}", json={"config": {
        "epochs": 6, "learning_rate": 0.001,
        "model_overrides": {"efficientnetv2b0": {"epochs": 12, "image_size": 300, "batch_size": 16, "learning_rate": 0.0003}},
    }})
    assert r.status_code == 200
    assert r.json()["config"]["model_overrides"] == {
        "efficientnetv2b0": {"epochs": 12, "image_size": 300, "batch_size": 16, "learning_rate": 0.0003}
    }
    with SessionLocal() as db:
        run = tq.start_run(db, "k-adv", "manual", project_ids=[p["id"]], force=True)
        jobs = {j.base_model: j for j in db.query(models.TrainingJob).filter_by(run_id=run.id)}
    base, tuned = jobs["mobilenetv3"], jobs["efficientnetv2b0"]
    assert (base.config["epochs"], base.config["image_size"], base.config["batch_size"]) == (6, 224, 32)
    assert (tuned.config["epochs"], tuned.config["image_size"], tuned.config["batch_size"]) == (12, 300, 16)
    assert tuned.config["learning_rate"] == 0.0003 and base.config["learning_rate"] == 0.001
    assert tuned.config["patience"] == base.config["patience"] == 3  # lo no indicado se hereda
    assert tuned.total_epochs == 12 + 5 and base.total_epochs == 6 + 5
    assert all("model_overrides" not in j.config and "base_models" not in j.config for j in jobs.values())


def test_overrides_are_validated_and_empty_ones_dropped():
    c = logged_in("ana")
    p = create_project(c, name="Validar")
    url = f"/api/projects/{p['id']}"
    assert c.patch(url, json={"config": {"model_overrides": {"inventado": {"epochs": 3}}}}).status_code == 422
    assert c.patch(url, json={"config": {"model_overrides": {"mobilenetv3": {"epochs": 500}}}}).status_code == 422
    assert c.patch(url, json={"config": {"model_overrides": {"mobilenetv3": {"image_size": 500}}}}).status_code == 422
    assert c.patch(url, json={"config": {"model_overrides": {"mobilenetv3": {"learning_rate": 0.5}}}}).status_code == 422
    r = c.patch(url, json={"config": {"model_overrides": {"mobilenetv3": {"epochs": None, "patience": None}, "resnet50v2": {"patience": 8}}}})
    assert r.json()["config"]["model_overrides"] == {"resnet50v2": {"patience": 8}}


def test_editor_can_set_overrides_and_creating_with_overrides_works():
    c = logged_in("ana")
    p = create_project(c, name="Nueva", base_models=["mobilenetv3"], config={"model_overrides": {"mobilenetv3": {"fine_tune_epochs": 0}}})
    assert p["config"]["model_overrides"] == {"mobilenetv3": {"fine_tune_epochs": 0}}  # 0 es un valor válido, no «vacío»
    bob = logged_in("bob")
    assert c.post(f"/api/projects/{p['id']}/members", json={"identifier": "bob", "role": "editor"}).status_code == 201
    r = bob.patch(f"/api/projects/{p['id']}", json={"config": {"model_overrides": {"mobilenetv3": {"epochs": 3}}}})
    assert r.status_code == 200 and r.json()["config"]["model_overrides"] == {"mobilenetv3": {"epochs": 3}}


def test_unchanged_dataset_but_changed_override_retrains_only_that_model():
    c = logged_in("ana")
    p = create_project(c, name="Cambios", base_models=["mobilenetv3", "efficientnetv2b0"])
    cats = {x["name"]: x["id"] for x in p["categories"]}
    seed_images(c, p, cats)
    with SessionLocal() as db:
        run = tq.start_run(db, "k1", "manual", project_ids=[p["id"]], force=True)
        run.status = "published"  # el entrenamiento terminó: se reabren las modificaciones
        for j in db.query(models.TrainingJob).filter_by(run_id=run.id):
            j.status = "completed"
            db.add(models.ModelVersion(
                project_id=p["id"], version=1 + (j.base_model == "efficientnetv2b0"), base_model=j.base_model, labels=["a", "b"],
                metrics={}, config=j.config, dataset_hash=db.get(models.DatasetSnapshot, j.snapshot_id).dataset_hash, dir_path="/x",
            ))
        db.commit()
    c.patch(f"/api/projects/{p['id']}", json={"config": {"model_overrides": {"efficientnetv2b0": {"epochs": 20}}}})
    with SessionLocal() as db:
        run = tq.start_run(db, "k2", "manual", project_ids=[p["id"]])
        jobs = db.query(models.TrainingJob).filter_by(run_id=run.id).all()
    assert [j.base_model for j in jobs] == ["efficientnetv2b0"]
