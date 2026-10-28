import io

from PIL import Image

from app import models
from app.config import get_settings
from app.db import SessionLocal
from app.storage import image_dir, thumb_dir
from tests.conftest import (
    create_project, logged_in, login, make_user, new_client, png_bytes, upload,
)


def setup_project(c):
    p = create_project(c)
    return p, {x["name"]: x["id"] for x in p["categories"]}


def test_batch_upload_all_formats_thumbnails_and_stats():
    c = logged_in("ana")
    p, cats = setup_project(c)
    buf = io.BytesIO()
    Image.fromarray(__import__("numpy").random.default_rng(1).integers(0, 255, (64, 64, 3), dtype="uint8")).save(buf, "JPEG")
    wbuf = io.BytesIO()
    Image.fromarray(__import__("numpy").random.default_rng(2).integers(0, 255, (64, 64, 3), dtype="uint8")).save(wbuf, "WEBP")
    files = [
        ("a.jpg", buf.getvalue(), "image/jpeg"),
        ("b.png", png_bytes("red", 3), "image/png"),
        ("c.webp", wbuf.getvalue(), "image/webp"),
    ]
    r = upload(c, p["id"], cats["rojo"], files)
    assert r.status_code == 201
    body = r.json()
    assert len(body["uploaded"]) == 3 and not body["rejected"]
    assert {i["format"] for i in body["uploaded"]} == {"jpg", "png", "webp"}
    assert len(list(thumb_dir(p["id"]).glob("*.jpg"))) == 3
    assert len(list(image_dir(p["id"]).iterdir())) == 3
    stats = c.get(f"/api/projects/{p['id']}/stats").json()
    assert stats["image_count"] == 3
    assert {x["name"]: x["image_count"] for x in stats["categories"]} == {"rojo": 3, "azul": 0}
    img_id = body["uploaded"][0]["id"]
    assert c.get(f"/api/images/{img_id}/thumb").headers["content-type"] == "image/jpeg"
    assert c.get(f"/api/images/{img_id}/file").status_code == 200


def test_duplicates_and_invalid_files_are_rejected():
    c = logged_in("ana")
    p, cats = setup_project(c)
    data = png_bytes("red", 5)
    assert upload(c, p["id"], cats["rojo"], [("a.png", data, "image/png")]).json()["uploaded"]
    r = upload(c, p["id"], cats["azul"], [("copy.png", data, "image/png")]).json()
    assert len(r["duplicates"]) == 1 and not r["uploaded"]  # mismo hash en el mismo proyecto
    gif = io.BytesIO()
    Image.new("RGB", (32, 32)).save(gif, "GIF")
    bad = [
        ("fake.jpg", b"esto no es una imagen", "image/jpeg"),
        ("anim.gif", gif.getvalue(), "image/gif"),
        ("empty.png", b"", "image/png"),
        ("tiny.png", png_bytes("red", 1, size=4), "image/png"),
        ("trunc.png", data[:200], "image/png"),
    ]
    r = upload(c, p["id"], cats["rojo"], bad).json()
    assert len(r["rejected"]) == 5 and not r["uploaded"]
    assert c.get(f"/api/projects/{p['id']}/stats").json()["image_count"] == 1


def test_size_limit_and_quota(monkeypatch):
    c = logged_in("ana")
    p, cats = setup_project(c)
    s = get_settings()
    monkeypatch.setattr(s, "max_upload_mb", 0)
    r = upload(c, p["id"], cats["rojo"], [("a.png", png_bytes("red", 1), "image/png")]).json()
    assert "Supera el máximo" in r["rejected"][0]["reason"]
    monkeypatch.undo()
    admin = logged_in("root", role=models.ROLE_ADMIN)
    with SessionLocal() as db:
        uid = db.query(models.User).filter_by(username="ana").one().id
    assert admin.patch(f"/api/admin/users/{uid}", json={"quota_mb": 0}).status_code == 200
    r = upload(c, p["id"], cats["rojo"], [("a.png", png_bytes("red", 2), "image/png")]).json()
    assert "Cuota" in r["rejected"][0]["reason"]


def test_projects_and_images_are_private_but_visible_to_admin():
    ana = logged_in("ana")
    bob = logged_in("bob")
    admin = logged_in("root", role=models.ROLE_ADMIN)
    p, cats = setup_project(ana)
    img = upload(ana, p["id"], cats["rojo"], [("a.png", png_bytes("red", 1), "image/png")]).json()["uploaded"][0]
    assert bob.get(f"/api/projects/{p['id']}").status_code == 404
    assert bob.get(f"/api/projects/{p['id']}/images").status_code == 404
    assert bob.get(f"/api/images/{img['id']}/file").status_code == 404
    assert bob.get(f"/api/images/{img['id']}/thumb").status_code == 404
    assert bob.delete(f"/api/images/{img['id']}").status_code == 404
    assert upload(bob, p["id"], cats["rojo"], [("b.png", png_bytes("red", 2), "image/png")]).status_code == 404
    assert bob.get("/api/projects").json() == []
    assert admin.get(f"/api/projects/{p['id']}").status_code == 200
    assert admin.get(f"/api/images/{img['id']}/file").status_code == 200
    assert [x["id"] for x in admin.get("/api/projects?all=true").json()] == [p["id"]]
    assert admin.get("/api/projects").json() == []  # por defecto sólo los propios


def test_move_delete_image_and_category_remove_files():
    c = logged_in("ana")
    p, cats = setup_project(c)
    ids = [
        upload(c, p["id"], cats["rojo"], [(f"{i}.png", png_bytes("red", i), "image/png")]).json()["uploaded"][0]["id"]
        for i in range(2)
    ]
    assert c.patch(f"/api/images/{ids[0]}", json={"category_id": cats["azul"]}).json()["category_id"] == cats["azul"]
    assert c.delete(f"/api/images/{ids[1]}").status_code == 204
    assert len(list(image_dir(p["id"]).iterdir())) == 1
    assert c.delete(f"/api/projects/{p['id']}/categories/{cats['azul']}").status_code == 204
    assert not list(image_dir(p["id"]).iterdir())
    assert c.get(f"/api/projects/{p['id']}/stats").json()["image_count"] == 0


def test_project_names_unique_per_user_and_validation():
    c = logged_in("ana")
    create_project(c, name="Uno")
    assert c.post("/api/projects", json={"name": "Uno"}).json()["code"] == "PROJECT_EXISTS"
    assert c.post("/api/projects", json={"name": "  "}).status_code == 422
    assert c.post("/api/projects", json={"name": "X", "base_model": "resnet"}).status_code == 422
    r = create_project(c, name="Dos", base_model="efficientnetb0", config={"epochs": 3})
    assert r["base_model"] == "efficientnetb0" and r["config"]["epochs"] == 3
    d = new_client()
    login(d, "ana")
    assert len(d.get("/api/projects").json()) == 2


def test_admin_delete_user_removes_projects_and_files():
    ana = logged_in("ana")
    admin = logged_in("root", role=models.ROLE_ADMIN)
    p, cats = setup_project(ana)
    upload(ana, p["id"], cats["rojo"], [("a.png", png_bytes("red", 1), "image/png")])
    assert image_dir(p["id"]).exists()
    with SessionLocal() as db:
        uid = db.query(models.User).filter_by(username="ana").one().id
    assert admin.delete(f"/api/admin/users/{uid}").status_code == 204
    assert not image_dir(p["id"]).exists()
    with SessionLocal() as db:
        assert db.query(models.Project).count() == 0 and db.query(models.Image).count() == 0


def test_admin_dashboard_numbers():
    ana = logged_in("ana")
    make_user("pen", status=models.PENDING)
    admin = logged_in("root", role=models.ROLE_ADMIN)
    p, cats = setup_project(ana)
    upload(ana, p["id"], cats["rojo"], [("a.png", png_bytes("red", 1), "image/png")])
    d = admin.get("/api/admin/dashboard").json()
    assert d["users"]["pending"] == 1 and d["users"]["approved"] == 2
    assert d["projects"] == 1 and d["images"] == 1 and d["storage"]["images_bytes"] > 0
    assert d["lock"]["open"] is True and "cpu_percent" in d["resources"]
    assert admin.get("/api/admin/audit").json()["total"] >= 1


def test_user_job_list_filters_by_owner_before_limit():
    """Un usuario siempre ve sus trabajos aunque otros tengan muchos más recientes."""
    from app.db import utcnow

    ana = logged_in("ana")
    bob = logged_in("bob")
    pa, pb = create_project(ana, name="A"), create_project(bob, name="B")
    with SessionLocal() as db:
        db.add(models.TrainingJob(project_id=pa["id"], status="completed", base_model="mobilenetv3", config={}, finished_at=utcnow()))
        for _ in range(250):
            db.add(models.TrainingJob(project_id=pb["id"], status="completed", base_model="mobilenetv3", config={}, finished_at=utcnow()))
        db.commit()
    mine = ana.get("/api/training/jobs").json()
    assert len(mine) == 1 and mine[0]["project_id"] == pa["id"]


def test_cannot_delete_user_with_active_training():
    ana = logged_in("ana")
    admin = logged_in("root", role=models.ROLE_ADMIN)
    p = create_project(ana)
    with SessionLocal() as db:
        db.add(models.TrainingJob(project_id=p["id"], status="training", base_model="mobilenetv3", config={}))
        uid = db.query(models.User).filter_by(username="ana").one().id
        db.commit()
    r = admin.delete(f"/api/admin/users/{uid}")
    assert r.status_code == 409 and r.json()["code"] == "TRAINING_ACTIVE"


def test_api_docs_disabled_by_default(client):
    assert client.get("/api/docs").status_code == 404
    assert client.get("/api/openapi.json").status_code == 404


def test_training_hints():
    from app.ml.train import training_hints

    rising = [{"epoch": i, "loss": 1 / i, "accuracy": 0.5, "val_loss": 1 / i, "val_accuracy": 0.5} for i in range(1, 6)]
    hints = training_hints(rising, ft_epochs=0)
    assert any("seguía mejorando" in h and "Épocas máximas" in h for h in hints)
    assert any("ajuste fino" in h for h in hints)
    overfit = [{"epoch": 1, "val_loss": 0.5, "accuracy": 0.7, "val_accuracy": 0.7},
               {"epoch": 2, "val_loss": 0.9, "accuracy": 0.99, "val_accuracy": 0.7}]
    assert any("sobreajuste" in h for h in training_hints(overfit, ft_epochs=5))


def test_new_project_defaults_include_fine_tuning():
    c = logged_in("ana")
    p = create_project(c, name="Def")
    assert p["config"]["fine_tune_epochs"] == 5 and p["config"]["epochs"] == 15


def test_multiple_base_models_queue_one_job_each_with_shared_snapshot():
    from app.services import training_queue as tq

    c = logged_in("ana")
    p = create_project(c, name="Multi", base_models=["mobilenetv3", "efficientnetv2b0", "resnet50v2"])
    assert p["base_models"] == ["mobilenetv3", "efficientnetv2b0", "resnet50v2"] and p["base_model"] == "mobilenetv3"
    assert c.post("/api/projects", json={"name": "Mal", "base_models": ["inventado"]}).status_code == 422
    cats = {x["name"]: x["id"] for x in p["categories"]}
    for name, kind in (("rojo", "red"), ("azul", "blue")):
        upload(c, p["id"], cats[name], [(f"{kind}{i}.png", png_bytes(kind, i), "image/png") for i in range(7)])
    # la selección queda guardada y se puede cambiar
    r = c.patch(f"/api/projects/{p['id']}", json={"base_models": ["efficientnetv2b0", "mobilenetv3"]})
    assert r.json()["base_models"] == ["efficientnetv2b0", "mobilenetv3"]
    r = c.patch(f"/api/projects/{p['id']}", json={"config": {"epochs": 3}})
    assert r.json()["base_models"] == ["efficientnetv2b0", "mobilenetv3"]  # cambiar config no la pierde
    with SessionLocal() as db:
        run = tq.start_run(db, "k1", "manual", project_ids=[p["id"]], force=True)
        jobs = db.query(models.TrainingJob).filter_by(run_id=run.id).all()
        assert sorted(j.base_model for j in jobs) == ["efficientnetv2b0", "mobilenetv3"]
        assert len({j.snapshot_id for j in jobs}) == 1  # mismo dataset congelado
        assert all("base_models" not in j.config and j.config["epochs"] == 3 for j in jobs)
    assert c.get("/api/architectures").json()["default"] == "mobilenetv3"
    items = c.get("/api/architectures").json()["items"]
    assert sum(i["modality"] == "image" for i in items) == 11 and sum(i["modality"] == "tabular" for i in items) == 4


def test_image_size_is_validated_and_saved():
    c = logged_in("ana")
    p = create_project(c, name="Res")
    assert p["config"]["image_size"] == 224
    assert c.patch(f"/api/projects/{p['id']}", json={"config": {"image_size": 300}}).json()["config"]["image_size"] == 300
    assert c.patch(f"/api/projects/{p['id']}", json={"config": {"image_size": 500}}).status_code == 422
