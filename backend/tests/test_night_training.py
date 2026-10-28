from datetime import datetime, timedelta

import pytest

from app import calendar_rules as cal
from app import models
from app.db import SessionLocal, utcnow
from app.ml.runner import claim_next_job
from app.services import estimates, night_plan
from app.services import training_queue as tq
from tests.conftest import TZ, create_project, logged_in, png_bytes, upload

TUESDAY_1AM = datetime(2026, 10, 27, 1, 0, tzinfo=TZ)
H = 3600


def seed(client, p, n=7):
    cats = {x["name"]: x["id"] for x in p["categories"]}
    for name, kind in (("rojo", "red"), ("azul", "blue")):
        r = upload(client, p["id"], cats[name], [(f"{p['id']}{kind}{i}.png", png_bytes(kind, i + 100 * p["id"]), "image/png") for i in range(n)])
        assert r.status_code == 201


def make_project(client, name, models_=("mobilenetv3",), quiet=True):
    p = create_project(client, name=name, base_models=list(models_))
    seed(client, p)
    if quiet:  # nadie trabajando en él ahora: no se pregunta
        set_activity(p["id"], minutes_ago=120)
    return p


def set_activity(project_id, minutes_ago, user_id=None):
    with SessionLocal() as db:
        p = db.get(models.Project, project_id)
        p.last_activity_at = utcnow() - timedelta(minutes=minutes_ago)
        if user_id is not None:
            p.last_activity_by_id = user_id
        db.commit()


@pytest.fixture
def fake_estimates(monkeypatch):
    """Duración fija por modelo base (segundos), en vez de la estimación real."""
    table: dict[str, float] = {}

    def fake(db, project_id, base_model, cfg, n_images):
        return table.get(base_model, 600.0), "prueba"

    monkeypatch.setattr(estimates, "estimate_job_seconds", fake)
    monkeypatch.setattr(night_plan, "estimate_job_seconds", fake)
    return table


def plan_at(when):
    cal.set_clock_override(when)
    with SessionLocal() as db:
        return night_plan.plan_night(db)


def night_jobs(project_id=None):
    with SessionLocal() as db:
        q = db.query(models.TrainingJob).filter_by(trigger="night")
        if project_id is not None:
            q = q.filter_by(project_id=project_id)
        return [(j.base_model, j.estimated_seconds, j.queue_order, j.status) for j in q.order_by(models.TrainingJob.id)]


# ------------------------------------------------------------------ estimación
def test_estimate_uses_formula_then_history():
    c = logged_in("ana")
    p = make_project(c, "Est")
    with SessionLocal() as db:
        cfg = {"epochs": 15, "fine_tune_epochs": 5, "image_size": 224}
        small, src = estimates.estimate_job_seconds(db, p["id"], "mobilenetv3", cfg, 100)
        big, _ = estimates.estimate_job_seconds(db, p["id"], "convnext_small", cfg, 100)
        hires, _ = estimates.estimate_job_seconds(db, p["id"], "mobilenetv3", {**cfg, "image_size": 384}, 100)
        assert src == "fórmula" and 0 < small < big
        # la parte variable crece con el cuadrado de la resolución (el costo fijo no)
        oh = estimates.fixed_overhead(100) * estimates.FORMULA_MARGIN
        assert hires - oh == pytest.approx((small - oh) * (384 / 224) ** 2)
        # un trabajo real de este proyecto: 1000 s con 100 imágenes y 10 épocas a 224 px
        snap = models.DatasetSnapshot(project_id=p["id"], dataset_hash="h", class_counts={}, total_images=100, path="")
        db.add(snap)
        db.flush()
        start = utcnow() - timedelta(hours=2)
        db.add(models.TrainingJob(
            project_id=p["id"], snapshot_id=snap.id, status="completed", base_model="mobilenetv3",
            config={"image_size": 224}, history=[{"epoch": i} for i in range(10)], total_epochs=20,
            started_at=start, finished_at=start + timedelta(seconds=1000),
        ))
        db.commit()
        est, src = estimates.estimate_job_seconds(db, p["id"], "mobilenetv3", cfg, 100)
        overhead = estimates.fixed_overhead(100)
        assert src == "proyecto" and est == pytest.approx(2 * (1000 - overhead) + overhead)
        # otro proyecto sin historial propio usa el del mismo modelo base
        other = make_project(c, "Otro")
        assert estimates.estimate_job_seconds(db, other["id"], "mobilenetv3", cfg, 100)[1] == "modelo"


# --------------------------------------------------------------- fin de semana
def test_weekend_queue_runs_longest_first(fake_estimates):
    fake_estimates.update({"mobilenetv3": 100, "efficientnetv2b0": 5000, "resnet50v2": 900})
    c = logged_in("ana")
    p = make_project(c, "Orden", ("mobilenetv3", "efficientnetv2b0", "resnet50v2"))
    with SessionLocal() as db:
        run = tq.start_run(db, "k-orden", "schedule", project_ids=[p["id"]], force=True)
        jobs = db.query(models.TrainingJob).filter_by(run_id=run.id).all()
        order = {j.base_model: tq.queue_position(db, j) for j in jobs}
    assert order == {"efficientnetv2b0": 1, "resnet50v2": 2, "mobilenetv3": 3}
    cal.set_clock_override(datetime(2026, 10, 24, 1, 0, tzinfo=TZ))  # sábado: nada se pospone
    first = claim_next_job()
    with SessionLocal() as db:
        assert db.get(models.TrainingJob, first).base_model == "efficientnetv2b0"


# ------------------------------------------------------------------- noche
def test_night_packs_short_models_and_leaves_long_ones(fake_estimates):
    fake_estimates.update({"mobilenetv3": 3.5 * H, "efficientnetv2b0": 3.4 * H, "convnext_small": 5 * H})
    c = logged_in("ana")
    p = make_project(c, "Noche", ("mobilenetv3", "efficientnetv2b0", "convnext_small"))
    result = plan_at(TUESDAY_1AM)  # quedan 7 h: caben las dos de ~3,5 h, la de 5 h no
    assert result["queued"] == 2
    jobs = night_jobs(p["id"])
    assert [j[0] for j in sorted(jobs, key=lambda j: j[2])] == ["efficientnetv2b0", "mobilenetv3"]  # cortos primero
    assert all(j[2] > 0 for j in jobs)
    assert plan_at(TUESDAY_1AM + timedelta(minutes=1)) is None or night_jobs(p["id"]) == jobs  # idempotente
    assert len(night_jobs(p["id"])) == 2


def test_capacity_counts_remaining_window(fake_estimates):
    fake_estimates.update({"mobilenetv3": 3.4 * H})
    c = logged_in("ana")
    make_project(c, "Tarde")
    assert plan_at(datetime(2026, 10, 27, 5, 0, tzinfo=TZ)) is None  # a las 05:00 quedan 3 h: no cabe
    assert night_jobs() == []
    c2 = logged_in("bob")
    fake_estimates.update({"efficientnetv2b0": 2 * H})
    make_project(c2, "Corto", ("efficientnetv2b0",))
    assert plan_at(datetime(2026, 10, 27, 5, 0, tzinfo=TZ))["queued"] == 1
    # otro modelo de 2 h ya no cabe en lo que queda (3 h - 2 h comprometidas)
    fake_estimates.update({"resnet50v2": 2 * H})
    make_project(logged_in("carl"), "Segundo", ("resnet50v2",))
    assert plan_at(datetime(2026, 10, 27, 5, 1, tzinfo=TZ)) is None


def test_no_night_training_on_weekend_outside_window_or_during_weekly_run(fake_estimates):
    c = logged_in("ana")
    p = make_project(c, "Nunca")
    assert plan_at(datetime(2026, 10, 24, 1, 0, tzinfo=TZ)) is None  # sábado
    assert plan_at(datetime(2026, 10, 27, 9, 0, tzinfo=TZ)) is None  # martes 09:00
    with SessionLocal() as db:
        db.add(models.WeeklyRun(week_key="manual-x", trigger="manual", status="running"))
        db.commit()
    assert plan_at(TUESDAY_1AM) is None
    assert night_jobs(p["id"]) == []


def test_only_the_trained_project_is_locked_at_night(fake_estimates):
    c = logged_in("ana")
    trained = make_project(c, "Entrena")
    with SessionLocal() as db:  # el otro proyecto no está marcado para entrenar
        other_id = create_project(c, name="Abierto")["id"]
        db.get(models.Project, other_id).auto_train = False
        db.commit()
    assert plan_at(TUESDAY_1AM)["queued"] == 1
    assert c.get("/api/schedule").json()["open"] is True  # no hay bloqueo general
    cats = {x["name"]: x["id"] for x in trained["categories"]}
    r = upload(c, trained["id"], cats["rojo"], [("x.png", png_bytes("red", 999), "image/png")])
    assert r.status_code == 423 and r.json()["code"] == "PROJECT_TRAINING"
    assert c.get(f"/api/projects/{trained['id']}").json()["training_lock"]["kind"] == "night"
    other = c.get(f"/api/projects/{other_id}").json()
    assert other["training_lock"] is None
    assert c.post(f"/api/projects/{other_id}/categories", json={"name": "nueva"}).status_code == 201
    # al terminar se publica, se desbloquea y el proyecto queda desmarcado, igual que tras la ejecución del sábado:
    # la persona revisa los resultados y vuelve a marcarlo si quiere otra ronda (se le avisa cuándo se ejecutaría)
    with SessionLocal() as db:
        for j in db.query(models.TrainingJob).filter_by(trigger="night"):
            j.status = "completed"
        db.commit()
        tq.finalize_runs(db)
        assert db.get(models.Project, trained["id"]).auto_train is False
        reset = db.query(models.Notification).filter_by(kind="training.optin_reset").all()
        assert len(reset) == 1 and "nocturno" in reset[0].body and "Entrena" in reset[0].title + reset[0].body
    assert upload(c, trained["id"], cats["rojo"], [("x.png", png_bytes("red", 999), "image/png")]).status_code == 201


def test_unchanged_models_are_not_retrained_at_night(fake_estimates):
    c = logged_in("ana")
    p = make_project(c, "Igual")
    assert plan_at(TUESDAY_1AM)["queued"] == 1
    with SessionLocal() as db:
        job = db.query(models.TrainingJob).filter_by(trigger="night").one()
        snap = db.get(models.DatasetSnapshot, job.snapshot_id)
        db.add(models.ModelVersion(
            project_id=p["id"], version=1, base_model=job.base_model, labels=["azul", "rojo"], metrics={},
            config=job.config, dataset_hash=snap.dataset_hash, dir_path="/x",
        ))
        job.status = "completed"
        db.commit()
        tq.finalize_runs(db)
        assert db.get(models.Project, p["id"]).auto_train is False  # entrenado: queda desmarcado
        db.get(models.Project, p["id"]).auto_train = True  # lo vuelve a marcar sin cambiar nada
        db.commit()
    assert plan_at(datetime(2026, 10, 28, 1, 0, tzinfo=TZ)) is None  # miércoles: marcado, pero nada cambió


# --------------------------------------------------------------- la pregunta
def test_recent_activity_asks_before_freezing(fake_estimates):
    ana = logged_in("ana")
    p = make_project(ana, "Activo", quiet=False)  # ana acaba de cargar imágenes
    assert plan_at(TUESDAY_1AM) is None or night_jobs(p["id"]) == []
    mine = ana.get("/api/night-checks/mine").json()
    assert len(mine) == 1 and mine[0]["project_id"] == p["id"] and mine[0]["status"] == "asking"
    with SessionLocal() as db:
        assert db.query(models.Notification).filter_by(kind="training.night_check").count() == 1
    bob = logged_in("bob")
    assert bob.post(f"/api/night-checks/{mine[0]['id']}/answer", json={"still_uploading": False}).status_code == 404
    # «sí, me faltan cargas»: no se entrena esta noche, y la pregunta no se repite
    r = ana.post(f"/api/night-checks/{mine[0]['id']}/answer", json={"still_uploading": True})
    assert r.status_code == 200 and r.json()["status"] == "skip"
    assert ana.post(f"/api/night-checks/{mine[0]['id']}/answer", json={"still_uploading": False}).status_code == 409
    plan_at(TUESDAY_1AM + timedelta(minutes=5))
    assert night_jobs(p["id"]) == [] and ana.get("/api/night-checks/mine").json() == []
    # la noche siguiente se vuelve a evaluar
    set_activity(p["id"], minutes_ago=120)
    assert plan_at(datetime(2026, 10, 28, 1, 0, tzinfo=TZ))["queued"] == 1


def test_answer_done_uploading_trains_tonight(fake_estimates):
    ana = logged_in("ana")
    p = make_project(ana, "Listo", quiet=False)
    plan_at(TUESDAY_1AM)
    check = ana.get("/api/night-checks/mine").json()[0]
    assert ana.post(f"/api/night-checks/{check['id']}/answer", json={"still_uploading": False}).json()["status"] == "go"
    assert plan_at(TUESDAY_1AM + timedelta(minutes=1))["queued"] == 1
    assert len(night_jobs(p["id"])) == 1


@pytest.mark.parametrize("kept_uploading, trains", [(True, False), (False, True)])
def test_no_answer_decides_by_activity(fake_estimates, kept_uploading, trains):
    ana = logged_in("ana")
    p = make_project(ana, "Silencio", quiet=False)
    plan_at(TUESDAY_1AM)
    with SessionLocal() as db:
        check = db.query(models.NightCheck).one()
        check.created_at = utcnow() - timedelta(minutes=30)  # venció el plazo para responder
        project = db.get(models.Project, p["id"])
        project.last_activity_at = utcnow() - (timedelta(minutes=1) if kept_uploading else timedelta(minutes=45))
        db.commit()
    plan_at(TUESDAY_1AM + timedelta(minutes=30))
    assert bool(night_jobs(p["id"])) is trains
    with SessionLocal() as db:
        assert db.query(models.NightCheck).one().status == ("go" if trains else "skip")


def test_viewers_are_not_asked(fake_estimates):
    ana = logged_in("ana")
    p = make_project(ana, "Lector", quiet=True)
    viewer = logged_in("vera")
    assert ana.post(f"/api/projects/{p['id']}/members", json={"identifier": "vera", "role": "viewer"}).status_code == 201
    with SessionLocal() as db:
        vid = db.query(models.User).filter_by(username="vera").one().id
        for row in db.query(models.AuditLog):  # lo que hizo ana (crear, cargar, compartir) fue hace rato
            row.created_at = utcnow() - timedelta(hours=2)
        db.commit()
    set_activity(p["id"], minutes_ago=2, user_id=vid)  # actividad atribuida a alguien sin permiso de edición
    assert plan_at(TUESDAY_1AM)["queued"] == 1  # nadie a quien preguntar: se entrena
    assert viewer.get("/api/night-checks/mine").json() == []


# ----------------------------------------------------------- inicio de cada trabajo
def test_night_job_that_no_longer_fits_is_postponed(fake_estimates):
    fake_estimates.update({"mobilenetv3": 3 * H})
    c = logged_in("ana")
    p = make_project(c, "Pospuesto")
    assert plan_at(TUESDAY_1AM)["queued"] == 1
    cal.set_clock_override(datetime(2026, 10, 27, 6, 0, tzinfo=TZ))  # los anteriores se atrasaron
    assert claim_next_job() is None
    with SessionLocal() as db:
        job = db.query(models.TrainingJob).filter_by(project_id=p["id"]).one()
        assert job.status == "cancelled" and job.phase == "Pospuesto para el fin de semana"
        tq.finalize_runs(db)
    assert c.get(f"/api/projects/{p['id']}").json()["training_lock"] is None  # se desbloquea


def test_night_job_that_fits_starts(fake_estimates):
    fake_estimates.update({"mobilenetv3": 1 * H})
    c = logged_in("ana")
    make_project(c, "Empieza")
    assert plan_at(TUESDAY_1AM)["queued"] == 1
    job_id = claim_next_job()
    with SessionLocal() as db:
        assert db.get(models.TrainingJob, job_id).status == "preparing"


# ------------------------------------------------------------------ admin
def test_admin_night_plan_preview(fake_estimates):
    fake_estimates.update({"mobilenetv3": 3.5 * H, "efficientnetv2b0": 3.4 * H, "convnext_small": 5 * H})
    c = logged_in("ana")
    make_project(c, "Vista", ("mobilenetv3", "efficientnetv2b0", "convnext_small"))
    admin = logged_in("root", role=models.ROLE_ADMIN)
    cal.set_clock_override(datetime(2026, 10, 27, 14, 0, tzinfo=TZ))  # de día: muestra la noche siguiente completa
    r = admin.get("/api/admin/night-plan").json()
    assert r["in_window"] is False and r["capacity_seconds"] == 8 * H and r["night"] == "2026-10-28"
    decisions = {i["base_model"]: i["decision"] for i in r["items"]}
    assert decisions == {"efficientnetv2b0": "tonight", "mobilenetv3": "tonight", "convnext_small": "too_long"}
    assert c.get("/api/admin/night-plan").status_code == 403
    with SessionLocal() as db:
        assert db.query(models.TrainingJob).count() == 0  # la vista previa no encola nada


# ------------------------------------------------------------- resumen de cada noche
def reports_at(when, client):
    cal.set_clock_override(when)
    return client.get("/api/admin/night-reports").json()


def test_quiet_night_is_recorded_and_owner_warned_once_a_week(fake_estimates):
    fake_estimates.update({"convnext_small": 30 * H})
    ana = logged_in("ana")
    p = make_project(ana, "Lento", ("convnext_small",))
    admin = logged_in("root", role=models.ROLE_ADMIN)
    assert reports_at(datetime(2026, 10, 26, 23, 0, tzinfo=TZ), admin) == []  # todavía no hay resúmenes
    assert plan_at(TUESDAY_1AM) is None
    plan_at(TUESDAY_1AM + timedelta(minutes=1))  # la segunda pasada no duplica nada
    [night] = reports_at(datetime(2026, 10, 27, 9, 0, tzinfo=TZ), admin)
    assert night["night"] == "2026-10-27" and not night["missing"] and not night["in_progress"]
    assert night["headline"].startswith("No se entrenó nada: 1 modelo supera el máximo")
    assert [(i["project_id"], i["decision"]) for i in night["items"]] == [(p["id"], "too_long")]

    def warnings():
        with SessionLocal() as db:
            return db.query(models.Notification).filter_by(kind="training.night_left_out").all()

    [w] = warnings()
    assert "Lento" in w.title and "sábado" in w.body
    plan_at(datetime(2026, 10, 28, 1, 0, tzinfo=TZ))  # miércoles: ya se le avisó esta semana
    assert len(warnings()) == 1
    plan_at(datetime(2026, 11, 2, 1, 0, tzinfo=TZ))  # lunes siguiente: se avisa de nuevo
    assert len(warnings()) == 2
    # jueves y viernes no hubo pasada del planificador: aparecen sin registro
    rows = reports_at(datetime(2026, 11, 2, 9, 0, tzinfo=TZ), admin)
    assert [(r["night"], r["missing"]) for r in rows] == [
        ("2026-11-02", False), ("2026-10-30", True), ("2026-10-29", True), ("2026-10-28", False), ("2026-10-27", False),
    ]
    assert "no corrió" in rows[1]["headline"]
    assert ana.get("/api/admin/night-reports").status_code == 403


def test_night_report_shows_results_and_reasons(fake_estimates):
    fake_estimates.update({"mobilenetv3": 1 * H})
    c = logged_in("ana")
    p = make_project(c, "Rápido")
    admin = logged_in("root", role=models.ROLE_ADMIN)
    assert plan_at(TUESDAY_1AM)["queued"] == 1
    [night] = reports_at(TUESDAY_1AM + timedelta(minutes=10), admin)
    assert night["in_progress"] and night["tone"] == "running"
    assert night["items"][0]["job"]["status"] == "pending"
    with SessionLocal() as db:
        db.query(models.TrainingJob).filter_by(project_id=p["id"]).one().status = "completed"
        db.commit()
        tq.finalize_runs(db)
    [night] = reports_at(datetime(2026, 10, 27, 9, 0, tzinfo=TZ), admin)
    assert night["headline"] == "Se entrenó 1 modelo." and night["tone"] == "ok"
    # sin proyectos marcados: queda registrado que no había nada pendiente
    with SessionLocal() as db:
        db.get(models.Project, p["id"]).auto_train = False
        db.commit()
    plan_at(datetime(2026, 10, 28, 1, 0, tzinfo=TZ))
    rows = reports_at(datetime(2026, 10, 28, 9, 0, tzinfo=TZ), admin)
    assert rows[0]["headline"].startswith("No se entrenó nada: no había modelos pendientes")
    # con una ejecución semanal en curso, la noche queda anotada
    with SessionLocal() as db:
        db.add(models.WeeklyRun(week_key="manual-y", trigger="manual", status="running"))
        db.commit()
    plan_at(datetime(2026, 10, 29, 1, 0, tzinfo=TZ))
    assert reports_at(datetime(2026, 10, 29, 9, 0, tzinfo=TZ), admin)[0]["headline"].startswith("Había un entrenamiento")
