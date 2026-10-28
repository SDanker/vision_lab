from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from app import calendar_rules as cal
from app import models
from app.db import SessionLocal, utcnow
from app.services import training_queue as tq
from app.tasks import ensure_weekly_run
from tests.conftest import (
    create_project, logged_in, make_user, new_client, login, png_bytes, upload, TZ,
)


def at(y, m, d, hh=0, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=TZ)


def test_window_states_across_the_week():
    ok = lambda dt, active=False: cal.window_state(active, dt)["open"]  # noqa: E731
    assert ok(at(2026, 10, 19, 0, 0))            # lunes 00:00 sin entrenamientos
    assert not ok(at(2026, 10, 19, 0, 0), True)  # lunes con entrenamiento activo
    assert ok(at(2026, 10, 23, 23, 59))          # viernes 23:59
    assert not ok(at(2026, 10, 24, 0, 0))        # sábado 00:00
    assert not ok(at(2026, 10, 24, 0, 5))        # sábado 00:05
    assert not ok(at(2026, 10, 24, 18, 0), False)  # sábado aunque no haya entrenamiento
    assert not ok(at(2026, 10, 25, 12, 0), False)  # domingo aunque terminara antes


def test_training_start_window_key():
    assert cal.week_key_for(at(2026, 10, 24, 0, 4)) is None
    assert cal.week_key_for(at(2026, 10, 24, 0, 5)) == "2026-10-24"
    assert cal.week_key_for(at(2026, 10, 25, 9, 0)) == "2026-10-24"
    assert cal.week_key_for(at(2026, 10, 26, 0, 0)) is None


def test_next_events_are_saturday():
    st = cal.window_state(False, at(2026, 10, 21, 12, 0))
    assert st["next_lock_at"].startswith("2026-10-24T00:00")
    assert st["next_training_start"].startswith("2026-10-24T00:05")
    assert st["timezone"] == "America/Santiago"


def test_chile_dst_changes_are_respected():
    # Verano (UTC-3) hasta el 5-abr-2026; invierno (UTC-4) después. 00:05 local del sábado:
    cal.set_clock_override(datetime(2026, 4, 4, 3, 5, tzinfo=timezone.utc))
    assert cal.now_local().strftime("%a %H:%M") == "Sat 00:05"
    cal.set_clock_override(datetime(2026, 4, 11, 4, 5, tzinfo=timezone.utc))
    assert cal.now_local().strftime("%a %H:%M") == "Sat 00:05"
    # Primavera: UTC-3 desde el 6-sep-2026
    cal.set_clock_override(datetime(2026, 9, 12, 3, 5, tzinfo=timezone.utc))
    assert cal.now_local().strftime("%a %H:%M") == "Sat 00:05"
    cal.set_clock_override(datetime(2026, 9, 5, 4, 5, tzinfo=timezone.utc))  # aún invierno
    assert cal.now_local().strftime("%a %H:%M") == "Sat 00:05"
    assert cal.week_key_for(cal.now_local()) == "2026-09-05"


def _project_with_images(c, per_class=6):
    p = create_project(c)
    cats = {x["name"]: x["id"] for x in p["categories"]}
    for name, kind in (("rojo", "red"), ("azul", "blue")):
        files = [(f"{kind}{i}.png", png_bytes(kind, i), "image/png") for i in range(per_class)]
        assert upload(c, p["id"], cats[name], files).status_code == 201
    return p, cats


def test_backend_blocks_modifications_on_weekend_and_while_training():
    c = logged_in("ana")
    p, cats = _project_with_images(c)
    img = c.get(f"/api/projects/{p['id']}/images").json()["items"][0]

    cal.set_clock_override(at(2026, 10, 24, 0, 5))  # sábado
    r = upload(c, p["id"], cats["rojo"], [("x.png", png_bytes("red", 99), "image/png")])
    assert r.status_code == 423 and r.json()["code"] == "UPLOADS_LOCKED"
    assert c.delete(f"/api/images/{img['id']}").status_code == 423
    assert c.patch(f"/api/images/{img['id']}", json={"category_id": cats["azul"]}).status_code == 423
    assert c.post(f"/api/projects/{p['id']}/categories", json={"name": "nuevo"}).status_code == 423
    assert c.delete(f"/api/projects/{p['id']}").status_code == 423
    assert c.get(f"/api/images/{img['id']}/thumb").status_code == 200  # la lectura sigue permitida
    st = c.get("/api/schedule").json()
    assert st["open"] is False and st["reason"] == "WEEKEND_LOCK"

    cal.set_clock_override(at(2026, 10, 25, 20, 0))  # domingo
    assert upload(c, p["id"], cats["rojo"], [("x.png", png_bytes("red", 98), "image/png")]).status_code == 423


def test_monday_reopens_only_after_training_and_publication():
    c = logged_in("ana")
    p, cats = _project_with_images(c)
    cal.set_clock_override(at(2026, 10, 24, 0, 5))
    assert ensure_weekly_run() is True
    with SessionLocal() as db:
        jobs = db.query(models.TrainingJob).all()
        assert len(jobs) == 1 and jobs[0].status == "pending"
        assert db.query(models.DatasetSnapshot).count() == 1
        job_id = jobs[0].id

    cal.set_clock_override(at(2026, 10, 26, 0, 0))  # lunes 00:00 con entrenamiento activo
    st = c.get("/api/schedule").json()
    assert st["open"] is False and st["reason"] == "TRAINING_ACTIVE"
    r = upload(c, p["id"], cats["rojo"], [("x.png", png_bytes("red", 77), "image/png")])
    assert r.status_code == 423

    with SessionLocal() as db:  # el entrenamiento termina (simulado), aún sin publicar
        job = db.get(models.TrainingJob, job_id)
        job.status, job.finished_at = "completed", utcnow()
        db.commit()
    assert c.get("/api/schedule").json()["open"] is False  # falta publicar resultados
    with SessionLocal() as db:
        assert tq.finalize_runs(db) == 1
    assert c.get("/api/schedule").json()["open"] is True
    assert upload(c, p["id"], cats["rojo"], [("x.png", png_bytes("red", 77), "image/png")]).status_code == 201


def test_weekly_run_is_idempotent_and_only_on_saturday_window():
    c = logged_in("ana")
    _project_with_images(c)
    cal.set_clock_override(at(2026, 10, 21, 12, 0))
    assert ensure_weekly_run() is False  # miércoles
    cal.set_clock_override(at(2026, 10, 24, 0, 4))
    assert ensure_weekly_run() is False  # 00:04: aún no
    cal.set_clock_override(at(2026, 10, 24, 0, 5))
    assert ensure_weekly_run() is True
    assert ensure_weekly_run() is False  # idempotente
    cal.set_clock_override(at(2026, 10, 25, 3, 0))
    assert ensure_weekly_run() is False  # el domingo no repite
    with SessionLocal() as db:
        assert db.query(models.TrainingJob).count() == 1


def test_missed_beat_is_covered_by_tick():
    c = logged_in("ana")
    _project_with_images(c)
    cal.set_clock_override(at(2026, 10, 24, 14, 30))  # beat estuvo caído a las 00:05
    assert ensure_weekly_run() is True


def test_weekly_run_skips_insufficient_projects_and_publishes_immediately():
    c = logged_in("ana")
    create_project(c, name="Vacío")
    cal.set_clock_override(at(2026, 10, 24, 0, 5))
    assert ensure_weekly_run() is True
    with SessionLocal() as db:
        run = db.query(models.WeeklyRun).one()
        assert run.status == "published" and run.summary["queued"] == 0
        assert len(run.summary["skipped"]) == 1
    cal.set_clock_override(at(2026, 10, 26, 0, 0))
    assert c.get("/api/schedule").json()["open"] is True


def test_stale_jobs_are_recovered_after_restart():
    c = logged_in("ana")
    p, _ = _project_with_images(c)
    cal.set_clock_override(at(2026, 10, 24, 0, 5))
    ensure_weekly_run()
    from datetime import timedelta

    with SessionLocal() as db:
        job = db.query(models.TrainingJob).one()
        job.status, job.attempts = "training", 1
        job.heartbeat_at = utcnow() - timedelta(minutes=30)
        db.commit()
        assert tq.recover_stale(db) == 1
        assert db.get(models.TrainingJob, job.id).status == "pending"
        job.status, job.attempts = "training", 5
        job.heartbeat_at = utcnow() - timedelta(minutes=30)
        db.commit()
        tq.recover_stale(db)
        assert db.get(models.TrainingJob, job.id).status == "error"
