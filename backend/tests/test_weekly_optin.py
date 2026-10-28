from datetime import datetime
from zoneinfo import ZoneInfo

from app import calendar_rules, models
from app.db import SessionLocal
from app.services import training_queue as tq
from tests.conftest import create_project, logged_in
from tests.test_sharing import seed_images


def make(client, name, **extra):
    p = create_project(client, name=name, base_models=["mobilenetv3"], **extra)
    return p, {c["name"]: c["id"] for c in p["categories"]}


def complete_run(db, run):
    for j in db.query(models.TrainingJob).filter_by(run_id=run.id):
        j.status = "completed"
    db.commit()


def notified(kind):
    with SessionLocal() as db:
        return [(n.user_id, n.title, n.body) for n in db.query(models.Notification).filter_by(kind=kind)]


def user_id(name):
    with SessionLocal() as db:
        return db.query(models.User).filter_by(username=name).one().id


# ------------------------------------------------------------------ al terminar la semana
def test_weekly_run_unmarks_the_projects_that_took_part_and_tells_their_people():
    ana, bob, carl = logged_in("ana"), logged_in("bob"), logged_in("carl")
    pa, cats = make(ana, "Semanal")
    seed_images(ana, pa, cats)
    ana.post(f"/api/projects/{pa['id']}/members", json={"identifier": "bob", "role": "editor"})
    ana.post(f"/api/projects/{pa['id']}/members", json={"identifier": "carl", "role": "viewer"})
    pb, _ = make(ana, "Fuera", auto_train=False)  # no estaba marcado cuando arrancó la ejecución

    with SessionLocal() as db:
        run = tq.start_run(db, "2026-10-24", "schedule")
        assert run.status == "running" and run.summary["project_ids"] == [pa["id"]]
        db.get(models.Project, pb["id"]).auto_train = True  # lo marcaron después: no participó, no se toca
        calendar_rules.set_clock_override(datetime(2026, 10, 26, 9, 0, tzinfo=ZoneInfo("America/Santiago")))  # lunes: se publican los resultados
        complete_run(db, run)
        assert tq.finalize_runs(db) == 1

    assert ana.get(f"/api/projects/{pa['id']}").json()["auto_train"] is False
    assert ana.get(f"/api/projects/{pb['id']}").json()["auto_train"] is True
    people = notified("training.optin_reset")
    assert sorted(u for u, _, _ in people) == sorted([user_id("ana"), user_id("bob")])  # propietario y editores, no el lector
    assert all("Semanal" in t and "marca" in b and "sábado 31-10-2026 a las 00:05" in b for _, t, b in people)
    with SessionLocal() as db:
        row = db.query(models.AuditLog).filter_by(action="project.weekly_optin_reset").one()
        assert row.target_id == str(pa["id"]) and row.detail == {"run": "2026-10-24"}
    assert ana.get(f"/api/projects/{pa['id']}").json()["next_training_start"] is None  # desmarcado: sin fecha


def test_a_week_with_nothing_to_train_also_resets_the_projects():
    ana = logged_in("ana")
    p, _ = make(ana, "Vacío")  # sin imágenes: se omite
    with SessionLocal() as db:
        run = tq.start_run(db, "2026-10-24", "schedule")
        assert run.status == "published" and run.summary["queued"] == 0 and run.summary["project_ids"] == [p["id"]]
    assert ana.get(f"/api/projects/{p['id']}").json()["auto_train"] is False
    assert len(notified("training.optin_reset")) == 1


def test_manual_runs_do_not_touch_the_checkbox():
    ana = logged_in("ana")
    p, cats = make(ana, "Manual")
    seed_images(ana, p, cats)
    with SessionLocal() as db:
        run = tq.start_run(db, "manual-1", "manual", project_ids=[p["id"]], force=True)
        complete_run(db, run)
        assert tq.finalize_runs(db) == 1
    assert ana.get(f"/api/projects/{p['id']}").json()["auto_train"] is True
    assert notified("training.optin_reset") == []


def test_a_failed_week_also_closes_the_cycle():
    ana = logged_in("ana")
    p, cats = make(ana, "Con error")
    seed_images(ana, p, cats)
    with SessionLocal() as db:
        run = tq.start_run(db, "2026-10-24", "schedule")
        for j in db.query(models.TrainingJob).filter_by(run_id=run.id):
            j.status = "error"
        db.commit()
        assert tq.finalize_runs(db) == 1
    assert ana.get(f"/api/projects/{p['id']}").json()["auto_train"] is False


def test_weekly_run_only_picks_marked_projects_so_the_user_must_opt_in_again():
    ana = logged_in("ana")
    p, cats = make(ana, "Optar")
    seed_images(ana, p, cats)
    with SessionLocal() as db:
        run = tq.start_run(db, "2026-10-24", "schedule")
        complete_run(db, run)
        tq.finalize_runs(db)
    with SessionLocal() as db:  # la semana siguiente nadie lo marcó: no entra
        assert tq.start_run(db, "2026-10-31", "schedule").summary["project_ids"] == []
    assert ana.patch(f"/api/projects/{p['id']}", json={"auto_train": True}).status_code == 200
    with SessionLocal() as db:  # lo volvió a marcar y guardó: entra
        assert tq.start_run(db, "2026-11-07", "schedule").summary["project_ids"] == [p["id"]]


# ------------------------------------------------------------------ al guardar marcado
def test_saving_a_marked_project_tells_the_execution_date():
    ana = logged_in("ana")
    p, _ = make(ana, "Fecha", auto_train=False)
    assert p["next_training_start"] is None
    r = ana.patch(f"/api/projects/{p['id']}", json={"auto_train": True})
    assert r.status_code == 200 and r.json()["next_training_start"].startswith("2026-10-24T00:05")
    sent = notified("training.scheduled")
    assert len(sent) == 1 and sent[0][0] == user_id("ana") and "sábado 24-10-2026 a las 00:05" in sent[0][2]
    ana.patch(f"/api/projects/{p['id']}", json={"auto_train": True})  # ya estaba marcado: no repite el aviso
    assert len(notified("training.scheduled")) == 1
    assert ana.patch(f"/api/projects/{p['id']}", json={"auto_train": False}).json()["next_training_start"] is None
    ana.patch(f"/api/projects/{p['id']}", json={"auto_train": True})  # lo desmarcó y lo volvió a marcar
    assert len(notified("training.scheduled")) == 2


def test_the_date_is_the_next_saturday_even_on_the_weekend():


    tz = ZoneInfo("America/Santiago")
    assert calendar_rules.next_training_start(datetime(2026, 10, 26, 9, 0, tzinfo=tz)).date().isoformat() == "2026-10-31"  # lunes
    assert calendar_rules.next_training_start(datetime(2026, 10, 24, 0, 1, tzinfo=tz)).date().isoformat() == "2026-10-24"  # antes de las 00:05
    assert calendar_rules.next_training_start(datetime(2026, 10, 24, 12, 0, tzinfo=tz)).date().isoformat() == "2026-10-31"  # ya arrancó
    assert calendar_rules.next_training_start(datetime(2026, 10, 25, 12, 0, tzinfo=tz)).date().isoformat() == "2026-10-31"  # domingo


# ----------------------------------------------------------------- entrenamiento nocturno
def test_the_notice_mentions_the_night_window_when_night_training_is_on(monkeypatch):
    from app.config import get_settings

    ana = logged_in("ana")
    p, _ = make(ana, "Noche", auto_train=False)
    got = ana.patch(f"/api/projects/{p['id']}", json={"auto_train": True}).json()
    assert got["night_window"] == "00:00–08:00" and got["next_training_start"].startswith("2026-10-24T00:05")
    body = notified("training.scheduled")[0][2]
    assert "sábado 24-10-2026 a las 00:05" in body and "entre las 00:00 y las 08:00" in body

    monkeypatch.setattr(get_settings(), "night_training_enabled", False)  # sin entrenamiento nocturno sólo cuenta el sábado
    assert ana.get(f"/api/projects/{p['id']}").json()["night_window"] is None
    ana.patch(f"/api/projects/{p['id']}", json={"auto_train": False})
    ana.patch(f"/api/projects/{p['id']}", json={"auto_train": True})
    assert "entre las" not in notified("training.scheduled")[1][2]

