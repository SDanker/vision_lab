"""Entrenamiento nocturno de lunes a viernes (por defecto 00:00–08:00, hora local).

Cada minuto (desde `tick`), dentro de la ventana:
1. Busca los modelos de proyectos marcados para entrenar cuyo dataset o configuración cambió desde su
   última versión, y estima cuánto tardaría cada uno (services/estimates.py).
2. Descarta los que superan NIGHT_MAX_JOB_HOURS: quedan para el fin de semana.
3. Reparte el tiempo que queda hasta el fin de la ventana entre los más cortos primero, para que quepan
   todos los posibles (p. ej. dos de 3 h y media en 8 h).
4. Si alguien modificó el proyecto hace poco, antes de congelarlo le pregunta si todavía le faltan cargas.

Cada proyecto entrenado de noche tiene su propia ejecución (trigger "night"): sólo se bloquea ese proyecto
y se publica y desbloquea apenas terminan sus modelos. El resto de los proyectos sigue abierto.

En la primera pasada de cada noche se guarda un resumen (NightReport), también cuando no se entrena nada, y se
avisa al dueño de cada proyecto cuyos modelos no caben de noche (una vez por semana y modelo).
"""
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import calendar_rules, events
from ..config import get_settings
from ..db import utcnow
from ..ml.architectures import label_of
from ..models import (
    ACTIVE_JOB_STATES,
    NC_ASKING,
    NC_GO,
    NC_SKIP,
    RUNNING_JOB_STATES,
    AuditLog,
    NightCheck,
    NightReport,
    Project,
    TrainingJob,
    User,
    WeeklyRun,
)
from . import training_queue as tq
from .access import EDIT, _LEVEL, _level, project_role, training_active
from .estimates import estimate_job_seconds
from .notifications import notify_user

log = logging.getLogger(__name__)


@dataclass
class Candidate:
    project: Project
    base_model: str
    estimated_seconds: float
    source: str


def run_key(night: str, project_id: int) -> str:
    return f"night-{night}-p{project_id}"


def _committed_seconds(db: Session) -> float:
    """Lo que aún falta de los trabajos nocturnos ya en cola o en proceso."""
    night_runs = select(WeeklyRun.id).where(WeeklyRun.trigger == tq.TRIGGER_NIGHT)
    jobs = db.scalars(
        select(TrainingJob).where(TrainingJob.status.in_(ACTIVE_JOB_STATES), TrainingJob.run_id.in_(night_runs))
    ).unique().all()
    total = 0.0
    for j in jobs:
        if j.status in RUNNING_JOB_STATES and j.eta_seconds is not None:
            total += max(j.eta_seconds, 0)
        elif j.status in RUNNING_JOB_STATES and j.started_at:
            total += max((j.estimated_seconds or 0) - (utcnow() - j.started_at).total_seconds(), 0)
        else:
            total += j.estimated_seconds or 0
    return total


def candidates(db: Session, night: str) -> tuple[list[Candidate], list[dict]]:
    """Modelos que necesitan entrenarse (proyectos marcados) y los proyectos que no se pueden entrenar."""
    out, skipped = [], []
    already = set(
        db.scalars(select(WeeklyRun.week_key).where(WeeklyRun.week_key.like(f"night-{night}-p%"))).all()
    )
    projects = db.scalars(select(Project).where(Project.auto_train.is_(True)).order_by(Project.id)).unique().all()
    for p in projects:
        if run_key(night, p.id) in already:
            continue  # ya se entrenó (o se está entrenando) esta noche
        digest, n_images, reason = tq.dataset_digest(db, p)
        if digest is None:
            skipped.append({"project_id": p.id, "project": p.name, "reason": reason})
            continue
        for base_model, cfg in tq.models_needing_training(db, p, digest):
            est, source = estimate_job_seconds(db, p.id, base_model, cfg, n_images)
            out.append(Candidate(p, base_model, est, source))
    return out, skipped


def _assign(cands: list[Candidate], capacity: float, max_job: float) -> tuple[list[Candidate], list[Candidate], list[Candidate]]:
    """(caben esta noche, demasiado largos, no caben por tiempo): los más cortos primero."""
    fit, too_long, no_room = [], [], []
    for c in sorted(cands, key=lambda c: c.estimated_seconds):
        if c.estimated_seconds > max_job:
            too_long.append(c)
        elif c.estimated_seconds <= capacity:
            fit.append(c)
            capacity -= c.estimated_seconds
        else:
            no_room.append(c)
    return fit, too_long, no_room


def _row(c: Candidate, decision: str) -> dict:
    return {
        "project_id": c.project.id, "project": c.project.name, "base_model": c.base_model,
        "model": label_of(c.base_model), "estimated_seconds": round(c.estimated_seconds), "source": c.source,
        "decision": decision,
    }


# ------------------------------------------------------------------ resumen de la noche
def _notify_left_out(db: Session, night: str, too_long: list[Candidate], no_room: list[Candidate]) -> list[str]:
    """Avisa al dueño qué modelos no se entrenan esta noche. Devuelve las claves «proyecto:modelo» avisadas.

    Cada modelo se avisa una sola vez por semana, para no repetir lo mismo de lunes a viernes.
    """
    d = date.fromisoformat(night)
    monday = (d - timedelta(days=d.weekday())).isoformat()
    already: set[str] = set()
    for r in db.scalars(select(NightReport).where(NightReport.night >= monday, NightReport.night < night)).all():
        already.update(r.notified or [])
    limit = tq.fmt_duration(get_settings().night_max_job_hours * 3600)
    lines: dict[int, list[str]] = {}
    projects: dict[int, Project] = {}
    long_ones: set[int] = set()
    sent: list[str] = []
    for c, too_big in [(c, True) for c in too_long] + [(c, False) for c in no_room]:
        key = f"{c.project.id}:{c.base_model}"
        if key in already:
            continue
        sent.append(key)
        projects[c.project.id] = c.project
        est = tq.fmt_duration(c.estimated_seconds)
        if too_big:
            long_ones.add(c.project.id)
            lines.setdefault(c.project.id, []).append(f"{label_of(c.base_model)} tardaría ~{est}, más que el máximo nocturno de {limit}.")
        else:
            lines.setdefault(c.project.id, []).append(f"{label_of(c.base_model)} (~{est}) no cabe en el tiempo que queda esta noche.")
    for pid, msgs in lines.items():
        p = projects[pid]
        tip = (
            " Si quieres que entre de noche, baja las épocas o el tamaño de imagen, o usa un modelo más rápido."
            if pid in long_ones else " Si otra noche queda espacio, se entrena entonces."
        )
        notify_user(
            db, p.owner, "training.night_left_out", f"«{p.name}» no se entrena esta noche",
            " ".join(msgs) + " Se entrenará el sábado, con el resto de la cola (del más largo al más corto)." + tip,
            meta={"project_id": pid, "night": night},
            email=False,
        )
    return sent


def _record_night(db: Session, night: str, capacity: float, busy: bool) -> NightReport:
    """Resumen de la noche: se crea en la primera pasada y no cambia (los resultados se leen de las ejecuciones)."""
    report = db.scalar(select(NightReport).where(NightReport.night == night))
    if report is not None:
        return report
    report = NightReport(night=night, capacity_seconds=max(capacity, 0), items=[], not_ready=[], notified=[])
    if busy:
        report.note = "Había un entrenamiento semanal o manual en curso, así que esta noche no se planificó nada."
    else:
        cands, skipped = candidates(db, night)
        fit, too_long, no_room = _assign(cands, max(capacity, 0), get_settings().night_max_job_hours * 3600)
        report.items = (
            [_row(c, "tonight") for c in fit] + [_row(c, "too_long") for c in too_long] + [_row(c, "no_room") for c in no_room]
        )
        report.not_ready = skipped
        report.notified = _notify_left_out(db, night, too_long, no_room)
    db.add(report)
    db.commit()
    return report


# ------------------------------------------------------------------ pregunta
def _people_to_ask(db: Session, project: Project, since: datetime) -> list[User]:
    """Quienes modificaron el proyecto recientemente y pueden decidir (propietario o editor)."""
    ids = {project.last_activity_by_id} if project.last_activity_by_id else set()
    ids |= set(
        db.scalars(
            select(AuditLog.actor_id).where(
                AuditLog.target_type == "project", AuditLog.target_id == str(project.id), AuditLog.created_at >= since
            )
        ).all()
    )
    people = [u for u in (db.get(User, i) for i in ids if i) if u is not None]
    return [u for u in people if _level(project_role(project, u)) >= _LEVEL[EDIT]]


def check_dict(c: NightCheck) -> dict:
    return {
        "id": c.id,
        "project_id": c.project_id,
        "project_name": c.project.name if c.project else None,
        "night": c.night,
        "status": c.status,
        "reason": c.reason,
        "created_at": c.created_at,
        "answered_at": c.answered_at,
        "timeout_minutes": get_settings().night_ask_timeout_minutes,
    }


def _ask(db: Session, project: Project, night: str, now) -> NightCheck | None:
    s = get_settings()
    people = _people_to_ask(db, project, now - timedelta(minutes=s.night_activity_minutes))
    if not people:
        return None  # nadie con permisos que preguntar: se entrena normalmente
    check = NightCheck(project_id=project.id, night=night, status=NC_ASKING, asked_user_ids=[u.id for u in people])
    db.add(check)
    db.flush()
    for u in people:
        notify_user(
            db, u, "training.night_check", f"¿Todavía te faltan cargas en «{project.name}»?",
            f"Esta noche se va a entrenar «{project.name}», pero vimos que estás trabajando en él. Si todavía te faltan "
            "cargas, responde «Sí, sigo cargando» y no se entrenará esta noche. Si no respondes en "
            f"{s.night_ask_timeout_minutes} minutos, se entrena si ya dejaste de cargar.",
            meta={"night_check_id": check.id, "project_id": project.id},
            email=False,
        )
    return check


def publish_check(check: NightCheck) -> None:
    events.publish({"type": "night_check", "user_ids": list(check.asked_user_ids or []), "check": check_dict(check)})


def _decide(db: Session, project: Project, night: str, now) -> bool:
    """¿Se puede congelar ya el proyecto? Crea o resuelve la pregunta nocturna si corresponde."""
    s = get_settings()
    check = db.scalar(select(NightCheck).where(NightCheck.project_id == project.id, NightCheck.night == night))
    if check is not None:
        if check.status == NC_SKIP:
            return False
        if check.status == NC_GO:
            return True
        if now - check.created_at < timedelta(minutes=s.night_ask_timeout_minutes):
            return False  # esperando respuesta
        # sin respuesta: si siguió trabajando después de la pregunta, no se entrena esta noche
        still = project.last_activity_at is not None and project.last_activity_at > check.created_at
        check.status = NC_SKIP if still else NC_GO
        check.reason = "Sin respuesta y siguió cargando" if still else "Sin respuesta y sin actividad"
        check.answered_at = now
        publish_check(check)
        return not still
    recent = project.last_activity_at and now - project.last_activity_at < timedelta(minutes=s.night_activity_minutes)
    if not recent:
        return True
    check = _ask(db, project, night, now)
    if check is None:
        return True
    publish_check(check)
    return False


# ------------------------------------------------------------------ planificación
def plan_night(db: Session) -> dict | None:
    """Una pasada del planificador. Idempotente: se puede llamar cada minuto.

    Devuelve un resumen sólo si encoló trabajos; si no (nada que entrenar, nada cabe o esperando respuestas), None.
    """
    local = calendar_rules.now_local()
    end = calendar_rules.night_window_end(local)
    if end is None:
        return None  # fuera de la ventana
    s = get_settings()
    night = calendar_rules.night_key(local)
    now = utcnow()
    capacity = (end - local).total_seconds() - _committed_seconds(db)
    busy = training_active(db)
    _record_night(db, night, capacity, busy)
    if busy:
        return None  # hay un entrenamiento semanal/manual en curso
    cands, _ = candidates(db, night)
    if not cands or capacity <= 0:
        return None
    fit, _, _ = _assign(cands, capacity, s.night_max_job_hours * 3600)
    by_project: dict[int, list[Candidate]] = {}
    for c in fit:
        by_project.setdefault(c.project.id, []).append(c)
    jobs: list[TrainingJob] = []
    started: list[str] = []
    for chosen in by_project.values():
        project = chosen[0].project
        if not _decide(db, project, night, now):
            continue
        run = WeeklyRun(week_key=run_key(night, project.id), trigger=tq.TRIGGER_NIGHT)
        db.add(run)
        db.flush()
        new_jobs, reason = tq.enqueue_project(
            db, project, run, tq.TRIGGER_NIGHT, only_models=[c.base_model for c in chosen]
        )
        run.summary = {"queued": len(new_jobs), "skipped": [], "project_ids": [project.id]}
        if not new_jobs:
            run.status, run.published_at = "published", utcnow()
            run.summary["skipped"] = [{"project_id": project.id, "project": project.name, "reason": reason}]
            continue
        jobs.extend(new_jobs)
        started.append(f"{project.name}: {', '.join(label_of(c.base_model) for c in chosen)}")
    db.commit()
    for job in jobs:
        tq.publish_job_event(job)
    if not jobs:
        return None
    log.info("Entrenamiento nocturno encolado: %s", " | ".join(started))
    tq.kick_queue()
    return {"queued": len(jobs), "projects": started}


def preview(db: Session) -> dict:
    """Qué se entrenaría la próxima noche (o en la actual) sin modificar nada: para administradores."""
    s = get_settings()
    local = calendar_rules.now_local()
    end = calendar_rules.night_window_end(local)
    if end is not None:
        capacity = (end - local).total_seconds() - _committed_seconds(db)
        night = calendar_rules.night_key(local)
    else:  # fuera de la ventana: la noche completa
        start = calendar_rules._hhmm(s.night_window_start)
        stop = calendar_rules._hhmm(s.night_window_end)
        capacity = (stop.hour * 60 + stop.minute - start.hour * 60 - start.minute) * 60
        night = calendar_rules.night_key(local + timedelta(days=1))
    cands, skipped = candidates(db, night)
    fit, too_long, no_room = _assign(cands, max(capacity, 0), s.night_max_job_hours * 3600)
    checks = {
        c.project_id: c
        for c in db.scalars(select(NightCheck).where(NightCheck.night == night)).unique().all()
    }

    def row(c: Candidate, decision: str) -> dict:
        check = checks.get(c.project.id)
        if decision == "tonight" and check is not None and check.status != NC_GO:
            decision = "asking" if check.status == NC_ASKING else "skipped_by_user"
        return _row(c, decision)

    return {
        "enabled": s.night_training_enabled,
        "in_window": end is not None,
        "night": night,
        "window": f"{s.night_window_start}–{s.night_window_end}",
        "capacity_seconds": round(max(capacity, 0)),
        "max_job_seconds": round(s.night_max_job_hours * 3600),
        "items": [row(c, "tonight") for c in fit]
        + [row(c, "too_long") for c in too_long]
        + [row(c, "no_room") for c in no_room],
        "not_ready": skipped,
    }


# ------------------------------------------------------------------ historial para administradores
def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def _report_dict(db: Session, night: str, r: NightReport | None, in_progress: bool) -> dict:
    if r is None:
        return {
            "night": night, "missing": True, "in_progress": False, "tone": "missing", "created_at": None,
            "capacity_seconds": 0, "items": [], "not_ready": [],
            "headline": "Sin registro: el planificador no corrió esa noche (¿estaba detenido el servidor o el worker?).",
        }
    checks = {c.project_id: c for c in db.scalars(select(NightCheck).where(NightCheck.night == night)).unique().all()}
    run_ids = select(WeeklyRun.id).where(WeeklyRun.week_key.like(f"night-{night}-p%"))
    jobs = db.scalars(select(TrainingJob).where(TrainingJob.run_id.in_(run_ids)).order_by(TrainingJob.id)).unique().all()
    by_model = {(j.project_id, j.base_model): j for j in jobs}

    def job_dict(j: TrainingJob) -> dict:
        return {"id": j.id, "status": j.status, "phase": j.phase}

    items = []
    for it in r.items or []:
        it = dict(it)
        j = by_model.pop((it["project_id"], it["base_model"]), None)
        if j is not None:
            it["job"] = job_dict(j)
        elif it["decision"] == "tonight":
            check = checks.get(it["project_id"])
            if check is not None and check.status == NC_SKIP:
                it["decision"] = "skipped_by_user"
            elif check is not None and check.status == NC_ASKING:
                it["decision"] = "asking"
            elif not in_progress:
                it["decision"] = "not_queued"
        items.append(it)
    for j in by_model.values():  # encolados en pasadas posteriores (p. ej. cambios hechos durante la noche)
        items.append({
            "project_id": j.project_id, "project": j.project.name if j.project else "", "base_model": j.base_model,
            "model": label_of(j.base_model), "estimated_seconds": round(j.estimated_seconds or 0), "source": None,
            "decision": "tonight", "job": job_dict(j),
        })

    done = sum(j.status == "completed" for j in jobs)
    failed = sum(j.status == "error" for j in jobs)
    postponed = sum(j.status == "cancelled" and j.phase == tq.POSTPONED_PHASE for j in jobs)
    active = sum(j.status in ACTIVE_JOB_STATES for j in jobs)
    count = lambda d: sum(i["decision"] == d for i in items)  # noqa: E731
    limit = tq.fmt_duration(get_settings().night_max_job_hours * 3600)
    if r.note:
        headline, tone = r.note, "idle"
    elif jobs:
        parts = []
        if done:
            parts.append(f"{'Se entrenó' if done == 1 else 'Se entrenaron'} {_plural(done, 'modelo', 'modelos')}")
        if active:
            parts.append(f"{_plural(active, 'modelo', 'modelos')} en cola o entrenando")
        if failed:
            parts.append(f"{_plural(failed, 'modelo terminó', 'modelos terminaron')} con error")
        if postponed:
            parts.append(f"{_plural(postponed, 'se pospuso', 'se pospusieron')} al sábado (ya no cabía)")
        cancelled = len(jobs) - done - active - failed - postponed
        if cancelled:
            parts.append(f"{_plural(cancelled, 'se canceló', 'se cancelaron')}")
        headline = "; ".join(parts) + "."
        tone = "error" if failed else ("running" if active else ("ok" if done else "idle"))
    elif not items:
        headline, tone = "No se entrenó nada: no había modelos pendientes (ningún proyecto marcado tiene cambios).", "idle"
    else:
        reasons = []
        if count("too_long"):
            reasons.append(f"{_plural(count('too_long'), 'modelo supera', 'modelos superan')} el máximo de {limit}")
        if count("no_room"):
            reasons.append(f"{_plural(count('no_room'), 'modelo no cabía', 'modelos no cabían')} en el tiempo disponible")
        if count("skipped_by_user"):
            reasons.append(f"{_plural(count('skipped_by_user'), 'proyecto seguía', 'proyectos seguían')} cargando")
        if count("asking"):
            reasons.append("esperando respuesta de quien estaba cargando")
        if count("not_queued"):
            reasons.append(f"{_plural(count('not_queued'), 'modelo no se encoló', 'modelos no se encolaron')}")
        if in_progress and count("tonight"):
            headline = f"Planificado: {_plural(count('tonight'), 'modelo', 'modelos')} esta noche."
        else:
            headline = "No se entrenó nada: " + ", ".join(reasons) + "."
        tone = "idle"
    return {
        "night": night, "missing": False, "in_progress": in_progress, "tone": tone, "created_at": r.created_at,
        "capacity_seconds": round(r.capacity_seconds or 0), "headline": headline, "items": items,
        "not_ready": r.not_ready or [],
    }


def reports(db: Session, limit: int = 10) -> list[dict]:
    """Las últimas noches hábiles, desde el primer resumen guardado; marca las que no tienen registro."""
    s = get_settings()
    first = db.scalar(select(func.min(NightReport.night)))
    if first is None:
        return []
    local = calendar_rules.now_local()
    current = calendar_rules.night_window_end(local) is not None
    window_start = datetime.combine(local.date(), calendar_rules._hhmm(s.night_window_start), tzinfo=local.tzinfo)
    d = local.date() if local >= window_start else local.date() - timedelta(days=1)
    nights: list[str] = []
    while d >= date.fromisoformat(first) and len(nights) < limit:
        if d.weekday() < 5:
            nights.append(d.isoformat())
        d -= timedelta(days=1)
    found = {r.night: r for r in db.scalars(select(NightReport).where(NightReport.night.in_(nights))).all()}
    today = calendar_rules.night_key(local)
    out = []
    for n in nights:
        r = found.get(n)
        if r is None and not s.night_training_enabled:
            continue  # desactivado: no es una falla
        if r is None and n == today and local - window_start < timedelta(minutes=5):
            continue  # la ventana recién empezó: el planificador aún no pasa
        out.append(_report_dict(db, n, r, in_progress=current and n == today))
    return out
