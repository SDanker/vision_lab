"""Cola de entrenamiento: snapshots, ejecución semanal, recuperación y publicación."""
import hashlib
import json
import logging
import os
import shutil
from datetime import timedelta
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .. import calendar_rules, events
from ..config import get_settings
from ..db import utcnow
from ..models import (
    ACTIVE_JOB_STATES,
    J_CANCELLED,
    J_COMPLETED,
    J_ERROR,
    J_PENDING,
    MEMBER_EDITOR,
    RUNNING_JOB_STATES,
    Category,
    DatasetSnapshot,
    Image,
    ModelVersion,
    Project,
    TrainingJob,
    TrainingLog,
    WeeklyRun,
)
from ..ml.architectures import label_of
from ..modalities import get_modality
from ..storage import data_dir, image_path
from .audit import audit
from .notifications import notify_user

log = logging.getLogger(__name__)

TRIGGER_NIGHT = "night"  # ejecuciones nocturnas de lunes a viernes (services/night_plan.py)
POSTPONED_PHASE = "Pospuesto para el fin de semana"  # trabajo nocturno que ya no cabía antes de empezar


def fmt_duration(seconds: float) -> str:
    m = max(int(round(seconds / 60)), 1)
    return f"{m // 60} h {m % 60:02d} min" if m >= 60 else f"{m} min"


# ----------------------------------------------------------------- serialización
def queue_position(db: Session, job: TrainingJob) -> int | None:
    """Mismo orden que claim_next_job: queue_order y luego id."""
    if job.status != J_PENDING:
        return 0 if job.status in RUNNING_JOB_STATES else None
    order = job.queue_order or 0.0
    ahead = db.scalar(
        select(func.count(TrainingJob.id)).where(
            TrainingJob.status == J_PENDING,
            (TrainingJob.queue_order < order) | ((TrainingJob.queue_order == order) & (TrainingJob.id < job.id)),
        )
    )
    return int(ahead or 0) + 1


def serialize_job(job: TrainingJob, position: int | None = None, snapshot: DatasetSnapshot | None = None) -> dict:
    elapsed = None
    if job.started_at:
        end = job.finished_at or utcnow()
        elapsed = max((end - job.started_at).total_seconds(), 0)
    d = {
        "id": job.id,
        "project_id": job.project_id,
        "project_name": job.project.name if job.project else None,
        "owner_id": job.project.owner_id if job.project else None,
        "status": job.status,
        "phase": job.phase,
        "base_model": job.base_model,
        "trigger": job.trigger,
        "config": job.config,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "epoch": job.epoch,
        "total_epochs": job.total_epochs,
        "percent": round(job.percent or 0.0, 1),
        "eta_seconds": job.eta_seconds,
        "estimated_seconds": job.estimated_seconds,
        "elapsed_seconds": elapsed,
        "metrics": job.metrics,
        "history": job.history or [],
        "cpu_percent": job.cpu_percent,
        "ram_mb": job.ram_mb,
        "error": job.error,
        "queue_position": position,
        "model_version_id": job.model_version_id,
        "attempts": job.attempts,
        "cancel_requested": job.cancel_requested,
    }
    if snapshot is not None:
        d["snapshot"] = {
            "id": snapshot.id,
            "dataset_hash": snapshot.dataset_hash,
            "class_counts": snapshot.class_counts,
            "excluded": snapshot.excluded,
            "total_images": snapshot.total_images,
            "created_at": snapshot.created_at,
        }
    return d


def publish_job_event(job: TrainingJob) -> None:
    project = job.project
    events.publish(
        {
            "type": "job",
            "owner_id": project.owner_id,
            "member_ids": [m.user_id for m in project.members],  # usuarios con quienes se compartió
            "job": serialize_job(job),
        }
    )


def add_log(db: Session, job_id: int, message: str, level: str = "info") -> None:
    db.add(TrainingLog(job_id=job_id, level=level, message=message[:4000]))


# --------------------------------------------------------------------- celery
def kick_queue() -> None:
    try:
        from ..celery_app import celery_app

        celery_app.send_task("vision.process_queue", queue="training")
    except Exception as exc:  # noqa: BLE001
        log.warning("No se pudo despertar la cola de entrenamiento: %s", exc)


# ------------------------------------------------------------------- snapshots
def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _dataset(db: Session, project: Project):
    """Categorías que cumplen el mínimo, sus imágenes y el hash del dataset (sin copiar archivos).

    Devuelve (eligible, counts, excluded, images, hash) o, si no se puede entrenar, (None, motivo).
    En datos tabulares las clases salen de una columna y `images` es el número de filas (ver services/tabular.py).
    """
    if get_modality(project.modality).info.tabular:
        from . import tabular

        return tabular.dataset(db, project)
    min_n = get_settings().min_images_per_class
    counts = dict(
        db.execute(
            select(Image.category_id, func.count(Image.id)).where(Image.project_id == project.id).group_by(Image.category_id)
        ).all()
    )
    cats: list[Category] = sorted(project.categories, key=lambda c: c.name)
    eligible = [c for c in cats if counts.get(c.id, 0) >= min_n]
    excluded = {c.name: counts.get(c.id, 0) for c in cats if counts.get(c.id, 0) < min_n}
    noun = get_modality(project.modality).info.unit
    if len(eligible) < 2:
        detail = ", ".join(f"{c.name}: {counts.get(c.id, 0)}" for c in cats) or "sin categorías"
        return None, (
            f"Para entrenar se necesitan al menos 2 categorías con {min_n} o más {noun} cada una. "
            f"Tu proyecto tiene {len(eligible)} que cumplen ({detail}). Agrega otra categoría con {noun}."
        )
    images = db.scalars(
        select(Image).where(Image.category_id.in_([c.id for c in eligible])).order_by(Image.id)
    ).all()
    label_of = {c.id: c.name for c in eligible}
    digest = hashlib.sha256()
    for line in sorted(f"{label_of[i.category_id]}:{i.sha256}" for i in images):
        digest.update(line.encode())
    return (eligible, counts, excluded, images, digest.hexdigest()), ""


def dataset_digest(db: Session, project: Project) -> tuple[str | None, int, str]:
    """(hash, nº de imágenes, motivo si no se puede entrenar): el mismo hash que tendría un snapshot ahora."""
    data, reason = _dataset(db, project)
    if data is None:
        return None, 0, reason
    return data[4], (data[3] if isinstance(data[3], int) else len(data[3])), ""


def models_needing_training(db: Session, project: Project, digest: str) -> list[tuple[str, dict]]:
    """Modelos seleccionados cuyo dataset o configuración cambió desde su última versión."""
    out = []
    for base_model in project_models(project):
        cfg = job_config_for(project, base_model)
        last = db.scalar(
            select(ModelVersion)
            .where(ModelVersion.project_id == project.id, ModelVersion.base_model == base_model)
            .order_by(ModelVersion.version.desc())
            .limit(1)
        )
        if not (last and last.dataset_hash == digest and last.config == cfg):
            out.append((base_model, cfg))
    return out


def create_snapshot(db: Session, project: Project, run_id: int | None) -> tuple[DatasetSnapshot | None, str]:
    if get_modality(project.modality).info.tabular:
        from . import tabular

        return tabular.create_snapshot(db, project, run_id)
    data, reason = _dataset(db, project)
    if data is None:
        return None, reason
    eligible, counts, excluded, images, digest = data
    label_of = {c.id: c.name for c in eligible}
    labels = [c.name for c in eligible]
    snap = DatasetSnapshot(
        project_id=project.id,
        run_id=run_id,
        dataset_hash=digest,
        class_counts={c.name: counts[c.id] for c in eligible},
        excluded=excluded or None,
        total_images=len(images),
        path="",
    )
    db.add(snap)
    db.flush()
    root = data_dir() / "snapshots" / str(snap.id)
    (root / "images").mkdir(parents=True, exist_ok=True)
    items = []
    for img in images:
        name = f"{img.id}.{img.fmt}"
        _link_or_copy(image_path(img), root / "images" / name)
        items.append({"file": f"images/{name}", "label": label_of[img.category_id], "sha256": img.sha256,
                      "duration": img.duration_s})
    (root / "manifest.json").write_text(
        json.dumps({"labels": labels, "dataset_hash": snap.dataset_hash, "items": items}), encoding="utf-8"
    )
    snap.path = str(root)
    return snap, ""


def project_models(project: Project) -> list[str]:
    """Modelos base seleccionados para el proyecto (uno o más)."""
    return list((project.config or {}).get("base_models") or [project.base_model])


def job_config_for(project: Project, base_model: str) -> dict:
    """Configuración efectiva de un modelo: valores generales + ajustes propios de ese modelo.

    Los ajustes por modelo (`model_overrides`) se aplican aquí, de modo que el entrenamiento y la
    comparación «sin cambios» siempre ven la configuración final y no necesitan saber de overrides.
    """
    raw = project.config or {}
    cfg = {k: v for k, v in raw.items() if k not in ("base_models", "model_overrides")}
    cfg.update({k: v for k, v in (raw.get("model_overrides") or {}).get(base_model, {}).items() if v is not None})
    cfg = get_modality(project.modality).job_config(cfg)
    return {**cfg, "base_model": base_model}


def enqueue_project(
    db: Session, project: Project, run: WeeklyRun, trigger: str, force: bool = False,
    only_models: list[str] | None = None,
) -> tuple[list[TrainingJob], str]:
    """Un único snapshot del dataset y un trabajo por cada modelo base seleccionado.

    Cada trabajo lleva su duración estimada y su lugar en la cola: de noche primero los más cortos (caben
    más en la ventana); el fin de semana y en los manuales, primero los más largos.
    `only_models` limita los modelos encolados (de noche, sólo los que caben en la ventana).
    """
    from .estimates import estimate_job_seconds

    snap, reason = create_snapshot(db, project, run.id)
    if snap is None:
        return [], reason
    jobs, unchanged = [], []
    for base_model in project_models(project):
        if only_models is not None and base_model not in only_models:
            continue
        cfg = job_config_for(project, base_model)
        if not force:
            last = db.scalar(
                select(ModelVersion)
                .where(ModelVersion.project_id == project.id, ModelVersion.base_model == base_model)
                .order_by(ModelVersion.version.desc())
                .limit(1)
            )
            if last and last.dataset_hash == snap.dataset_hash and last.config == cfg:
                unchanged.append(f"{label_of(base_model)} (v{last.version})")
                continue
        estimate, source = estimate_job_seconds(db, project.id, base_model, cfg, snap.total_images)
        job = TrainingJob(
            project_id=project.id,
            run_id=run.id,
            snapshot_id=snap.id,
            status=J_PENDING,
            base_model=base_model,
            config=cfg,
            trigger=trigger,
            total_epochs=int(cfg.get("epochs", 10)) + int(cfg.get("fine_tune_epochs", 0)),
            estimated_seconds=round(estimate),
            queue_order=estimate if trigger == TRIGGER_NIGHT else -estimate,
        )
        db.add(job)
        db.flush()
        add_log(
            db, job.id,
            f"Trabajo encolado ({trigger}) con {label_of(base_model)}. Dataset {snap.dataset_hash[:12]} con {snap.total_images} "
            f"{get_modality(project.modality).sample_unit or get_modality(project.modality).info.unit}. "
            f"Duración estimada: {fmt_duration(estimate)} (según {source}).",
        )
        jobs.append(job)
    if not jobs:
        shutil.rmtree(snap.path, ignore_errors=True)
        db.delete(snap)
        return [], f"Sin cambios desde la última versión de: {', '.join(unchanged)}; no se reentrena."
    note = f" Sin cambios (omitidos): {', '.join(unchanged)}." if unchanged else ""
    return jobs, note.strip()


def start_run(
    db: Session, week_key: str, trigger: str, project_ids: list[int] | None = None, force: bool = False
) -> WeeklyRun | None:
    run = WeeklyRun(week_key=week_key, trigger=trigger)
    try:
        with db.begin_nested():
            db.add(run)
    except IntegrityError:
        return None  # ya existe (idempotente ante ticks duplicados)
    q = select(Project).order_by(Project.id)
    if project_ids is not None:
        q = q.where(Project.id.in_(project_ids))
    else:
        q = q.where(Project.auto_train.is_(True))
    jobs, skipped = [], []
    projects = db.scalars(q).unique().all()
    for project in projects:
        new_jobs, reason = enqueue_project(db, project, run, trigger, force=force)
        jobs.extend(new_jobs)
        if not new_jobs:
            skipped.append({"project_id": project.id, "project": project.name, "reason": reason})
            notify_user(
                db, project.owner, "training.skipped", f"Entrenamiento omitido: {project.name}", reason, email=False
            )
        elif reason:
            skipped.append({"project_id": project.id, "project": project.name, "reason": reason})
    run.summary = {"queued": len(jobs), "skipped": skipped, "project_ids": [p.id for p in projects]}
    if not jobs:
        run.status = "published"
        run.published_at = utcnow()
        reset_weekly_optin(db, run)  # nada que esperar: la semana ya terminó
    db.commit()
    for job in jobs:
        publish_job_event(job)
    if jobs:
        kick_queue()
    return run


# ---------------------------------------------------------------- mantenimiento
def reset_weekly_optin(db: Session, run: WeeklyRun) -> int:
    """Cierra el ciclo: desmarca los proyectos que participaron en una ejecución automática (semanal o nocturna).

    Así cada ronda empieza igual: la persona revisa los resultados, ajusta el modelo y, si quiere que se entrene
    de nuevo, vuelve a marcar «Incluir en el entrenamiento semanal automático» y guarda (al guardar se le avisa
    cuándo se ejecutaría). Las ejecuciones manuales de un administrador no tocan la casilla.
    """
    ids = (run.summary or {}).get("project_ids") or []
    if run.trigger not in ("schedule", TRIGGER_NIGHT) or not ids:
        return 0
    when = calendar_rules.describe_next_execution()
    which = "nocturno" if run.trigger == TRIGGER_NIGHT else "semanal"
    projects = db.scalars(select(Project).where(Project.id.in_(ids), Project.auto_train.is_(True))).unique().all()
    for p in projects:
        p.auto_train = False
        audit(db, "project.weekly_optin_reset", None, "project", p.id, {"run": run.week_key})
        for person in [p.owner, *(m.user for m in p.members if m.role == MEMBER_EDITOR)]:
            notify_user(
                db, person, "training.optin_reset", f"Revisa «{p.name}» y vuelve a marcarlo",
                f"Terminó el entrenamiento {which} y «{p.name}» quedó desmarcado. Revisa los resultados, ajusta el "
                "modelo y, si quieres que se entrene otra vez, marca «Incluir en el entrenamiento semanal automático» "
                f"y guarda: se ejecutaría {when}.",
                email=False,
            )
    return len(projects)


def finalize_runs(db: Session) -> int:
    """Publica los resultados de ejecuciones cuyos trabajos ya terminaron."""
    done = 0
    for run in db.scalars(select(WeeklyRun).where(WeeklyRun.status == "running")).all():
        jobs = db.scalars(select(TrainingJob).where(TrainingJob.run_id == run.id)).unique().all()
        if any(j.status in ACTIVE_JOB_STATES for j in jobs):
            continue
        ok = [j for j in jobs if j.status == J_COMPLETED]
        failed = [j for j in jobs if j.status == J_ERROR]
        run.status = "published"
        run.published_at = utcnow()
        run.summary = {**(run.summary or {}), "completed": len(ok), "failed": len(failed)}
        reset_weekly_optin(db, run)
        for j in jobs:
            # el propietario y las personas con quienes se compartió el proyecto
            people = [j.project.owner, *(m.user for m in j.project.members)]
            if j.status == J_COMPLETED:
                for person in people:
                    notify_user(
                        db, person, "training.published",
                        f"Resultados publicados: {j.project.name}",
                        "El entrenamiento terminó y ya puedes revisar métricas y descargar el modelo.",
                    )
            elif j.status == J_ERROR:
                for person in people:
                    notify_user(
                        db, person, "training.failed", f"Entrenamiento con error: {j.project.name}",
                        (j.error or "")[:300],
                    )
        done += 1
    if done:
        db.commit()
    return done


def recover_stale(db: Session) -> int:
    s = get_settings()
    limit = utcnow() - timedelta(seconds=s.heartbeat_stale_seconds)
    n = 0
    for job in db.scalars(select(TrainingJob).where(TrainingJob.status.in_(RUNNING_JOB_STATES))).unique().all():
        beat = job.heartbeat_at or job.started_at or job.created_at
        if beat > limit:
            continue
        if job.cancel_requested:
            job.status, job.finished_at = J_CANCELLED, utcnow()
        elif job.attempts >= s.max_job_attempts:
            job.status, job.finished_at = J_ERROR, utcnow()
            job.error = "El trabajo se interrumpió repetidamente (posible falta de memoria o reinicios del servidor)."
        else:
            job.status = J_PENDING
            add_log(db, job.id, "Trabajo recuperado tras interrupción del worker; se reintentará.", "warning")
        n += 1
    if n:
        db.commit()
    return n


def has_pending_and_idle(db: Session) -> bool:
    pending = db.scalar(select(func.count(TrainingJob.id)).where(TrainingJob.status == J_PENDING))
    running = db.scalar(select(func.count(TrainingJob.id)).where(TrainingJob.status.in_(RUNNING_JOB_STATES)))
    return bool(pending) and not running


def cancel_job(db: Session, job: TrainingJob) -> None:
    if job.status == J_PENDING:
        job.status, job.finished_at = J_CANCELLED, utcnow()
        add_log(db, job.id, "Trabajo cancelado por un administrador.", "warning")
    elif job.status in RUNNING_JOB_STATES:
        job.cancel_requested = True
        add_log(db, job.id, "Cancelación solicitada; se detendrá al terminar el paso actual.", "warning")
