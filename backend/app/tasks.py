import logging
import shutil
from datetime import timedelta

from sqlalchemy import select, text

from . import calendar_rules
from .celery_app import celery_app
from .db import SessionLocal, engine, utcnow
from .models import ACTIVE_JOB_STATES, DatasetSnapshot, TrainingJob
from .services import hf, night_plan, training_queue as tq
from .storage import data_dir

log = logging.getLogger(__name__)
ADVISORY_KEY = 424242  # garantiza un único entrenamiento intensivo a la vez


def ensure_weekly_run() -> bool:
    """Crea la ejecución semanal si estamos en la ventana (sáb >= 00:05 / dom) y no existe."""
    key = calendar_rules.week_key_for(calendar_rules.now_local())
    if not key:
        return False
    with SessionLocal() as db:
        return tq.start_run(db, key, "schedule") is not None


@celery_app.task(name="vision.weekly_start")
def weekly_start():
    created = ensure_weekly_run()
    log.info("Cola semanal %s", "creada" if created else "ya existía")
    return created


@celery_app.task(name="vision.tick")
def tick():
    """Latido cada minuto: recupera trabajos, cubre arranques perdidos y publica resultados."""
    ensure_weekly_run()  # cubre caídas de beat durante el sábado 00:05
    with SessionLocal() as db:
        tq.recover_stale(db)
        tq.finalize_runs(db)
        try:
            hf.recover_stale(db)  # importaciones de Hugging Face cuyo worker murió
        except Exception:  # noqa: BLE001
            db.rollback()
            log.exception("Fallo al revisar las importaciones de Hugging Face")
        try:
            night_plan.plan_night(db)  # lunes a viernes de noche: modelos cortos que caben en la ventana
        except Exception:  # noqa: BLE001 - el latido nunca debe caerse por el planificador
            db.rollback()
            log.exception("Fallo del planificador nocturno")
        if tq.has_pending_and_idle(db):
            tq.kick_queue()


@celery_app.task(name="vision.cleanup")
def cleanup(keep_per_project: int = 2):
    """Elimina snapshots antiguos (hardlinks) y directorios de trabajo huérfanos."""
    with SessionLocal() as db:
        in_use = {
            j.snapshot_id for j in db.scalars(select(TrainingJob).where(TrainingJob.status.in_(ACTIVE_JOB_STATES)))
        }
        by_project: dict[int, list[DatasetSnapshot]] = {}
        for snap in db.scalars(select(DatasetSnapshot).order_by(DatasetSnapshot.id.desc())):
            by_project.setdefault(snap.project_id, []).append(snap)
        removed = 0
        for snaps in by_project.values():
            for snap in snaps[keep_per_project:]:
                if snap.id in in_use:
                    continue
                shutil.rmtree(snap.path, ignore_errors=True)
                snap.path = ""
                removed += 1
        db.commit()
    work = data_dir() / "work"
    if work.exists():
        for d in work.iterdir():
            if d.is_dir() and utcnow().timestamp() - d.stat().st_mtime > timedelta(days=7).total_seconds():
                shutil.rmtree(d, ignore_errors=True)
    return removed


@celery_app.task(name="vision.process_queue")
def process_queue():
    """Procesa trabajos pendientes de uno en uno. Un lock consultivo de PostgreSQL
    impide que dos entrenamientos intensivos coincidan aunque haya varios workers."""
    from .ml.runner import claim_next_job, run_job

    conn = engine.connect()
    locked = True
    try:
        if engine.dialect.name == "postgresql":
            locked = bool(conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": ADVISORY_KEY}).scalar())
        if not locked:
            return "otro worker ya entrena"
        while True:
            job_id = claim_next_job()
            if job_id is None:
                break
            run_job(job_id)
    finally:
        if engine.dialect.name == "postgresql" and locked:
            conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": ADVISORY_KEY})
        conn.close()


@celery_app.task(name="vision.hf_import")
def hf_import(import_id: int):
    """Descarga un dataset de Hugging Face hacia un proyecto; corre en el worker liviano (cola «default»)."""
    hf.run_import(import_id)


@celery_app.task(name="vision.build_similarity", bind=True, max_retries=None)
def build_similarity(self, model_id: int):
    """Calcula (o completa) el índice de «lo más parecido» de un modelo; corre en el worker liviano.

    Los índices se calculan de uno en uno (usan mucha CPU y memoria): si hay otro en curso, espera su turno.
    """
    from .services import similarity

    if not similarity.acquire_build_lock():
        similarity.mark_queued(model_id)
        raise self.retry(countdown=60)
    try:
        return similarity.build_index(model_id)
    finally:
        similarity.release_build_lock()
