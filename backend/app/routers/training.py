import asyncio
import json

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_db
from ..deps import require_approved
from ..errors import ApiError
from ..events import CHANNEL
from ..models import ROLE_ADMIN, DatasetSnapshot, TrainingJob, TrainingLog, User
from ..services.access import accessible_project_ids, get_project, lock_state
from ..services.training_queue import queue_position, serialize_job

router = APIRouter(prefix="/api", tags=["training"])


def _visible_jobs(db: Session, user: User, project_id: int | None = None):
    q = select(TrainingJob).order_by(TrainingJob.id.desc())
    if project_id is not None:
        q = q.where(TrainingJob.project_id == project_id)
    if user.role != ROLE_ADMIN:  # filtrado en SQL, antes del límite (proyectos propios y compartidos)
        q = q.where(TrainingJob.project_id.in_(accessible_project_ids(user)))
    return db.scalars(q.limit(200)).unique().all()


@router.get("/schedule")
def schedule(user: User = Depends(require_approved), db: Session = Depends(get_db)):
    return lock_state(db)


@router.get("/training/jobs")
def list_jobs(project_id: int | None = None, user: User = Depends(require_approved), db: Session = Depends(get_db)):
    if project_id is not None:
        get_project(db, user, project_id)
    return [serialize_job(j, queue_position(db, j)) for j in _visible_jobs(db, user, project_id)]


@router.get("/training/jobs/{job_id}")
def read_job(job_id: int, user: User = Depends(require_approved), db: Session = Depends(get_db)):
    job = db.get(TrainingJob, job_id)
    if job is None:
        raise ApiError(404, "JOB_NOT_FOUND", "Trabajo no encontrado.")
    get_project(db, user, job.project_id)
    snap = db.get(DatasetSnapshot, job.snapshot_id) if job.snapshot_id else None
    return serialize_job(job, queue_position(db, job), snap)


@router.get("/training/jobs/{job_id}/logs")
def job_logs(job_id: int, limit: int = 300, user: User = Depends(require_approved), db: Session = Depends(get_db)):
    job = db.get(TrainingJob, job_id)
    if job is None:
        raise ApiError(404, "JOB_NOT_FOUND", "Trabajo no encontrado.")
    get_project(db, user, job.project_id)
    rows = db.scalars(
        select(TrainingLog).where(TrainingLog.job_id == job_id).order_by(TrainingLog.id.desc()).limit(min(limit, 1000))
    ).all()
    return [{"id": r.id, "level": r.level, "message": r.message, "created_at": r.created_at} for r in reversed(rows)]


@router.get("/training/stream")
async def stream(request: Request, user: User = Depends(require_approved)):
    """SSE: eventos de progreso. Cada usuario sólo recibe los de sus proyectos."""
    uid, is_admin = user.id, user.role == ROLE_ADMIN
    client = aioredis.Redis.from_url(get_settings().redis_url)

    async def gen():
        pubsub = client.pubsub()
        await pubsub.subscribe(CHANNEL)
        yield "retry: 3000\n\n"
        try:
            last_ping = asyncio.get_event_loop().time()
            while True:
                if await request.is_disconnected():
                    break
                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if msg and msg["type"] == "message":
                    try:
                        payload = json.loads(msg["data"])
                    except ValueError:
                        continue
                    if is_admin or payload.get("owner_id") == uid or uid in payload.get("member_ids", ()) or uid in payload.get("user_ids", ()):
                        yield f"event: {payload.get('type', 'message')}\ndata: {json.dumps(payload)}\n\n"
                now = asyncio.get_event_loop().time()
                if now - last_ping > 15:
                    last_ping = now
                    yield ": ping\n\n"
        finally:
            await pubsub.unsubscribe(CHANNEL)
            await pubsub.aclose()
            await client.aclose()

    return StreamingResponse(
        gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@router.get("/architectures")
def architectures(user: User = Depends(require_approved)):
    """Catálogo de modelos base disponibles para entrenar."""
    from ..ml.architectures import DEFAULT_ARCH, public_catalog

    return {"default": DEFAULT_ARCH, "items": public_catalog()}
