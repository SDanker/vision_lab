import time
from pathlib import Path

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_db, utcnow
from ..deps import require_admin
from ..errors import ApiError
from ..models import (
    ACTIVE_JOB_STATES,
    APPROVED,
    J_CANCELLED,
    J_COMPLETED,
    J_ERROR,
    J_PENDING,
    PENDING,
    REJECTED,
    REQ_APPROVED,
    REQ_PENDING,
    REQ_REJECTED,
    REQ_STATUSES,
    ROLE_ADMIN,
    SUSPENDED,
    USER_STATUSES,
    ApiToken,
    ApiTokenRequest,
    AuditLog,
    DatasetSnapshot,
    Image,
    ModelExport,
    ModelVersion,
    Project,
    TrainingJob,
    User,
    UserStatusHistory,
    WeeklyRun,
)
from ..schemas import BulkApproveIn, EnqueueIn, ReviewIn, TokenApproveIn, UserPatchIn
from ..services import api_tokens as tokens_svc, night_plan, resources, training_queue as tq
from ..services.access import lock_state
from ..services.audit import audit
from ..services.notifications import notify_user, smtp_configured
from ..services.users import change_status, ensure_not_last_admin, user_to_dict
from ..storage import delete_project_files, user_image_bytes, user_model_bytes, user_quota_bytes

router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(require_admin)])


def _get_user(db: Session, user_id: int) -> User:
    u = db.get(User, user_id)
    if u is None:
        raise ApiError(404, "USER_NOT_FOUND", "Usuario no encontrado.")
    return u


@router.get("/dashboard")
def dashboard(db: Session = Depends(get_db)):
    by_status = dict(db.execute(select(User.status, func.count(User.id)).group_by(User.status)).all())
    jobs = dict(db.execute(select(TrainingJob.status, func.count(TrainingJob.id)).group_by(TrainingJob.status)).all())
    images = db.scalar(select(func.count(Image.id))) or 0
    img_bytes = db.scalar(select(func.coalesce(func.sum(Image.size_bytes), 0))) or 0
    model_bytes = db.scalar(select(func.coalesce(func.sum(ModelExport.size_bytes), 0))) or 0
    last_run = db.scalar(select(WeeklyRun).order_by(WeeklyRun.id.desc()).limit(1))
    return {
        "users": {
            "total": sum(by_status.values()),
            "pending": by_status.get(PENDING, 0),
            "approved": by_status.get(APPROVED, 0),
            "rejected": by_status.get(REJECTED, 0),
            "suspended": by_status.get(SUSPENDED, 0),
        },
        "projects": db.scalar(select(func.count(Project.id))) or 0,
        "images": images,
        "storage": {"images_bytes": int(img_bytes), "models_bytes": int(model_bytes)},
        "training": {
            "queued": jobs.get(J_PENDING, 0),
            "active": sum(v for k, v in jobs.items() if k in ACTIVE_JOB_STATES and k != J_PENDING),
            "completed": jobs.get(J_COMPLETED, 0),
            "failed": jobs.get(J_ERROR, 0),
            "cancelled": jobs.get(J_CANCELLED, 0),
        },
        "models": db.scalar(select(func.count(ModelVersion.id))) or 0,
        "token_requests_pending": db.scalar(
            select(func.count(ApiTokenRequest.id)).where(ApiTokenRequest.status == REQ_PENDING)
        ) or 0,
        "lock": lock_state(db),
        "last_run": {
            "week_key": last_run.week_key, "status": last_run.status, "started_at": last_run.started_at,
            "published_at": last_run.published_at, "summary": last_run.summary,
        }
        if last_run
        else None,
        "smtp_configured": smtp_configured(),
        "resources": resources.snapshot(),
    }


@router.get("/resources")
def get_resources():
    return resources.snapshot()


# ---------------------------------------------------------------- usuarios
@router.get("/users")
def list_users(status: str | None = None, q: str | None = None, db: Session = Depends(get_db)):
    query = select(User).order_by(User.created_at.desc())
    if status:
        if status not in USER_STATUSES:
            raise ApiError(422, "BAD_STATUS", "Estado inválido.")
        query = query.where(User.status == status)
    if q:
        like = f"%{q.lower()}%"
        query = query.where(
            or_(func.lower(User.email).like(like), func.lower(User.username).like(like),
                func.lower(User.first_name).like(like), func.lower(User.last_name).like(like))
        )
    out = []
    for u in db.scalars(query.limit(500)).unique().all():
        d = user_to_dict(u)
        d["storage_bytes"] = user_image_bytes(db, u.id) + user_model_bytes(db, u.id)
        d["quota_bytes"] = user_quota_bytes(u)
        d["projects"] = db.scalar(select(func.count(Project.id)).where(Project.owner_id == u.id)) or 0
        out.append(d)
    return out


@router.get("/registrations")
def registrations(status: str | None = None, db: Session = Depends(get_db)):
    """Sección «Solicitudes de registro»."""
    query = select(User).order_by(User.created_at.desc())
    query = query.where(User.status == status) if status else query.where(User.role != ROLE_ADMIN)
    return [user_to_dict(u) for u in db.scalars(query.limit(500)).unique().all()]


def _action(action: str):
    def handler(user_id: int, data: ReviewIn, request: Request, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
        target = _get_user(db, user_id)
        change_status(db, target, admin, action, data.note, request)
        db.commit()
        return user_to_dict(target)

    return handler


for _name in ("approve", "reject", "suspend", "reactivate"):
    router.add_api_route(f"/users/{{user_id}}/{_name}", _action(_name), methods=["POST"], name=_name)


@router.post("/users/bulk-approve")
def bulk_approve(data: BulkApproveIn, request: Request, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    if not data.confirm:
        raise ApiError(400, "CONFIRMATION_REQUIRED", "Debes confirmar la aprobación múltiple.")
    results = []
    for uid in dict.fromkeys(data.user_ids):
        target = db.get(User, uid)
        try:
            if target is None:
                raise ApiError(404, "USER_NOT_FOUND", "Usuario no encontrado.")
            with db.begin_nested():
                change_status(db, target, admin, "approve", None, request)
            results.append({"id": uid, "ok": True})
        except ApiError as exc:
            results.append({"id": uid, "ok": False, "error": exc.message})
    audit(db, "user.bulk_approved", admin, "user", None, {"ids": data.user_ids}, request)
    db.commit()
    return {"results": results, "approved": sum(r["ok"] for r in results)}


@router.get("/users/{user_id}/history")
def user_history(user_id: int, db: Session = Depends(get_db)):
    _get_user(db, user_id)
    rows = db.scalars(
        select(UserStatusHistory).where(UserStatusHistory.user_id == user_id).order_by(UserStatusHistory.id.desc())
    ).unique().all()
    return [
        {"id": r.id, "from_status": r.from_status, "to_status": r.to_status, "note": r.note,
         "actor": r.actor.username if r.actor else None, "created_at": r.created_at}
        for r in rows
    ]


@router.patch("/users/{user_id}")
def patch_user(user_id: int, data: UserPatchIn, request: Request, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    target = _get_user(db, user_id)
    if data.role and data.role != target.role:
        if target.id == admin.id:
            raise ApiError(403, "SELF_ACTION", "No puedes cambiar tu propio rol.")
        if data.role == ROLE_ADMIN and target.status != APPROVED:
            raise ApiError(409, "NOT_APPROVED", "Sólo se puede promover a administrador a usuarios aprobados.")
        if data.role != ROLE_ADMIN:
            ensure_not_last_admin(db, target)
        audit(db, "user.role_changed", admin, "user", target.id, {"from": target.role, "to": data.role}, request)
        target.role = data.role
        target.token_version += 1
    if data.clear_quota:
        target.quota_mb = None
    elif data.quota_mb is not None:
        target.quota_mb = data.quota_mb
    if data.clear_quota or data.quota_mb is not None:
        audit(db, "user.quota_changed", admin, "user", target.id, {"quota_mb": target.quota_mb}, request)
    db.commit()
    return user_to_dict(target)


@router.delete("/users/{user_id}", status_code=204)
def delete_user(user_id: int, request: Request, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    target = _get_user(db, user_id)
    if target.id == admin.id:
        raise ApiError(403, "SELF_ACTION", "No puedes eliminar tu propia cuenta.")
    ensure_not_last_admin(db, target)
    busy = db.scalar(
        select(func.count(TrainingJob.id))
        .join(Project, Project.id == TrainingJob.project_id)
        .where(Project.owner_id == target.id, TrainingJob.status.in_(ACTIVE_JOB_STATES))
    )
    if busy:
        raise ApiError(
            409, "TRAINING_ACTIVE",
            "Este usuario tiene entrenamientos en cola o en proceso. Cancélalos antes de eliminar la cuenta.",
        )
    for p in db.scalars(select(Project).where(Project.owner_id == target.id)).unique().all():
        dirs = [m.dir_path for m in db.scalars(select(ModelVersion).where(ModelVersion.project_id == p.id))]
        delete_project_files(p.id, dirs)
        db.delete(p)
    audit(db, "user.deleted", admin, "user", target.id, {"username": target.username}, request)
    db.flush()
    db.delete(target)
    db.commit()


# ------------------------------------------------------------- tokens de API
@router.get("/token-requests")
def token_requests(status: str | None = None, db: Session = Depends(get_db)):
    """Solicitudes de tokens con escritura (las pendientes primero)."""
    q = select(ApiTokenRequest)
    if status:
        if status not in REQ_STATUSES:
            raise ApiError(422, "BAD_STATUS", "Estado inválido.")
        q = q.where(ApiTokenRequest.status == status)
    q = q.order_by((ApiTokenRequest.status != REQ_PENDING), ApiTokenRequest.id.desc()).limit(300)
    return [tokens_svc.request_to_dict(r, with_owner=True) for r in db.scalars(q).unique().all()]


def _get_request(db: Session, request_id: int) -> ApiTokenRequest:
    r = db.get(ApiTokenRequest, request_id)
    if r is None:
        raise ApiError(404, "REQUEST_NOT_FOUND", "Solicitud no encontrada.")
    return r


def _review_request(req: ApiTokenRequest, admin: User) -> None:
    if req.user_id == admin.id:
        raise ApiError(403, "SELF_ACTION", "No puedes resolver tu propia solicitud; debe hacerlo otro administrador.")
    if req.status != REQ_PENDING:
        raise ApiError(409, "INVALID_STATE", "Esta solicitud ya fue resuelta.")
    if req.user.status != APPROVED:
        raise ApiError(409, "USER_NOT_APPROVED", "El usuario no tiene la cuenta aprobada.")
    req.reviewed_at, req.reviewed_by_id = utcnow(), admin.id


@router.post("/token-requests/{request_id}/approve")
def approve_token_request(
    request_id: int, data: TokenApproveIn, request: Request,
    admin: User = Depends(require_admin), db: Session = Depends(get_db),
):
    """El usuario podrá generar un token con escritura que actúa dentro de sus propios privilegios."""
    req = _get_request(db, request_id)
    _review_request(req, admin)
    s = get_settings()
    days = min(data.expires_days or s.api_write_default_days, s.api_write_max_days)
    req.status, req.expires_days, req.review_note = REQ_APPROVED, days, data.note
    audit(db, "apitoken.write_approved", admin, "api_token_request", req.id,
          {"user": req.user.username, "name": req.name, "expires_days": days}, request)
    notify_user(
        db, req.user, "apitoken.write_approved", "Tu token de API con escritura fue aprobado",
        f"Ya puedes generar «{req.name}» desde Cuenta → Tokens de API. Vence a los {days} días de generarlo."
        + (f" Nota del administrador: {data.note}" if data.note else ""),
    )
    db.commit()
    return tokens_svc.request_to_dict(req, with_owner=True)


@router.post("/token-requests/{request_id}/reject")
def reject_token_request(
    request_id: int, data: ReviewIn, request: Request,
    admin: User = Depends(require_admin), db: Session = Depends(get_db),
):
    req = _get_request(db, request_id)
    _review_request(req, admin)
    req.status, req.review_note = REQ_REJECTED, data.note
    audit(db, "apitoken.write_rejected", admin, "api_token_request", req.id,
          {"user": req.user.username, "name": req.name, "note": data.note}, request)
    notify_user(
        db, req.user, "apitoken.write_rejected", "Tu solicitud de token con escritura fue rechazada",
        f"«{req.name}» no fue aprobado." + (f" Motivo: {data.note}" if data.note else ""),
    )
    db.commit()
    return tokens_svc.request_to_dict(req, with_owner=True)


@router.get("/api-tokens")
def list_api_tokens(db: Session = Depends(get_db)):
    """Todos los tokens de la plataforma (nunca su valor, sólo el prefijo)."""
    rows = db.scalars(select(ApiToken).order_by(ApiToken.id.desc()).limit(500)).unique().all()
    return [tokens_svc.token_to_dict(t, with_owner=True) for t in rows]


@router.post("/api-tokens/{token_id}/revoke")
def revoke_api_token(token_id: int, request: Request, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    token = db.get(ApiToken, token_id)
    if token is None:
        raise ApiError(404, "TOKEN_NOT_FOUND", "Token no encontrado.")
    tokens_svc.revoke_token(db, token, admin, request)
    db.commit()
    return tokens_svc.token_to_dict(token, with_owner=True)


# ------------------------------------------------------------ auditoría / cola
@router.get("/audit")
def audit_log(page: int = 1, page_size: int = 50, action: str | None = None, db: Session = Depends(get_db)):
    page_size = max(1, min(page_size, 200))
    q = select(AuditLog).order_by(AuditLog.id.desc())
    if action:
        q = q.where(AuditLog.action.like(f"{action}%"))
    total = db.scalar(select(func.count()).select_from(q.subquery()))
    rows = db.scalars(q.offset((max(page, 1) - 1) * page_size).limit(page_size)).all()
    return {
        "total": total,
        "items": [
            {"id": r.id, "actor": r.actor_username, "action": r.action, "target_type": r.target_type,
             "target_id": r.target_id, "detail": r.detail, "ip": r.ip, "created_at": r.created_at}
            for r in rows
        ],
    }


@router.get("/night-plan")
def night_plan_preview(db: Session = Depends(get_db)):
    """Qué modelos se entrenarían esta noche (o la próxima), con su duración estimada, y cuáles quedan para el sábado."""
    return night_plan.preview(db)


@router.get("/night-reports")
def night_reports(limit: int = Query(10, ge=1, le=60), db: Session = Depends(get_db)):
    """Resumen de las últimas noches hábiles: qué se entrenó, o por qué no se entrenó nada."""
    return night_plan.reports(db, limit)


@router.get("/queue")
def queue(db: Session = Depends(get_db)):
    jobs = db.scalars(select(TrainingJob).order_by(TrainingJob.id.desc()).limit(100)).unique().all()
    runs = db.scalars(select(WeeklyRun).order_by(WeeklyRun.id.desc()).limit(10)).all()
    return {
        "jobs": [tq.serialize_job(j, tq.queue_position(db, j)) for j in jobs],
        "runs": [
            {"id": r.id, "week_key": r.week_key, "trigger": r.trigger, "status": r.status,
             "started_at": r.started_at, "published_at": r.published_at, "summary": r.summary}
            for r in runs
        ],
        "lock": lock_state(db),
    }


@router.post("/training/enqueue")
def enqueue(data: EnqueueIn, request: Request, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    """Entrenamiento manual (resolución de errores). Bloquea las cargas hasta su publicación."""
    key = f"manual-{int(time.time())}"
    ids = list(db.scalars(select(Project.id))) if data.all else data.project_ids
    run = tq.start_run(db, key, "manual", project_ids=ids, force=True)
    if run is None:
        raise ApiError(409, "RUN_EXISTS", "Ya hay una ejecución creada en este instante; reintenta.")
    if not (run.summary or {}).get("queued"):
        reasons = " | ".join(f"{s['project']}: {s['reason']}" for s in (run.summary or {}).get("skipped", []))
        raise ApiError(409, "NOTHING_QUEUED", f"No se encoló ningún entrenamiento. {reasons or 'No hay proyectos para entrenar.'}")
    audit(db, "training.manual_enqueue", admin, "run", run.id, {"projects": ids}, request)
    db.commit()
    return {"run_id": run.id, "summary": run.summary}


@router.post("/training/{job_id}/cancel")
def cancel(job_id: int, request: Request, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    job = db.get(TrainingJob, job_id)
    if job is None:
        raise ApiError(404, "JOB_NOT_FOUND", "Trabajo no encontrado.")
    tq.cancel_job(db, job)
    audit(db, "training.cancelled", admin, "job", job.id, None, request)
    db.commit()
    tq.publish_job_event(job)
    return tq.serialize_job(job, tq.queue_position(db, job))


@router.post("/training/{job_id}/retry")
def retry(job_id: int, request: Request, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    job = db.get(TrainingJob, job_id)
    if job is None:
        raise ApiError(404, "JOB_NOT_FOUND", "Trabajo no encontrado.")
    if job.status not in (J_ERROR, J_CANCELLED):
        raise ApiError(409, "INVALID_STATE", "Sólo se pueden reintentar trabajos con error o cancelados.")
    snap = db.get(DatasetSnapshot, job.snapshot_id) if job.snapshot_id else None
    if snap is None or not snap.path or not Path(snap.path).exists():
        raise ApiError(409, "SNAPSHOT_GONE", "El snapshot del dataset ya no existe; encola un entrenamiento nuevo.")
    job.status, job.attempts, job.error, job.cancel_requested = J_PENDING, 0, None, False
    job.percent, job.finished_at, job.epoch = 0.0, None, 0
    job.history = []
    run = db.get(WeeklyRun, job.run_id) if job.run_id else None
    if run and run.status == "published":
        run.status, run.published_at = "running", None
    tq.add_log(db, job.id, "Reintento solicitado por un administrador.")
    audit(db, "training.retried", admin, "job", job.id, None, request)
    db.commit()
    tq.publish_job_event(job)
    tq.kick_queue()
    return tq.serialize_job(job, tq.queue_position(db, job))
