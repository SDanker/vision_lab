"""Reglas de negocio sobre cuentas: transiciones de estado, historial y salvaguardas."""
from fastapi import Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db import utcnow
from ..errors import ApiError
from ..models import (
    APPROVED,
    PENDING,
    REJECTED,
    ROLE_ADMIN,
    SUSPENDED,
    User,
    UserStatusHistory,
)
from .api_tokens import revoke_all_for_user
from .audit import audit
from .notifications import notify_user

LABELS = {
    "approve": (APPROVED, "user.approved", "Tu cuenta fue aprobada"),
    "reject": (REJECTED, "user.rejected", "Tu solicitud fue rechazada"),
    "suspend": (SUSPENDED, "user.suspended", "Tu cuenta fue suspendida"),
    "reactivate": (APPROVED, "user.reactivated", "Tu cuenta fue reactivada"),
}
ALLOWED_FROM = {
    "approve": (PENDING, REJECTED),
    "reject": (PENDING,),
    "suspend": (APPROVED,),
    "reactivate": (SUSPENDED,),
}


def active_admin_count(db: Session, excluding_id: int | None = None) -> int:
    q = select(func.count(User.id)).where(User.role == ROLE_ADMIN, User.status == APPROVED)
    if excluding_id is not None:
        q = q.where(User.id != excluding_id)
    return int(db.scalar(q) or 0)


def ensure_not_last_admin(db: Session, target: User) -> None:
    if target.role == ROLE_ADMIN and target.status == APPROVED:
        if active_admin_count(db, excluding_id=target.id) == 0:
            raise ApiError(
                409, "LAST_ADMIN", "No se puede modificar o eliminar la última cuenta administrativa activa."
            )


def change_status(
    db: Session, target: User, actor: User, action: str, note: str | None, request: Request | None = None
) -> User:
    new_status, audit_action, title = LABELS[action]
    if target.id == actor.id:
        raise ApiError(403, "SELF_ACTION", "No puedes cambiar el estado de tu propia cuenta.")
    if target.status not in ALLOWED_FROM[action]:
        raise ApiError(
            409, "INVALID_TRANSITION", f"No se puede ejecutar '{action}' sobre una cuenta {target.status}."
        )
    if action == "suspend":
        ensure_not_last_admin(db, target)

    old = target.status
    target.status = new_status
    target.reviewed_at = utcnow()
    target.reviewed_by_id = actor.id
    target.review_note = note
    if action in ("suspend", "reject"):
        target.token_version += 1  # invalida las sesiones activas
        revoke_all_for_user(db, target.id)  # y los tokens de API
    db.add(UserStatusHistory(user_id=target.id, from_status=old, to_status=new_status, actor_id=actor.id, note=note))
    audit(db, audit_action, actor, "user", target.id, {"from": old, "to": new_status, "note": note}, request)
    body = title + "."
    if note and action in ("reject", "suspend"):
        body += f" Motivo: {note}"
    notify_user(db, target, audit_action, title, body)
    db.flush()
    return target


def user_to_dict(u: User) -> dict:
    return {
        "id": u.id,
        "first_name": u.first_name,
        "last_name": u.last_name,
        "email": u.email,
        "username": u.username,
        "role": u.role,
        "status": u.status,
        "reason": u.reason,
        "quota_mb": u.quota_mb,
        "created_at": u.created_at,
        "reviewed_at": u.reviewed_at,
        "review_note": u.review_note,
        "reviewed_by": u.reviewed_by.username if u.reviewed_by else None,
    }
