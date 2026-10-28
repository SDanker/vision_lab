"""Tokens de API personales: creación, revocación y serialización.

Un token siempre actúa como su dueño: hereda su rol, sus proyectos propios y compartidos y el calendario
semanal. Los de lectura (`read`) los crea cada usuario; los de escritura (`write`) salen de una solicitud
que debe aprobar un administrador.
"""
from datetime import timedelta

from fastapi import Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db import utcnow
from ..errors import ApiError
from ..models import (
    REQ_APPROVED,
    REQ_PENDING,
    TOKEN_READ,
    TOKEN_WRITE,
    ApiToken,
    ApiTokenRequest,
    User,
)
from ..security import new_api_token
from .audit import audit
from .notifications import notify_user


def active_tokens(db: Session, user_id: int, scope: str | None = None) -> list[ApiToken]:
    now = utcnow()
    q = select(ApiToken).where(ApiToken.user_id == user_id, ApiToken.revoked_at.is_(None))
    if scope:
        q = q.where(ApiToken.scope == scope)
    return [t for t in db.scalars(q).unique().all() if t.is_active(now)]


def count_open_write_requests(db: Session, user_id: int) -> int:
    """Solicitudes que todavía pueden terminar en un token: pendientes o aprobadas sin generar."""
    return int(
        db.scalar(
            select(func.count(ApiTokenRequest.id)).where(
                ApiTokenRequest.user_id == user_id,
                ApiTokenRequest.status.in_((REQ_PENDING, REQ_APPROVED)),
                ApiTokenRequest.token_id.is_(None),
            )
        )
        or 0
    )


def issue_token(
    db: Session, user: User, name: str, scope: str, days: int | None, actor: User | None = None,
    request: Request | None = None,
) -> tuple[ApiToken, str]:
    """Crea el token y devuelve (registro, valor en claro). El valor en claro no se vuelve a poder leer."""
    plain, prefix, digest = new_api_token()
    token = ApiToken(
        user_id=user.id, name=name, prefix=prefix, token_hash=digest, scope=scope,
        expires_at=utcnow() + timedelta(days=days) if days else None,
    )
    db.add(token)
    db.flush()
    audit(
        db, "apitoken.created", actor or user, "api_token", token.id,
        {"name": name, "scope": scope, "prefix": prefix, "expires_days": days}, request,
    )
    return token, plain


def revoke_token(db: Session, token: ApiToken, actor: User, request: Request | None = None) -> None:
    if token.revoked_at is not None:
        return
    token.revoked_at = utcnow()
    token.revoked_by_id = actor.id
    audit(
        db, "apitoken.revoked", actor, "api_token", token.id,
        {"owner": token.user.username, "name": token.name, "scope": token.scope}, request,
    )
    if actor.id != token.user_id:
        notify_user(
            db, token.user, "apitoken.revoked", "Un administrador revocó uno de tus tokens de API",
            f"El token «{token.name}» ({token.prefix}…) dejó de funcionar.",
        )


def revoke_all_for_user(db: Session, user_id: int) -> None:
    """Al suspender o rechazar una cuenta sus tokens dejan de servir, igual que sus sesiones."""
    now = utcnow()
    for t in db.scalars(select(ApiToken).where(ApiToken.user_id == user_id, ApiToken.revoked_at.is_(None))).unique():
        t.revoked_at = now


def token_to_dict(t: ApiToken, with_owner: bool = False) -> dict:
    now = utcnow()
    state = "revoked" if t.revoked_at else "expired" if t.expires_at and t.expires_at <= now else "active"
    d = {
        "id": t.id, "name": t.name, "prefix": t.prefix, "scope": t.scope, "state": state,
        "created_at": t.created_at, "expires_at": t.expires_at, "last_used_at": t.last_used_at,
        "last_used_ip": t.last_used_ip, "revoked_at": t.revoked_at,
    }
    if with_owner:
        d["owner"] = {"id": t.user.id, "username": t.user.username, "name": t.user.full_name}
    return d


def request_to_dict(r: ApiTokenRequest, with_owner: bool = False) -> dict:
    d = {
        "id": r.id, "name": r.name, "reason": r.reason, "status": r.status, "expires_days": r.expires_days,
        "created_at": r.created_at, "reviewed_at": r.reviewed_at, "review_note": r.review_note,
        "reviewed_by": r.reviewed_by.username if r.reviewed_by else None,
        "token_id": r.token_id,
        "can_generate": r.status == REQ_APPROVED and r.token_id is None,
    }
    if with_owner:
        d["owner"] = {"id": r.user.id, "username": r.user.username, "name": r.user.full_name, "role": r.user.role}
    return d


def check_read_quota(db: Session, user: User, max_tokens: int) -> None:
    if len(active_tokens(db, user.id, TOKEN_READ)) >= max_tokens:
        raise ApiError(
            409, "TOKEN_LIMIT",
            f"Ya tienes {max_tokens} tokens de lectura activos. Revoca alguno antes de crear otro.",
        )


def check_write_quota(db: Session, user: User, max_tokens: int) -> None:
    if len(active_tokens(db, user.id, TOKEN_WRITE)) + count_open_write_requests(db, user.id) >= max_tokens:
        raise ApiError(
            409, "TOKEN_LIMIT",
            "Ya tienes un token con escritura o una solicitud en curso. Revoca el token o espera la respuesta "
            "del administrador antes de pedir otro.",
        )
