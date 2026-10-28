"""Tokens de API del propio usuario. Se gestionan sólo con la sesión web, nunca con otro token."""
from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_db
from ..deps import require_session
from ..errors import ApiError
from ..models import (
    REQ_APPROVED,
    REQ_CANCELLED,
    REQ_PENDING,
    TOKEN_READ,
    TOKEN_WRITE,
    ApiToken,
    ApiTokenRequest,
    User,
)
from ..schemas import TokenCreateIn, TokenRequestIn
from ..services import api_tokens as svc
from ..services.audit import audit
from ..services.notifications import notify_admins

router = APIRouter(prefix="/api/tokens", tags=["tokens"])


def _no_store(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"  # la respuesta puede traer el token en claro


def _own_token(db: Session, user: User, token_id: int) -> ApiToken:
    t = db.get(ApiToken, token_id)
    if t is None or t.user_id != user.id:
        raise ApiError(404, "TOKEN_NOT_FOUND", "Token no encontrado.")
    return t


def _own_request(db: Session, user: User, request_id: int, lock: bool = False) -> ApiTokenRequest:
    q = select(ApiTokenRequest).where(ApiTokenRequest.id == request_id)
    if lock:
        # evita que dos clics generen dos tokens a partir de la misma aprobación; `of` porque PostgreSQL
        # no permite FOR UPDATE sobre el lado nullable de los LEFT JOIN de las relaciones cargadas
        q = q.with_for_update(of=ApiTokenRequest)
    r = db.scalars(q).unique().one_or_none()
    if r is None or r.user_id != user.id:
        raise ApiError(404, "REQUEST_NOT_FOUND", "Solicitud no encontrada.")
    return r


@router.get("")
def my_tokens(user: User = Depends(require_session), db: Session = Depends(get_db)):
    s = get_settings()
    tokens = db.scalars(
        select(ApiToken).where(ApiToken.user_id == user.id).order_by(ApiToken.id.desc()).limit(100)
    ).unique().all()
    reqs = db.scalars(
        select(ApiTokenRequest).where(ApiTokenRequest.user_id == user.id).order_by(ApiTokenRequest.id.desc()).limit(50)
    ).unique().all()
    return {
        "tokens": [svc.token_to_dict(t) for t in tokens],
        "requests": [svc.request_to_dict(r) for r in reqs],
        "limits": {"read": s.api_max_read_tokens, "write": s.api_max_write_tokens},
    }


@router.post("", status_code=201)
def create_read_token(
    data: TokenCreateIn, request: Request, response: Response,
    user: User = Depends(require_session), db: Session = Depends(get_db),
):
    """Token de sólo lectura: ve proyectos, progreso y modelos a los que el usuario ya tiene acceso."""
    svc.check_read_quota(db, user, get_settings().api_max_read_tokens)
    token, plain = svc.issue_token(db, user, data.name, TOKEN_READ, data.expires_days, request=request)
    db.commit()
    _no_store(response)
    return {**svc.token_to_dict(token), "token": plain}


@router.delete("/{token_id}", status_code=204)
def revoke_my_token(token_id: int, request: Request, user: User = Depends(require_session), db: Session = Depends(get_db)):
    svc.revoke_token(db, _own_token(db, user, token_id), user, request)
    db.commit()


# ---------------------------------------------------------- token con escritura
@router.post("/requests", status_code=201)
def request_write_token(
    data: TokenRequestIn, request: Request, user: User = Depends(require_session), db: Session = Depends(get_db)
):
    """Pide un token con escritura. Un administrador debe aprobarlo antes de que se pueda generar."""
    svc.check_write_quota(db, user, get_settings().api_max_write_tokens)
    req = ApiTokenRequest(user_id=user.id, name=data.name, reason=data.reason, status=REQ_PENDING)
    db.add(req)
    db.flush()
    audit(db, "apitoken.write_requested", user, "api_token_request", req.id, {"name": data.name}, request)
    notify_admins(
        db, "apitoken.write_requested", "Solicitud de token de API con escritura",
        f"{user.full_name} (@{user.username}) pide un token con escritura: «{data.name}». Motivo: {data.reason}",
        {"request_id": req.id},
    )
    db.commit()
    return svc.request_to_dict(req)


@router.post("/requests/{request_id}/generate", status_code=201)
def generate_write_token(
    request_id: int, request: Request, response: Response,
    user: User = Depends(require_session), db: Session = Depends(get_db),
):
    """Crea el token con escritura de una solicitud aprobada y lo muestra una sola vez."""
    req = _own_request(db, user, request_id, lock=True)
    if req.status != REQ_APPROVED:
        raise ApiError(409, "REQUEST_NOT_APPROVED", "Esta solicitud no está aprobada.")
    if req.token_id is not None:
        raise ApiError(409, "TOKEN_ALREADY_ISSUED", "El token de esta solicitud ya fue generado. Pide uno nuevo si lo perdiste.")
    token, plain = svc.issue_token(db, user, req.name, TOKEN_WRITE, req.expires_days, request=request)
    req.token_id = token.id
    db.commit()
    _no_store(response)
    return {**svc.token_to_dict(token), "token": plain}


@router.delete("/requests/{request_id}", status_code=204)
def cancel_request(request_id: int, request: Request, user: User = Depends(require_session), db: Session = Depends(get_db)):
    req = _own_request(db, user, request_id)
    if req.status not in (REQ_PENDING, REQ_APPROVED) or req.token_id is not None:
        raise ApiError(409, "INVALID_STATE", "Sólo se pueden cancelar solicitudes pendientes o aprobadas sin generar.")
    req.status = REQ_CANCELLED
    audit(db, "apitoken.write_cancelled", user, "api_token_request", req.id, None, request)
    db.commit()
