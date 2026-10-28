from datetime import timedelta

from fastapi import Depends, Request
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from . import ratelimit
from .config import get_settings
from .db import get_db, utcnow
from .errors import ApiError
from .models import APPROVED, ROLE_ADMIN, SUSPENDED, TOKEN_READ, ApiToken, User
from .security import API_TOKEN_PREFIX, decode_token, hash_api_token
from .services.audit import client_ip

COOKIE = "vision_session"
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
LAST_USED_REFRESH = timedelta(minutes=5)  # no se escribe en la BD en cada petición


def bearer_token(request: Request) -> str | None:
    """Token de la cabecera `Authorization: Bearer ...`, si la hay."""
    scheme, _, value = request.headers.get("authorization", "").partition(" ")
    value = value.strip()
    return value if scheme.lower() == "bearer" and value else None


def _user_from_api_token(request: Request, db: Session, raw: str) -> User:
    s = get_settings()
    ip = client_ip(request) or "unknown"
    now = utcnow()
    token = None
    if raw.startswith(API_TOKEN_PREFIX):
        token = db.scalar(select(ApiToken).where(ApiToken.token_hash == hash_api_token(raw)))
    if token is None or not token.is_active(now):
        ratelimit.enforce(
            f"tokfail:{ip}", s.api_token_fail_max, s.login_window_seconds,
            "Demasiados tokens inválidos desde esta dirección. Inténtalo más tarde.",
        )
        raise ApiError(401, "INVALID_API_TOKEN", "El token de API no es válido, venció o fue revocado.")
    user = token.user
    if user.status == SUSPENDED:
        raise ApiError(401, "SESSION_INVALID", "Tu cuenta está suspendida.")
    ratelimit.enforce(
        f"tok:{token.id}", s.api_token_rate_per_minute, 60,
        "Demasiadas peticiones con este token. Espera un minuto.",
    )
    if token.scope == TOKEN_READ and request.method not in SAFE_METHODS:
        raise ApiError(
            403, "TOKEN_READ_ONLY",
            "Este token es de sólo lectura. Solicita un token con escritura en tu cuenta (requiere aprobación).",
        )
    if token.last_used_at is None or now - token.last_used_at > LAST_USED_REFRESH:
        db.execute(update(ApiToken).where(ApiToken.id == token.id).values(last_used_at=now, last_used_ip=ip))
        db.commit()
    request.state.auth_kind = "token"
    request.state.api_token_id = token.id
    return user


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    raw = bearer_token(request)
    if raw is not None:  # con token de API se ignora la cookie: nunca se mezclan
        return _user_from_api_token(request, db, raw)
    token = request.cookies.get(COOKIE)
    payload = decode_token(token) if token else None
    if not payload:
        raise ApiError(401, "NOT_AUTHENTICATED", "Debes iniciar sesión.")
    user = db.get(User, int(payload["sub"]))
    if user is None or user.token_version != payload.get("tv") or user.status == SUSPENDED:
        raise ApiError(401, "SESSION_INVALID", "Tu sesión ya no es válida. Inicia sesión nuevamente.")
    request.state.auth_kind = "session"
    return user


def require_approved(user: User = Depends(get_current_user)) -> User:
    """Se evalúa en cada acceso protegido: el estado se lee siempre de la BD."""
    if user.status != APPROVED:
        raise ApiError(
            403, "ACCOUNT_NOT_APPROVED", "Tu cuenta no está aprobada para usar la plataforma."
        )
    return user


def forbid_api_token(request: Request) -> None:
    if getattr(request.state, "auth_kind", "session") == "token":
        raise ApiError(
            403, "SESSION_REQUIRED",
            "Esta acción sólo puede hacerse iniciando sesión en la web, no con un token de API.",
        )


def require_session(request: Request, user: User = Depends(require_approved)) -> User:
    """Usuario aprobado autenticado con la sesión web (los tokens de API quedan fuera)."""
    forbid_api_token(request)
    return user


def require_admin(request: Request, user: User = Depends(require_approved)) -> User:
    forbid_api_token(request)  # la administración nunca se hace con tokens de API
    if user.role != ROLE_ADMIN:
        raise ApiError(403, "ADMIN_ONLY", "Esta acción requiere permisos de administrador.")
    return user
