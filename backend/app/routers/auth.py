from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .. import ratelimit
from ..config import get_settings
from ..db import get_db, utcnow
from ..deps import COOKIE, get_current_user
from ..errors import ApiError
from ..models import APPROVED, PENDING, REJECTED, SUSPENDED, Notification, User, UserStatusHistory
from ..schemas import LoginIn, RegisterIn
from ..security import create_token, hash_password, verify_password
from ..services.audit import audit, client_ip
from ..services.notifications import notify_admins
from ..services.users import user_to_dict

router = APIRouter(prefix="/api/auth", tags=["auth"])

PENDING_MESSAGE = (
    "Tu cuenta fue creada correctamente y se encuentra pendiente de aprobación por un "
    "administrador. Recibirás una notificación cuando tu solicitud sea revisada."
)


@router.post("/register", status_code=201)
def register(data: RegisterIn, request: Request, db: Session = Depends(get_db)):
    s = get_settings()
    if not s.allow_registration:
        raise ApiError(403, "REGISTRATION_DISABLED", "El registro público está deshabilitado.")
    ip = client_ip(request) or "unknown"
    ratelimit.enforce(
        f"register:{ip}", s.register_max_attempts, s.register_window_seconds,
        "Demasiados registros desde esta dirección. Inténtalo más tarde.",
    )
    email = data.email.lower()
    username = data.username.lower()
    if db.scalar(select(User.id).where(func.lower(User.email) == email)):
        raise ApiError(409, "EMAIL_TAKEN", "Ese correo electrónico ya está registrado.")
    if db.scalar(select(User.id).where(func.lower(User.username) == username)):
        raise ApiError(409, "USERNAME_TAKEN", "Ese nombre de usuario ya está en uso.")

    user = User(
        first_name=data.first_name,
        last_name=data.last_name,
        email=email,
        username=username,
        password_hash=hash_password(data.password),
        status=PENDING,
        role="USER",
        reason=(data.reason or "").strip() or None,
    )
    db.add(user)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise ApiError(409, "DUPLICATE", "El correo o el usuario ya están registrados.") from None
    db.add(UserStatusHistory(user_id=user.id, from_status=None, to_status=PENDING, note="Registro"))
    audit(db, "user.registered", None, "user", user.id, {"username": username}, request)
    notify_admins(
        db,
        "registration.pending",
        "Nueva solicitud de registro",
        f"{user.full_name} (@{username}) solicitó acceso y está pendiente de aprobación.",
        {"user_id": user.id},
    )
    db.add(
        Notification(
            user_id=user.id, kind="registration.created", title="Solicitud recibida", body=PENDING_MESSAGE
        )
    )
    db.commit()
    return {"message": PENDING_MESSAGE, "status": PENDING}


@router.post("/login")
def login(data: LoginIn, request: Request, response: Response, db: Session = Depends(get_db)):
    s = get_settings()
    ip = client_ip(request) or "unknown"
    ident = data.identifier.strip().lower()
    msg = "Demasiados intentos de inicio de sesión. Inténtalo más tarde."
    ratelimit.enforce(f"login-ip:{ip}", s.login_max_attempts * 4, s.login_window_seconds, msg)
    ratelimit.enforce(f"login:{ip}:{ident}", s.login_max_attempts, s.login_window_seconds, msg)
    # por cuenta, sin importar la IP: si alguien falsea su IP para esquivar los límites de arriba, igual se frena
    ratelimit.enforce(f"login-ident:{ident}", s.login_max_attempts * 3, s.login_window_seconds, msg)

    user = db.scalar(
        select(User).where((func.lower(User.email) == ident) | (func.lower(User.username) == ident))
    )
    ok = verify_password(user.password_hash if user else None, data.password)
    if not user or not ok:
        audit(db, "auth.login_failed", None, "user", user.id if user else None, {"identifier": ident}, request)
        db.commit()
        raise ApiError(401, "BAD_CREDENTIALS", "Usuario o contraseña incorrectos.")
    if user.status == SUSPENDED:
        raise ApiError(
            403, "ACCOUNT_SUSPENDED",
            "Tu cuenta está suspendida. Contacta a un administrador." + (f" Motivo: {user.review_note}" if user.review_note else ""),
        )
    ratelimit.clear(f"login:{ip}:{ident}")
    ratelimit.clear(f"login-ident:{ident}")
    user.last_login_at = utcnow()
    audit(db, "auth.login", user, "user", user.id, None, request)
    db.commit()
    response.set_cookie(
        COOKIE,
        create_token(user.id, user.token_version),
        max_age=s.session_hours * 3600,
        httponly=True,
        samesite="lax",
        secure=s.cookie_secure,
        path="/",
    )
    return _me(user)


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(COOKIE, path="/")
    return {"ok": True}


def _me(user: User) -> dict:
    d = user_to_dict(user)
    d["can_use_platform"] = user.status == APPROVED
    if user.status == PENDING:
        d["status_message"] = PENDING_MESSAGE
    elif user.status == REJECTED:
        d["status_message"] = "Tu solicitud de registro fue rechazada." + (
            f" Motivo: {user.review_note}" if user.review_note else ""
        )
    return d


@router.get("/me")
def me(user: User = Depends(get_current_user)):
    return _me(user)
