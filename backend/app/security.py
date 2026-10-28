import hashlib
import re
import secrets
from datetime import timedelta

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from .config import get_settings
from .db import utcnow

_ph = PasswordHasher()  # Argon2id por defecto
_DUMMY_HASH = _ph.hash("dummy-password-for-timing")
ALGO = "HS256"


def hash_password(password: str) -> str:
    return _ph.hash(password)


def verify_password(password_hash: str | None, password: str) -> bool:
    try:
        return _ph.verify(password_hash or _DUMMY_HASH, password) and password_hash is not None
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def password_error(password: str) -> str | None:
    if len(password) < 10:
        return "La contraseña debe tener al menos 10 caracteres."
    if len(password) > 128:
        return "La contraseña es demasiado larga."
    if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        return "La contraseña debe incluir letras y números."
    return None


def create_token(user_id: int, token_version: int) -> str:
    s = get_settings()
    now = utcnow()
    payload = {
        "sub": str(user_id),
        "tv": token_version,
        "iat": now,
        "exp": now + timedelta(hours=s.session_hours),
    }
    return jwt.encode(payload, s.secret_key, algorithm=ALGO)


API_TOKEN_PREFIX = "vl_"


def hash_api_token(token: str) -> str:
    """SHA-256 basta: el token tiene 256 bits de entropía, no es una contraseña elegida por una persona."""
    return hashlib.sha256(token.encode()).hexdigest()


def new_api_token() -> tuple[str, str, str]:
    """(token en claro, prefijo visible, hash). El token en claro se muestra una única vez."""
    token = API_TOKEN_PREFIX + secrets.token_urlsafe(32)
    return token, token[:11], hash_api_token(token)


def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, get_settings().secret_key, algorithms=[ALGO])
    except jwt.PyJWTError:
        return None
