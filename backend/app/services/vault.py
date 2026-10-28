"""Cifrado de secretos de los usuarios (por ahora, el token de Hugging Face).

Un token de HF hay que poder usarlo más adelante (a diferencia de una contraseña, no basta con guardar su hash),
así que se guarda cifrado con Fernet (AES-128-CBC + HMAC). La clave sale de HF_ENCRYPTION_KEY; si no está definida se
deriva de SECRET_KEY. Las dos claves se aceptan al descifrar: definir HF_ENCRYPTION_KEY más adelante no deja a nadie
sin su conexión.
"""
import base64

from cryptography.fernet import Fernet, InvalidToken, MultiFernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from ..config import get_settings


def _derived() -> Fernet:
    raw = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"vision-lab:secrets:v1").derive(
        get_settings().secret_key.encode()
    )
    return Fernet(base64.urlsafe_b64encode(raw))


def _ring() -> MultiFernet:
    configured = get_settings().hf_encryption_key.strip()
    if configured.startswith("CHANGE_ME"):  # .env.example copiado sin editar: se comporta como «sin definir»
        configured = ""
    keys = [_derived()]
    if configured:
        try:
            keys.insert(0, Fernet(configured.encode()))  # la dedicada cifra; la derivada sólo sirve para descifrar
        except ValueError as exc:
            raise RuntimeError(
                "HF_ENCRYPTION_KEY no es válida: debe ser una clave Fernet (32 bytes en base64 url-safe). "
                "Genérala con: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
            ) from exc
    return MultiFernet(keys)


def encrypt(plain: str) -> str:
    return _ring().encrypt(plain.encode()).decode()


def decrypt(token: str) -> str | None:
    """None si el valor no se puede descifrar (cambió la clave): el usuario debe reconectar su cuenta."""
    try:
        return _ring().decrypt(token.encode()).decode()
    except InvalidToken:
        return None
