"""Notificaciones internas + correo SMTP opcional (sin SMTP todo sigue funcionando)."""
import logging
import smtplib
import threading
from email.message import EmailMessage

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import APPROVED, ROLE_ADMIN, Notification, User

log = logging.getLogger(__name__)


def smtp_configured() -> bool:
    return bool(get_settings().smtp_host)


def _send_email(to: str, subject: str, body: str) -> None:
    s = get_settings()
    try:
        msg = EmailMessage()
        msg["From"] = s.smtp_from or s.smtp_user or "vision-lab@localhost"
        msg["To"] = to
        msg["Subject"] = subject
        msg.set_content(body)
        with smtplib.SMTP(s.smtp_host, s.smtp_port, timeout=15) as smtp:
            if s.smtp_starttls:
                smtp.starttls()
            if s.smtp_user:
                smtp.login(s.smtp_user, s.smtp_password)
            smtp.send_message(msg)
    except Exception as exc:  # noqa: BLE001 - el correo nunca debe romper el flujo
        log.warning("Fallo al enviar correo a %s: %s", to, exc)


def _email_async(to: str, subject: str, body: str) -> None:
    if smtp_configured():
        threading.Thread(target=_send_email, args=(to, subject, body), daemon=True).start()


def notify_user(
    db: Session, user: User, kind: str, title: str, body: str = "", meta: dict | None = None,
    email: bool = True,
) -> None:
    db.add(Notification(user_id=user.id, kind=kind, title=title, body=body, meta=meta))
    if email:
        _email_async(user.email, f"[Vision Lab AI] {title}", f"{body}\n\n{get_settings().public_url}")


def notify_admins(db: Session, kind: str, title: str, body: str = "", meta: dict | None = None) -> None:
    admins = db.scalars(
        select(User).where(User.role == ROLE_ADMIN, User.status == APPROVED)
    ).all()
    for admin in admins:
        notify_user(db, admin, kind, title, body, meta)
