from fastapi import APIRouter, Depends
from sqlalchemy import func, select, text, update
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import get_current_user, require_approved
from ..errors import ApiError
from ..modalities import describe_all
from ..models import Notification, User

router = APIRouter(prefix="/api", tags=["misc"])


@router.get("/health")
def health(db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))
    return {"status": "ok"}


@router.get("/modalities")
def modalities(_: User = Depends(require_approved)):
    """Tipos de dato que se pueden entrenar, con los textos y formatos que necesita la interfaz."""
    return describe_all()


@router.get("/notifications")
def list_notifications(unread_only: bool = False, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    q = select(Notification).where(Notification.user_id == user.id).order_by(Notification.id.desc()).limit(100)
    if unread_only:
        q = q.where(Notification.read.is_(False))
    rows = db.scalars(q).all()
    unread = db.scalar(
        select(func.count(Notification.id)).where(Notification.user_id == user.id, Notification.read.is_(False))
    )
    return {
        "unread": unread,
        "items": [
            {"id": n.id, "kind": n.kind, "title": n.title, "body": n.body, "read": n.read,
             "created_at": n.created_at, "meta": n.meta}
            for n in rows
        ],
    }


@router.post("/notifications/{notification_id}/read")
def mark_read(notification_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    n = db.get(Notification, notification_id)
    if n is None or n.user_id != user.id:
        raise ApiError(404, "NOT_FOUND", "Notificación no encontrada.")
    n.read = True
    db.commit()
    return {"ok": True}


@router.post("/notifications/read-all")
def mark_all_read(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    db.execute(
        update(Notification).where(Notification.user_id == user.id, Notification.read.is_(False)).values(read=True)
    )
    db.commit()
    return {"ok": True}
