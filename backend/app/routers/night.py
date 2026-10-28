from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import calendar_rules
from ..db import get_db, utcnow
from ..deps import require_approved
from ..errors import ApiError
from ..models import NC_ASKING, NC_GO, NC_SKIP, NightCheck, User
from ..services.access import EDIT, _LEVEL, _level, project_role
from ..services.audit import audit
from ..services.night_plan import check_dict, publish_check

router = APIRouter(prefix="/api/night-checks", tags=["night"])


class AnswerIn(BaseModel):
    still_uploading: bool


@router.get("/mine")
def my_checks(user: User = Depends(require_approved), db: Session = Depends(get_db)):
    """Preguntas nocturnas pendientes para mí («¿todavía te faltan cargas?»)."""
    rows = db.scalars(
        select(NightCheck).where(NightCheck.status == NC_ASKING, NightCheck.night == calendar_rules.night_key())
    ).unique().all()
    return [check_dict(c) for c in rows if user.id in (c.asked_user_ids or [])]


@router.post("/{check_id}/answer")
def answer(
    check_id: int, data: AnswerIn, request: Request,
    user: User = Depends(require_approved), db: Session = Depends(get_db),
):
    check = db.get(NightCheck, check_id)
    if check is None or user.id not in (check.asked_user_ids or []):
        raise ApiError(404, "NIGHT_CHECK_NOT_FOUND", "Pregunta no encontrada.")
    if _level(project_role(check.project, user)) < _LEVEL[EDIT]:
        raise ApiError(403, "PROJECT_FORBIDDEN", "Ya no tienes permiso de edición en este proyecto.")
    if check.status != NC_ASKING:
        msg = (
            "Ya se decidió: el proyecto no se entrenará esta noche."
            if check.status == NC_SKIP
            else "Ya se decidió: el proyecto se entrena esta noche."
        )
        raise ApiError(409, "NIGHT_CHECK_DECIDED", msg, {"status": check.status})
    check.status = NC_SKIP if data.still_uploading else NC_GO
    check.answered_by_id = user.id
    check.answered_at = utcnow()
    check.reason = "Todavía faltan cargas" if data.still_uploading else "Ya terminó de cargar"
    audit(
        db, "training.night_answer", user, "project", check.project_id,
        {"night": check.night, "still_uploading": data.still_uploading}, request,
    )
    db.commit()
    publish_check(check)
    return check_dict(check)
