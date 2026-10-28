"""Datos tabulares: columnas del CSV, columna a predecir y columnas de entrada de un proyecto."""
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import require_approved
from ..errors import ApiError
from ..models import Project, User
from ..services import tabular
from ..services.access import EDIT, ensure_modifications_open, get_project
from ..services.audit import audit

router = APIRouter(prefix="/api/projects", tags=["tabular"])


class ColumnsIn(BaseModel):
    target: str = Field(min_length=1, max_length=120)
    features: list[str] | None = Field(default=None, max_length=300)  # None = las columnas sugeridas


def _tabular_project(db: Session, user: User, project_id: int, need: str = "view") -> Project:
    project = get_project(db, user, project_id, need)
    if not tabular.is_tabular(project):
        raise ApiError(409, "NOT_APPLICABLE", "Este proyecto no es de datos tabulares.")
    return project


@router.get("/{project_id}/tabular")
def read_tabular(project_id: int, user: User = Depends(require_approved), db: Session = Depends(get_db)):
    return tabular.summary(db, _tabular_project(db, user, project_id))


@router.put("/{project_id}/tabular/columns")
def set_columns(
    project_id: int, data: ColumnsIn, request: Request,
    user: User = Depends(require_approved), db: Session = Depends(get_db),
):
    project = _tabular_project(db, user, project_id, EDIT)
    ensure_modifications_open(db, project, user)
    sm = tabular.summary(db, project)
    names = {c["name"] for c in sm["columns"]}
    if not names:
        raise ApiError(409, "NO_DATA", "Sube primero un archivo CSV.")
    target = data.target.strip()
    if target not in names:
        raise ApiError(422, "COLUMN_NOT_FOUND", f"La columna «{target}» no existe en la tabla.")
    features = None
    if data.features is not None:
        features = list(dict.fromkeys(data.features))
        unknown = [f for f in features if f not in names]
        if unknown:
            raise ApiError(422, "COLUMN_NOT_FOUND", f"Estas columnas no existen en la tabla: {', '.join(unknown[:5])}.")
        if target in features:
            raise ApiError(422, "TARGET_IN_FEATURES", "La columna a predecir no puede ser también una columna de entrada.")
        if not features:
            raise ApiError(422, "NO_FEATURES", "Elige al menos una columna de entrada.")
    cfg = {k: v for k, v in (project.config or {}).items() if k not in ("tabular_target", "tabular_features")}
    cfg["tabular_target"] = target
    if features is not None:
        cfg["tabular_features"] = features
    project.config = cfg  # asignación nueva: SQLAlchemy no detecta cambios dentro de un JSON mutable
    audit(db, "project.tabular_columns", user, "project", project.id, {"target": target, "features": len(features or [])}, request)
    db.commit()
    return tabular.summary(db, project)
