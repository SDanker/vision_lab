from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import calendar_rules
from ..db import utcnow
from ..errors import ApiError
from ..models import (
    ACTIVE_JOB_STATES,
    MEMBER_EDITOR,
    ROLE_ADMIN,
    ModelVersion,
    Project,
    ProjectMember,
    TrainingJob,
    User,
    WeeklyRun,
)

# Niveles de acceso a un proyecto, de menor a mayor
VIEW, EDIT, OWNER = "view", "edit", "owner"
_LEVEL = {VIEW: 1, EDIT: 2, OWNER: 3}


def project_role(project: Project, user: User) -> str | None:
    """Rol efectivo del usuario: owner (propietario o admin), editor, viewer o None."""
    if project.owner_id == user.id or user.role == ROLE_ADMIN:
        return "owner"
    for m in project.members:
        if m.user_id == user.id:
            return m.role
    return None


def _level(role: str | None) -> int:
    if role == "owner":
        return _LEVEL[OWNER]
    if role == MEMBER_EDITOR:
        return _LEVEL[EDIT]
    return _LEVEL[VIEW] if role else 0


def get_project(db: Session, user: User, project_id: int, need: str = VIEW) -> Project:
    """Proyecto privado: propietario, administradores y usuarios con los que se compartió.

    `need` indica lo que se quiere hacer: ver, editar (cargar datos, ajustar parámetros) o gestionar
    (sólo propietario/administrador). Quien no tiene acceso al proyecto recibe 404, como si no existiera;
    quien lo ve pero no tiene permiso suficiente recibe 403.
    """
    project = db.get(Project, project_id)
    role = project_role(project, user) if project else None
    if project is None or role is None:
        raise ApiError(404, "PROJECT_NOT_FOUND", "Proyecto no encontrado.")
    if _level(role) < _LEVEL[need]:
        msg = (
            "Tu acceso a este proyecto es de sólo lectura."
            if need == EDIT
            else "Sólo el propietario del proyecto puede hacer esto."
        )
        raise ApiError(403, "PROJECT_FORBIDDEN", msg)
    return project


def accessible_project_ids(user: User):
    """Subconsulta con los ids de proyectos propios o compartidos con el usuario."""
    return select(Project.id).where(
        (Project.owner_id == user.id)
        | Project.id.in_(select(ProjectMember.project_id).where(ProjectMember.user_id == user.id))
    )


def get_model(db: Session, user: User, model_id: int) -> ModelVersion:
    model = db.get(ModelVersion, model_id)
    if model is None:
        raise ApiError(404, "MODEL_NOT_FOUND", "Modelo no encontrado.")
    get_project(db, user, model.project_id)
    return model


NIGHT = "night"  # = training_queue.TRIGGER_NIGHT (sin importar ese módulo aquí)


def training_active(db: Session) -> bool:
    """Entrenamiento que bloquea a todos: el semanal y los manuales. Los nocturnos sólo bloquean su proyecto."""
    night_runs = select(WeeklyRun.id).where(WeeklyRun.trigger == NIGHT)
    if db.scalar(
        select(TrainingJob.id)
        .where(
            TrainingJob.status.in_(ACTIVE_JOB_STATES),
            TrainingJob.run_id.is_(None) | TrainingJob.run_id.not_in(night_runs),
        )
        .limit(1)
    ):
        return True
    return bool(
        db.scalar(select(WeeklyRun.id).where(WeeklyRun.status == "running", WeeklyRun.trigger != NIGHT).limit(1))
    )


def night_lock(db: Session, project: Project) -> dict | None:
    """Si el proyecto se está entrenando en la ejecución nocturna, el detalle del bloqueo (sólo de ese proyecto)."""
    run = db.scalar(
        select(WeeklyRun).where(
            WeeklyRun.trigger == NIGHT, WeeklyRun.status == "running", WeeklyRun.week_key.like(f"night-%-p{project.id}")
        )
    )
    if run is None:
        return None
    pending = db.scalars(
        select(TrainingJob).where(TrainingJob.run_id == run.id, TrainingJob.status.in_(ACTIVE_JOB_STATES))
    ).unique().all()
    remaining = sum(
        max(j.eta_seconds if j.eta_seconds is not None else (j.estimated_seconds or 0), 0) for j in pending
    )
    return {
        "kind": NIGHT,
        "message": "Este proyecto se está entrenando esta noche. Podrás cargar y modificar datos cuando termine.",
        "remaining_seconds": round(remaining),
    }


def lock_state(db: Session) -> dict:
    return calendar_rules.window_state(training_active(db))


def ensure_modifications_open(db: Session, project: Project | None = None, actor: User | None = None) -> None:
    """Verificación del calendario en el backend (no sólo en la interfaz).

    Además del bloqueo general (fin de semana, entrenamiento semanal o manual), de noche se bloquea sólo el
    proyecto que se está entrenando: su dataset ya está congelado y el resto sigue abierto.
    Si se permite la modificación, queda registrada como actividad del proyecto (se guarda con el commit de
    la operación): el planificador nocturno pregunta antes de congelar un proyecto en el que alguien trabaja.
    """
    state = lock_state(db)
    if not state["open"]:
        raise ApiError(423, "UPLOADS_LOCKED", state["message"], {"lock": state})
    if project is not None and (lock := night_lock(db, project)):
        raise ApiError(423, "PROJECT_TRAINING", lock["message"], {"lock": lock})
    if project is not None and actor is not None:
        project.last_activity_at = utcnow()
        project.last_activity_by_id = actor.id
