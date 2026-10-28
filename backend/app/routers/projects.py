from fastapi import APIRouter, Depends, Request
from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .. import calendar_rules
from ..config import get_settings
from ..db import get_db
from ..deps import require_approved
from ..errors import ApiError
from ..models import (
    APPROVED,
    ROLE_ADMIN,
    Category,
    Image,
    ModelVersion,
    Project,
    ProjectMember,
    TrainingJob,
    User,
)
from ..modalities import get_modality
from ..schemas import CategoryIn, MemberIn, MemberPatchIn, ProjectIn, ProjectPatchIn, check_models_fit
from ..services import tabular
from ..services.access import (
    EDIT,
    OWNER,
    accessible_project_ids,
    ensure_modifications_open,
    night_lock,
    get_project,
    project_role,
)
from ..services.audit import audit
from ..services.notifications import notify_user
from ..services.training_queue import project_models
from ..storage import delete_project_files

router = APIRouter(prefix="/api/projects", tags=["projects"])

ROLE_LABEL = {"viewer": "sólo lectura", "editor": "editor"}


def _no_categories_in_tables(project: Project) -> None:
    if tabular.is_tabular(project):
        raise ApiError(409, "NOT_APPLICABLE", "En datos tabulares las categorías salen de los valores de la columna a predecir.")


def member_dict(m: ProjectMember) -> dict:
    return {
        "user_id": m.user_id,
        "username": m.user.username,
        "name": m.user.full_name,
        "role": m.role,
        "added_at": m.created_at,
    }


def project_dict(db: Session, p: Project, user: User | None = None) -> dict:
    counts = dict(
        db.execute(
            select(Image.category_id, func.count(Image.id))
            .where(Image.project_id == p.id)
            .group_by(Image.category_id)
        ).all()
    )
    size = db.scalar(select(func.coalesce(func.sum(Image.size_bytes), 0)).where(Image.project_id == p.id))
    latest = db.scalar(
        select(ModelVersion).where(ModelVersion.project_id == p.id).order_by(ModelVersion.version.desc()).limit(1)
    )
    return {
        "id": p.id,
        "name": p.name,
        "description": p.description,
        "base_model": p.base_model,
        "base_models": project_models(p),
        "config": p.config,
        "auto_train": p.auto_train,
        # cuándo se ejecutaría el próximo entrenamiento semanal si el proyecto está marcado
        "next_training_start": calendar_rules.next_training_start().isoformat() if p.auto_train else None,
        # entrenamiento nocturno (lun–vie): si está activo, un proyecto marcado puede entrenarse antes del sábado
        "night_window": f"{get_settings().night_window_start}–{get_settings().night_window_end}" if get_settings().night_training_enabled else None,
        "modality": p.modality or "image",
        "created_at": p.created_at,
        "owner": p.owner.username,
        "owner_id": p.owner_id,
        # rol de quien consulta: owner | editor | viewer
        "my_role": (project_role(p, user) if user else None),
        # de noche se bloquea sólo el proyecto que se entrena
        "training_lock": night_lock(db, p),
        "members": [member_dict(m) for m in sorted(p.members, key=lambda m: m.user.username)],
        "categories": [{"id": c.id, "name": c.name, "image_count": counts.get(c.id, 0)} for c in p.categories],
        "image_count": sum(counts.values()),
        "min_images_per_class": get_settings().min_images_per_class,
        "size_bytes": int(size or 0),
        "latest_model": {"id": latest.id, "version": latest.version, "accuracy": latest.metrics.get("test_accuracy")}
        if latest
        else None,
    }


@router.get("")
def list_projects(
    all: bool = False, user: User = Depends(require_approved), db: Session = Depends(get_db)
):
    """Proyectos propios y compartidos conmigo. Un administrador puede pedir todos con all=true."""
    q = select(Project).order_by(Project.created_at.desc())
    if not (all and user.role == ROLE_ADMIN):
        q = q.where(Project.id.in_(accessible_project_ids(user)))
    return [project_dict(db, p, user) for p in db.scalars(q).unique().all()]


@router.post("", status_code=201)
def create_project(
    data: ProjectIn, request: Request, user: User = Depends(require_approved), db: Session = Depends(get_db)
):
    names = list(dict.fromkeys(n.strip() for n in data.categories if n.strip()))
    if get_modality(data.modality).info.tabular:
        names = [tabular.DATA_CATEGORY]  # las categorías salen de la columna a predecir, no de carpetas
    project = Project(
        owner_id=user.id,
        name=data.name,
        description=data.description,
        base_model=data.base_model,
        config={**data.config.to_dict(), "base_models": data.base_models},
        auto_train=data.auto_train,
        modality=data.modality,
    )
    project.categories = [Category(name=n) for n in names]
    db.add(project)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise ApiError(409, "PROJECT_EXISTS", "Ya tienes un proyecto con ese nombre.") from None
    audit(db, "project.created", user, "project", project.id, {"name": project.name}, request)
    db.commit()
    return project_dict(db, project, user)


@router.get("/{project_id}")
def read_project(project_id: int, user: User = Depends(require_approved), db: Session = Depends(get_db)):
    return project_dict(db, get_project(db, user, project_id), user)


@router.patch("/{project_id}")
def update_project(
    project_id: int, data: ProjectPatchIn, request: Request,
    user: User = Depends(require_approved), db: Session = Depends(get_db),
):
    project = get_project(db, user, project_id, EDIT)
    if (data.name is not None or data.description is not None) and project_role(project, user) != "owner":
        raise ApiError(403, "PROJECT_FORBIDDEN", "Sólo el propietario puede cambiar el nombre y la descripción.")
    ensure_modifications_open(db, project, user)
    changed: list[str] = []
    was_auto = project.auto_train
    if data.name is not None:
        project.name = data.name.strip()
        changed.append("name")
    if data.description is not None:
        project.description = data.description
        changed.append("description")
    models_now = project_models(project)
    if data.config is not None:
        new_config = {**data.config.to_dict(), "base_models": models_now}
        for key in ("tabular_target", "tabular_features"):  # se cambian en la pestaña de datos, no al guardar el resto
            if key not in new_config and key in (project.config or {}):
                new_config[key] = project.config[key]
        project.config = new_config
        changed.append("config")
    if data.base_models:
        try:
            check_models_fit(data.base_models, tabular.is_tabular(project))
        except ValueError as exc:
            raise ApiError(422, "MODEL_MODALITY_MISMATCH", str(exc)) from None
        project.base_model = data.base_models[0]
        project.config = {**project.config, "base_models": data.base_models}
        changed.append("base_models")
    if data.auto_train is not None:
        project.auto_train = data.auto_train
        changed.append("auto_train")
    audit(db, "project.updated", user, "project", project.id, {"changed": changed}, request)
    if project.auto_train and not was_auto:  # volvió a marcarlo: se le avisa cuándo se ejecutará
        when = calendar_rules.describe_next_execution()
        notify_user(
            db, user, "training.scheduled", f"«{project.name}» se entrenará pronto",
            f"Quedó marcado para el entrenamiento semanal: se ejecutaría {when}. Las cargas y modificaciones se "
            "bloquean desde el sábado a las 00:00 hasta que se publiquen los resultados.",
            email=False,
        )
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise ApiError(409, "PROJECT_EXISTS", "Ya tienes un proyecto con ese nombre.") from None
    return project_dict(db, project, user)


@router.delete("/{project_id}", status_code=204)
def delete_project(
    project_id: int, request: Request, user: User = Depends(require_approved), db: Session = Depends(get_db)
):
    project = get_project(db, user, project_id, OWNER)
    ensure_modifications_open(db, project, user)
    model_dirs = [m.dir_path for m in db.scalars(select(ModelVersion).where(ModelVersion.project_id == project.id))]
    audit(db, "project.deleted", user, "project", project.id, {"name": project.name}, request)
    db.delete(project)
    db.commit()
    delete_project_files(project_id, model_dirs)


@router.post("/{project_id}/categories", status_code=201)
def add_category(
    project_id: int, data: CategoryIn, user: User = Depends(require_approved), db: Session = Depends(get_db)
):
    project = get_project(db, user, project_id, EDIT)
    _no_categories_in_tables(project)
    ensure_modifications_open(db, project, user)
    cat = Category(project_id=project.id, name=data.name)
    db.add(cat)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise ApiError(409, "CATEGORY_EXISTS", "Esa categoría ya existe en el proyecto.") from None
    return {"id": cat.id, "name": cat.name, "image_count": 0}


@router.patch("/{project_id}/categories/{category_id}")
def rename_category(
    project_id: int, category_id: int, data: CategoryIn,
    user: User = Depends(require_approved), db: Session = Depends(get_db),
):
    project = get_project(db, user, project_id, EDIT)
    _no_categories_in_tables(project)
    ensure_modifications_open(db, project, user)
    cat = db.get(Category, category_id)
    if not cat or cat.project_id != project.id:
        raise ApiError(404, "CATEGORY_NOT_FOUND", "Categoría no encontrada.")
    cat.name = data.name
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise ApiError(409, "CATEGORY_EXISTS", "Esa categoría ya existe en el proyecto.") from None
    return {"id": cat.id, "name": cat.name}


@router.delete("/{project_id}/categories/{category_id}", status_code=204)
def delete_category(
    project_id: int, category_id: int, user: User = Depends(require_approved), db: Session = Depends(get_db)
):
    from ..storage import delete_image_files

    # borra también todas las imágenes de la categoría: sólo el propietario
    project = get_project(db, user, project_id, OWNER)
    _no_categories_in_tables(project)
    ensure_modifications_open(db, project, user)
    cat = db.get(Category, category_id)
    if not cat or cat.project_id != project.id:
        raise ApiError(404, "CATEGORY_NOT_FOUND", "Categoría no encontrada.")
    images = db.scalars(select(Image).where(Image.category_id == cat.id)).all()
    for img in images:
        delete_image_files(img)
    db.delete(cat)
    db.commit()


@router.get("/{project_id}/stats")
def project_stats(project_id: int, user: User = Depends(require_approved), db: Session = Depends(get_db)):
    p = get_project(db, user, project_id)
    d = project_dict(db, p, user)
    jobs = db.scalar(select(func.count(TrainingJob.id)).where(TrainingJob.project_id == p.id))
    return {**d, "training_jobs": jobs}


# ------------------------------------------------------------------ compartir
@router.get("/{project_id}/members")
def list_members(project_id: int, user: User = Depends(require_approved), db: Session = Depends(get_db)):
    p = get_project(db, user, project_id)
    return {
        "owner": {"user_id": p.owner_id, "username": p.owner.username, "name": p.owner.full_name},
        "members": [member_dict(m) for m in sorted(p.members, key=lambda m: m.user.username)],
        "my_role": project_role(p, user),
    }


@router.post("/{project_id}/members", status_code=201)
def add_member(
    project_id: int, data: MemberIn, request: Request,
    user: User = Depends(require_approved), db: Session = Depends(get_db),
):
    project = get_project(db, user, project_id, OWNER)
    ident = data.identifier.strip().lower()
    target = db.scalar(
        select(User).where(or_(func.lower(User.username) == ident, func.lower(User.email) == ident))
    )
    if target is None or target.status != APPROVED:
        raise ApiError(404, "USER_NOT_FOUND", "No hay ningún usuario activo con ese nombre de usuario o correo.")
    if target.id == project.owner_id:
        raise ApiError(409, "ALREADY_OWNER", "Esa persona ya es la propietaria del proyecto.")
    if any(m.user_id == target.id for m in project.members):
        raise ApiError(409, "ALREADY_MEMBER", "El proyecto ya está compartido con ese usuario.")
    member = ProjectMember(project_id=project.id, user_id=target.id, role=data.role, added_by_id=user.id)
    project.members.append(member)
    audit(
        db, "project.shared", user, "project", project.id,
        {"with": target.username, "role": data.role}, request,
    )
    notify_user(
        db, target, "project.shared", f"{user.username} compartió contigo el proyecto «{project.name}»",
        f"Tu acceso es de {ROLE_LABEL[data.role]}. Lo verás en tu lista de proyectos.",
        meta={"project_id": project.id},
    )
    db.commit()
    db.refresh(member)
    return member_dict(member)


def _member_of(project: Project, user_id: int) -> ProjectMember:
    m = next((m for m in project.members if m.user_id == user_id), None)
    if m is None:
        raise ApiError(404, "MEMBER_NOT_FOUND", "Ese usuario no tiene acceso al proyecto.")
    return m


@router.patch("/{project_id}/members/{user_id}")
def change_member_role(
    project_id: int, user_id: int, data: MemberPatchIn, request: Request,
    user: User = Depends(require_approved), db: Session = Depends(get_db),
):
    project = get_project(db, user, project_id, OWNER)
    member = _member_of(project, user_id)
    member.role = data.role
    audit(
        db, "project.share_role", user, "project", project.id,
        {"with": member.user.username, "role": data.role}, request,
    )
    db.commit()
    return member_dict(member)


@router.delete("/{project_id}/members/{user_id}", status_code=204)
def remove_member(
    project_id: int, user_id: int, request: Request,
    user: User = Depends(require_approved), db: Session = Depends(get_db),
):
    """El propietario quita a cualquiera; cada miembro puede abandonar el proyecto por sí mismo."""
    project = get_project(db, user, project_id)
    leaving = user_id == user.id and project.owner_id != user.id
    if not leaving and project_role(project, user) != "owner":
        raise ApiError(403, "PROJECT_FORBIDDEN", "Sólo el propietario puede quitar el acceso a otras personas.")
    member = _member_of(project, user_id)
    audit(
        db, "project.unshared", user, "project", project.id,
        {"with": member.user.username, "left": leaving}, request,
    )
    project.members.remove(member)
    db.commit()
