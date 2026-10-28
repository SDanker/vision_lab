"""Hugging Face: cuenta de cada usuario (token cifrado) e importación de datasets a un proyecto de imágenes.

Todo se hace con la sesión web: un token de API (vl_…) no puede conectar ni usar la cuenta de Hugging Face de nadie.
"""
from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import ratelimit
from ..config import get_settings
from ..db import get_db, utcnow
from ..deps import require_session
from ..errors import ApiError
from ..models import HF_ACTIVE, HF_CANCELLED, HF_PENDING, HfConnection, HfImport, Project, User
from ..schemas import HfSourceIn, HfTokenIn
from ..services import hf, vault
from ..services.access import EDIT, ensure_modifications_open, get_project, project_role
from ..services.audit import audit
from ..storage import user_image_bytes, user_quota_bytes

router = APIRouter(prefix="/api", tags=["huggingface"])


def _no_store(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"


def _hf(call):
    """Ejecuta una llamada al Hub y deja los errores conocidos como respuestas HTTP claras."""
    try:
        return call()
    except hf.HfError as exc:
        raise exc.as_api() from None


# ------------------------------------------------------------------------------ cuenta
@router.get("/integrations/huggingface")
def hf_status(response: Response, user: User = Depends(require_session), db: Session = Depends(get_db)):
    _no_store(response)
    return hf.status_dict(hf.connection_of(db, user.id))


@router.put("/integrations/huggingface")
def hf_connect(
    data: HfTokenIn, request: Request, response: Response,
    user: User = Depends(require_session), db: Session = Depends(get_db),
):
    """Guarda el token (cifrado) tras comprobar con Hugging Face que es válido. El token nunca se devuelve."""
    _no_store(response)
    ratelimit.enforce(f"hfconnect:{user.id}", 10, 900, "Demasiados intentos de conectar Hugging Face. Espera unos minutos.")
    username = _hf(lambda: hf.hub_whoami(data.token))
    conn = hf.connection_of(db, user.id)
    if conn is None:
        conn = HfConnection(user_id=user.id)
        db.add(conn)
    conn.token_enc = vault.encrypt(data.token)
    conn.token_hint = hf.token_hint(data.token)
    conn.hf_username = username[:100]
    conn.created_at, conn.last_used_at = utcnow(), None
    audit(db, "hf.connected", user, "user", user.id, {"hf_user": username}, request)
    db.commit()
    return hf.status_dict(conn)


@router.delete("/integrations/huggingface", status_code=204)
def hf_disconnect(request: Request, user: User = Depends(require_session), db: Session = Depends(get_db)):
    conn = hf.connection_of(db, user.id)
    if conn is not None:
        audit(db, "hf.disconnected", user, "user", user.id, {"hf_user": conn.hf_username}, request)
        db.delete(conn)
        db.commit()


# ------------------------------------------------------------------------------ importación
def _image_project(db: Session, user: User, project_id: int) -> Project:
    project = get_project(db, user, project_id, EDIT)
    if project.modality != "image":
        raise ApiError(
            409, "NOT_APPLICABLE",
            "La importación desde Hugging Face por ahora sólo funciona en proyectos de imágenes.",
        )
    return project


def _opts(data: HfSourceIn) -> hf.HfOpts:
    return hf.HfOpts(
        image_column=data.image_column, label_source=data.label_source, label_column=data.label_column,
        label_phrase=data.label_phrase.model_dump() if data.label_phrase else None,
        default_category=data.default_category, include=data.include, rest=data.rest,
        other_label=data.other_label or "otros", max_items=data.max_items, max_per_category=data.max_per_category,
    )


def _plan_for(db: Session, user: User, project: Project, data: HfSourceIn) -> tuple[hf.Plan, hf.HfOpts]:
    token = _hf(lambda: hf.token_for(db, user.id))
    owner = project.owner
    free = user_quota_bytes(owner) - user_image_bytes(db, owner.id)
    opts = _opts(data)
    return _hf(lambda: hf.build_plan(data.source, token, free, opts)), opts


@router.post("/projects/{project_id}/hf/inspect")
def hf_inspect(
    project_id: int, data: HfSourceIn, user: User = Depends(require_session), db: Session = Depends(get_db),
):
    """Revisa un dataset sin descargarlo: tamaño, categorías y si cabe en el proyecto."""
    project = _image_project(db, user, project_id)
    ratelimit.enforce(f"hfinspect:{user.id}", 30, 300, "Demasiadas consultas a Hugging Face. Espera unos minutos.")
    plan, _ = _plan_for(db, user, project, data)
    db.commit()  # token_for anota el último uso
    return hf.plan_dict(plan)


@router.post("/projects/{project_id}/hf/imports", status_code=201)
def hf_start_import(
    project_id: int, data: HfSourceIn, request: Request,
    user: User = Depends(require_session), db: Session = Depends(get_db),
):
    project = _image_project(db, user, project_id)
    ensure_modifications_open(db, project, user)
    if hf.active_count(db, user.id) >= get_settings().hf_import_active_per_user:
        raise ApiError(409, "IMPORT_IN_PROGRESS", "Ya tienes una importación en curso. Espera a que termine o cancélala.")
    plan, opts = _plan_for(db, user, project, data)
    if plan.blockers:
        raise ApiError(422, "IMPORT_BLOCKED", plan.blockers[0], {"blockers": plan.blockers})
    imp = HfImport(
        project_id=project.id, user_id=user.id, repo_id=plan.repo_id, revision=plan.revision, status=HF_PENDING,
        phase="En cola", total=min(plan.items, opts.max_items) if (plan.items and opts.max_items) else (plan.items or 0),
        options=opts.stored(plan),
    )
    db.add(imp)
    db.flush()
    audit(
        db, "hf.import_started", user, "project", project.id,
        {
            "repo": plan.repo_id, "revision": plan.revision, "layout": plan.layout, "bytes": plan.effective_bytes,
            "private": plan.private, "label_source": plan.label_source, "include": len(opts.include) if opts.include is not None else None,
            "max_items": opts.max_items, "max_per_category": opts.max_per_category,
        },
        request,
    )
    db.commit()
    hf.kick_import(imp.id)
    db.refresh(imp)
    return hf.import_dict(imp)


@router.get("/projects/{project_id}/hf/imports")
def hf_imports(project_id: int, user: User = Depends(require_session), db: Session = Depends(get_db)):
    get_project(db, user, project_id)
    rows = db.scalars(
        select(HfImport).where(HfImport.project_id == project_id).order_by(HfImport.id.desc()).limit(10)
    ).unique().all()
    return [hf.import_dict(i) for i in rows]


@router.delete("/hf/imports/{import_id}")
def hf_cancel_import(
    import_id: int, request: Request, user: User = Depends(require_session), db: Session = Depends(get_db)
):
    """Cancela una importación: quien la inició o el propietario del proyecto. Lo ya importado se conserva."""
    imp = db.get(HfImport, import_id)
    project = db.get(Project, imp.project_id) if imp else None
    role = project_role(project, user) if project else None
    if imp is None or role is None:
        raise ApiError(404, "IMPORT_NOT_FOUND", "Importación no encontrada.")
    if imp.user_id != user.id and role != "owner":
        raise ApiError(403, "PROJECT_FORBIDDEN", "Sólo quien la inició o el propietario del proyecto puede cancelarla.")
    if imp.status not in HF_ACTIVE:
        raise ApiError(409, "INVALID_STATE", "Esta importación ya terminó.")
    if imp.status == HF_PENDING:
        imp.status, imp.phase, imp.finished_at = HF_CANCELLED, "Cancelada", utcnow()
    else:
        imp.cancel_requested, imp.phase = True, "Cancelando…"
    audit(db, "hf.import_cancelled", user, "project", imp.project_id, {"repo": imp.repo_id}, request)
    db.commit()
    return hf.import_dict(imp)
