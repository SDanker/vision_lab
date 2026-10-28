from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_db
from ..deps import require_approved
from ..errors import ApiError
from ..modalities import MEDIA, get_modality
from ..models import Category, Image, Project, User
from ..schemas import ImageMoveIn
from ..services import tabular
from ..services.access import EDIT, VIEW, ensure_modifications_open, get_project
from ..services.audit import audit
from ..storage import (
    InvalidImage,
    delete_image_files,
    image_path,
    thumb_path,
    user_image_bytes,
    user_quota_bytes,
)

router = APIRouter(prefix="/api", tags=["images"])

MAX_FILES_PER_REQUEST = 200
# Lo subido por usuarios nunca se interpreta como página: sin scripts ni recursos y en un sandbox aparte
FILE_HEADERS = {"Cache-Control": "private, max-age=3600", "Content-Security-Policy": "default-src 'none'; sandbox"}


def image_dict(img: Image) -> dict:
    return {
        "id": img.id,
        "project_id": img.project_id,
        "category_id": img.category_id,
        "name": img.original_name,
        "size_bytes": img.size_bytes,
        "width": img.width,
        "height": img.height,
        "format": img.fmt,
        "duration_s": img.duration_s,
        "sha256": img.sha256,
        "created_at": img.created_at,
    }


async def _read_limited(file: UploadFile, limit: int) -> bytes | None:
    chunks, total = [], 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


@router.post("/projects/{project_id}/images", status_code=201)
async def upload_images(
    project_id: int,
    request: Request,
    category_id: int | None = Form(None),  # datos tabulares: sin categoría (los CSV van a la categoría interna)
    files: list[UploadFile] = File(...),
    user: User = Depends(require_approved),
    db: Session = Depends(get_db),
):
    project = get_project(db, user, project_id, EDIT)
    ensure_modifications_open(db, project, user)
    mod = get_modality(project.modality)
    if category_id is None and mod.info.tabular:
        cat = tabular.ensure_data_category(db, project)
    else:
        cat = db.get(Category, category_id) if category_id is not None else None
        if not cat or cat.project_id != project.id:
            raise ApiError(404, "CATEGORY_NOT_FOUND", "Categoría no encontrada.")
    if len(files) > MAX_FILES_PER_REQUEST:
        raise ApiError(413, "TOO_MANY_FILES", f"Máximo {MAX_FILES_PER_REQUEST} archivos por carga.")

    s = get_settings()
    max_mb = mod.max_upload_mb(s)
    limit = max_mb * 1024 * 1024
    owner = project.owner
    quota = user_quota_bytes(owner)
    used = user_image_bytes(db, owner.id)
    uploaded, duplicates, rejected = [], [], []

    for f in files:
        name = f.filename or "sin-nombre"
        data = await _read_limited(f, limit)
        if data is None:
            rejected.append({"filename": name, "reason": f"Supera el máximo de {max_mb} MB."})
            continue
        try:  # validar puede ser lento (ffmpeg, PDF…): fuera del bucle de eventos
            v = await run_in_threadpool(mod.validate, data, name)
        except InvalidImage as exc:
            rejected.append({"filename": name, "reason": str(exc)})
            continue
        if db.scalar(select(Image.id).where(Image.project_id == project.id, Image.sha256 == v.sha256)):
            duplicates.append({"filename": name, "reason": "Archivo duplicado (mismo contenido)."})
            continue
        if used + len(data) > quota:
            rejected.append({"filename": name, "reason": "Cuota de almacenamiento excedida."})
            continue
        stored = await run_in_threadpool(mod.save, project.id, v)
        img = Image(
            project_id=project.id, category_id=cat.id, original_name=name[:255], stored_name=stored,
            sha256=v.sha256, size_bytes=len(data), fmt=v.ext, **mod.item_fields(v),
        )
        try:
            with db.begin_nested():
                db.add(img)
        except IntegrityError:
            delete_image_files(img)
            duplicates.append({"filename": name, "reason": "Archivo duplicado (mismo contenido)."})
            continue
        used += len(data)
        uploaded.append(image_dict(img))
    audit(
        db, "images.uploaded", user, "project", project.id,
        {"uploaded": len(uploaded), "duplicates": len(duplicates), "rejected": len(rejected)}, request,
    )
    db.commit()
    return {"uploaded": uploaded, "duplicates": duplicates, "rejected": rejected}


@router.get("/projects/{project_id}/images")
def list_images(
    project_id: int,
    category_id: int | None = None,
    page: int = 1,
    page_size: int = 60,
    user: User = Depends(require_approved),
    db: Session = Depends(get_db),
):
    project = get_project(db, user, project_id)
    page_size = max(1, min(page_size, 200))
    q = select(Image).where(Image.project_id == project.id)
    if category_id:
        q = q.where(Image.category_id == category_id)
    total = db.scalar(select(func.count()).select_from(q.subquery()))
    rows = db.scalars(q.order_by(Image.id.desc()).offset((max(page, 1) - 1) * page_size).limit(page_size)).all()
    return {"total": total, "page": page, "page_size": page_size, "items": [image_dict(i) for i in rows]}


def _owned_image(db: Session, user: User, image_id: int, need: str = VIEW) -> Image:
    img = db.get(Image, image_id)
    if img is None:
        raise ApiError(404, "IMAGE_NOT_FOUND", "Imagen no encontrada.")
    get_project(db, user, img.project_id, need)
    return img


@router.get("/images/{image_id}/thumb")
def get_thumb(image_id: int, user: User = Depends(require_approved), db: Session = Depends(get_db)):
    img = _owned_image(db, user, image_id)
    path = thumb_path(img)
    if not path.exists():
        raise ApiError(404, "FILE_MISSING", "Miniatura no disponible.")
    return FileResponse(path, media_type="image/jpeg", headers=FILE_HEADERS)


@router.get("/images/{image_id}/file")
def get_file(image_id: int, user: User = Depends(require_approved), db: Session = Depends(get_db)):
    img = _owned_image(db, user, image_id)
    path = image_path(img)
    if not path.exists():
        raise ApiError(404, "FILE_MISSING", "Archivo no disponible.")
    media = MEDIA.get(img.fmt, "application/octet-stream")
    return FileResponse(path, media_type=media, headers=FILE_HEADERS)


@router.delete("/images/{image_id}", status_code=204)
def delete_image(
    image_id: int, request: Request, user: User = Depends(require_approved), db: Session = Depends(get_db)
):
    img = _owned_image(db, user, image_id, EDIT)
    ensure_modifications_open(db, db.get(Project, img.project_id), user)
    delete_image_files(img)
    audit(db, "image.deleted", user, "image", img.id, {"project_id": img.project_id}, request)
    db.delete(img)
    db.commit()


@router.patch("/images/{image_id}")
def move_image(
    image_id: int, data: ImageMoveIn, user: User = Depends(require_approved), db: Session = Depends(get_db)
):
    img = _owned_image(db, user, image_id, EDIT)
    project = db.get(Project, img.project_id)
    if tabular.is_tabular(project):
        raise ApiError(409, "NOT_APPLICABLE", "Los archivos de datos tabulares no se agrupan en categorías.")
    ensure_modifications_open(db, project, user)
    cat = db.get(Category, data.category_id)
    if not cat or cat.project_id != img.project_id:
        raise ApiError(404, "CATEGORY_NOT_FOUND", "Categoría no encontrada.")
    img.category_id = cat.id
    db.commit()
    return image_dict(img)
