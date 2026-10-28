import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_db
from ..deps import require_approved
from ..errors import ApiError
from ..modalities import InvalidInput, get_modality
from ..models import ALL_EXPORT_FORMATS, ModelVersion, Project, TrainingJob, User
from ..services import inference, similarity
from ..services.access import OWNER, get_model, get_project, project_role
from ..services.audit import audit
from ..storage import delete_model_files

router = APIRouter(prefix="/api", tags=["models"])
log = logging.getLogger(__name__)

EXTRA_FILES = {
    "labels": ("labels.json", "application/json"),
    "metadata": ("metadata.json", "application/json"),
    "metrics": ("metrics.json", "application/json"),
    "inference": ("inference.py", "text/x-python"),
    "readme": ("README.md", "text/markdown"),
    "all": ("bundle.zip", "application/zip"),
}
MEDIA = {
    "keras": "application/octet-stream",
    "h5": "application/x-hdf5",
    "savedmodel": "application/zip",
    "tflite": "application/octet-stream",
    "onnx": "application/octet-stream",
    "joblib": "application/octet-stream",
}


def model_dict(m: ModelVersion, detail: bool = False, role: str | None = None) -> dict:
    d = {
        "my_role": role,
        "can_delete": role == "owner",  # borrar un modelo es cosa del propietario del proyecto (o de un administrador)
        "id": m.id,
        "project_id": m.project_id,
        "version": m.version,
        "base_model": m.base_model,
        "labels": m.labels,
        "created_at": m.created_at,
        "published": m.published,
        "accuracy": m.metrics.get("test_accuracy"),
        "top5": m.metrics.get("test_top5_accuracy"),
        "macro_f1": m.metrics.get("macro_f1"),
        "epochs_run": m.metrics.get("epochs_run"),
        "dataset_hash": m.dataset_hash,
        "modality": m.config.get("modality", "image"),
        "exports": [
            {"format": e.fmt, "status": e.status, "filename": e.filename, "size_bytes": e.size_bytes,
             "validation": e.validation, "error": e.error}
            for e in sorted(m.exports, key=lambda e: ALL_EXPORT_FORMATS.index(e.fmt))
        ],
    }
    if detail:
        d["metrics"] = m.metrics
        d["config"] = m.config
        if d["modality"] == "tabular":
            d["input_spec"] = _tabular_spec(m)
        else:
            s = int(m.config.get("image_size", 224))
            d["input_spec"] = {"shape": [1, s, s, 3], "dtype": "float32", "range": [0, 255]}
    return d


def _tabular_spec(m: ModelVersion) -> dict:
    """Columnas de entrada del modelo (tipo, categorías conocidas, rango): con ellas la interfaz arma el formulario."""
    try:
        meta = json.loads((Path(m.dir_path) / "metadata.json").read_text(encoding="utf-8"))["tabular"]
        return {"kind": "tabular", "target": meta.get("target"), "features": meta["features"]}
    except (OSError, ValueError, KeyError):
        return {"kind": "tabular", "target": None, "features": []}


@router.get("/projects/{project_id}/models")
def list_models(project_id: int, user: User = Depends(require_approved), db: Session = Depends(get_db)):
    project = get_project(db, user, project_id)
    role = project_role(project, user)

    rows = db.scalars(select(ModelVersion).where(ModelVersion.project_id == project_id).order_by(ModelVersion.version.desc())).all()
    jobs = {j.id: j for j in db.scalars(select(TrainingJob).where(TrainingJob.id.in_([m.job_id for m in rows if m.job_id]))).unique()}
    out = []
    for m in rows:
        j = jobs.get(m.job_id)
        secs = (j.finished_at - j.started_at).total_seconds() if j and j.started_at and j.finished_at else None
        out.append({**model_dict(m, role=role), "train_seconds": secs})
    return out


@router.get("/models")
def all_models(user: User = Depends(require_approved), db: Session = Depends(get_db)):
    from ..models import ROLE_ADMIN
    from ..services.access import accessible_project_ids

    q = select(ModelVersion, Project).join(Project, Project.id == ModelVersion.project_id).order_by(ModelVersion.created_at.desc())
    if user.role != ROLE_ADMIN:
        q = q.where(Project.id.in_(accessible_project_ids(user)))
    return [
        {**model_dict(m, role=project_role(p, user)), "project_name": p.name, "project_owner": p.owner.username}
        for m, p in db.execute(q.limit(300)).unique().all()
    ]


@router.get("/models/{model_id}")
def read_model(model_id: int, user: User = Depends(require_approved), db: Session = Depends(get_db)):
    from ..ml.train import training_hints

    m = get_model(db, user, model_id)
    d = model_dict(m, detail=True, role=project_role(db.get(Project, m.project_id), user))
    if "hints" not in d["metrics"] and m.job_id:  # modelos entrenados antes de existir los diagnósticos
        job = db.get(TrainingJob, m.job_id)
        if job and job.history:
            d["metrics"] = {**d["metrics"], "hints": training_hints(job.history, int(m.config.get("fine_tune_epochs", 0)))}
    return d


@router.delete("/models/{model_id}", status_code=204)
def delete_model(model_id: int, request: Request, user: User = Depends(require_approved), db: Session = Depends(get_db)):
    """Borra un modelo ya entrenado y sus archivos. Sólo el propietario del proyecto o un administrador.

    El trabajo de entrenamiento que lo generó se conserva en el historial (sólo pierde la referencia).
    """
    m = get_model(db, user, model_id)
    project = get_project(db, user, m.project_id, need=OWNER)
    dir_path = m.dir_path
    db.execute(update(TrainingJob).where(TrainingJob.model_version_id == m.id).values(model_version_id=None))
    audit(
        db, "model.deleted", user, "model", m.id,
        {"project": project.name, "project_id": project.id, "version": m.version, "base_model": m.base_model}, request,
    )
    db.delete(m)  # sus exportaciones se borran en cascada y dejan de contar en la cuota del propietario
    db.commit()
    inference.evict(dir_path)
    delete_model_files(dir_path)  # tras confirmar la BD: si falla queda una carpeta huérfana, nunca un modelo sin archivos


@router.get("/models/{model_id}/download/{fmt}")
def download(model_id: int, fmt: str, request: Request, user: User = Depends(require_approved), db: Session = Depends(get_db)):
    m = get_model(db, user, model_id)
    base = Path(m.dir_path)
    if fmt in ALL_EXPORT_FORMATS:
        export = next((e for e in m.exports if e.fmt == fmt), None)
        if export is None or export.status != "ok":
            raise ApiError(409, "EXPORT_NOT_VALID", f"El formato {fmt} no está disponible: no pasó la validación.")
        filename, media = export.filename, MEDIA[fmt]
    elif fmt in EXTRA_FILES:
        filename, media = EXTRA_FILES[fmt]
    else:
        raise ApiError(404, "FORMAT_UNKNOWN", "Formato desconocido.")
    path = base / filename
    if not path.is_file():
        raise ApiError(404, "FILE_MISSING", "El archivo ya no está disponible en el servidor.")
    audit(db, "model.downloaded", user, "model", m.id, {"format": fmt}, request)
    db.commit()
    return FileResponse(path, media_type=media, filename=f"{_slug(m)}-{filename}")


def _slug(m: ModelVersion) -> str:
    return f"modelo-p{m.project_id}-v{m.version}"


@router.post("/models/{model_id}/predict")
async def predict(
    model_id: int, request: Request, file: UploadFile = File(...), similar: int = Query(0, ge=0, le=8),
    user: User = Depends(require_approved), db: Session = Depends(get_db),
):
    """Clasifica un archivo de prueba (no se guarda). Con `similar=N` añade los N archivos del proyecto más parecidos."""
    from starlette.concurrency import run_in_threadpool

    m = get_model(db, user, model_id)
    mod = get_modality(m.config.get("modality"))
    max_mb = mod.max_upload_mb(get_settings())
    data = await file.read(max_mb * 1024 * 1024 + 1)
    if len(data) > max_mb * 1024 * 1024:
        raise ApiError(413, "FILE_TOO_LARGE", f"El archivo supera el máximo de {max_mb} MB.")
    ok = {e.fmt: e.status == "ok" for e in m.exports}
    try:
        result = await run_in_threadpool(mod.predict, m.dir_path, m.labels, data, file.filename or "", ok, m.config)
    except InvalidInput as exc:
        raise ApiError(422, exc.code, exc.message) from None
    except RuntimeError as exc:
        raise ApiError(409, "MODEL_UNAVAILABLE", str(exc)) from None
    out = {**result, "model_id": m.id, "model_version": m.version, "project_id": m.project_id, "modality": mod.info.id}
    if similar:
        out.update(await _similar_block(db, m, mod, data, similar))
    return out


async def _similar_block(db: Session, m: ModelVersion, mod, data: bytes, k: int) -> dict:
    """Lo más parecido del proyecto. Un fallo aquí nunca impide devolver la clasificación."""
    from starlette.concurrency import run_in_threadpool

    try:
        query = await run_in_threadpool(mod.embed_file, m.dir_path, data, m.config)
        state = similarity.index_state(db, m)
        if state != "ready":
            similarity.kick_build(m.id)  # índice nuevo o desactualizado: se completa en segundo plano
        items = similarity.search(db, m, query, k) if state != "missing" else []
        return {"similar": items, "similar_status": {"ready": "ready", "stale": "updating", "missing": "building"}[state]}
    except Exception as exc:  # noqa: BLE001
        log.warning("«Lo más parecido» no disponible para el modelo %s: %s", m.id, exc)
        return {"similar": [], "similar_status": "unavailable"}


@router.get("/models/{model_id}/similar-status")
def similar_status(model_id: int, user: User = Depends(require_approved), db: Session = Depends(get_db)):
    """Estado del índice de parecidos del modelo (para saber cuándo reintentar)."""
    m = get_model(db, user, model_id)
    state = similarity.index_state(db, m)
    building = similarity.get_status(m.id)
    return {
        "state": "building" if building.get("state") == "building" else {"ready": "ready", "stale": "ready", "missing": "missing"}[state],
        "up_to_date": state == "ready",
        "done": building.get("done"), "total": building.get("total"), "error": building.get("error"),
    }
