"""Estimación de la duración de un entrenamiento.

Costo de un trabajo ≈ imágenes × épocas × (lado/224)² × tasa + costo fijo (preparar datos, evaluar y exportar).
La tasa (segundos por imagen y época a 224 px) se aprende del historial de trabajos completados en este
servidor; mientras no hay historial se usa una tasa de referencia según la velocidad del modelo base.
"""
from statistics import median

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..ml.architectures import ARCHITECTURES
from ..models import J_COMPLETED, DatasetSnapshot, TrainingJob

# segundos por imagen y época a 224 px en CPU (referencia inicial; el historial la reemplaza)
SPEED_RATE = {"muy rápido": 0.02, "rápido": 0.035, "medio": 0.07, "lento": 0.18, "muy lento": 0.4}
FORMULA_MARGIN = 1.5  # sin historial se estima con holgura
HISTORY_SAMPLES = 8  # trabajos recientes que se consideran


def fixed_overhead(n_images: int) -> float:
    """Preparar el dataset, evaluar y exportar/validar 5 formatos: no depende de las épocas."""
    return 120.0 + 0.02 * n_images


def _units(n_images: int, epochs: int, image_size: int) -> float:
    return max(n_images, 1) * max(epochs, 1) * (image_size / 224) ** 2


def _rate_of(job: TrainingJob, snap: DatasetSnapshot | None) -> float | None:
    if not (job.started_at and job.finished_at and snap and snap.total_images):
        return None
    epochs_run = len(job.history or []) or job.total_epochs
    size = int((job.config or {}).get("image_size", 224))
    work = (job.finished_at - job.started_at).total_seconds() - fixed_overhead(snap.total_images)
    if work <= 0 or not epochs_run:
        return None
    return work / _units(snap.total_images, epochs_run, size)


def _history_rate(db: Session, base_model: str, project_id: int | None) -> float | None:
    q = (
        select(TrainingJob, DatasetSnapshot)
        .join(DatasetSnapshot, DatasetSnapshot.id == TrainingJob.snapshot_id)
        .where(TrainingJob.status == J_COMPLETED, TrainingJob.base_model == base_model)
        .order_by(TrainingJob.finished_at.desc())
        .limit(HISTORY_SAMPLES)
    )
    if project_id is not None:
        q = q.where(TrainingJob.project_id == project_id)
    rates = [r for job, snap in db.execute(q).unique().all() if (r := _rate_of(job, snap))]
    return median(rates) if rates else None


def estimate_job_seconds(db: Session, project_id: int, base_model: str, cfg: dict, n_images: int) -> tuple[float, str]:
    """Devuelve (segundos, fuente). Fuente: «proyecto», «modelo» o «fórmula».

    Se usa el máximo de épocas configurado (el early stopping suele cortar antes): la estimación es
    conservadora, que es lo que conviene para decidir si cabe en la noche.
    """
    if cfg.get("modality") == "tabular":  # scikit-learn: segundos, no horas; `n_images` son filas
        from ..ml import tabular

        return tabular.estimate_seconds(base_model, n_images, len(cfg.get("tabular_features") or ()) or 10), "fórmula"
    epochs = int(cfg.get("epochs", 10)) + int(cfg.get("fine_tune_epochs", 0))
    size = int(cfg.get("image_size", 224))
    units = _units(n_images, epochs, size)
    for source, scope in (("proyecto", project_id), ("modelo", None)):
        rate = _history_rate(db, base_model, scope)
        if rate:
            return units * rate + fixed_overhead(n_images), source
    speed = ARCHITECTURES.get(base_model, {}).get("speed", "medio")
    rate = SPEED_RATE.get(speed, SPEED_RATE["medio"])
    return (units * rate + fixed_overhead(n_images)) * FORMULA_MARGIN, "fórmula"
