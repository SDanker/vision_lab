"""Orquestación de un trabajo de entrenamiento: preparar → entrenar → validar → exportar."""
import json
import logging
import shutil
import traceback
from pathlib import Path

import numpy as np
from sqlalchemy import func, select

from .. import calendar_rules
from ..config import get_settings
from ..db import SessionLocal, engine, utcnow
from ..models import (
    J_CANCELLED,
    J_COMPLETED,
    J_ERROR,
    J_EXPORTING,
    J_PENDING,
    J_PREPARING,
    J_TRAINING,
    J_VALIDATING,
    DatasetSnapshot,
    ModelExport,
    ModelVersion,
    TrainingJob,
)
from ..services import similarity, training_queue as tq
from ..storage import data_dir
from .reporter import Cancelled, Reporter
from ..modalities import get_modality
from .architectures import label_of
from .templates import README

log = logging.getLogger(__name__)


def _postpone_if_late(db, job: TrainingJob) -> bool:
    """Un trabajo nocturno sólo se inicia si su duración estimada cabe antes del fin de la ventana.

    Si ya no cabe (los anteriores tardaron más de lo previsto), no se inicia: queda para el fin de semana,
    que reentrena todo modelo sin versión al día. Un trabajo que ya empezó nunca se corta.
    """
    if job.trigger != tq.TRIGGER_NIGHT:
        return False
    now = calendar_rules.now_local()
    end = calendar_rules.night_window_end(now)
    if end is not None and (now.timestamp() + (job.estimated_seconds or 0)) <= end.timestamp():
        return False
    job.status, job.phase, job.finished_at = J_CANCELLED, tq.POSTPONED_PHASE, utcnow()
    tq.add_log(
        db, job.id,
        f"No se inicia: su duración estimada ({tq.fmt_duration(job.estimated_seconds or 0)}) ya no cabe antes del fin "
        "de la ventana nocturna. Se entrenará en el entrenamiento del fin de semana.",
        "warning",
    )
    return True


def claim_next_job() -> int | None:
    """Siguiente trabajo según queue_order (de noche: el más corto; fin de semana: el más largo)."""
    with SessionLocal() as db:
        while True:
            q = (
                select(TrainingJob)
                .where(TrainingJob.status == J_PENDING)
                .order_by(TrainingJob.queue_order, TrainingJob.id)
                .limit(1)
            )
            if engine.dialect.name == "postgresql":
                q = q.with_for_update(skip_locked=True, of=TrainingJob)
            job = db.scalars(q).first()
            if job is None:
                return None
            if not _postpone_if_late(db, job):
                break
            db.commit()
            tq.publish_job_event(job)
        job.status = J_PREPARING
        job.phase = "Preparando datos"
        job.attempts += 1
        job.started_at = utcnow()
        job.finished_at = None
        job.error = None
        job.cancel_requested = False
        job.heartbeat_at = utcnow()
        db.commit()
        tq.publish_job_event(job)
        return job.id


def _friendly_error(exc: BaseException) -> str:
    name = type(exc).__name__
    text = str(exc)
    if isinstance(exc, MemoryError) or "ResourceExhausted" in name or "OOM" in text or "out of memory" in text.lower():
        return "Memoria insuficiente durante el entrenamiento. Reduce el batch_size del proyecto o aumenta el límite de RAM del worker."
    return f"{name}: {text}"[:2000]


def _set_status(rep: Reporter, status: str, phase: str, percent: float, **extra) -> None:
    rep.update(force=True, status=status, phase=phase, percent=percent, **extra)
    rep.log(phase)


def run_job(job_id: int) -> None:
    rep = Reporter(job_id)
    rep.start_heartbeat()
    work: Path | None = None
    try:
        with SessionLocal() as db:
            job = db.get(TrainingJob, job_id)
            snap = db.get(DatasetSnapshot, job.snapshot_id)
            cfg, base_name = dict(job.config), job.base_model
            project_id, snap_path, snap_hash, snap_id = job.project_id, snap.path, snap.dataset_hash, snap.id
            project_name = job.project.name
        work = data_dir() / "work" / str(job_id)
        _execute(rep, job_id, project_id, project_name, cfg, base_name, snap_path, snap_hash, snap_id, work)
    except Cancelled:
        rep.update(force=True, status=J_CANCELLED, phase="Cancelado", finished_at=utcnow(), eta_seconds=None)
        rep.log("Entrenamiento cancelado.", "warning")
        if work:
            shutil.rmtree(work, ignore_errors=True)
    except BaseException as exc:  # noqa: BLE001
        log.error("Trabajo %s falló: %s", job_id, traceback.format_exc())
        rep.log(traceback.format_exc()[-3500:], "error")
        rep.update(force=True, status=J_ERROR, phase="Error", error=_friendly_error(exc), finished_at=utcnow(), eta_seconds=None)
        if not isinstance(exc, Exception):
            raise
    finally:
        rep.stop_heartbeat()
        try:
            import tensorflow as tf

            tf.keras.backend.clear_session()
        except Exception:  # noqa: BLE001
            pass
        with SessionLocal() as db:
            tq.finalize_runs(db)


def _execute(rep, job_id, project_id, project_name, cfg, base_name, snap_path, snap_hash, snap_id, work: Path):
    if get_modality(cfg.get("modality")).info.tabular:  # scikit-learn: sin TensorFlow ni imágenes
        from . import tabular_runner

        return tabular_runner.execute(rep, job_id, project_id, project_name, cfg, base_name, snap_path, snap_hash, snap_id, work)
    from . import data as data_mod
    from . import export as export_mod
    from . import train as train_mod

    train_mod.configure_tf()
    settings = get_settings()
    size = int(cfg.get("image_size", 224))
    mod = get_modality(cfg.get("modality"))

    # 1. Preparación: división + decodificación progresiva a memmaps en disco
    _set_status(rep, J_PREPARING, "Preparando datos", 1.0, total_epochs=int(cfg.get("epochs", 10)) + int(cfg.get("fine_tune_epochs", 0)))
    labels, items = data_mod.load_manifest(snap_path)
    splits = data_mod.split_items(items, labels)
    marker = work / ".prepared"
    if not marker.exists():
        shutil.rmtree(work, ignore_errors=True)
        total = sum(len(v) for v in splits.values())
        done = 0
        for name, subset in splits.items():
            def prog(i, n, base=done):
                rep.update(percent=1.0 + 7.0 * (base + i) / max(total, 1))
                rep.raise_if_cancelled()  # se llama una vez por bloque de imágenes
            mod.write_split(subset, work, name, prog, cfg, settings)
            done += len(subset)
        marker.write_text("ok")
    rep.log(
        f"Dataset: {len(splits['train'])} entrenamiento / {len(splits['val'])} validación / {len(splits['test'])} prueba ({mod.info.unit}), "
        f"{len(labels)} categorías." + mod.dataset_note(cfg)
    )
    rep.raise_if_cancelled()

    # 2. Entrenamiento
    _set_status(rep, J_TRAINING, "Entrenando", train_mod.P_PREPARE_END)
    model, info = train_mod.train(rep, cfg, base_name, labels, work)
    rep.raise_if_cancelled()

    # 3. Validación en el conjunto de prueba
    _set_status(rep, J_VALIDATING, "Evaluando en el conjunto de prueba", train_mod.P_TRAIN_END, eta_seconds=None)
    metrics = train_mod.evaluate(model, work, labels, int(cfg.get("batch_size", 32)), size)
    metrics.update({k: info[k] for k in ("epochs_run", "best_epoch", "train_images", "val_images", "hints")})
    metrics["val_accuracy"] = max((r.get("val_accuracy", 0) for r in rep.history), default=None)
    rep.log(f"Exactitud en prueba: {metrics['test_accuracy']:.4f} · top-{metrics['top_k']}: {metrics['test_top5_accuracy']:.4f} (loss {metrics['test_loss']:.4f}).")
    for hint in metrics["hints"]:
        rep.log(f"Sugerencia: {hint}", "warning")
    rep.raise_if_cancelled()

    # 4. Exportación y validación de los 5 formatos
    _set_status(rep, J_EXPORTING, "Exportando y validando formatos", train_mod.P_VALIDATE_END, metrics={**(rep.history[-1] if rep.history else {}), **{"test_accuracy": metrics["test_accuracy"]}})
    with SessionLocal() as db:
        version = (db.scalar(select(func.max(ModelVersion.version)).where(ModelVersion.project_id == project_id)) or 0) + 1
    out = data_dir() / "models" / str(project_id) / f"v{version}"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    X = np.load(work / "test_x.npy", mmap_mode="r")
    sel = np.linspace(0, len(X) - 1, num=min(8, len(X))).astype(int)
    samples = np.asarray(X[sel], dtype=np.float32)
    exports = export_mod.export_all(model, out, samples, lambda m, level="info": rep.log(m, level))
    rep.update(force=True, percent=97.0)

    created = utcnow()
    meta = {
        "project": project_name,
        "version": version,
        "created_at": created.isoformat(),
        "base_model": base_name,
        "base_model_label": label_of(base_name),
        "framework": "TensorFlow/Keras",
        "input": {"name": "image", "shape": [1, size, size, 3], "dtype": "float32", "range": [0, 255],
                  "layout": "NHWC", "color": "RGB", "resize": f"{size}x{size} bilineal (sin conservar proporción)",
                  "normalization": "incluida en el modelo"},
        **mod.metadata(cfg),
        "output": {"name": "probs", "dtype": "float32", "activation": "softmax", "classes": len(labels)},
        "labels": labels,
        "dataset": {"hash": snap_hash, "snapshot_id": snap_id,
                    "train": info["train_images"], "val": info["val_images"], "test": metrics["test_images"]},
        "training_config": cfg,
        "metrics": {k: metrics[k] for k in ("test_accuracy", "test_top5_accuracy", "test_loss", "macro_f1", "val_accuracy")},
        "exports": [{k: e[k] for k in ("fmt", "status", "filename", "size_bytes", "validation")} for e in exports],
        "low_confidence_threshold": settings.low_confidence_threshold,
    }
    (out / "labels.json").write_text(json.dumps({"labels": labels}, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps({**metrics, "history": rep.history}, ensure_ascii=False, indent=2), encoding="utf-8")
    mod.export_assets(out)  # inference.py y, si hace falta, el código de características usado al entrenar
    (out / "README.md").write_text(
        README.format(project=project_name, version=version, size=size, created=created.strftime("%Y-%m-%d %H:%M UTC"),
                      base_model=label_of(base_name), labels=", ".join(labels), accuracy=metrics["test_accuracy"]),
        encoding="utf-8",
    )
    if (work / "checkpoints" / "best.weights.h5").exists():
        (out / "checkpoints").mkdir(exist_ok=True)
        shutil.copy2(work / "checkpoints" / "best.weights.h5", out / "checkpoints" / "best.weights.h5")
    export_mod.build_bundle(out)

    all_ok = all(e["status"] == "ok" for e in exports)
    with SessionLocal() as db:
        mv = ModelVersion(
            project_id=project_id, job_id=job_id, snapshot_id=snap_id, version=version, base_model=base_name,
            labels=labels, metrics={k: v for k, v in metrics.items() if k != "history"}, config=cfg,
            dataset_hash=snap_hash, dir_path=str(out), published=any(e["status"] == "ok" for e in exports),
        )
        for e in exports:
            mv.exports.append(ModelExport(**e))
        db.add(mv)
        db.flush()
        job = db.get(TrainingJob, job_id)
        job.model_version_id = mv.id
        tq.add_log(db, job_id, f"Modelo v{version} registrado ({sum(e['status'] == 'ok' for e in exports)}/5 formatos validados).")
        db.commit()
    shutil.rmtree(work, ignore_errors=True)  # datos temporales; los checkpoints quedan en el modelo
    try:  # recortar el ONNX necesita memoria (un ConvNeXt pesa 190 MB): se hace aquí, no en el worker liviano de 1 GB
        similarity.ensure_embedder(str(out), size)
    except Exception as exc:  # noqa: BLE001 - «lo más parecido» nunca debe romper un entrenamiento
        log.warning("No se pudo preparar el vector de parecidos del modelo %s: %s", mv.id, exc)
    similarity.kick_build(mv.id)  # deja listo «lo más parecido» (en el worker liviano) antes de que lo pidan
    if all_ok:
        rep.update(force=True, status=J_COMPLETED, phase="Completado", percent=100.0, finished_at=utcnow(), eta_seconds=0,
                   metrics={"test_accuracy": metrics["test_accuracy"], "test_loss": metrics["test_loss"],
                            **{k: v for k, v in (rep.history[-1] if rep.history else {}).items() if k != "epoch"}},
                   model_version_id=mv.id)
    else:
        failed = ", ".join(e["fmt"] for e in exports if e["status"] != "ok")
        rep.update(force=True, status=J_ERROR, phase="Exportación incompleta", percent=100.0, finished_at=utcnow(), eta_seconds=0,
                   error=f"Formatos que no pasaron la validación: {failed}. El modelo v{version} quedó disponible en los formatos válidos.",
                   model_version_id=mv.id)
