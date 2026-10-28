"""Datos tabulares de un proyecto: perfil de sus CSV, columnas elegidas y dataset de entrenamiento.

Los proyectos tabulares guardan cada CSV como una fila de `Image` (en una única categoría interna «Datos»): así
reutilizan la carga, los duplicados, la cuota y los bloqueos del calendario. Las categorías a predecir salen de los
valores de la columna objetivo, no de carpetas.
"""
import hashlib
import json
import shutil
from functools import lru_cache
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..ml import tabular as tab
from ..ml import tabular_features as feat
from ..models import Category, DatasetSnapshot, Image, Project
from ..storage import data_dir, image_path

DATA_CATEGORY = "Datos"  # categoría interna que agrupa los CSV del proyecto (no se muestra)


def is_tabular(project: Project) -> bool:
    return (project.modality or "image") == "tabular"


def project_files(db: Session, project: Project) -> list[Image]:
    return list(db.scalars(select(Image).where(Image.project_id == project.id).order_by(Image.id)))


def ensure_data_category(db: Session, project: Project) -> Category:
    cat = db.scalar(select(Category).where(Category.project_id == project.id).order_by(Category.id).limit(1))
    if cat is None:
        cat = Category(project_id=project.id, name=DATA_CATEGORY)
        db.add(cat)
        db.flush()
    return cat


def load_frame(paths: list[Path]):
    """Todos los CSV como una sola tabla de textos: columnas = unión de las de cada archivo (lo que falta queda vacío)."""
    import pandas as pd

    s = get_settings()
    frames, total = [], 0
    for path in paths:
        df = tab.read_table(path.read_bytes(), s.max_tabular_rows, s.max_tabular_columns)
        total += len(df)
        if total > s.max_tabular_rows:
            raise tab.InvalidTable(f"Entre todos los archivos hay más de {s.max_tabular_rows:,} filas.".replace(",", "."))
        frames.append(df)
    if not frames:
        raise tab.InvalidTable("Sube al menos un archivo CSV.")
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True, sort=False).fillna("")


@lru_cache(maxsize=8)
def _profile(key: tuple[tuple[str, str], ...]) -> dict:
    """Perfil de los CSV, en caché por contenido (sha256): abrir la pestaña no vuelve a leer la tabla."""
    return tab.profile(load_frame([Path(p) for _, p in key]))


def profile_of(files: list[Image]) -> dict:
    return _profile(tuple((f.sha256, str(image_path(f))) for f in files))


def selection(project: Project, columns: list[dict]) -> tuple[str | None, list[str], list[str]]:
    """(columna objetivo, columnas de entrada efectivas, columnas sugeridas) según lo guardado y lo que hay en la tabla."""
    cfg = project.config or {}
    names = {c["name"] for c in columns}
    target = cfg.get("tabular_target")
    target = target if target in names else None
    suggested = tab.suggest_features(columns, target)
    chosen = cfg.get("tabular_features")
    features = [n for n in chosen if n in names and n != target] if chosen else suggested
    return target, features, suggested


def summary(db: Session, project: Project) -> dict:
    """Todo lo que la pestaña de datos necesita: columnas, vista previa, selección y por qué (no) se puede entrenar."""
    s = get_settings()
    files = project_files(db, project)
    out = {
        "n_files": len(files), "n_rows": 0, "n_columns": 0, "columns": [], "preview": None, "target": None, "features": [],
        "suggested_features": [], "classes": None, "min_per_class": s.min_images_per_class, "max_classes": tab.MAX_CLASSES,
        "problems": [], "ready": False,
    }
    if not files:
        out["problems"].append("Sube al menos un archivo CSV.")
        return out
    try:
        prof = profile_of(files)
    except (tab.InvalidTable, OSError) as exc:
        out["problems"].append(f"No se pudieron leer los archivos: {exc}")
        return out
    columns = prof["columns"]
    target, features, suggested = selection(project, columns)
    out.update(n_rows=prof["n_rows"], n_columns=len(columns), columns=columns, preview=prof["preview"], target=target,
               features=features, suggested_features=suggested)
    if target is None:
        out["problems"].append("Elige la columna que el modelo debe predecir.")
        return out
    col = next(c for c in columns if c["name"] == target)
    if col["values"] is None:
        out["problems"].append(
            f"«{target}» tiene {col['unique']} valores distintos: para clasificar necesita entre 2 y {tab.MAX_CLASSES}. "
            "Elige otra columna."
        )
        return out
    classes = [{"label": v["value"], "count": v["count"], "used": v["count"] >= s.min_images_per_class} for v in col["values"]]
    out["classes"] = classes
    if sum(c["used"] for c in classes) < 2:
        out["problems"].append(
            f"Para entrenar se necesitan al menos 2 valores de «{target}» con {s.min_images_per_class} o más filas cada uno."
        )
    if not features:
        out["problems"].append("Elige al menos una columna de entrada.")
    out["ready"] = not out["problems"]
    return out


def _plan(db: Session, project: Project) -> tuple[dict | None, str]:
    sm = summary(db, project)
    if sm["problems"]:
        return None, " ".join(sm["problems"])
    used = [c for c in sm["classes"] if c["used"]]
    counts = {c["label"]: c["count"] for c in sorted(used, key=lambda c: c["label"])}
    excluded = {c["label"]: c["count"] for c in sm["classes"] if not c["used"]}
    digest = hashlib.sha256()
    for line in sorted(f.sha256 for f in project_files(db, project)):
        digest.update(line.encode())
    digest.update(json.dumps([sm["target"], sorted(sm["features"]), list(counts)], ensure_ascii=False).encode())
    return {
        "target": sm["target"], "features": sm["features"], "counts": counts, "excluded": excluded,
        "n_rows": sum(counts.values()), "digest": digest.hexdigest(),
    }, ""


def dataset(db: Session, project: Project):
    """Misma forma que `training_queue._dataset`: (clases, filas por clase, excluidas, nº de filas, hash) o (None, motivo)."""
    plan, reason = _plan(db, project)
    if plan is None:
        return None, reason
    return (list(plan["counts"]), plan["counts"], plan["excluded"], plan["n_rows"], plan["digest"]), ""


def create_snapshot(db: Session, project: Project, run_id: int | None) -> tuple[DatasetSnapshot | None, str]:
    """Congela las filas de las clases que cumplen el mínimo, con sólo las columnas elegidas (`data.csv`)."""
    plan, reason = _plan(db, project)
    if plan is None:
        return None, reason
    try:
        df = load_frame([image_path(f) for f in project_files(db, project)])
    except (tab.InvalidTable, OSError) as exc:
        return None, f"No se pudieron leer los archivos: {exc}"
    target, features, labels = plan["target"], plan["features"], list(plan["counts"])
    sub = df[[target, *features]].copy()
    sub[target] = sub[target].map(feat.normalize_label)
    sub = sub[sub[target].isin(labels)]
    snap = DatasetSnapshot(
        project_id=project.id, run_id=run_id, dataset_hash=plan["digest"], class_counts=plan["counts"],
        excluded=plan["excluded"] or None, total_images=len(sub), path="",
    )
    db.add(snap)
    db.flush()
    root = data_dir() / "snapshots" / str(snap.id)
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    sub.to_csv(root / "data.csv", index=False, lineterminator="\n")
    (root / "manifest.json").write_text(
        json.dumps({"labels": labels, "dataset_hash": plan["digest"], "items": [],
                    "tabular": {"target": target, "features": features, "rows": len(sub)}}, ensure_ascii=False),
        encoding="utf-8",
    )
    snap.path = str(root)
    return snap, ""
