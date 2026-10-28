"""«Lo más parecido»: un vector de características por cada imagen/audio del proyecto y búsqueda por coseno.

El vector es la salida del pooling global del modelo, justo la entrada de su capa de clasificación. No hace falta
reentrenar ni cambiar lo exportado: se recorta `model.onnx` en ese tensor y queda un `embedder.onnx` liviano
(sin TensorFlow) que además funciona con modelos ya entrenados. Se guarda junto al modelo:

  embedder.onnx   el modelo recortado
  similar.npz     ids de `Image` + vectores normalizados (float16); se actualiza de forma incremental
"""
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import SessionLocal

log = logging.getLogger(__name__)

INDEX_FILE, EMBEDDER_FILE = "similar.npz", "embedder.onnx"
BUILD_TTL = 180  # segundos que vale el estado «construyendo» si el worker no responde


class EmbedderUnavailable(Exception):
    pass


# --------------------------------------------------------------------------- embedder
def ensure_embedder(model_dir: str, size: int) -> Path:
    """Devuelve `embedder.onnx`, creándolo (y validándolo) la primera vez.

    Se recorta el mismo grafo en memoria, sin duplicar los pesos: un ConvNeXt Small (190 MB) hecho de otra forma
    llegaba a tres copias a la vez y no cabía en el worker liviano (1 GB).
    """
    import onnx
    from onnx import TensorProto, helper

    d = Path(model_dir)
    src, target = d / "model.onnx", d / EMBEDDER_FILE
    if not src.is_file():
        raise EmbedderUnavailable("El modelo no tiene un ONNX válido.")
    if target.is_file() and target.stat().st_mtime_ns >= src.stat().st_mtime_ns:
        return target

    model = onnx.load(str(src))
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    head = next((n for n in reversed(g.node) if n.op_type in ("MatMul", "Gemm")), None)
    if head is None:
        raise EmbedderUnavailable("No se encontró la capa de clasificación en el ONNX.")
    emb_name = head.input[0]  # el vector que entra a la capa final
    w = inits.get(head.input[1])  # su matriz de pesos: sirve para comprobar la dimensión del vector
    head_dim = None
    if w is not None and len(w.dims) == 2:
        transposed = head.op_type == "Gemm" and any(a.name == "transB" and a.i == 1 for a in head.attribute)
        head_dim = int(w.dims[1] if transposed else w.dims[0])
    inputs = [i for i in g.input if i.name not in inits]
    if len(inputs) != 1:
        raise EmbedderUnavailable("El ONNX tiene un número inesperado de entradas.")

    needed, keep = {emb_name}, set()
    for idx in range(len(g.node) - 1, -1, -1):  # sólo los nodos de los que depende ese tensor
        n = g.node[idx]
        if any(o in needed for o in n.output):
            keep.add(idx)
            needed.update(i for i in n.input if i)
    for idx in range(len(g.node) - 1, -1, -1):  # se borra en el mismo objeto: los pesos no se copian
        if idx not in keep:
            del g.node[idx]
    for idx in range(len(g.initializer) - 1, -1, -1):
        if g.initializer[idx].name not in needed:
            del g.initializer[idx]
    del g.output[:]
    g.output.append(helper.make_tensor_value_info(emb_name, TensorProto.FLOAT, [None, None]))
    del g.value_info[:]
    onnx.checker.check_model(model)

    tmp = target.with_name("embedder.tmp.onnx")
    onnx.save(model, str(tmp))
    del model, g, inits, w  # libera la copia en memoria antes de abrirlo con ONNX Runtime
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])
        x = np.random.default_rng(0).uniform(0, 255, (2, size, size, 3)).astype(np.float32)
        e = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        if e.ndim != 2 or e.shape[0] != 2 or not np.isfinite(e).all():
            raise EmbedderUnavailable("La salida del modelo recortado no es un vector por muestra.")
        if head_dim is not None and head_dim != e.shape[1]:  # debe coincidir con la capa de clasificación
            raise EmbedderUnavailable("El vector no coincide con la capa de clasificación.")
        del sess
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, target)
    return target


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def embed_rows(model_dir: str, x: np.ndarray, size: int) -> np.ndarray:
    """Un vector normalizado por fila de `x` (N, S, S, 3), forma (N, D)."""
    from . import inference

    ensure_embedder(model_dir, size)
    sess = inference._session(Path(model_dir), "embedder")
    out = np.asarray(sess.run(None, {sess.get_inputs()[0].name: x})[0], dtype=np.float32).reshape(len(x), -1)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def embed_mean(model_dir: str, x: np.ndarray, size: int) -> np.ndarray:
    """Vector normalizado de un archivo: promedio de las filas de `x` (un cuadro por imagen, varios clips por audio)."""
    return _normalize(embed_rows(model_dir, x, size).mean(axis=0))


# ------------------------------------------------------------------------------ índice
_loaded: "OrderedDict[tuple, tuple[np.ndarray, np.ndarray]]" = OrderedDict()
_loaded_lock = threading.Lock()


def load_index(model_dir: str) -> tuple[np.ndarray, np.ndarray] | None:
    p = Path(model_dir) / INDEX_FILE
    try:
        key = (str(p), p.stat().st_mtime_ns)
    except FileNotFoundError:
        return None
    with _loaded_lock:
        if key in _loaded:
            _loaded.move_to_end(key)
            return _loaded[key]
    try:
        with np.load(p) as z:
            data = (z["ids"].astype(np.int64), z["emb"])
    except Exception:  # noqa: BLE001 - un índice dañado se reconstruye
        return None
    with _loaded_lock:
        _loaded[key] = data
        while len(_loaded) > 2:  # 50 000 archivos × 1 280 dimensiones ≈ 120 MB por índice
            _loaded.popitem(last=False)
    return data


def index_state(db: Session, model) -> str:
    """ready: al día · stale: hay archivos nuevos o borrados · missing: aún no existe."""
    from ..models import Image

    idx = load_index(model.dir_path)
    if idx is None:
        return "missing"
    current = set(db.scalars(select(Image.id).where(Image.project_id == model.project_id)))
    return "ready" if current == set(int(i) for i in idx[0]) else "stale"


CHECKPOINT_EVERY = 2000  # un índice grande puede tardar minutos: se guarda el avance por si el worker se reinicia


def _save_index(model_dir: str, ordered_ids: list[int], have: dict) -> None:
    ids = [i for i in ordered_ids if i in have]
    dim = len(next(iter(have.values()))) if have else 1
    emb = np.stack([have[i] for i in ids]) if ids else np.zeros((0, dim), np.float16)
    final = Path(model_dir) / INDEX_FILE
    tmp = final.with_name("similar.tmp.npz")
    np.savez(tmp, ids=np.array(ids, dtype=np.int64), emb=emb)
    os.replace(tmp, final)


def build_index(model_id: int) -> int:
    """Calcula los vectores que faltan (y descarta los de archivos borrados). Devuelve cuántos calculó.

    Es reanudable: lo ya guardado no se recalcula, así que si el worker se reinicia a mitad basta volver a pedirlo.
    """
    from ..models import Image, ModelVersion
    from ..modalities import get_modality
    from ..storage import image_path

    with SessionLocal() as db:
        m = db.get(ModelVersion, model_id)
        if m is None:
            return 0
        cfg, model_dir, pid = dict(m.config), m.dir_path, m.project_id
        rows = [(i.id, str(image_path(i))) for i in db.scalars(select(Image).where(Image.project_id == pid).order_by(Image.id))]
    mod = get_modality(cfg.get("modality"))
    if mod.info.tabular:  # una tabla no tiene «archivos parecidos» que mostrar
        return 0
    ordered = [i for i, _ in rows]
    try:
        ensure_embedder(model_dir, int(cfg.get("image_size", 224)))
        old = load_index(model_dir)
        have = {int(i): e for i, e in zip(old[0], old[1])} if old is not None else {}
        todo = [(i, p) for i, p in rows if i not in have]
        _set_status(model_id, "building", done=0, total=len(todo))
        for n, (iid, vec) in enumerate(mod.embed_files(model_dir, todo, cfg), 1):
            if vec is not None:
                have[iid] = vec.astype(np.float16)
            if n % 100 == 0:
                _set_status(model_id, "building", done=n, total=len(todo))
                refresh_build_lock()
            if n % CHECKPOINT_EVERY == 0:
                _save_index(model_dir, ordered, have)
        _save_index(model_dir, ordered, have)
        _set_status(model_id, "idle")
        return len(todo)
    except Exception as exc:  # noqa: BLE001
        log.error("Falló el índice de parecidos del modelo %s: %s", model_id, exc)
        _set_status(model_id, "error", error=str(exc)[:300])
        raise


def search(db: Session, model, query: np.ndarray, k: int) -> list[dict]:
    """Los `k` archivos del proyecto más parecidos al vector (ya normalizado) `query`."""
    from ..models import Category, Image
    from ..modalities import get_modality

    idx = load_index(model.dir_path)
    if idx is None or len(idx[0]) == 0:
        return []
    ids, emb = idx
    sims = np.empty(len(emb), dtype=np.float32)
    for s in range(0, len(emb), 8192):  # por bloques: un proyecto de 50 000 archivos no debe disparar la memoria
        sims[s : s + 8192] = emb[s : s + 8192].astype(np.float32) @ query
    want = min(k * 3 + 3, len(sims))  # margen por si algún archivo se borró desde el último índice
    top = np.argpartition(-sims, want - 1)[:want] if want < len(sims) else np.arange(len(sims))
    order = top[np.argsort(-sims[top])]
    rows = {i.id: i for i in db.scalars(select(Image).where(Image.id.in_([int(ids[j]) for j in order])))}
    cats = {c.id: c.name for c in db.scalars(select(Category).where(Category.project_id == model.project_id))}
    kind = get_modality(model.config.get("modality")).info.preview
    out = []
    for j in order:
        img = rows.get(int(ids[j]))
        if img is None or img.project_id != model.project_id:
            continue
        out.append({
            "image_id": img.id, "name": img.original_name, "category": cats.get(img.category_id),
            "similarity": round(float(max(min(sims[j], 1.0), -1.0)), 4), "kind": kind, "duration_s": img.duration_s,
        })
        if len(out) >= k:
            break
    return out


# ------------------------------------------------------------------------------ estado
_mem: dict[str, tuple[float, dict]] = {}


def _redis():
    url = get_settings().redis_url
    if not url:
        return None
    import redis

    return redis.Redis.from_url(url, socket_connect_timeout=1, socket_timeout=1)


def _set_status(model_id: int, state: str, **extra) -> None:
    key = f"sim:{model_id}"
    payload = {"state": state, **extra}
    ttl = BUILD_TTL if state == "building" else 600  # «building» caduca solo si el worker se cae
    try:
        r = _redis()
        if r is not None:
            if state == "idle":
                r.delete(key)
            else:
                r.set(key, json.dumps(payload), ex=ttl)
            return
    except Exception:  # noqa: BLE001
        pass
    if state == "idle":
        _mem.pop(key, None)
    else:
        _mem[key] = (time.time() + ttl, payload)


def get_status(model_id: int) -> dict:
    key = f"sim:{model_id}"
    try:
        r = _redis()
        if r is not None:
            raw = r.get(key)
            return json.loads(raw) if raw else {"state": "idle"}
    except Exception:  # noqa: BLE001
        pass
    expires, payload = _mem.get(key, (0.0, None))
    return payload if payload and expires > time.time() else {"state": "idle"}


def kick_build(model_id: int) -> None:
    """Pide el cálculo en segundo plano (worker liviano). Si ya hay uno en curso no duplica."""
    if get_status(model_id).get("state") == "building":
        return
    _set_status(model_id, "building", done=0, total=0)
    try:
        from ..celery_app import celery_app

        celery_app.send_task("vision.build_similarity", args=[model_id], queue="default")
    except Exception as exc:  # noqa: BLE001
        log.warning("No se pudo encolar el índice de parecidos: %s", exc)


# ------------------------------------------------------------- un índice a la vez
# Calcular un índice grande usa mucha CPU y memoria: los trabajos se turnan con un candado compartido entre workers.
LOCK_KEY, LOCK_TTL = "sim:build-lock", 900  # se renueva mientras avanza; si el worker muere, caduca solo
_local_lock = threading.Lock()  # sin Redis (pruebas)


def acquire_build_lock() -> bool:
    try:
        r = _redis()
        if r is not None:
            return bool(r.set(LOCK_KEY, "1", nx=True, ex=LOCK_TTL))
    except Exception:  # noqa: BLE001
        pass
    return _local_lock.acquire(blocking=False)


def refresh_build_lock() -> None:
    try:
        r = _redis()
        if r is not None:
            r.expire(LOCK_KEY, LOCK_TTL)
    except Exception:  # noqa: BLE001
        pass


def release_build_lock() -> None:
    try:
        r = _redis()
        if r is not None:
            r.delete(LOCK_KEY)
            return
    except Exception:  # noqa: BLE001
        pass
    try:
        _local_lock.release()
    except RuntimeError:
        pass


def mark_queued(model_id: int) -> None:
    """Mientras espera su turno, la interfaz sigue viendo «preparando»."""
    _set_status(model_id, "building", done=0, total=0)
