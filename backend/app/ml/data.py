"""Datos de entrenamiento: división estratificada y lectura progresiva (memmap en disco)."""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .preprocess import IMG_SIZE, load_array


def load_manifest(snapshot_dir: str) -> tuple[list[str], list[dict]]:
    m = json.loads((Path(snapshot_dir) / "manifest.json").read_text(encoding="utf-8"))
    items = [{**it, "path": str(Path(snapshot_dir) / it["file"])} for it in m["items"]]
    return m["labels"], items


def split_items(items: list[dict], labels: list[str], seed: int = 42, val_frac=0.15, test_frac=0.15):
    """División estratificada train/val/test, determinista por semilla."""
    rng = np.random.default_rng(seed)
    out = {"train": [], "val": [], "test": []}
    for idx, label in enumerate(labels):
        group = [dict(it, y=idx) for it in items if it["label"] == label]
        order = rng.permutation(len(group))
        group = [group[i] for i in order]
        n = len(group)
        n_test = max(1, round(n * test_frac))
        n_val = max(1, round(n * val_frac))
        n_train = n - n_val - n_test
        if n_train < 1:
            raise ValueError(f"La categoría '{label}' tiene muy pocas imágenes ({n}) para dividir el dataset.")
        out["train"] += group[:n_train]
        out["val"] += group[n_train : n_train + n_val]
        out["test"] += group[n_train + n_val :]
    return out


def write_split(items: list[dict], out_dir: Path, name: str, progress=None, size: int = IMG_SIZE, workers: int | None = None) -> int:
    """Decodifica y redimensiona cada imagen una sola vez escribiendo en un memmap en disco.
    Se decodifica en paralelo (Pillow libera el GIL) en bloques acotados: en memoria sólo
    hay un bloque de imágenes a la vez, nunca el dataset completo."""
    out_dir.mkdir(parents=True, exist_ok=True)
    x_path, y_path = out_dir / f"{name}_x.npy", out_dir / f"{name}_y.npy"
    X = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.uint8, shape=(len(items), size, size, 3))
    y = np.array([it["y"] for it in items], dtype=np.int32)
    workers = workers or min(8, os.cpu_count() or 1)
    block = 64 * workers
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(items), block):
            chunk = items[start : start + block]
            for j, arr in enumerate(pool.map(lambda it: load_array(it["path"], size), chunk)):
                X[start + j] = arr
            if progress:
                progress(start + len(chunk), len(items))
    X.flush()
    del X
    np.save(y_path, y)
    return len(items)



def write_split_audio(
    items: list[dict], out_dir: Path, name: str, progress=None, size: int = IMG_SIZE,
    clip_seconds: float = 5.0, max_clips: int = 20, workers: int | None = None,
) -> int:
    """Audio: cada archivo se corta en clips y cada clip se convierte en un espectrograma-imagen.
    La división train/val/test ya se hizo por ARCHIVO, así que los clips de un mismo audio
    nunca quedan repartidos entre entrenamiento y prueba."""
    from .audio_features import load_audio, n_clips_for, spectrogram_image, split_clips

    out_dir.mkdir(parents=True, exist_ok=True)
    counts = [n_clips_for(float(it.get("duration") or clip_seconds), clip_seconds, max_clips) for it in items]
    total = int(sum(counts))
    X = np.lib.format.open_memmap(out_dir / f"{name}_x.npy", mode="w+", dtype=np.uint8, shape=(total, size, size, 3))
    y = np.repeat(np.array([it["y"] for it in items], dtype=np.int32), counts)

    def featurize(job):
        it, n = job
        signal = load_audio(it["path"], max_seconds=n * clip_seconds)
        return [spectrogram_image(c, size) for c in split_clips(signal, clip_seconds, n)]

    workers = workers or min(8, os.cpu_count() or 1)
    pos = 0
    jobs = list(zip(items, counts))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(jobs), 16 * workers):
            chunk = jobs[start : start + 16 * workers]
            for imgs in pool.map(featurize, chunk):
                for img in imgs:
                    X[pos] = img
                    pos += 1
            if progress:
                progress(start + len(chunk), len(items))
    X.flush()
    del X
    np.save(out_dir / f"{name}_y.npy", y)
    return total

def make_dataset(out_dir: Path, name: str, batch_size: int, shuffle: bool, augment=None, size: int = IMG_SIZE):
    import tensorflow as tf

    X = np.load(out_dir / f"{name}_x.npy", mmap_mode="r")
    y = np.load(out_dir / f"{name}_y.npy")
    n = len(y)

    def gen():
        idx = np.arange(n)
        if shuffle:
            np.random.shuffle(idx)
        for start in range(0, n, batch_size):
            sel = np.sort(idx[start : start + batch_size])
            yield X[sel].astype(np.float32), y[sel]

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec((None, size, size, 3), tf.float32),
            tf.TensorSpec((None,), tf.int32),
        ),
    )
    if augment is not None:
        ds = ds.map(lambda a, b: (augment(a, training=True), b), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(2), n, int(np.ceil(n / batch_size))
