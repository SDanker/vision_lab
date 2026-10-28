"""Imágenes (JPG, PNG, WEBP): la modalidad original de la plataforma."""
from pathlib import Path

from ..storage import InvalidImage, save_image_files, validate_image_bytes
from .base import InvalidInput, Modality, ModalityInfo


class ImageModality(Modality):
    info = ModalityInfo(
        id="image", label="Imágenes", icon="🖼", description="Clasifica fotos (JPG, PNG, WEBP).",
        unit="imágenes", unit_one="imagen", extensions=("jpg", "jpeg", "png", "webp"), accept="image/jpeg,image/png,image/webp", upload_help="",
        preview="image", windowed=False,
        resolution_label="Resolución de entrada",
        resolution_hint="Más resolución distingue mejor detalles finos (p. ej. tipos de avión), pero entrena más lento y usa más memoria.",
    )
    media = {"jpg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
    augmentation = "image"

    def max_upload_mb(self, settings) -> int:
        return settings.max_upload_mb

    def validate(self, data: bytes, filename: str):
        return validate_image_bytes(data)

    def save(self, project_id: int, v) -> str:
        return save_image_files(project_id, v)

    def item_fields(self, v) -> dict:
        return {"width": v.width, "height": v.height, "duration_s": None}

    def job_config(self, cfg: dict) -> dict:
        cfg = dict(cfg)
        cfg.pop("clip_seconds", None)  # no aplica a imágenes (y así no cambia la comparación «sin cambios»)
        return cfg

    def write_split(self, items, out_dir: Path, name: str, progress, cfg: dict, settings) -> int:
        from ..ml import data
        from ..ml.preprocess import IMG_SIZE

        return data.write_split(items, out_dir, name, progress, size=int(cfg.get("image_size", IMG_SIZE)))

    def export_assets(self, out_dir: Path) -> None:
        from ..ml.templates import INFERENCE_PY

        (out_dir / "inference.py").write_text(INFERENCE_PY, encoding="utf-8")

    def predict(self, model_dir, labels, data, filename, ok, cfg) -> dict:
        from ..services import inference

        try:
            validate_image_bytes(data)
        except InvalidImage as exc:
            raise InvalidInput("INVALID_IMAGE", str(exc)) from None
        return inference.predict(
            model_dir, labels, data, ok.get("onnx", False), ok.get("keras", False), int(cfg.get("image_size", 224))
        )

    def embed_file(self, model_dir, source, cfg):
        from ..ml.preprocess import load_array, to_model_input
        from ..services import similarity

        size = int(cfg.get("image_size", 224))
        return similarity.embed_mean(model_dir, to_model_input(load_array(source, size)), size)

    def embed_files(self, model_dir, items, cfg):
        """Decodifica en paralelo (Pillow libera el GIL) y calcula los vectores por lotes."""
        import logging
        import os
        from concurrent.futures import ThreadPoolExecutor

        import numpy as np

        from ..ml.preprocess import load_array
        from ..services import similarity

        size = int(cfg.get("image_size", 224))
        batch = max(4, int(8 * (224 / size) ** 2))  # lotes chicos: el worker liviano tiene 1 GB; a mayor resolución, más chicos

        def load(item):
            try:
                return load_array(item[1], size)
            except Exception as exc:  # noqa: BLE001
                logging.getLogger(__name__).warning("No se pudo leer %s: %s", item[1], exc)
                return None

        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
            for start in range(0, len(items), batch):
                chunk = items[start : start + batch]
                arrays = list(pool.map(load, chunk))
                good = [(it[0], a) for it, a in zip(chunk, arrays) if a is not None]
                vecs = similarity.embed_rows(model_dir, np.stack([a for _, a in good]).astype(np.float32), size) if good else []
                got = {iid: v for (iid, _), v in zip(good, vecs)}
                for it in chunk:
                    yield it[0], got.get(it[0])
