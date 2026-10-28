"""Preprocesado compartido por entrenamiento e inferencia (mismo resultado en ambos)."""
import io

import numpy as np
from PIL import Image, ImageOps

IMG_SIZE = 224  # valor por defecto; cada proyecto puede elegir otro (TrainingConfig.image_size)


def to_rgb(im: Image.Image) -> Image.Image:
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        return bg
    return im.convert("RGB")


def load_array(source, size: int = IMG_SIZE) -> np.ndarray:
    """Devuelve uint8 (size, size, 3). `source` es una ruta o bytes."""
    opener = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
    with Image.open(opener) as im:
        im = to_rgb(im).resize((size, size), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def to_model_input(arr: np.ndarray) -> np.ndarray:
    """Los modelos exportados incluyen su normalización: entrada float32 0-255 NHWC."""
    return arr.astype(np.float32)[None, ...]
