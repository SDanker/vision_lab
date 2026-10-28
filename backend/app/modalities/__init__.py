"""Registro de modalidades. Para añadir un tipo de dato: subclase de `Modality` + una línea en REGISTRY."""
from dataclasses import asdict

from .audio import AudioModality
from .base import InvalidInput, Modality, ModalityInfo
from .image import ImageModality
from .tabular import TabularModality

REGISTRY: dict[str, Modality] = {m.info.id: m for m in (ImageModality(), AudioModality(), TabularModality())}
DEFAULT = "image"

# extensión -> tipo MIME de todos los archivos que se pueden guardar en un proyecto
MEDIA: dict[str, str] = {ext: mime for m in REGISTRY.values() for ext, mime in m.media.items()}


def get_modality(modality_id: str | None) -> Modality:
    mid = modality_id or DEFAULT
    try:
        return REGISTRY[mid]
    except KeyError:
        raise ValueError(f"Tipo de dato desconocido: {mid}") from None


def describe_all() -> list[dict]:
    return [asdict(m.info) for m in REGISTRY.values()]


__all__ = ["DEFAULT", "InvalidInput", "MEDIA", "Modality", "ModalityInfo", "REGISTRY", "describe_all", "get_modality"]
