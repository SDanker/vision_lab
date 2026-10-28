"""Contrato de un tipo de dato (modalidad) de la plataforma.

Todo lo que cambia de un tipo de dato a otro (cómo se valida y guarda un archivo, cómo se convierte en muestras
de entrenamiento, cómo se predice y qué se exporta junto al modelo) vive en una subclase de `Modality`. El resto
del código pregunta al registro (`get_modality`) en lugar de ramificar con `if audio`.

Las importaciones pesadas (TensorFlow, ffmpeg…) se hacen dentro de los métodos para que importar el registro
sea barato y no cree ciclos con `storage` o `ml`.
"""
from dataclasses import dataclass
from pathlib import Path


class InvalidInput(Exception):
    """Archivo de prueba no válido para el modelo; el router lo traduce a 422 con este código."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


@dataclass(frozen=True)
class ModalityInfo:
    """Lo que necesita la interfaz para presentar el tipo de dato (se sirve en GET /api/modalities)."""

    id: str
    label: str
    icon: str
    description: str
    unit: str  # plural: «imágenes», «audios»
    unit_one: str  # singular
    extensions: tuple[str, ...]  # extensiones aceptadas (para filtrar al arrastrar carpetas)
    accept: str  # atributo `accept` del selector de archivos
    upload_help: str  # nota bajo la zona de carga
    preview: str  # cómo se muestra un archivo: image | audio | table
    windowed: bool  # el archivo se corta en clips/ventanas de duración configurable
    tabular: bool = False  # filas de una tabla: no se convierten en imágenes ni usan las redes de visión
    window_label: str = ""
    window_hint: str = ""
    resolution_label: str = "Resolución de entrada"
    resolution_hint: str = ""
    test_hint: str = ""  # frase extra bajo «Probar el modelo»


class Modality:
    info: ModalityInfo
    media: dict[str, str] = {}  # extensión -> tipo MIME, para servir el archivo original
    augmentation = "image"  # "image" (voltear/rotar…) o "spec" (SpecAugment sobre espectrogramas)
    sample_unit: str | None = None  # cómo se llaman las muestras de entrenamiento si no son los archivos («filas»)

    # ----------------------------------------------------------------- carga de archivos
    def max_upload_mb(self, settings) -> int:
        raise NotImplementedError

    def validate(self, data: bytes, filename: str):
        """Valida de verdad el contenido (no sólo la extensión). Lanza `storage.InvalidImage`."""
        raise NotImplementedError

    def save(self, project_id: int, v) -> str:
        """Guarda el original y su miniatura. Devuelve el nombre almacenado."""
        raise NotImplementedError

    def item_fields(self, v) -> dict:
        """Columnas propias del tipo en la fila `Image`: width, height, duration_s."""
        return {"width": 0, "height": 0, "duration_s": None}

    # ------------------------------------------------------------------- entrenamiento
    def job_config(self, cfg: dict) -> dict:
        """Ajusta la configuración efectiva de un trabajo (qué claves aplican a este tipo)."""
        return cfg

    def write_split(self, items: list[dict], out_dir: Path, name: str, progress, cfg: dict, settings) -> int:
        """Convierte los archivos de una partición en muestras N×N×3 (memmap) y devuelve cuántas son."""
        raise NotImplementedError

    def dataset_note(self, cfg: dict) -> str:
        return ""

    def metadata(self, cfg: dict) -> dict:
        """Bloque específico del tipo para metadata.json."""
        return {"modality": self.info.id}

    def export_assets(self, out_dir: Path) -> None:
        """Escribe junto al modelo exportado el código para reproducir la entrada (inference.py y demás)."""
        raise NotImplementedError

    # ------------------------------------------------------------------------ uso
    def predict(self, model_dir: str, labels: list[str], data: bytes, filename: str, ok: dict, cfg: dict) -> dict:
        """Clasifica un archivo de prueba. `ok` indica qué formatos exportados son válidos (onnx/keras)."""
        raise NotImplementedError

    def embed_file(self, model_dir: str, source, cfg: dict):
        """Vector de características normalizado de un archivo (ruta o bytes), para «lo más parecido»."""
        raise NotImplementedError

    def embed_files(self, model_dir: str, items: list, cfg: dict):
        """Genera `(id, vector | None)` por cada `(id, ruta)`. Una modalidad puede agrupar el trabajo para ir más rápido."""
        import logging

        for iid, path in items:
            try:
                yield iid, self.embed_file(model_dir, path, cfg)
            except Exception as exc:  # noqa: BLE001 - un archivo dañado no detiene el resto
                logging.getLogger(__name__).warning("No se pudo calcular el vector de %s: %s", iid, exc)
                yield iid, None
