import hashlib
import io
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .config import get_settings
from .models import Image as ImageRow
from .models import ModelExport, ModelVersion, Project, User

ALLOWED = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}
THUMB_SIZE = 256


def data_dir() -> Path:
    return Path(get_settings().data_dir)


def image_dir(project_id: int) -> Path:
    return data_dir() / "images" / str(project_id)


def thumb_dir(project_id: int) -> Path:
    return data_dir() / "thumbs" / str(project_id)


def image_path(img: ImageRow) -> Path:
    return image_dir(img.project_id) / img.stored_name


def thumb_path(img: ImageRow) -> Path:
    return thumb_dir(img.project_id) / (Path(img.stored_name).stem + ".jpg")


@dataclass
class ValidatedImage:
    data: bytes
    sha256: str
    fmt: str
    ext: str
    width: int
    height: int


class InvalidImage(Exception):
    pass


# Firmas de ejecutables y scripts que no tienen cabida dentro de una imagen o un audio. Son lo bastante largas
# como para que no aparezcan por azar en datos comprimidos legítimos.
_PAYLOAD_RE = re.compile(
    rb"This program cannot be run in DOS mode"  # ejecutable de Windows (PE)
    rb"|\x7fELF[\x01\x02][\x01\x02]\x01"  # ejecutable de Linux (ELF)
    # «<%@» a secas son 3 bytes y aparecen por azar en ~7 % de los PNG de 1 MB: se exige la directiva completa de ASP
    rb"|<\?php|<script|<%@\s*(?:page|language|import|assembly|control|master|register|webservice|implements|reference|outputcache|application)\b"
    rb"|#!\s*/(?:usr/|bin/)",  # scripts web / shell
    re.IGNORECASE,
)
_ARCHIVE_STARTS = (b"PK\x03\x04", b"MZ", b"Rar!", b"7z\xbc\xaf", b"\x1f\x8b", b"#!")
PAYLOAD_MESSAGE = "El archivo contiene código ejecutable o datos ocultos y fue rechazado."


def _png_end(data: bytes) -> int | None:
    """Posición donde termina el último chunk (IEND) de un PNG, recorriendo los chunks (no buscando texto)."""
    pos = 8
    while pos + 12 <= len(data):
        length = int.from_bytes(data[pos : pos + 4], "big")
        ctype = data[pos + 4 : pos + 8]
        pos += 12 + length
        if ctype == b"IEND":
            return pos
    return None


def _trailing_data(data: bytes, fmt: str) -> bytes:
    """Bytes que sobran después del final real de la imagen (un truco habitual para esconder un ejecutable)."""
    if fmt == "PNG":
        end = _png_end(data)
        return data[end:] if end is not None else b""
    if fmt == "WEBP":
        end = int.from_bytes(data[4:8], "little") + 8
        return data[end + (end & 1) :]
    if fmt == "JPEG":
        end = data.rfind(b"\xff\xd9")
        tail = data[end + 2 :] if end >= 0 else b""
        # algunas cámaras añaden relleno; sólo molesta lo que parece un archivo comprimido o un ejecutable
        return tail if tail.lstrip(b"\x00\r\n ").startswith(_ARCHIVE_STARTS) else b""
    return b""


def reject_embedded_payload(data: bytes, fmt: str | None = None) -> None:
    """Rechaza archivos que llevan un ejecutable/script escondido (polyglots) o datos tras el final de la imagen."""
    if _PAYLOAD_RE.search(data) or (fmt and _trailing_data(data, fmt)):
        raise InvalidImage(PAYLOAD_MESSAGE)


def check_audio_payload(data: bytes) -> None:
    # un audio es binario: si empieza como texto es un script o una lista que ffmpeg seguiría (#EXTM3U, ffconcat…)
    if not data[:512].translate(None, bytes(range(32, 127)) + b"\t\r\n"):
        raise InvalidImage("El archivo no es un audio válido.")
    reject_embedded_payload(data)


def validate_image_bytes(data: bytes) -> ValidatedImage:
    s = get_settings()
    if not data:
        raise InvalidImage("Archivo vacío.")
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format or ""
            if fmt not in ALLOWED:
                raise InvalidImage("Formato no permitido (sólo JPG, PNG o WEBP).")
            w, h = im.size
            if w < 8 or h < 8:
                raise InvalidImage("La imagen es demasiado pequeña (mínimo 8x8).")
            if w * h > s.max_image_pixels:
                raise InvalidImage("La imagen tiene demasiados píxeles.")
            im.verify()
        with Image.open(io.BytesIO(data)) as im:  # verify() deja la imagen inutilizable
            im.load()
    except InvalidImage:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        raise InvalidImage("El archivo no es una imagen válida o está dañado.") from None
    reject_embedded_payload(data, fmt)
    return ValidatedImage(data, hashlib.sha256(data).hexdigest(), fmt, ALLOWED[fmt], w, h)


def save_image_files(project_id: int, v: ValidatedImage) -> str:
    """Guarda el original y su miniatura. Devuelve el nombre almacenado."""
    stored = f"{uuid.uuid4().hex}.{v.ext}"
    image_dir(project_id).mkdir(parents=True, exist_ok=True)
    thumb_dir(project_id).mkdir(parents=True, exist_ok=True)
    path = image_dir(project_id) / stored
    tmp = path.with_suffix(".part")
    tmp.write_bytes(v.data)
    os.replace(tmp, path)
    with Image.open(io.BytesIO(v.data)) as im:
        try:
            im = ImageOps.exif_transpose(im)
        except (SyntaxError, ValueError, OSError, KeyError, TypeError):
            pass  # EXIF dañado (la imagen en sí es válida): la miniatura sale sin girar en vez de fallar la carga
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
        im.thumbnail((THUMB_SIZE, THUMB_SIZE))
        im.save(thumb_dir(project_id) / (Path(stored).stem + ".jpg"), "JPEG", quality=80)
    return stored


def delete_image_files(img: ImageRow) -> None:
    for p in (image_path(img), thumb_path(img)):
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def delete_project_files(project_id: int, extra_dirs: list[str]) -> None:
    shutil.rmtree(image_dir(project_id), ignore_errors=True)
    shutil.rmtree(thumb_dir(project_id), ignore_errors=True)
    for d in extra_dirs:
        shutil.rmtree(d, ignore_errors=True)


def delete_model_files(dir_path: str) -> None:
    """Borra la carpeta de un modelo, pero sólo si está dentro de DATA_DIR/models (nunca una ruta cualquiera)."""
    root = (data_dir() / "models").resolve()
    target = Path(dir_path).resolve()
    if root in target.parents:
        shutil.rmtree(target, ignore_errors=True)


def user_quota_bytes(user: User) -> int:
    mb = user.quota_mb if user.quota_mb is not None else get_settings().default_user_quota_mb
    return mb * 1024 * 1024


def user_image_bytes(db: Session, user_id: int) -> int:
    return int(
        db.scalar(
            select(func.coalesce(func.sum(ImageRow.size_bytes), 0))
            .join(Project, Project.id == ImageRow.project_id)
            .where(Project.owner_id == user_id)
        )
        or 0
    )


def user_model_bytes(db: Session, user_id: int) -> int:
    return int(
        db.scalar(
            select(func.coalesce(func.sum(ModelExport.size_bytes), 0))
            .join(ModelVersion, ModelVersion.id == ModelExport.model_id)
            .join(Project, Project.id == ModelVersion.project_id)
            .where(Project.owner_id == user_id)
        )
        or 0
    )


# ------------------------------------------------------------------------------ audio
AUDIO_EXT = {"wav", "mp3", "ogg", "flac", "m4a", "aac", "opus", "webm"}
AUDIO_MEDIA = {
    "wav": "audio/wav", "mp3": "audio/mpeg", "ogg": "audio/ogg", "flac": "audio/flac",
    "m4a": "audio/mp4", "aac": "audio/aac", "opus": "audio/ogg", "webm": "audio/webm",
}
MIN_AUDIO_SECONDS = 0.3


@dataclass
class ValidatedAudio:
    data: bytes
    sha256: str
    ext: str
    duration: float
    signal: object  # np.ndarray mono 16 kHz (para la miniatura)


def validate_audio_bytes(data: bytes, filename: str) -> ValidatedAudio:
    """Valida decodificando realmente el archivo con ffmpeg (no basta con la extensión)."""
    from .ml.audio_features import SAMPLE_RATE, AudioError, load_audio

    if not data:
        raise InvalidImage("Archivo vacío.")
    ext = Path(filename or "").suffix.lower().lstrip(".")
    if ext not in AUDIO_EXT:
        raise InvalidImage("Formato no permitido (WAV, MP3, OGG, FLAC, M4A, AAC, OPUS o WEBM).")
    check_audio_payload(data)
    try:
        signal = load_audio(data, max_seconds=get_settings().max_audio_seconds)
    except AudioError as exc:
        raise InvalidImage(f"El archivo no es un audio válido: {exc}") from None
    duration = len(signal) / SAMPLE_RATE
    if duration < MIN_AUDIO_SECONDS:
        raise InvalidImage(f"El audio es demasiado corto (mínimo {MIN_AUDIO_SECONDS} s).")
    return ValidatedAudio(data, hashlib.sha256(data).hexdigest(), ext, round(duration, 3), signal)


def save_audio_files(project_id: int, v: ValidatedAudio) -> str:
    from .ml.audio_features import thumbnail

    stored = f"{uuid.uuid4().hex}.{v.ext}"
    image_dir(project_id).mkdir(parents=True, exist_ok=True)
    thumb_dir(project_id).mkdir(parents=True, exist_ok=True)
    path = image_dir(project_id) / stored
    tmp = path.with_suffix(".part")
    tmp.write_bytes(v.data)
    os.replace(tmp, path)
    thumbnail(v.signal).save(thumb_dir(project_id) / (Path(stored).stem + ".jpg"), "JPEG", quality=80)
    return stored


# ------------------------------------------------------------------------- tablas (CSV)
TABLE_EXT = {"csv", "tsv"}


@dataclass
class ValidatedTable:
    data: bytes
    sha256: str
    ext: str
    rows: int
    columns: int


def validate_table_bytes(data: bytes, filename: str) -> ValidatedTable:
    """Valida leyendo de verdad la tabla (separador, encabezado, filas); no basta con la extensión."""
    from .ml.tabular import InvalidTable, read_table

    ext = Path(filename or "").suffix.lower().lstrip(".")
    if ext not in TABLE_EXT:
        raise InvalidImage("Formato no permitido (sólo CSV o TSV).")
    reject_embedded_payload(data)
    s = get_settings()
    try:
        df = read_table(data, s.max_tabular_rows, s.max_tabular_columns)
    except InvalidTable as exc:
        raise InvalidImage(str(exc)) from None
    return ValidatedTable(data, hashlib.sha256(data).hexdigest(), "csv", len(df), int(df.shape[1]))


def save_table_file(project_id: int, v: ValidatedTable) -> str:
    stored = f"{uuid.uuid4().hex}.csv"
    image_dir(project_id).mkdir(parents=True, exist_ok=True)
    path = image_dir(project_id) / stored
    tmp = path.with_suffix(".part")
    tmp.write_bytes(v.data)
    os.replace(tmp, path)
    return stored
