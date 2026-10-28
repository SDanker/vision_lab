"""Importación de datasets desde Hugging Face a un proyecto de imágenes.

Dos formas de dataset (las que usa casi todo el Hub):
  · «files»   carpetas con imágenes sueltas (imagefolder); cada carpeta es una categoría.
  · «parquet» tablas con una columna de imagen y otra de etiqueta (ClassLabel); es el formato por defecto del Hub.

Todo el acceso a la red está en las funciones `hub_*` de este módulo: las pruebas las reemplazan y nada más
del servicio habla con Hugging Face. El token de cada usuario sólo se descifra en el momento de usarlo.
"""
import errno
import logging
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from time import monotonic

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import SessionLocal, utcnow
from ..errors import ApiError
from ..models import (
    HF_ACTIVE,
    HF_CANCELLED,
    HF_COMPLETED,
    HF_ERROR,
    HF_PENDING,
    HF_RUNNING,
    Category,
    HfConnection,
    HfImport,
    Image,
    Project,
    User,
)
from ..modalities import get_modality
from ..storage import InvalidImage, data_dir, delete_image_files, user_image_bytes, user_quota_bytes
from . import hf_labels, vault
from .access import ensure_modifications_open
from .hf_labels import clean_label
from .audit import audit
from .notifications import notify_user

log = logging.getLogger(__name__)

IMG_EXT = {"jpg", "jpeg", "png", "webp"}
# carpetas que sólo organizan el dataset (separaciones, contenedores): no son una categoría
IGNORED_DIRS = {
    "train", "training", "test", "testing", "val", "valid", "validation", "dev", "eval", "evaluation",
    "data", "dataset", "datasets", "images", "image", "imgs", "img", "raw", "files",
}
LABEL_HINTS = ("label", "labels", "class", "category", "target", "fine_label", "coarse_label", "label_name")
TOKEN_RE = re.compile(r"^hf_[A-Za-z0-9]{16,128}$")
REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}(/[A-Za-z0-9][A-Za-z0-9._-]{0,95})?$")
URL_STOP = {"tree", "blob", "viewer", "resolve", "discussions", "commits", "raw", "settings", "embed", "edit", "upload"}
MAX_TREE_ENTRIES = 600_000  # tope de archivos que se recorren al revisar un repositorio
MAX_FOOTERS_PREVIEW = 12  # cuántos parquet se consultan para contar filas al revisar (más = se omite el conteo)
DOWNLOAD_THREADS = 6
CHECKPOINT_EVERY = 25  # elementos entre cada guardado de progreso
TRANSIENT = {"HF_UNREACHABLE", "HF_UNAVAILABLE"}  # se reintentan antes de darse por vencidos
MAX_ROW_GROUP_BYTES = 512 * 1024 * 1024  # un grupo de filas se lee entero: más que esto no cabe en el worker liviano (1 GB)


class HfError(Exception):
    """Error con mensaje listo para mostrar al usuario."""

    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status

    def as_api(self) -> ApiError:
        return ApiError(self.status, self.code, self.message)


class _Cancelled(Exception):
    pass


class _LimitReached(Exception):
    """Se llegó al máximo de imágenes pedido (o todas las categorías elegidas están completas): fin normal."""


SAMPLE_ROWS = 2000  # filas de muestra para proponer y previsualizar las categorías
MAX_SAMPLE_COLUMNS = 16
LABEL_SOURCES = ("auto", "column", "phrase", "none")


@dataclass
class HfOpts:
    """Lo que el usuario decide al importar: de dónde sale la categoría, qué se queda y hasta cuánto."""

    image_column: str | None = None
    label_source: str = "auto"  # auto (la que declare el dataset) | column | phrase (extraída de un texto) | none
    label_column: str | None = None
    label_phrase: dict | None = None  # {"column", "after", "words", "until"}
    default_category: str | None = None  # destino de lo que no tiene categoría
    include: list[str] | None = None  # None = todas las categorías
    rest: str = "skip"  # qué pasa con las no incluidas: skip (omitir) | other (juntar en other_label)
    other_label: str = "otros"
    max_items: int | None = None
    max_per_category: int | None = None

    def stored(self, plan: "Plan") -> dict:
        """Lo que se guarda en la importación para que el worker haga lo mismo que mostró la vista previa."""
        return {
            "layout": plan.layout, "image_column": plan.image_column, "label_column": plan.label_column,
            "class_names": plan.class_names, "label_phrase": plan.label_phrase,
            "default_category": clean_label(self.default_category), "include": self.include, "rest": self.rest,
            "other_label": clean_label(self.other_label) or "otros", "max_items": self.max_items,
            "max_per_category": self.max_per_category,
        }


# ------------------------------------------------------------------------------ entrada del usuario
def parse_repo_id(raw: str) -> str:
    """«usuario/dataset», «datasets/usuario/dataset» o la URL de huggingface.co. Cualquier otro sitio se rechaza."""
    s = (raw or "").strip()
    m = re.match(r"^https?://(?:www\.)?huggingface\.co/(.*)$", s, re.IGNORECASE)
    if m:
        segs = [x for x in re.split(r"[/?#]", m.group(1).split("?")[0].split("#")[0]) if x]
        if not segs or segs[0] != "datasets" or len(segs) < 2:
            raise HfError("INVALID_REPO", "Pega la dirección de un dataset de Hugging Face (huggingface.co/datasets/…).")
        rest = segs[1:]
        s = "/".join(rest[:2]) if len(rest) >= 2 and rest[1] not in URL_STOP else rest[0]
    elif re.match(r"^[a-z][a-z0-9+.-]*://", s, re.IGNORECASE):
        raise HfError("INVALID_REPO", "Sólo se aceptan direcciones de huggingface.co.")
    else:
        s = s.removeprefix("datasets/").strip("/")
    if not REPO_RE.match(s):
        raise HfError("INVALID_REPO", "Escribe el dataset como «usuario/nombre» (por ejemplo uoft-cs/cifar10).")
    return s


def valid_token_format(token: str) -> bool:
    return bool(TOKEN_RE.match(token))


def token_hint(token: str) -> str:
    return f"hf_…{token[-4:]}"


# ------------------------------------------------------------------------------ red (reemplazable en pruebas)
def _translate(exc: Exception, repo: str | None = None, token: bool = False) -> HfError | None:
    """Traduce los errores de huggingface_hub a un mensaje claro; None si no se reconoce."""
    import requests
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError, RevisionNotFoundError

    if isinstance(exc, GatedRepoError):
        if token:
            return HfError(
                "HF_GATED",
                f"Este dataset es restringido. Entra a huggingface.co/datasets/{repo}, acepta sus condiciones con tu "
                "cuenta de Hugging Face y vuelve a intentarlo.", 403,
            )
        return HfError(
            "HF_GATED",
            "Este dataset es restringido. Conecta tu cuenta de Hugging Face en «Mi cuenta» y acepta sus condiciones "
            "en huggingface.co.", 403,
        )
    if isinstance(exc, RepositoryNotFoundError):
        msg = (
            "No existe ese dataset, o tu cuenta de Hugging Face no tiene acceso."
            if token
            else "No existe ese dataset o es privado. Si es privado, conecta tu cuenta de Hugging Face en «Mi cuenta»."
        )
        return HfError("HF_NOT_FOUND", msg, 404)
    if isinstance(exc, RevisionNotFoundError):
        return HfError("HF_NOT_FOUND", "No se encontró esa versión del dataset.", 404)
    # whoami() convierte el 401 en un requests.HTTPError genérico («Invalid user token»), que no es un HfHubHTTPError
    if isinstance(exc, (HfHubHTTPError, requests.exceptions.HTTPError)):
        code = getattr(getattr(exc, "response", None), "status_code", None)
        if code is None and "invalid user token" in str(exc).lower():
            code = 401
        if code in (401, 403):
            return HfError("HF_TOKEN_INVALID", "Hugging Face rechazó el token (venció, se revocó o no tiene permiso de lectura).", 401 if code == 401 else 403)
        if code == 429:
            return HfError("HF_RATE_LIMIT", "Hugging Face limitó las solicitudes. Espera unos minutos y reintenta.", 429)
        if code and code >= 500:
            return HfError("HF_UNAVAILABLE", "Hugging Face no responde en este momento. Reintenta más tarde.", 502)
    if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return HfError("HF_UNREACHABLE", "El servidor no pudo conectarse con Hugging Face. Reintenta más tarde.", 502)
    return None


@contextmanager
def _guard(repo: str | None = None, token: str | None = None):
    """Los errores conocidos de huggingface_hub salen como HfError (sin filtrar el token); el resto se propaga igual."""
    try:
        yield
    except HfError:
        raise
    except Exception as exc:  # noqa: BLE001
        err = _translate(exc, repo, bool(token))
        if err is None:
            raise
        raise err from None


def _tok(token: str | None):
    """`False` = anónimo de verdad: huggingface_hub no debe caer en un HF_TOKEN del entorno del servidor."""
    return token or False


def hub_whoami(token: str) -> str:
    from huggingface_hub import HfApi

    with _guard(None, token):
        return str(HfApi().whoami(token=token)["name"])


def hub_dataset_info(repo_id: str, token: str | None) -> dict:
    from huggingface_hub import HfApi

    with _guard(repo_id, token):
        info = HfApi().dataset_info(repo_id, token=_tok(token))
    return {"id": info.id, "sha": info.sha, "private": bool(info.private), "gated": bool(info.gated)}


def hub_list_files(repo_id: str, revision: str, token: str | None) -> list[tuple[str, int]]:
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    out: list[tuple[str, int]] = []
    with _guard(repo_id, token):
        for entry in HfApi().list_repo_tree(repo_id, repo_type="dataset", revision=revision, recursive=True, token=_tok(token)):
            if isinstance(entry, RepoFile):
                out.append((entry.path, int(entry.size or 0)))
                if len(out) > MAX_TREE_ENTRIES:
                    raise HfError("HF_TOO_MANY_FILES", "El repositorio tiene demasiados archivos para importarlo.")
    return out


@dataclass
class ParquetInfo:
    rows: int
    columns: dict[str, str]  # nombre -> image | int | bool | string | other (según el tipo de Arrow)
    features: dict = field(default_factory=dict)  # metadatos de «datasets»: nombre -> {"_type": "ClassLabel", "names": [...]}


def _parquet_info_from(pf) -> ParquetInfo:
    import json

    import pyarrow as pa

    cols: dict[str, str] = {}
    for f in pf.schema_arrow:
        t = f.type
        if pa.types.is_struct(t) and "bytes" in {x.name for x in t}:
            cols[f.name] = "image"
        elif pa.types.is_boolean(t):
            cols[f.name] = "bool"
        elif pa.types.is_integer(t):
            cols[f.name] = "int"
        elif pa.types.is_string(t) or pa.types.is_large_string(t):
            cols[f.name] = "string"
        else:
            cols[f.name] = "other"
    features: dict = {}
    raw = (pf.schema_arrow.metadata or {}).get(b"huggingface")
    if raw:
        try:
            features = json.loads(raw).get("info", {}).get("features", {}) or {}
        except (ValueError, AttributeError):
            features = {}
    return ParquetInfo(pf.metadata.num_rows, cols, features if isinstance(features, dict) else {})


def hub_parquet_info(repo_id: str, revision: str, token: str | None, path: str) -> ParquetInfo:
    """Lee sólo el pie del parquet (rangos HTTP), sin bajar el archivo."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    with _guard(repo_id, token), HfFileSystem(token=_tok(token)).open(f"datasets/{repo_id}@{revision}/{path}", "rb") as f:
        return _parquet_info_from(pq.ParquetFile(f))


def hub_sample(repo_id: str, revision: str, token: str | None, path: str, columns: list[str], max_rows: int) -> dict[str, list]:
    """Primeras filas de unas columnas (sólo esas: lecturas por rango HTTP, sin bajar las imágenes)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    with _guard(repo_id, token), HfFileSystem(token=_tok(token)).open(f"datasets/{repo_id}@{revision}/{path}", "rb") as f:
        return _sample_from(pq.ParquetFile(f), columns, max_rows, pa)


def _sample_from(pf, columns: list[str], max_rows: int, pa) -> dict[str, list]:
    columns = [c for c in columns if c in pf.schema_arrow.names]
    tables, rows = [], 0
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(g, columns=columns)
        tables.append(t)
        rows += t.num_rows
        if rows >= max_rows:
            break
    if not tables:
        return {c: [] for c in columns}
    table = pa.concat_tables(tables).slice(0, max_rows)
    return {c: table.column(c).to_pylist() for c in columns}


def hub_download(repo_id: str, revision: str, token: str | None, path: str, dest: Path) -> Path:
    from huggingface_hub import hf_hub_download

    with _guard(repo_id, token):
        return Path(
            hf_hub_download(repo_id=repo_id, filename=path, repo_type="dataset", revision=revision, token=_tok(token), local_dir=str(dest))
        )


# ------------------------------------------------------------------------------ cuenta del usuario
def connection_of(db: Session, user_id: int) -> HfConnection | None:
    return db.scalar(select(HfConnection).where(HfConnection.user_id == user_id))


def token_for(db: Session, user_id: int) -> str | None:
    """Token descifrado del usuario (None si no conectó su cuenta). Si no se puede descifrar, hay que reconectar."""
    conn = connection_of(db, user_id)
    if conn is None:
        return None
    token = vault.decrypt(conn.token_enc)
    if token is None:
        raise HfError("HF_RECONNECT", "No se pudo leer tu token de Hugging Face guardado. Vuelve a conectar tu cuenta en «Mi cuenta».", 409)
    conn.last_used_at = utcnow()
    return token


def status_dict(conn: HfConnection | None) -> dict:
    if conn is None:
        return {"connected": False}
    return {
        "connected": True,
        "username": conn.hf_username,
        "hint": conn.token_hint,
        "connected_at": conn.created_at,
        "last_used_at": conn.last_used_at,
        "needs_reconnect": vault.decrypt(conn.token_enc) is None,
    }


# ------------------------------------------------------------------------------ plan de importación
@dataclass
class Plan:
    repo_id: str
    revision: str
    private: bool
    gated: bool
    layout: str  # files | parquet
    files: list[tuple[str, int]]  # imágenes (files) o parquets (parquet), con su tamaño
    total_bytes: int
    items: int | None  # imágenes que se esperan (None = no se sabe todavía)
    categories: dict[str, int] | None = None  # sólo «files»: nombre -> cantidad
    unlabeled: int = 0
    image_columns: list[str] = field(default_factory=list)
    label_columns: list[str] = field(default_factory=list)
    image_column: str | None = None
    label_column: str | None = None
    class_names: list[str] | None = None
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    items_estimate: int | None = None  # filas aproximadas cuando hay demasiadas tablas para contarlas todas
    effective_bytes: int = 0  # lo que se descargaría con el límite de imágenes aplicado
    label_source: str = "none"  # lo que se aplica de verdad: dataset (ClassLabel/columna) | column | phrase | folders | none
    label_phrase: dict | None = None
    text_columns: list[str] = field(default_factory=list)
    label_preview: dict | None = None  # reparto de categorías (sobre una muestra en tablas; exacto en carpetas)
    suggestions: list[dict] = field(default_factory=list)


def class_of(value, names: list[str] | None) -> str | None:
    """Categoría de un valor de la columna de etiquetas (un ClassLabel trae números y la lista de nombres aparte)."""
    if value is None or (isinstance(value, int) and not isinstance(value, bool) and value < 0):  # -1 = sin etiqueta
        return None
    if isinstance(value, bool):
        return clean_label(value)
    if isinstance(value, int) and names:
        return clean_label(names[value]) if 0 <= value < len(names) else clean_label(value)
    return clean_label(value)


def label_from_path(path: str) -> str | None:
    """Categoría de una imagen según su carpeta; None si sólo está dentro de carpetas organizativas (train/, images/…)."""
    for part in reversed(path.split("/")[:-1]):
        if part.lower() not in IGNORED_DIRS:
            return clean_label(part)
    return None


def _ext(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


def _hidden(path: str) -> bool:
    """Archivos ocultos y rutas raras: un repositorio ajeno no decide dónde se escribe nada en el servidor."""
    if path.startswith("/") or "\\" in path or "\x00" in path:
        return True
    return any(p in ("", "..") or p.startswith(".") or p == "__MACOSX" for p in path.split("/"))


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def build_plan(source: str, token: str | None, free_bytes: int, opts: HfOpts | None = None) -> Plan:
    s = get_settings()
    opts = opts or HfOpts()
    repo = parse_repo_id(source)
    info = hub_dataset_info(repo, token)
    repo, rev = info["id"], info["sha"]
    files = [(p, sz) for p, sz in hub_list_files(repo, rev, token) if not _hidden(p)]
    media = sorted((p, sz) for p, sz in files if _ext(p) in IMG_EXT)
    shards = sorted((p, sz) for p, sz in files if _ext(p) == "parquet")
    media_bytes, shard_bytes = sum(sz for _, sz in media), sum(sz for _, sz in shards)
    if not media and not shards:
        raise HfError("HF_NO_IMAGES", "Este dataset no tiene imágenes (JPG, PNG o WEBP) ni tablas parquet con imágenes.", 422)

    if media and (not shards or media_bytes >= shard_bytes):
        plan = _plan_files(repo, rev, info, media, opts)
    else:
        plan = _plan_parquet(repo, rev, info, shards, token, opts)

    # con un máximo de imágenes se descarga sólo una parte: el tamaño y el espacio se calculan sobre esa parte
    plan.effective_bytes = plan.total_bytes
    known = plan.items if plan.items is not None else plan.items_estimate
    if opts.max_items and known and opts.max_items < known:
        plan.effective_bytes = int(plan.total_bytes * opts.max_items / known)
    wanted = min(plan.items, opts.max_items) if (plan.items is not None and opts.max_items) else plan.items
    cap = s.hf_import_max_gb * 1024**3
    if plan.effective_bytes > cap:
        plan.blockers.append(
            f"Se descargarían {fmt_bytes(plan.effective_bytes)} y el máximo por importación es {s.hf_import_max_gb:g} GB."
            + ("" if opts.max_items else " Pon un máximo de imágenes más bajo para importar sólo una parte.")
        )
    if wanted is not None and wanted > s.hf_import_max_items:
        plan.blockers.append(f"Tiene {wanted:,} imágenes y el máximo por importación es {s.hf_import_max_items:,}.".replace(",", "."))
    if plan.effective_bytes > free_bytes:
        plan.blockers.append(f"No cabe en tu espacio libre ({fmt_bytes(max(free_bytes, 0))}); necesita unos {fmt_bytes(plan.effective_bytes)}.")
    if plan.categories is not None and len(plan.categories) > s.hf_import_max_categories:
        plan.blockers.append(f"Tiene {len(plan.categories)} categorías y el máximo es {s.hf_import_max_categories}.")
    if plan.class_names is not None and len(plan.class_names) > s.hf_import_max_categories:
        plan.blockers.append(f"Tiene {len(plan.class_names)} categorías y el máximo es {s.hf_import_max_categories}.")
    distinct = (plan.label_preview or {}).get("distinct", 0)
    if distinct > s.hf_import_max_categories:
        plan.blockers.append(f"La regla genera {distinct} categorías distintas y el máximo es {s.hf_import_max_categories}.")
    _check_filters(plan, opts)
    return plan


def _check_filters(plan: Plan, opts: HfOpts) -> None:
    """Avisos sobre los filtros elegidos, calculados con el reparto de la vista previa."""
    values = [v["value"].lower() for v in (plan.label_preview or {}).get("values", [])]
    if opts.include is not None and values:
        keep = {x.lower() for x in opts.include}
        if not keep:
            plan.blockers.append("No quedó ninguna categoría marcada para importar.")
        elif not keep & set(values) and (plan.label_preview or {}).get("exact"):
            plan.blockers.append("Ninguna de las categorías marcadas existe en el dataset.")
        elif not keep & set(values):
            plan.warnings.append("Ninguna de las categorías marcadas aparece en la muestra; puede que no haya nada que importar.")


def _files_label_preview(cats: dict[str, int], unlabeled: int) -> dict:
    top = sorted(cats.items(), key=lambda kv: -kv[1])
    total = sum(cats.values()) + unlabeled
    return {
        "exact": True, "sample_rows": total, "matched": sum(cats.values()), "unmatched": unlabeled, "distinct": len(cats),
        "values": [{"value": k, "count": v} for k, v in top[:60]],
    }


def _plan_files(repo, rev, info, media, opts: HfOpts) -> Plan:
    if opts.label_source in ("column", "phrase"):
        raise HfError(
            "HF_BAD_LABEL",
            "Elegir la categoría desde una columna o un texto sólo sirve en datasets de tablas (parquet); aquí la categoría sale de las carpetas.", 422,
        )
    cats: dict[str, int] = {}
    unlabeled = 0
    for path, _ in media:
        label = label_from_path(path)
        if label is None:
            unlabeled += 1
        else:
            cats[label] = cats.get(label, 0) + 1
    plan = Plan(
        repo, rev, info["private"], info["gated"], "files", media, sum(sz for _, sz in media), len(media),
        categories=dict(sorted(cats.items())), unlabeled=unlabeled, label_source="folders",
    )
    default = clean_label(opts.default_category)
    if unlabeled and default:
        plan.categories[default] = plan.categories.get(default, 0) + unlabeled
        plan.unlabeled = 0
    elif unlabeled:
        plan.warnings.append(f"{unlabeled} imágenes no están dentro de una carpeta con nombre de categoría y se omitirán (o indica una categoría de destino).")
    if not plan.categories and not default:
        plan.blockers.append("No se pudo deducir ninguna categoría de las carpetas. Indica una categoría de destino.")
    plan.label_preview = _files_label_preview(plan.categories, plan.unlabeled)
    return plan


def _plan_parquet(repo, rev, info, shards, token, opts: HfOpts) -> Plan:
    first = hub_parquet_info(repo, rev, token, shards[0][0])
    feats = first.features
    images = [
        c for c, kind in first.columns.items()
        if feats.get(c, {}).get("_type") == "Image" or (c not in feats and kind == "image")
    ]
    if not images:
        raise HfError("HF_NO_IMAGES", "Este dataset no tiene una columna de imágenes.", 422)
    img_col = opts.image_column if opts.image_column in images else images[0]

    # columnas que podrían dar la categoría: las que declara el dataset (ClassLabel o por nombre) y, tras mirar una
    # muestra de filas, cualquier columna que parezca de categorías (pocos valores que se repiten)
    scalar = [c for c, kind in first.columns.items() if kind in ("int", "string", "bool") and c not in images]
    classlabel = [c for c in first.columns if feats.get(c, {}).get("_type") == "ClassLabel"]
    hinted = [c for c in scalar if c not in classlabel and c.lower() in LABEL_HINTS]
    wanted = [c for c in (opts.label_column, (opts.label_phrase or {}).get("column")) if c in scalar]
    cols = list(dict.fromkeys(wanted + classlabel + hinted + scalar))[:MAX_SAMPLE_COLUMNS]
    sample = hub_sample(repo, rev, token, shards[0][0], cols, SAMPLE_ROWS) if cols else {}
    stats = hf_labels.column_stats(sample)
    sample_rows = max((len(v) for v in sample.values()), default=0)
    extra = [c for c in cols if c not in classlabel + hinted and c in stats and hf_labels.is_categorical(stats[c], sample_rows)]
    auto = classlabel + hinted
    label_columns = auto + extra
    text_columns = [c for c in scalar if first.columns[c] == "string" and c in stats and stats[c]["avg_len"] > 40]

    lab_col, phrase, source = None, None, "none"
    if opts.label_source == "phrase":
        ph = opts.label_phrase or {}
        if first.columns.get(ph.get("column")) != "string":
            raise HfError("HF_BAD_LABEL", "La columna de texto elegida no existe en este dataset.", 422)
        phrase, source = ph, "phrase"
    elif opts.label_source == "column":
        if opts.label_column not in scalar:
            raise HfError("HF_BAD_LABEL", "La columna elegida no existe o no sirve como categoría.", 422)
        lab_col, source = opts.label_column, "column"
    elif opts.label_source == "auto":
        lab_col = opts.label_column if opts.label_column in label_columns else (auto[0] if auto else None)
        source = "dataset" if lab_col in auto else ("column" if lab_col else "none")
    names = None
    if lab_col is not None and feats.get(lab_col, {}).get("_type") == "ClassLabel":
        raw = feats[lab_col].get("names")
        names = [str(n) for n in raw] if isinstance(raw, list) else None

    items = None
    if len(shards) <= MAX_FOOTERS_PREVIEW:
        items = first.rows + sum(hub_parquet_info(repo, rev, token, p).rows for p, _ in shards[1:])
    plan = Plan(
        repo, rev, info["private"], info["gated"], "parquet", shards, sum(sz for _, sz in shards), items,
        image_columns=images, label_columns=label_columns, image_column=img_col, label_column=lab_col, class_names=names,
        items_estimate=first.rows * len(shards), label_source=source, label_phrase=phrase, text_columns=text_columns,
    )
    plan.suggestions = [dict(column=c, **s) for c in text_columns for s in hf_labels.suggest_phrases(sample.get(c, []))][:8]

    # reparto de categorías sobre la muestra, con la misma función que usará la importación
    labels = None
    if phrase is not None:
        labels = [hf_labels.extract_phrase(t, phrase.get("after", ""), phrase.get("words"), phrase.get("until")) for t in sample.get(phrase["column"], [])]
    elif lab_col is not None and lab_col in sample:
        labels = [class_of(v, names) for v in sample[lab_col]]
    default = clean_label(opts.default_category)
    if labels is not None:
        counted = hf_labels.count_labels(labels)
        matched = sum(c for _, c in counted)
        plan.label_preview = {
            "exact": False, "sample_rows": len(labels), "matched": matched, "unmatched": len(labels) - matched,
            "distinct": len(counted), "values": [{"value": v, "count": c} for v, c in counted[:60]],
        }
        if labels and matched == 0 and phrase is not None:
            plan.blockers.append(
                f"La frase «{phrase.get('after', '')}» no aparece en ningún texto de la muestra ({len(labels)} filas). Revisa cómo está escrita en el dataset."
            )
        elif labels and matched < 0.5 * len(labels):
            plan.warnings.append(f"Sólo {matched} de {len(labels)} filas de la muestra tienen categoría con esta regla; el resto se omitirá{' (o irá a la categoría de destino)' if not default else ''}.")
        elif labels and matched < len(labels) and not default:
            plan.warnings.append(f"{len(labels) - matched} de {len(labels)} filas de la muestra no tienen categoría y se omitirán (o indica una categoría de destino).")
        if matched and len(counted) == 1:
            plan.warnings.append("Con esta regla sale una sola categoría: no sirve para entrenar un clasificador (hacen falta al menos dos).")
    elif not default:
        plan.blockers.append(
            "El dataset no trae etiquetas: elige de qué columna o texto sacar la categoría, o indica la categoría a la que irán todas las imágenes."
        )
    return plan


def plan_dict(plan: Plan) -> dict:
    cats = plan.categories
    return {
        "repo_id": plan.repo_id,
        "revision": plan.revision,
        "private": plan.private,
        "gated": plan.gated,
        "layout": plan.layout,
        "files": len(plan.files),
        "items": plan.items,
        "total_bytes": plan.total_bytes,
        "categories": [{"name": n, "count": c} for n, c in cats.items()] if cats is not None else None,
        "category_count": len(cats) if cats is not None else (len(plan.class_names) if plan.class_names else None),
        "class_names": (plan.class_names or [])[:40] if plan.class_names else None,
        "unlabeled": plan.unlabeled,
        "image_columns": plan.image_columns,
        "label_columns": plan.label_columns,
        "image_column": plan.image_column,
        "label_column": plan.label_column,
        "label_source": plan.label_source,
        "label_phrase": plan.label_phrase,
        "text_columns": plan.text_columns,
        "suggestions": plan.suggestions,
        "label_preview": plan.label_preview,
        "effective_bytes": plan.effective_bytes,
        "blockers": plan.blockers,
        "warnings": plan.warnings,
        "can_import": not plan.blockers,
    }


# ------------------------------------------------------------------------------ ciclo de vida
def import_dict(imp: HfImport) -> dict:
    pct = None
    if imp.status == HF_COMPLETED:
        pct = 100.0
    elif imp.total:
        pct = round(min(imp.processed / imp.total, 1.0) * 100, 1)
    return {
        "id": imp.id,
        "project_id": imp.project_id,
        "repo_id": imp.repo_id,
        "revision": imp.revision,
        "status": imp.status,
        "phase": imp.phase,
        "percent": pct,
        "total": imp.total,
        "processed": imp.processed,
        "imported": imp.imported,
        "duplicates": imp.duplicates,
        "rejected": imp.rejected,
        "unlabeled": imp.unlabeled,
        "skipped": imp.skipped,
        "rejected_sample": imp.rejected_sample or [],
        "total_bytes": imp.total_bytes,
        "cancel_requested": imp.cancel_requested,
        "error": imp.error,
        "user": imp.user.username if imp.user else None,
        "created_at": imp.created_at,
        "started_at": imp.started_at,
        "finished_at": imp.finished_at,
    }


def active_count(db: Session, user_id: int) -> int:
    return len(db.scalars(select(HfImport.id).where(HfImport.user_id == user_id, HfImport.status.in_(HF_ACTIVE))).all())


def kick_import(import_id: int) -> None:
    try:
        from ..celery_app import celery_app

        celery_app.send_task("vision.hf_import", args=[import_id], queue="default")
    except Exception as exc:  # noqa: BLE001
        log.warning("No se pudo encolar la importación %s: %s", import_id, exc)
        with SessionLocal() as db:
            imp = db.get(HfImport, import_id)
            if imp is not None and imp.status == HF_PENDING:
                imp.status, imp.finished_at = HF_ERROR, utcnow()
                imp.error = "No se pudo poner la importación en la cola. Reintenta en unos minutos."
                db.commit()


def recover_stale(db: Session) -> int:
    """Marca como fallidas las importaciones cuyo worker murió (reinicio del servidor) o que nadie recogió."""
    s = get_settings()
    now = utcnow()
    n = 0
    for imp in db.scalars(select(HfImport).where(HfImport.status.in_(HF_ACTIVE))).unique().all():
        last = imp.heartbeat_at or imp.started_at or imp.created_at
        limit = timedelta(seconds=s.hf_import_stale_seconds) if imp.status == HF_RUNNING else timedelta(hours=1)
        if now - last > limit:
            imp.status, imp.finished_at = HF_ERROR, now
            imp.error = "La importación se interrumpió (probablemente se reinició el servidor). Vuelve a importar: se omiten las imágenes que ya están."
            n += 1
    if n:
        db.commit()
    return n


class _Runner:
    """Ejecuta una importación: descarga, valida cada imagen como en una carga normal y la guarda."""

    def __init__(self, db: Session, imp: HfImport, project: Project, user: User):
        self.db, self.imp, self.project, self.user = db, imp, project, user
        self.mod = get_modality(project.modality)
        s = get_settings()
        self.settings = s
        self.limit = self.mod.max_upload_mb(s) * 1024 * 1024
        owner = project.owner
        self.quota, self.used = user_quota_bytes(owner), user_image_bytes(db, owner.id)
        self.initial_used = self.used
        self.hashes = set(db.scalars(select(Image.sha256).where(Image.project_id == project.id)))
        self.cats = {c.name.lower(): c.id for c in project.categories}
        o = imp.options or {}
        self.default = clean_label(o.get("default_category"))
        self.phrase = o.get("label_phrase")
        self.include = {x.lower() for x in o["include"]} if o.get("include") is not None else None
        self.rest = o.get("rest") or "skip"
        self.other = clean_label(o.get("other_label")) or "otros"
        self.max_items, self.max_per = o.get("max_items"), o.get("max_per_category")
        self.per_cat: dict[str, int] = {}  # imágenes importadas en esta ejecución, por categoría
        self.byte_cap = int(s.hf_import_max_gb * 1024**3)
        self.unsaved: list[Image] = []
        self.rejected_sample: list[dict] = []
        self.last_checkpoint = monotonic()
        self.tmp = data_dir() / "work" / f"hf-{imp.id}"

    # ---- progreso
    def phase(self, text: str) -> None:
        self.imp.phase = text[:200]

    def checkpoint(self) -> None:
        imp = self.imp
        imp.heartbeat_at = utcnow()
        imp.rejected_sample = self.rejected_sample[:10]
        ensure_modifications_open(self.db, self.project, self.user)  # el calendario puede cerrarse a mitad de camino
        self.db.commit()
        self.unsaved.clear()
        self.last_checkpoint = monotonic()
        self.db.refresh(imp, ["cancel_requested"])
        if imp.cancel_requested:
            raise _Cancelled

    def tick(self) -> None:
        if len(self.unsaved) >= CHECKPOINT_EVERY or monotonic() - self.last_checkpoint > 3:
            self.checkpoint()

    # ---- un elemento
    def reject(self, name: str, reason: str) -> None:
        self.imp.rejected += 1
        if len(self.rejected_sample) < 10:
            self.rejected_sample.append({"name": name[:120], "reason": reason})

    def category(self, label: str) -> int | None:
        cid = self.cats.get(label.lower())
        if cid is None:
            if len(self.cats) >= self.settings.hf_import_max_categories:
                return None
            cat = Category(project_id=self.project.id, name=label)
            self.db.add(cat)
            self.db.flush()
            cid = self.cats[label.lower()] = cat.id
        return cid

    def decide(self, label: str) -> str | None:
        """Categoría con la que se importa una imagen según los filtros elegidos; None = no se importa."""
        if self.include is not None and label.lower() not in self.include:
            if self.rest != "other":
                return None
            label = self.other
        if self.max_per and self.per_cat.get(label.lower(), 0) >= self.max_per:
            return None
        return label

    def _all_full(self) -> bool:
        """Con categorías elegidas y un máximo por categoría, cuando todas están completas no hay nada más que bajar."""
        return bool(self.include and self.max_per and self.rest != "other" and all(self.per_cat.get(k, 0) >= self.max_per for k in self.include))

    def ingest(self, data: bytes | None, name: str, label: str | None) -> None:
        imp = self.imp
        imp.processed += 1
        label = label or self.default
        if label is None:
            imp.unlabeled += 1
            return
        label = self.decide(label)
        if label is None:  # fuera por los filtros o porque su categoría ya llegó al máximo
            imp.skipped += 1
            return
        if not data:
            return self.reject(name, "Sin contenido.")
        if len(data) > self.limit:
            return self.reject(name, f"Supera el máximo de {self.limit // (1024 * 1024)} MB.")
        try:
            v = self.mod.validate(data, name)
        except InvalidImage as exc:
            return self.reject(name, str(exc))
        if v.sha256 in self.hashes:
            imp.duplicates += 1
            return
        if self.used + len(data) > self.quota:
            raise ApiError(413, "QUOTA_EXCEEDED", "Se alcanzó tu cuota de almacenamiento.")
        if self.used + len(data) - self.initial_used > self.byte_cap:  # el tamaño real superó lo anunciado por el dataset
            raise HfError("HF_TOO_BIG", f"Se superó el máximo de {self.settings.hf_import_max_gb:g} GB por importación.", 422)
        cid = self.category(label)
        if cid is None:
            return self.reject(name, "Demasiadas categorías en el proyecto.")
        try:
            stored = self.mod.save(self.project.id, v)
        except (OSError, SyntaxError, ValueError) as exc:  # una imagen que no se puede guardar no tira abajo toda la importación
            if getattr(exc, "errno", None) == errno.ENOSPC:  # pero un disco lleno sí: afecta a todas
                raise HfError("HF_NO_DISK", "El servidor se quedó sin espacio en disco.", 507) from exc
            return self.reject(name, f"No se pudo guardar la imagen ({type(exc).__name__}).")
        img = Image(
            project_id=self.project.id, category_id=cid, original_name=(name or f"{imp.processed}.{v.ext}")[:255],
            stored_name=stored, sha256=v.sha256, size_bytes=len(data), fmt=v.ext, **self.mod.item_fields(v),
        )
        try:
            with self.db.begin_nested():
                self.db.add(img)
        except IntegrityError:
            delete_image_files(img)
            imp.duplicates += 1
            return
        self.hashes.add(v.sha256)
        self.used += len(data)
        imp.imported += 1
        self.per_cat[label.lower()] = self.per_cat.get(label.lower(), 0) + 1
        self.unsaved.append(img)
        if (self.max_items and imp.imported >= self.max_items) or self._all_full():
            raise _LimitReached

    # ---- las dos formas de dataset
    def run_files(self, plan: Plan, token: str | None) -> None:
        imp = self.imp
        todo = [(p, sz) for p, sz in plan.files]
        imp.total = len(todo)
        self.phase("Descargando imágenes")
        self.checkpoint()

        def fetch(job):
            path, label = job
            err: Exception | None = None
            for attempt in range(3):  # un corte de red puntual no debe perder la imagen
                try:
                    local = hub_download(imp.repo_id, imp.revision, token, path, self.tmp / "dl")
                    data = local.read_bytes()
                    local.unlink(missing_ok=True)
                    return path, label, data
                except HfError as exc:
                    if exc.code not in TRANSIENT:
                        return path, label, exc
                    err = exc
                except Exception as exc:  # noqa: BLE001
                    err = exc
                time.sleep(1.5 * (attempt + 1))
            if isinstance(err, HfError):
                return path, label, err
            return path, label, HfError("HF_FILE", f"No se pudo descargar el archivo ({type(err).__name__}).", 502)

        with ThreadPoolExecutor(max_workers=DOWNLOAD_THREADS) as pool:
            for start in range(0, len(todo), DOWNLOAD_THREADS * 8):
                jobs = []
                for path, size in todo[start : start + DOWNLOAD_THREADS * 8]:
                    # lo que no se va a importar (sin categoría, filtrado, categoría completa, demasiado grande) no se descarga
                    label = label_from_path(path) or self.default
                    if label is None:
                        imp.processed += 1
                        imp.unlabeled += 1
                    elif self.decide(label) is None:
                        imp.processed += 1
                        imp.skipped += 1
                    elif size > self.limit:
                        imp.processed += 1
                        self.reject(path.rsplit("/", 1)[-1], f"Supera el máximo de {self.limit // (1024 * 1024)} MB.")
                    else:
                        jobs.append((path, label))
                for path, label, data in pool.map(fetch, jobs):
                    if isinstance(data, HfError):
                        if data.code != "HF_FILE":  # la cuenta, el dataset o la conexión fallan para todos: se detiene
                            raise data
                        imp.processed += 1
                        self.reject(path.rsplit("/", 1)[-1], data.message)
                    else:
                        self.ingest(data, path.rsplit("/", 1)[-1], label)
                    self.tick()
                self.tick()
                shutil.rmtree(self.tmp / "dl", ignore_errors=True)  # incluye los metadatos que deja huggingface_hub

    def run_parquet(self, plan: Plan, token: str | None) -> None:
        import pyarrow.parquet as pq

        imp, opts = self.imp, self.imp.options or {}
        img_col, lab_col, names = opts.get("image_column"), opts.get("label_column"), opts.get("class_names")
        phrase = self.phrase
        read_cols = [img_col] + ([phrase["column"]] if phrase else [lab_col] if lab_col else [])
        self.phase("Revisando las tablas")
        total = 0
        for path, _ in plan.files:
            total += hub_parquet_info(imp.repo_id, imp.revision, token, path).rows
            if total > self.settings.hf_import_max_items:
                raise HfError("HF_TOO_BIG", f"El dataset tiene más de {self.settings.hf_import_max_items:,} imágenes.".replace(",", "."), 422)
        imp.total = total
        self.checkpoint()
        self.tmp.mkdir(parents=True, exist_ok=True)
        for n, (path, size) in enumerate(plan.files, 1):
            if shutil.disk_usage(self.tmp).free < size * 2:
                raise HfError("HF_NO_DISK", "No hay espacio libre en el servidor para descargar este dataset.", 507)
            self.phase(f"Descargando tabla {n} de {len(plan.files)}")
            self.checkpoint()
            local = hub_download(imp.repo_id, imp.revision, token, path, self.tmp / "dl")
            try:
                pf = pq.ParquetFile(local)
                biggest = max((pf.metadata.row_group(g).total_byte_size for g in range(pf.num_row_groups)), default=0)
                if biggest > MAX_ROW_GROUP_BYTES:
                    raise HfError(
                        "HF_TOO_BIG",
                        f"Este dataset guarda las filas en bloques de {fmt_bytes(biggest)}, demasiado grandes para importarlos aquí.", 422,
                    )
                cols = set(pf.schema_arrow.names)
                if not set(read_cols) <= cols:
                    imp.processed += pf.metadata.num_rows
                    self.reject(path, "La tabla no tiene las columnas elegidas.")
                    continue
                self.phase(f"Procesando tabla {n} de {len(plan.files)}")
                for batch in pf.iter_batches(batch_size=32, columns=read_cols):
                    for row in batch.to_pylist():
                        cell = row[img_col]
                        data = cell.get("bytes") if isinstance(cell, dict) else cell
                        if phrase:
                            label = hf_labels.extract_phrase(row.get(phrase["column"]), phrase.get("after", ""), phrase.get("words"), phrase.get("until"))
                        else:
                            label = class_of(row.get(lab_col), names) if lab_col else None
                        raw_name = (cell.get("path") if isinstance(cell, dict) else None) or ""
                        self.ingest(data, raw_name.rsplit("/", 1)[-1] or f"{imp.processed + 1}", label)
                        self.tick()
            finally:
                shutil.rmtree(self.tmp / "dl", ignore_errors=True)


def run_import(import_id: int) -> None:
    """Punto de entrada del worker (tarea `vision.hf_import`)."""
    with SessionLocal() as db:
        imp = db.get(HfImport, import_id)
        if imp is None or imp.status != HF_PENDING:  # cancelada antes de empezar, o ya recogida por otro worker
            return
        project, user = db.get(Project, imp.project_id), db.get(User, imp.user_id)
        imp.status, imp.started_at, imp.heartbeat_at, imp.phase = HF_RUNNING, utcnow(), utcnow(), "Preparando"
        db.commit()
        runner = _Runner(db, imp, project, user)
        token: str | None = None
        outcome = HF_COMPLETED
        try:
            token = token_for(db, user.id)
            ensure_modifications_open(db, project, user)
            plan = _replan(imp, token)
            if imp.options.get("layout") == "files":
                runner.run_files(plan, token)
            else:
                runner.run_parquet(plan, token)
            imp.phase = "Listo"
        except _LimitReached:
            imp.phase = "Listo (se alcanzó el máximo pedido)"
        except _Cancelled:
            outcome, imp.phase = HF_CANCELLED, "Cancelada"
        except (HfError, ApiError) as exc:
            outcome, imp.error = HF_ERROR, _scrub(getattr(exc, "message", str(exc)), token)
            imp.phase = "Detenida"
        except Exception as exc:  # noqa: BLE001 - un fallo inesperado no debe dejar la importación «en curso» para siempre
            log.exception("Fallo inesperado en la importación %s", import_id)
            db.rollback()  # vuelve al último guardado; lo que no se guardó dejó archivos sin fila y se borran
            for img in runner.unsaved:
                delete_image_files(img)
            runner.unsaved.clear()
            outcome, imp.phase = HF_ERROR, "Detenida"
            imp.error = _scrub(f"Error inesperado ({type(exc).__name__}). Revisa los registros del servidor.", token)
        finally:
            shutil.rmtree(runner.tmp, ignore_errors=True)
        _finish(db, imp, project, user, outcome, runner)


def _scrub(message: str, token: str | None) -> str:
    return message.replace(token, "hf_…") if token else message


def _replan(imp: HfImport, token: str | None) -> Plan:
    """Vuelve a listar los archivos de la versión fijada al crear la importación (no se guarda la lista)."""
    files = [(p, sz) for p, sz in hub_list_files(imp.repo_id, imp.revision, token) if not _hidden(p)]
    if imp.options.get("layout") == "files":
        chosen = sorted((p, sz) for p, sz in files if _ext(p) in IMG_EXT)
    else:
        chosen = sorted((p, sz) for p, sz in files if _ext(p) == "parquet")
    if not chosen:
        raise HfError("HF_NO_IMAGES", "El dataset ya no tiene los archivos esperados.", 422)
    return Plan(imp.repo_id, imp.revision, False, False, imp.options["layout"], chosen, sum(sz for _, sz in chosen), len(chosen))


def _finish(db: Session, imp: HfImport, project: Project, user: User, outcome: str, runner: _Runner) -> None:
    imp.status, imp.finished_at, imp.heartbeat_at = outcome, utcnow(), utcnow()
    imp.rejected_sample = runner.rejected_sample[:10]
    imp.total_bytes = runner.used - runner.initial_used
    detail = {
        "repo": imp.repo_id, "status": outcome, "imported": imp.imported, "duplicates": imp.duplicates,
        "rejected": imp.rejected, "unlabeled": imp.unlabeled,
    }
    audit(db, "hf.import_finished", user, "project", project.id, detail)
    if outcome == HF_COMPLETED:
        notify_user(
            db, user, "hf.import_done", "Importación de Hugging Face terminada",
            f"{imp.imported} imágenes importadas a «{project.name}» desde {imp.repo_id}"
            + (f" ({imp.duplicates} repetidas omitidas)." if imp.duplicates else "."),
            {"project_id": project.id, "import_id": imp.id}, email=False,
        )
    elif outcome == HF_ERROR:
        notify_user(
            db, user, "hf.import_failed", "La importación de Hugging Face se detuvo",
            f"«{imp.repo_id}» en «{project.name}»: {imp.error}", {"project_id": project.id, "import_id": imp.id}, email=False,
        )
    db.commit()
