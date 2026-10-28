from datetime import datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base, UTCDateTime, utcnow

# Estados de usuario
PENDING, APPROVED, REJECTED, SUSPENDED = "PENDING", "APPROVED", "REJECTED", "SUSPENDED"
USER_STATUSES = (PENDING, APPROVED, REJECTED, SUSPENDED)
ROLE_USER, ROLE_ADMIN = "USER", "ADMIN"

# Tokens de API: alcance y estados de la solicitud de escritura
TOKEN_READ, TOKEN_WRITE = "read", "write"
REQ_PENDING, REQ_APPROVED, REQ_REJECTED, REQ_CANCELLED = "pending", "approved", "rejected", "cancelled"
REQ_STATUSES = (REQ_PENDING, REQ_APPROVED, REQ_REJECTED, REQ_CANCELLED)

# Rol de un usuario dentro de un proyecto compartido (el propietario es "owner", sólo derivado)
MEMBER_VIEWER, MEMBER_EDITOR = "viewer", "editor"
MEMBER_ROLES = (MEMBER_VIEWER, MEMBER_EDITOR)

# Importaciones de datasets desde Hugging Face
HF_PENDING, HF_RUNNING, HF_COMPLETED, HF_ERROR, HF_CANCELLED = "pending", "running", "completed", "error", "cancelled"
HF_ACTIVE = (HF_PENDING, HF_RUNNING)

# Estados de entrenamiento
J_PENDING, J_PREPARING, J_TRAINING, J_VALIDATING, J_EXPORTING = (
    "pending",
    "preparing",
    "training",
    "validating",
    "exporting",
)
J_COMPLETED, J_ERROR, J_CANCELLED = "completed", "error", "cancelled"
ACTIVE_JOB_STATES = (J_PENDING, J_PREPARING, J_TRAINING, J_VALIDATING, J_EXPORTING)
RUNNING_JOB_STATES = (J_PREPARING, J_TRAINING, J_VALIDATING, J_EXPORTING)
TERMINAL_JOB_STATES = (J_COMPLETED, J_ERROR, J_CANCELLED)

from .ml.architectures import ARCHITECTURES as _ARCH  # noqa: E402

BASE_MODELS = tuple(_ARCH)
EXPORT_FORMATS = ("keras", "h5", "savedmodel", "tflite", "onnx")  # imágenes y audio
TABULAR_EXPORT_FORMATS = ("onnx", "joblib")  # datos tabulares (scikit-learn)
ALL_EXPORT_FORMATS = (*EXPORT_FORMATS, "joblib")


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    first_name: Mapped[str] = mapped_column(String(80))
    last_name: Mapped[str] = mapped_column(String(80))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    username: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(10), default=ROLE_USER)
    status: Mapped[str] = mapped_column(String(10), default=PENDING, index=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    token_version: Mapped[int] = mapped_column(Integer, default=0)
    quota_mb: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    reviewed_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    reviewed_by_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    review_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_login_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)

    reviewed_by: Mapped["User | None"] = relationship(remote_side=[id], lazy="joined")

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"


class UserStatusHistory(Base):
    __tablename__ = "user_status_history"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    from_status: Mapped[str | None] = mapped_column(String(10), nullable=True)
    to_status: Mapped[str] = mapped_column(String(10))
    actor_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    actor: Mapped[User | None] = relationship(foreign_keys=[actor_id], lazy="joined")


class ApiToken(Base):
    """Token personal para usar la API sin cookie. Sólo se guarda su hash; el valor se muestra una vez."""

    __tablename__ = "api_tokens"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(80))
    prefix: Mapped[str] = mapped_column(String(16))  # primeros caracteres, para reconocerlo en listados
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    scope: Mapped[str] = mapped_column(String(5), default=TOKEN_READ)  # read | write
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    last_used_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    last_used_ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    revoked_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    user: Mapped[User] = relationship(foreign_keys=[user_id], lazy="joined")

    def is_active(self, now: datetime) -> bool:
        return self.revoked_at is None and (self.expires_at is None or self.expires_at > now)


class ApiTokenRequest(Base):
    """Solicitud de un token con escritura: el administrador la aprueba y luego el usuario lo genera."""

    __tablename__ = "api_token_requests"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(80))
    reason: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(10), default=REQ_PENDING, index=True)
    expires_days: Mapped[int | None] = mapped_column(Integer, nullable=True)  # vigencia fijada al aprobar
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    reviewed_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    reviewed_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    review_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    token_id: Mapped[int | None] = mapped_column(Integer, nullable=True)  # token generado a partir de esta solicitud

    user: Mapped[User] = relationship(foreign_keys=[user_id], lazy="joined")
    reviewed_by: Mapped[User | None] = relationship(foreign_keys=[reviewed_by_id], lazy="joined")


class HfConnection(Base):
    """Cuenta de Hugging Face de un usuario. El token se guarda cifrado (services/vault.py) y nunca sale del servidor."""

    __tablename__ = "hf_connections"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), unique=True, index=True)
    token_enc: Mapped[str] = mapped_column(Text)
    token_hint: Mapped[str] = mapped_column(String(16))  # «hf_…abcd», para reconocerlo en la pantalla
    hf_username: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    last_used_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)


class HfImport(Base):
    """Descarga de un dataset de Hugging Face hacia un proyecto (corre en segundo plano en el worker liviano)."""

    __tablename__ = "hf_imports"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    repo_id: Mapped[str] = mapped_column(String(200))
    revision: Mapped[str] = mapped_column(String(80))  # commit fijado al crearla: el contenido no cambia a mitad de camino
    status: Mapped[str] = mapped_column(String(10), default=HF_PENDING, index=True)
    phase: Mapped[str | None] = mapped_column(String(200), nullable=True)
    options: Mapped[dict] = mapped_column(JSON, default=dict)  # layout, columnas elegidas, cuántos elementos se esperan
    total: Mapped[int] = mapped_column(Integer, default=0)  # elementos a procesar (0 = aún no se sabe)
    processed: Mapped[int] = mapped_column(Integer, default=0)
    imported: Mapped[int] = mapped_column(Integer, default=0)
    duplicates: Mapped[int] = mapped_column(Integer, default=0)
    rejected: Mapped[int] = mapped_column(Integer, default=0)
    unlabeled: Mapped[int] = mapped_column(Integer, default=0)
    skipped: Mapped[int] = mapped_column(Integer, default=0, server_default="0")  # fuera por los filtros o por los máximos
    total_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    rejected_sample: Mapped[list | None] = mapped_column(JSON, nullable=True)  # primeros motivos de rechazo
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)

    user: Mapped[User] = relationship(foreign_keys=[user_id], lazy="joined")


class Notification(Base):
    __tablename__ = "notifications"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(40))
    title: Mapped[str] = mapped_column(String(200))
    body: Mapped[str] = mapped_column(Text, default="")
    read: Mapped[bool] = mapped_column(Boolean, default=False)
    meta: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    actor_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    actor_username: Mapped[str | None] = mapped_column(String(32), nullable=True)
    action: Mapped[str] = mapped_column(String(60), index=True)
    target_type: Mapped[str | None] = mapped_column(String(40), nullable=True)
    target_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    detail: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(120))
    description: Mapped[str] = mapped_column(Text, default="")
    base_model: Mapped[str] = mapped_column(String(20), default="mobilenetv3")
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    auto_train: Mapped[bool] = mapped_column(Boolean, default=True)
    modality: Mapped[str] = mapped_column(String(10), default="image", server_default="image")  # id de app/modalities (image, audio…)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    # última modificación del dataset o la configuración: el planificador nocturno no entrena a ciegas
    # un proyecto en el que alguien está trabajando (le pregunta antes)
    last_activity_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    last_activity_by_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    owner: Mapped[User] = relationship(foreign_keys=[owner_id], lazy="joined")
    categories: Mapped[list["Category"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", order_by="Category.name"
    )
    members: Mapped[list["ProjectMember"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (UniqueConstraint("owner_id", "name", name="uq_project_owner_name"),)


class ProjectMember(Base):
    """Usuario con el que se comparte un proyecto. El propietario no aparece aquí."""

    __tablename__ = "project_members"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(10), default=MEMBER_EDITOR)  # viewer | editor
    added_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    project: Mapped[Project] = relationship(back_populates="members")
    user: Mapped[User] = relationship(foreign_keys=[user_id], lazy="joined")

    __table_args__ = (UniqueConstraint("project_id", "user_id", name="uq_member_project_user"),)


class Category(Base):
    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    project: Mapped[Project] = relationship(back_populates="categories")

    __table_args__ = (UniqueConstraint("project_id", "name", name="uq_category_project_name"),)


class Image(Base):
    __tablename__ = "images"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    category_id: Mapped[int] = mapped_column(
        ForeignKey("categories.id", ondelete="CASCADE"), index=True
    )
    original_name: Mapped[str] = mapped_column(String(255))
    stored_name: Mapped[str] = mapped_column(String(80))
    sha256: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    fmt: Mapped[str] = mapped_column(String(10))
    duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)  # sólo audio
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    __table_args__ = (UniqueConstraint("project_id", "sha256", name="uq_image_project_hash"),)


class WeeklyRun(Base):
    __tablename__ = "weekly_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    week_key: Mapped[str] = mapped_column(String(40), unique=True)
    trigger: Mapped[str] = mapped_column(String(20), default="schedule")
    status: Mapped[str] = mapped_column(String(20), default="running")  # running | published
    started_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    published_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class DatasetSnapshot(Base):
    __tablename__ = "dataset_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    run_id: Mapped[int | None] = mapped_column(
        ForeignKey("weekly_runs.id", ondelete="SET NULL"), nullable=True
    )
    dataset_hash: Mapped[str] = mapped_column(String(64))
    class_counts: Mapped[dict] = mapped_column(JSON)
    excluded: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    total_images: Mapped[int] = mapped_column(Integer)
    path: Mapped[str] = mapped_column(String(300))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    run_id: Mapped[int | None] = mapped_column(
        ForeignKey("weekly_runs.id", ondelete="SET NULL"), nullable=True
    )
    snapshot_id: Mapped[int | None] = mapped_column(
        ForeignKey("dataset_snapshots.id", ondelete="SET NULL"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(12), default=J_PENDING, index=True)
    phase: Mapped[str | None] = mapped_column(String(40), nullable=True)
    base_model: Mapped[str] = mapped_column(String(20))
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    trigger: Mapped[str] = mapped_column(String(20), default="schedule")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    epoch: Mapped[int] = mapped_column(Integer, default=0)
    total_epochs: Mapped[int] = mapped_column(Integer, default=0)
    percent: Mapped[float] = mapped_column(Float, default=0.0)
    eta_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    history: Mapped[list | None] = mapped_column(JSON, nullable=True)
    cpu_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    ram_mb: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_version_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # duración estimada al encolar (services/estimates.py) y orden en la cola: menor = antes
    estimated_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    queue_order: Mapped[float] = mapped_column(Float, default=0.0, server_default="0")

    project: Mapped[Project] = relationship(lazy="joined")

    __table_args__ = (Index("ix_jobs_project_created", "project_id", "created_at"),)


# Pregunta nocturna «¿todavía te faltan cargas?»
NC_ASKING, NC_GO, NC_SKIP = "asking", "go", "skip"


class NightCheck(Base):
    """Se crea cuando un proyecto que cabe en la noche tuvo actividad reciente: se pregunta antes de congelarlo."""

    __tablename__ = "night_checks"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    night: Mapped[str] = mapped_column(String(10))  # fecha local de la noche (AAAA-MM-DD)
    status: Mapped[str] = mapped_column(String(10), default=NC_ASKING)
    asked_user_ids: Mapped[list] = mapped_column(JSON, default=list)
    answered_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    reason: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    answered_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)

    project: Mapped[Project] = relationship(lazy="joined")

    __table_args__ = (UniqueConstraint("project_id", "night", name="uq_night_check_project_night"),)


class NightReport(Base):
    """Resumen de una noche hábil: lo que se planificó en la primera pasada, también cuando no se entrenó nada.

    Los resultados (qué se encoló, terminó o se pospuso) se leen en vivo de las ejecuciones nocturnas.
    """

    __tablename__ = "night_reports"

    id: Mapped[int] = mapped_column(primary_key=True)
    night: Mapped[str] = mapped_column(String(10), unique=True)  # fecha local (AAAA-MM-DD)
    capacity_seconds: Mapped[float] = mapped_column(Float, default=0)
    note: Mapped[str | None] = mapped_column(String(300), nullable=True)
    items: Mapped[list] = mapped_column(JSON, default=list)
    not_ready: Mapped[list] = mapped_column(JSON, default=list)
    notified: Mapped[list] = mapped_column(JSON, default=list)  # "proyecto:modelo" avisados al dueño
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class TrainingLog(Base):
    __tablename__ = "training_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(
        ForeignKey("training_jobs.id", ondelete="CASCADE"), index=True
    )
    level: Mapped[str] = mapped_column(String(10), default="info")
    message: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    job_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    snapshot_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    version: Mapped[int] = mapped_column(Integer)
    base_model: Mapped[str] = mapped_column(String(20))
    labels: Mapped[list] = mapped_column(JSON)
    metrics: Mapped[dict] = mapped_column(JSON)
    config: Mapped[dict] = mapped_column(JSON)
    dataset_hash: Mapped[str] = mapped_column(String(64))
    dir_path: Mapped[str] = mapped_column(String(300))
    published: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    exports: Mapped[list["ModelExport"]] = relationship(
        back_populates="model", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (UniqueConstraint("project_id", "version", name="uq_model_project_version"),)


class ModelExport(Base):
    __tablename__ = "model_exports"

    id: Mapped[int] = mapped_column(primary_key=True)
    model_id: Mapped[int] = mapped_column(
        ForeignKey("model_versions.id", ondelete="CASCADE"), index=True
    )
    fmt: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(10))  # ok | failed
    filename: Mapped[str | None] = mapped_column(String(120), nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    validation: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    model: Mapped[ModelVersion] = relationship(back_populates="exports")
