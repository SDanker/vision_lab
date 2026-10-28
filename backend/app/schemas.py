import re
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator

from .security import password_error

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")


class RegisterIn(BaseModel):
    first_name: str = Field(min_length=1, max_length=80)
    last_name: str = Field(min_length=1, max_length=80)
    email: EmailStr
    username: str
    password: str
    password_confirm: str
    reason: str | None = Field(default=None, max_length=500)

    @field_validator("first_name", "last_name")
    @classmethod
    def strip_names(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("No puede estar vacío.")
        return v

    @field_validator("username")
    @classmethod
    def check_username(cls, v: str) -> str:
        if not USERNAME_RE.match(v):
            raise ValueError("El usuario debe tener 3-32 caracteres: letras, números, . _ -")
        return v

    @model_validator(mode="after")
    def check_passwords(self):
        if self.password != self.password_confirm:
            raise ValueError("Las contraseñas no coinciden.")
        err = password_error(self.password)
        if err:
            raise ValueError(err)
        return self


class LoginIn(BaseModel):
    identifier: str = Field(min_length=1, max_length=255)
    password: str = Field(min_length=1, max_length=256)


class ReviewIn(BaseModel):
    note: str | None = Field(default=None, max_length=500)


class BulkApproveIn(BaseModel):
    user_ids: list[int] = Field(min_length=1, max_length=500)
    confirm: bool = False


class UserPatchIn(BaseModel):
    role: Literal["USER", "ADMIN"] | None = None
    quota_mb: int | None = Field(default=None, ge=0, le=10_000_000)
    clear_quota: bool = False


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    first_name: str
    last_name: str
    email: str
    username: str
    role: str
    status: str
    reason: str | None
    quota_mb: int | None
    created_at: datetime
    reviewed_at: datetime | None
    review_note: str | None
    reviewed_by: str | None = None


class TrainingConfig(BaseModel):
    epochs: int = Field(default=15, ge=1, le=100)
    fine_tune_epochs: int = Field(default=5, ge=0, le=50)
    batch_size: int = Field(default=32, ge=2, le=64)
    learning_rate: float = Field(default=1e-3, gt=0, le=0.1)
    patience: int = Field(default=3, ge=1, le=20)
    augmentation: bool = True
    # lado de la imagen de entrada; más resolución distingue mejor detalles finos pero entrena más lento
    image_size: Literal[160, 224, 260, 300, 384] = 224
    # sólo proyectos de audio: duración de cada clip (los audios largos se cortan en varios clips)
    clip_seconds: float = Field(default=5.0, ge=1.0, le=10.0)
    # sólo proyectos de datos tabulares: columna a predecir y columnas de entrada (None = las sugeridas)
    tabular_target: str | None = Field(default=None, max_length=120)
    tabular_features: list[str] | None = Field(default=None, max_length=300)
    # configuración avanzada: parámetros propios de un modelo base (clave = id del modelo).
    # Lo que no se indica en un modelo se hereda de los valores generales de arriba.
    model_overrides: dict[str, "ModelOverride"] = Field(default_factory=dict)

    def to_dict(self) -> dict:
        """Dict listo para guardar: sin campos vacíos dentro de los ajustes por modelo.

        Las claves de datos tabulares sólo se guardan si tienen valor: así la configuración de los proyectos de
        imágenes y audio no cambia (y no se les marca el modelo como «modificado»).
        """
        d = self.model_dump(exclude={"model_overrides", "tabular_target", "tabular_features"})
        for key in ("tabular_target", "tabular_features"):
            if getattr(self, key) is not None:
                d[key] = getattr(self, key)
        overrides = {k: v.model_dump(exclude_none=True) for k, v in self.model_overrides.items()}
        d["model_overrides"] = {k: v for k, v in overrides.items() if v}
        return d

    @field_validator("model_overrides")
    @classmethod
    def check_override_models(cls, v: dict) -> dict:
        from .ml.architectures import all_model_ids

        bad = [m for m in v if m not in all_model_ids()]
        if bad:
            raise ValueError(f"Modelo base no soportado en la configuración avanzada: {', '.join(bad)}")
        return v


class ModelOverride(BaseModel):
    """Mismos parámetros que TrainingConfig, todos opcionales: None = usar el valor general."""

    epochs: int | None = Field(default=None, ge=1, le=100)
    fine_tune_epochs: int | None = Field(default=None, ge=0, le=50)
    batch_size: int | None = Field(default=None, ge=2, le=64)
    learning_rate: float | None = Field(default=None, gt=0, le=0.1)
    patience: int | None = Field(default=None, ge=1, le=20)
    augmentation: bool | None = None
    image_size: Literal[160, 224, 260, 300, 384] | None = None
    clip_seconds: float | None = Field(default=None, ge=1.0, le=10.0)
    # ajuste fino: cuántas capas finales del modelo base se descongelan y con qué tasa (learning_rate × factor).
    # Vacío = el valor propio de la arquitectura (ver ml/architectures.py).
    unfreeze_layers: int | None = Field(default=None, ge=1, le=1000)
    fine_tune_lr_factor: float | None = Field(default=None, gt=0, le=1)


TrainingConfig.model_rebuild()


def _check_models(v: list[str] | None) -> list[str] | None:
    if v is None:
        return v
    from .ml.architectures import all_model_ids

    out = list(dict.fromkeys(v))
    if not out:
        raise ValueError("Selecciona al menos un modelo base.")
    bad = [m for m in out if m not in all_model_ids()]
    if bad:
        raise ValueError(f"Modelo base no soportado: {', '.join(bad)}")
    return out


def check_models_fit(models: list[str], tabular: bool) -> None:
    """Los modelos de scikit-learn sólo sirven para datos tabulares y las redes de visión para el resto."""
    from .ml.architectures import is_tabular_model

    wrong = [m for m in models if is_tabular_model(m) != tabular]
    if wrong:
        kind = "datos tabulares" if tabular else "imágenes o audio"
        raise ValueError(f"Estos modelos no sirven para proyectos de {kind}: {', '.join(wrong)}.")


class ProjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)
    modality: str = "image"  # tipo de dato (ver app/modalities); no se puede cambiar después de crear el proyecto
    base_model: str | None = None
    # uno o más modelos: cada entrenamiento genera una versión por modelo con el mismo dataset
    base_models: list[str] | None = Field(default=None, max_length=11)
    config: TrainingConfig = TrainingConfig()
    auto_train: bool = True
    categories: list[str] = Field(default_factory=list, max_length=100)

    @field_validator("name")
    @classmethod
    def strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("El nombre no puede estar vacío.")
        return v

    @field_validator("modality")
    @classmethod
    def check_modality(cls, v: str) -> str:
        from .modalities import REGISTRY

        if v not in REGISTRY:
            raise ValueError(f"Tipo de dato no soportado. Opciones: {', '.join(REGISTRY)}.")
        return v

    @model_validator(mode="after")
    def resolve_models(self):
        from .ml.architectures import DEFAULT_ARCH
        from .ml.tabular import DEFAULT_TABULAR_MODEL
        from .modalities import REGISTRY

        tabular = REGISTRY[self.modality].info.tabular
        default = DEFAULT_TABULAR_MODEL if tabular else DEFAULT_ARCH
        self.base_models = _check_models(self.base_models or [self.base_model or default])
        check_models_fit(self.base_models, tabular)
        self.base_model = self.base_models[0]
        return self


class ProjectPatchIn(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)
    base_model: str | None = None
    base_models: list[str] | None = Field(default=None, max_length=11)
    config: TrainingConfig | None = None
    auto_train: bool | None = None

    @model_validator(mode="after")
    def resolve_models(self):
        if self.base_models is None and self.base_model is not None:
            self.base_models = [self.base_model]
        self.base_models = _check_models(self.base_models)
        if self.base_models:
            self.base_model = self.base_models[0]
        return self


class CategoryIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)

    @field_validator("name")
    @classmethod
    def strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("El nombre no puede estar vacío.")
        return v


class ImageMoveIn(BaseModel):
    category_id: int


class MemberIn(BaseModel):
    identifier: str = Field(min_length=1, max_length=255)  # usuario o correo
    role: Literal["viewer", "editor"] = "editor"


class MemberPatchIn(BaseModel):
    role: Literal["viewer", "editor"]


class EnqueueIn(BaseModel):
    project_ids: list[int] = Field(default_factory=list, max_length=500)
    all: bool = False

    @model_validator(mode="after")
    def need_target(self):
        if not self.all and not self.project_ids:
            raise ValueError("Indica al menos un proyecto o usa all=true.")
        return self


class TokenCreateIn(BaseModel):
    """Token de lectura: lo crea el propio usuario."""

    name: str = Field(min_length=1, max_length=80)
    expires_days: int | None = Field(default=None, ge=1, le=730)  # None = no vence

    @field_validator("name")
    @classmethod
    def strip_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("El nombre no puede estar vacío.")
        return v


class TokenRequestIn(BaseModel):
    """Solicitud de un token con escritura."""

    name: str = Field(min_length=1, max_length=80)
    reason: str = Field(min_length=5, max_length=500)

    @field_validator("name", "reason")
    @classmethod
    def strip_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("No puede estar vacío.")
        return v


class HfTokenIn(BaseModel):
    token: str = Field(min_length=10, max_length=200)

    @field_validator("token")
    @classmethod
    def check_token(cls, v: str) -> str:
        from .services.hf import valid_token_format

        v = v.strip()
        if not valid_token_format(v):
            raise ValueError("Ese no parece un token de Hugging Face (empieza con «hf_»). Créalo en huggingface.co/settings/tokens.")
        return v


class HfPhraseIn(BaseModel):
    """Sacar la categoría de un texto: lo que sigue a una frase (N palabras, o hasta un delimitador)."""

    column: str = Field(min_length=1, max_length=120)
    after: str = Field(min_length=1, max_length=80)
    words: int | None = Field(default=1, ge=1, le=6)
    until: str | None = Field(default=None, max_length=40)

    @field_validator("after")
    @classmethod
    def strip_after(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Escribe la frase que antecede al valor.")
        return v

    @field_validator("until")
    @classmethod
    def blank_until(cls, v: str | None) -> str | None:
        return v if v and v.strip() else None  # el delimitador puede llevar espacios (« with the »): no se recorta

    @model_validator(mode="after")
    def need_a_cut(self):
        if self.until is None and self.words is None:
            self.words = 1
        return self


class HfSourceIn(BaseModel):
    """Dataset a revisar o importar y qué hacer con él: «usuario/nombre» o su dirección en huggingface.co."""

    source: str = Field(min_length=1, max_length=300)
    image_column: str | None = Field(default=None, max_length=120)
    label_column: str | None = Field(default=None, max_length=120)
    default_category: str | None = Field(default=None, max_length=80)  # destino de lo que no trae etiqueta
    # de dónde sale la categoría: la que declare el dataset, una columna, un texto o ninguna
    label_source: Literal["auto", "column", "phrase", "none"] = "auto"
    label_phrase: HfPhraseIn | None = None
    # filtros y máximos
    include: list[Annotated[str, Field(min_length=1, max_length=80)]] | None = Field(default=None, max_length=2000)
    rest: Literal["skip", "other"] = "skip"  # qué pasa con lo no incluido: omitirlo o juntarlo en other_label
    other_label: str | None = Field(default=None, max_length=80)
    max_items: int | None = Field(default=None, ge=1, le=10_000_000)
    max_per_category: int | None = Field(default=None, ge=1, le=10_000_000)

    @model_validator(mode="after")
    def check_source(self):
        if self.label_source == "phrase" and self.label_phrase is None:
            raise ValueError("Indica la columna de texto y la frase de la que sale la categoría.")
        if self.label_source == "column" and not self.label_column:
            raise ValueError("Indica la columna de la que sale la categoría.")
        return self


class TokenApproveIn(BaseModel):
    note: str | None = Field(default=None, max_length=500)
    expires_days: int | None = Field(default=None, ge=1, le=365)  # None = vigencia predeterminada
