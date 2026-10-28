from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    database_url: str = "postgresql+psycopg://vision:vision@db:5432/vision"
    redis_url: str = "redis://redis:6379/0"
    secret_key: str = ""
    data_dir: str = "/data"
    timezone: str = "America/Santiago"

    cookie_secure: bool = False
    session_hours: int = 12
    allow_registration: bool = True
    api_docs: bool = False  # /api/docs (Swagger); activar sólo en desarrollo

    default_user_quota_mb: int = 102400
    max_upload_mb: int = 15
    max_audio_upload_mb: int = 50
    max_audio_seconds: int = 600  # se usan como máximo los primeros 10 min de cada audio
    max_clips_per_audio: int = 20  # un audio largo aporta como máximo 20 clips al entrenamiento
    max_tabular_upload_mb: int = 50
    max_tabular_rows: int = 500_000  # filas por proyecto de datos tabulares (entre todos sus CSV)
    max_tabular_columns: int = 300
    max_tabular_predict_rows: int = 1000  # filas por consulta en «Probar»
    max_image_pixels: int = 50_000_000
    min_images_per_class: int = 10
    low_confidence_threshold: float = 0.6

    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    smtp_starttls: bool = True
    public_url: str = "http://localhost:8080"

    pretrained_weights: str = "imagenet"  # "imagenet" | "none" (none solo para pruebas)
    train_threads: int = 0
    heartbeat_stale_seconds: int = 180
    max_job_attempts: int = 3
    total_ram_limit_gb: float = 20.0

    # Entrenamiento nocturno de lunes a viernes (hora local): modelos cortos que caben en la ventana
    night_training_enabled: bool = True
    night_window_start: str = "00:00"
    night_window_end: str = "08:00"
    night_max_job_hours: float = 4.0  # sólo modelos cuya duración estimada no supera esto
    night_activity_minutes: int = 15  # actividad más reciente que esto → se pregunta antes de congelar
    night_ask_timeout_minutes: int = 20  # sin respuesta: se decide según si siguió cargando

    login_max_attempts: int = 8
    login_window_seconds: int = 900
    register_max_attempts: int = 5
    register_window_seconds: int = 3600

    # Tokens de API personales (Authorization: Bearer vl_...)
    api_max_read_tokens: int = 5  # tokens de lectura activos por usuario
    api_max_write_tokens: int = 1  # tokens con escritura activos por usuario (cada uno requiere aprobación)
    api_token_rate_per_minute: int = 300  # peticiones por minuto y por token
    api_token_fail_max: int = 30  # tokens inválidos por IP dentro de login_window_seconds
    api_write_default_days: int = 90  # vigencia propuesta al aprobar un token con escritura
    api_write_max_days: int = 365

    # Importación de datasets desde Hugging Face
    # Clave Fernet (base64 de 32 bytes) con la que se cifran los tokens de HF de los usuarios. Si está vacía se
    # deriva de SECRET_KEY: funciona igual, pero rotar SECRET_KEY obligaría a reconectar las cuentas.
    hf_encryption_key: str = ""
    hf_import_max_gb: float = 5.0  # tamaño máximo de una importación
    hf_import_max_items: int = 200_000  # imágenes por importación
    hf_import_max_categories: int = 2000
    hf_import_active_per_user: int = 1  # importaciones simultáneas (pendientes + en curso) por usuario
    hf_import_stale_seconds: int = 1800  # sin señal de vida durante este tiempo = el worker murió (una tabla grande tarda en bajar)


@lru_cache
def get_settings() -> Settings:
    return Settings()


def check_secret(settings: Settings) -> None:
    key = settings.secret_key
    if len(key) < 32 or key.startswith("CHANGE_ME"):
        raise RuntimeError(
            "SECRET_KEY no configurada. Ejecuta scripts/init-env.sh (o .ps1) para generar .env."
        )
