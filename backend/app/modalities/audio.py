"""Audio (WAV, MP3, OGG, FLAC, M4A, AAC, OPUS, WEBM): cada clip se convierte en un espectrograma log-mel."""
import shutil
from pathlib import Path

from ..storage import AUDIO_EXT, AUDIO_MEDIA, InvalidImage, check_audio_payload, save_audio_files, validate_audio_bytes
from .base import InvalidInput, Modality, ModalityInfo


class AudioModality(Modality):
    info = ModalityInfo(
        id="audio", label="Audio", icon="🎵",
        description="Clasifica sonidos (WAV, MP3, OGG, FLAC, M4A…) mediante espectrogramas.",
        unit="audios", unit_one="audio", extensions=("wav", "mp3", "ogg", "flac", "m4a", "aac", "opus", "webm"), accept=".wav,.mp3,.ogg,.flac,.m4a,.aac,.opus,.webm,audio/*",
        upload_help="WAV, MP3, OGG, FLAC, M4A, AAC, OPUS o WEBM. Los audios largos se cortan en clips para entrenar.",
        preview="audio", windowed=True,
        window_label="Duración de cada clip (segundos)",
        window_hint="Los audios se cortan en clips de esta duración; cada clip es una muestra de entrenamiento. 2–5 s suele funcionar bien.",
        resolution_label="Resolución del espectrograma",
        resolution_hint="Tamaño de la imagen del espectrograma. 224 es suficiente en la mayoría de los casos.",
        test_hint=" Cada audio se analiza por segmentos y se promedia.",
    )
    media = AUDIO_MEDIA
    augmentation = "spec"

    def max_upload_mb(self, settings) -> int:
        return settings.max_audio_upload_mb

    def validate(self, data: bytes, filename: str):
        return validate_audio_bytes(data, filename)

    def save(self, project_id: int, v) -> str:
        return save_audio_files(project_id, v)

    def item_fields(self, v) -> dict:
        return {"width": 0, "height": 0, "duration_s": v.duration}

    def job_config(self, cfg: dict) -> dict:
        return {**cfg, "modality": "audio"}

    def write_split(self, items, out_dir: Path, name: str, progress, cfg: dict, settings) -> int:
        from ..ml import data
        from ..ml.preprocess import IMG_SIZE

        return data.write_split_audio(
            items, out_dir, name, progress, size=int(cfg.get("image_size", IMG_SIZE)),
            clip_seconds=float(cfg.get("clip_seconds", 5.0)), max_clips=settings.max_clips_per_audio,
        )

    def dataset_note(self, cfg: dict) -> str:
        return f" Cada audio se corta en clips de {float(cfg.get('clip_seconds', 5.0)):g} s; cada clip es una muestra."

    def metadata(self, cfg: dict) -> dict:
        from ..ml import audio_features as af

        return {
            "modality": "audio",
            "audio": {
                "sample_rate": af.SAMPLE_RATE, "clip_seconds": float(cfg.get("clip_seconds", 5.0)), "n_fft": af.N_FFT,
                "win_length": af.WIN_LENGTH, "hop_length": af.HOP_LENGTH, "n_mels": af.N_MELS, "fmin": af.FMIN,
                "fmax": af.FMAX, "top_db": af.TOP_DB,
                "description": "audio mono 16 kHz -> clips de clip_seconds -> log-mel (dB, top_db) -> imagen uint8 NxN x3; "
                               "la predicción de un audio largo es el promedio de sus clips (ver audio_features.py)",
            },
        }

    def export_assets(self, out_dir: Path) -> None:
        from ..ml import audio_features as af
        from ..ml.templates import INFERENCE_AUDIO_PY

        # el mismo código de características usado al entrenar viaja con el modelo
        shutil.copy2(Path(af.__file__), out_dir / "audio_features.py")
        (out_dir / "inference.py").write_text(INFERENCE_AUDIO_PY, encoding="utf-8")

    def predict(self, model_dir, labels, data, filename, ok, cfg) -> dict:
        from ..ml.audio_features import AudioError
        from ..services import inference

        if Path(filename or "").suffix.lower().lstrip(".") not in AUDIO_EXT:
            raise InvalidInput("INVALID_AUDIO", "Formato no permitido (WAV, MP3, OGG, FLAC, M4A, AAC, OPUS o WEBM).")
        try:
            check_audio_payload(data)
        except InvalidImage as exc:
            raise InvalidInput("INVALID_AUDIO", str(exc)) from None
        try:
            return inference.predict_audio(
                model_dir, labels, data, ok.get("onnx", False), ok.get("keras", False),
                int(cfg.get("image_size", 224)), float(cfg.get("clip_seconds", 5.0)),
            )
        except AudioError as exc:
            raise InvalidInput("INVALID_AUDIO", f"El archivo no es un audio válido: {exc}") from None

    def embed_file(self, model_dir, source, cfg):
        import numpy as np

        from ..config import get_settings
        from ..ml import audio_features as af
        from ..services import similarity

        size, clip, cap = int(cfg.get("image_size", 224)), float(cfg.get("clip_seconds", 5.0)), get_settings().max_clips_per_audio
        signal = af.load_audio(source, max_seconds=cap * clip)  # mismos clips que en entrenamiento
        x = np.stack([af.spectrogram_image(c, size) for c in af.split_clips(signal, clip, max_clips=cap)]).astype(np.float32)
        return similarity.embed_mean(model_dir, x, size)
