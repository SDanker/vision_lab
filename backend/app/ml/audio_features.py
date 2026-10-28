"""Audio -> espectrograma log-mel -> imagen, sólo con numpy, Pillow y ffmpeg.

Este archivo NO importa nada de la plataforma: se copia tal cual junto a cada modelo de audio
exportado, de modo que la inferencia fuera de Vision Lab calcula exactamente la misma entrada
que se usó para entrenar.
"""
import math
import os
import shutil
import subprocess
import tempfile
import wave

import numpy as np
from PIL import Image

SAMPLE_RATE = 16000
N_FFT = 512
WIN_LENGTH = 400  # 25 ms
HOP_LENGTH = 160  # 10 ms
N_MELS = 128
FMIN = 20.0
FMAX = 8000.0
TOP_DB = 80.0


class AudioError(Exception):
    pass


# ----------------------------------------------------------------------------- carga
def _decode_ffmpeg(path: str, max_seconds: float | None) -> np.ndarray:
    # El archivo lo sube un usuario: sólo se permite leer ese archivo local (nada de http, tcp, listas que
    # apunten a otros recursos…) y se descarta el vídeo.
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "file", "-i", path, "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE)]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-f", "f32le", "pipe:1"]
    proc = subprocess.run(cmd, capture_output=True, timeout=300)
    if proc.returncode != 0:
        msg = proc.stderr.decode(errors="replace").strip().splitlines()
        raise AudioError(msg[-1] if msg else "No se pudo decodificar el audio.")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def _decode_wav(path: str) -> np.ndarray:
    """Respaldo sin ffmpeg: WAV PCM de 16 bits, con remuestreo lineal."""
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise AudioError("Sin ffmpeg sólo se admiten WAV PCM de 16 bits.")
        ch, sr = w.getnchannels(), w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    if sr != SAMPLE_RATE and len(data):
        n = int(round(len(data) * SAMPLE_RATE / sr))
        data = np.interp(np.linspace(0, len(data) - 1, n), np.arange(len(data)), data).astype(np.float32)
    return data


def load_audio(source, max_seconds: float | None = None) -> np.ndarray:
    """Devuelve una señal mono float32 a 16 kHz. `source` es una ruta o bytes."""
    tmp = None
    try:
        if isinstance(source, (bytes, bytearray)):
            fd, tmp = tempfile.mkstemp(suffix=".audio")
            with os.fdopen(fd, "wb") as fh:
                fh.write(source)
            path = tmp
        else:
            path = str(source)
        if shutil.which("ffmpeg"):
            data = _decode_ffmpeg(path, max_seconds)
        else:
            data = _decode_wav(path)
            if max_seconds:
                data = data[: int(max_seconds * SAMPLE_RATE)]
    finally:
        if tmp:
            os.unlink(tmp)
    if not len(data) or not np.isfinite(data).all():
        raise AudioError("El audio está vacío o dañado.")
    return data


# ------------------------------------------------------------------------- características
def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


_FILTERS: dict = {}


def mel_filterbank() -> np.ndarray:
    if "fb" not in _FILTERS:
        freqs = np.linspace(0, SAMPLE_RATE / 2, N_FFT // 2 + 1)
        mels = np.linspace(_hz_to_mel(FMIN), _hz_to_mel(FMAX), N_MELS + 2)
        hz = _mel_to_hz(mels)
        fb = np.zeros((N_MELS, len(freqs)), dtype=np.float32)
        for i in range(N_MELS):
            lo, mid, hi = hz[i], hz[i + 1], hz[i + 2]
            up = (freqs - lo) / max(mid - lo, 1e-9)
            down = (hi - freqs) / max(hi - mid, 1e-9)
            fb[i] = np.maximum(0, np.minimum(up, down))
        _FILTERS["fb"] = fb
    return _FILTERS["fb"]


def logmel(signal: np.ndarray) -> np.ndarray:
    """Espectrograma log-mel en dB, forma (N_MELS, frames)."""
    if len(signal) < N_FFT:
        signal = np.pad(signal, (0, N_FFT - len(signal)))
    n_frames = 1 + (len(signal) - N_FFT) // HOP_LENGTH
    idx = np.arange(N_FFT)[None, :] + HOP_LENGTH * np.arange(n_frames)[:, None]
    window = np.zeros(N_FFT, dtype=np.float32)
    start = (N_FFT - WIN_LENGTH) // 2
    window[start : start + WIN_LENGTH] = np.hanning(WIN_LENGTH)
    frames = signal[idx] * window
    power = np.abs(np.fft.rfft(frames, n=N_FFT, axis=1)) ** 2
    mel = power @ mel_filterbank().T
    return (10.0 * np.log10(np.maximum(mel, 1e-10))).T.astype(np.float32)


def spectrogram_image(clip: np.ndarray, size: int) -> np.ndarray:
    """Imagen uint8 (size, size, 3): frecuencias bajas abajo, tiempo de izquierda a derecha."""
    db = logmel(clip)
    top = float(db.max())
    db = np.clip(db, top - TOP_DB, top)
    img = ((db - (top - TOP_DB)) / TOP_DB * 255.0).astype(np.uint8)[::-1]
    im = Image.fromarray(img, mode="L").resize((size, size), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.uint8)
    return np.stack([arr, arr, arr], axis=-1)


def n_clips_for(duration_s: float, clip_seconds: float, max_clips: int) -> int:
    return int(min(max_clips, max(1, round(duration_s / clip_seconds))))


def split_clips(signal: np.ndarray, clip_seconds: float, n: int | None = None, max_clips: int = 60) -> list[np.ndarray]:
    """Corta en clips de `clip_seconds` (el último se rellena con silencio)."""
    length = int(clip_seconds * SAMPLE_RATE)
    if n is None:
        n = n_clips_for(len(signal) / SAMPLE_RATE, clip_seconds, max_clips)
    total = n * length
    if len(signal) < total:
        signal = np.pad(signal, (0, total - len(signal)))
    return [signal[i * length : (i + 1) * length] for i in range(n)]


def thumbnail(signal: np.ndarray, clip_seconds: float = 5.0, size: int = 256) -> Image.Image:
    """Miniatura en color del espectrograma del primer tramo (para la galería)."""
    gray = spectrogram_image(signal[: int(clip_seconds * SAMPLE_RATE)], size)[..., 0].astype(np.float32) / 255.0
    stops = np.array([[13, 8, 135], [126, 3, 168], [204, 71, 120], [248, 149, 64], [240, 249, 33]], dtype=np.float32)
    pos = gray * (len(stops) - 1)
    lo = np.floor(pos).astype(int).clip(0, len(stops) - 2)
    frac = (pos - lo)[..., None]
    rgb = stops[lo] * (1 - frac) + stops[lo + 1] * frac
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


def duration_label(seconds: float) -> str:
    m, s = divmod(int(math.ceil(seconds)), 60)
    return f"{m}:{s:02d}"
