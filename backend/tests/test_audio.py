"""Proyectos de audio: características, validación de cargas y entrenamiento real de extremo a extremo.
Los tonos sintéticos son sólo fixtures de prueba."""
import io
import json
import subprocess
import sys
import wave
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

from app import calendar_rules as cal
from app import models
from app.db import SessionLocal
from app.ml import audio_features as af
from app.ml.runner import claim_next_job, run_job
from app.storage import image_dir, thumb_dir
from app.tasks import ensure_weekly_run
from tests.conftest import TZ, create_project, logged_in, png_bytes, upload


def wav_bytes(kind: str, seed: int = 0, seconds: float = 6.0, sr: int = 22050) -> bytes:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    if kind == "grave":
        sig = np.sin(2 * np.pi * rng.uniform(220, 420) * t)
    else:
        sig = np.sin(2 * np.pi * rng.uniform(2500, 3500) * t)
    sig = 0.6 * sig + 0.05 * rng.normal(size=len(t))
    pcm = (np.clip(sig, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def to_mp3(data: bytes, tmp_path: Path) -> bytes:
    src, dst = tmp_path / "in.wav", tmp_path / "out.mp3"
    src.write_bytes(data)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), str(dst)], check=True)
    return dst.read_bytes()


def test_features_shapes_and_clips():
    sig = af.load_audio(wav_bytes("grave", seconds=7.3))
    assert abs(len(sig) / af.SAMPLE_RATE - 7.3) < 0.05  # remuestreado a 16 kHz
    clips = af.split_clips(sig, 2.0)
    assert len(clips) == 4 and all(len(c) == 2 * af.SAMPLE_RATE for c in clips)  # el último se rellena
    img = af.spectrogram_image(clips[0], 160)
    assert img.shape == (160, 160, 3) and img.dtype == np.uint8 and img.max() > 240  # rango completo (tras redimensionar)
    # un tono agudo concentra la energía más arriba (filas superiores) que uno grave
    hi = af.spectrogram_image(af.load_audio(wav_bytes("agudo"))[: 2 * af.SAMPLE_RATE], 128)[..., 0]
    lo = af.spectrogram_image(af.load_audio(wav_bytes("grave"))[: 2 * af.SAMPLE_RATE], 128)[..., 0]
    assert hi.mean(axis=1).argmax() < lo.mean(axis=1).argmax()
    assert af.n_clips_for(100.0, 5.0, 20) == 20 and af.n_clips_for(0.5, 5.0, 20) == 1


def test_audio_upload_validation_formats_and_thumbnails(tmp_path):
    c = logged_in("ana")
    p = create_project(c, name="Sonidos", modality="audio", categories=("grave", "agudo"))
    assert p["modality"] == "audio"
    cats = {x["name"]: x["id"] for x in p["categories"]}
    wav = wav_bytes("grave", 1)
    files = [
        ("a.wav", wav, "audio/wav"),
        ("b.mp3", to_mp3(wav_bytes("grave", 2), tmp_path), "audio/mpeg"),
        ("copia.wav", wav, "audio/wav"),                            # duplicado
        ("falso.wav", b"esto no es audio", "audio/wav"),             # no decodifica
        ("corto.wav", wav_bytes("grave", 3, seconds=0.1), "audio/wav"),
        ("foto.png", png_bytes("red", 1), "image/png"),              # imagen en proyecto de audio
    ]
    r = upload(c, p["id"], cats["grave"], files).json()
    assert sorted(u["format"] for u in r["uploaded"]) == ["mp3", "wav"]
    assert all(abs(u["duration_s"] - 6.0) < 0.2 for u in r["uploaded"])
    assert len(r["duplicates"]) == 1 and len(r["rejected"]) == 3
    assert len(list(thumb_dir(p["id"]).glob("*.jpg"))) == 2 and len(list(image_dir(p["id"]).iterdir())) == 2
    item = r["uploaded"][0]
    assert c.get(f"/api/images/{item['id']}/thumb").headers["content-type"] == "image/jpeg"
    assert c.get(f"/api/images/{item['id']}/file").headers["content-type"].startswith("audio/")
    # un proyecto de imágenes no acepta audio
    pi = create_project(c, name="Fotos")
    r = upload(c, pi["id"], pi["categories"][0]["id"], [("a.wav", wav, "audio/wav")]).json()
    assert len(r["rejected"]) == 1 and not r["uploaded"]


def test_audio_training_export_and_inference(tmp_path):
    c = logged_in("ana")
    p = create_project(
        c, name="Tonos", modality="audio", categories=("grave", "agudo"), base_models=["mobilenetv3small"],
        config={"epochs": 8, "fine_tune_epochs": 0, "batch_size": 8, "image_size": 160, "clip_seconds": 2.0, "patience": 8},
    )
    cats = {x["name"]: x["id"] for x in p["categories"]}
    for name in ("grave", "agudo"):
        files = [(f"{name}{i}.wav", wav_bytes(name, i), "audio/wav") for i in range(10)]
        assert len(upload(c, p["id"], cats[name], files).json()["uploaded"]) == 10

    cal.set_clock_override(datetime(2026, 10, 24, 0, 5, tzinfo=TZ))
    assert ensure_weekly_run() is True
    job_id = claim_next_job()
    run_job(job_id)
    with SessionLocal() as db:
        job = db.get(models.TrainingJob, job_id)
        assert job.status == "completed", job.error
        assert job.config["modality"] == "audio" and job.config["clip_seconds"] == 2.0
        mv = db.query(models.ModelVersion).one()
        assert [e.status for e in mv.exports] == ["ok"] * 5
        # 6 s por audio en clips de 2 s -> 3 clips por archivo; 1-2 archivos de prueba por clase
        assert mv.metrics["test_images"] % 3 == 0 and mv.metrics["train_images"] % 3 == 0
        out, model_id = Path(mv.dir_path), mv.id
        logs = " ".join(l.message for l in db.query(models.TrainingLog).filter_by(job_id=job_id))
        assert "clips de 2 s" in logs
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert meta["modality"] == "audio" and meta["audio"]["clip_seconds"] == 2.0 and meta["input"]["shape"] == [1, 160, 160, 3]
    assert (out / "audio_features.py").exists()

    cal.set_clock_override(datetime(2026, 10, 26, 9, 0, tzinfo=TZ))
    r = c.post(f"/api/models/{model_id}/predict", files={"file": ("x.wav", wav_bytes("agudo", 99, seconds=5), "audio/wav")})
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["modality"] == "audio" and res["prediction"] in ("grave", "agudo")
    hits = 0
    for k, kind in enumerate(["grave", "agudo"] * 3):
        pr = c.post(f"/api/models/{model_id}/predict", files={"file": ("t.wav", wav_bytes(kind, 500 + k, seconds=4), "audio/wav")}).json()
        hits += pr["prediction"] == kind
    assert hits >= 5, f"sólo {hits}/6 aciertos"
    assert len(res["segments"]) == 2 and res["segments"][0]["start"] == 0 and res["duration_s"] == 5.0
    mp3 = to_mp3(wav_bytes("grave", 98, seconds=4), tmp_path)
    r_mp3 = c.post(f"/api/models/{model_id}/predict", files={"file": ("x.mp3", mp3, "audio/mpeg")})
    assert r_mp3.status_code == 200 and r_mp3.json()["prediction"] in ("grave", "agudo")  # MP3 se decodifica
    assert c.post(f"/api/models/{model_id}/predict", files={"file": ("x.png", png_bytes("red", 1), "image/png")}).status_code == 422

    # «lo más parecido» con audio (sobre el ONNX realmente exportado): un grave nuevo se parece a los graves del proyecto
    from app.services import similarity

    assert similarity.build_index(model_id) == 20
    sim = c.post(f"/api/models/{model_id}/predict?similar=3", files={"file": ("n.wav", wav_bytes("grave", 700, seconds=4), "audio/wav")}).json()
    assert sim["similar_status"] == "ready" and [s["kind"] for s in sim["similar"]] == ["audio"] * 3
    assert [s["category"] for s in sim["similar"]] == ["grave"] * 3 and all(s["duration_s"] for s in sim["similar"])

    # el paquete descargado funciona fuera de la plataforma (ONNX Runtime + audio_features.py)
    bundle = tmp_path / "bundle"
    with zipfile.ZipFile(out / "bundle.zip") as z:
        z.extractall(bundle)
        assert {"audio_features.py", "inference.py", "model.onnx", "model.tflite"} <= set(z.namelist())
    sample = tmp_path / "s.wav"
    sample.write_bytes(wav_bytes("agudo", 77, seconds=4))
    proc = subprocess.run([sys.executable, "inference.py", str(sample)], cwd=bundle, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    api_pred = c.post(f"/api/models/{model_id}/predict", files={"file": ("s.wav", sample.read_bytes(), "audio/wav")}).json()["prediction"]
    assert proc.stdout.strip().splitlines()[0].split()[0] == api_pred  # el script entregado coincide con la plataforma
