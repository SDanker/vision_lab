from app.modalities import MEDIA, REGISTRY, describe_all, get_modality
from tests.conftest import create_project, logged_in


def test_registry_describes_the_original_types_for_the_ui():
    infos = {m["id"]: m for m in describe_all()}
    assert {"image", "audio"} <= set(infos)
    assert infos["image"]["unit"] == "imágenes" and "png" in infos["image"]["extensions"] and not infos["image"]["windowed"]
    assert infos["audio"]["unit"] == "audios" and infos["audio"]["windowed"] and infos["audio"]["preview"] == "audio"
    assert MEDIA["png"] == "image/png" and MEDIA["wav"] == "audio/wav"
    assert get_modality(None).info.id == "image" and get_modality("audio") is REGISTRY["audio"]


def test_modalities_endpoint_requires_login_and_lists_types():
    c = logged_in("ana")
    ids = [m["id"] for m in c.get("/api/modalities").json()]
    assert ids[:2] == ["image", "audio"]
    from tests.conftest import new_client

    assert new_client().get("/api/modalities").status_code == 401


def test_project_creation_validates_the_type():
    c = logged_in("ana")
    assert create_project(c, name="Audio", modality="audio")["modality"] == "audio"
    assert create_project(c, name="Imagen")["modality"] == "image"
    r = c.post("/api/projects", json={"name": "Raro", "categories": ["a", "b"], "modality": "holograma"})
    assert r.status_code == 422 and "no soportado" in r.text


def test_job_config_keeps_each_types_keys():
    img, aud = get_modality("image"), get_modality("audio")
    assert "clip_seconds" not in img.job_config({"clip_seconds": 5.0, "epochs": 3}) and "modality" not in img.job_config({})
    assert aud.job_config({"epochs": 3})["modality"] == "audio"
