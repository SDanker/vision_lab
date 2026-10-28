"""Cada arquitectura del catálogo debe construirse y exportarse en los 5 formatos con validación.
Se usan pesos aleatorios (weights=None): la conversión no depende de los pesos y así la prueba es rápida."""
import numpy as np
import pytest

from app.ml.architectures import ARCHITECTURES


def test_every_architecture_has_a_valid_recommended_tuning():
    from app.ml.architectures import TUNING, public_catalog
    from app.schemas import ModelOverride

    assert set(TUNING) == set(ARCHITECTURES)
    for arch, t in TUNING.items():
        assert ModelOverride(**t).model_dump(exclude_none=True) == t, arch  # cumple los límites de la API
    image_items = [item for item in public_catalog() if item["modality"] == "image"]
    assert {item["id"] for item in image_items} == set(ARCHITECTURES)
    assert all(item["tuning"] == TUNING[item["id"]] for item in image_items)


def test_fine_tune_settings_use_architecture_values_and_explicit_overrides():
    from app.ml.architectures import TUNING, fine_tune_settings

    assert fine_tune_settings("densenet121") == (70, 0.01)
    assert fine_tune_settings("convnext_small") == (20, 0.03)
    assert fine_tune_settings("mobilenetv3") == (30, 0.01)  # el medido: igual que antes de personalizarlo
    assert fine_tune_settings("modelo_desconocido") == (30, 0.01)
    assert fine_tune_settings("densenet121", {"unfreeze_layers": 12, "fine_tune_lr_factor": 0.5}) == (12, 0.5)
    assert fine_tune_settings("densenet121", {"unfreeze_layers": None}) == (70, 0.01)
    assert all(fine_tune_settings(a)[0] == t["unfreeze_layers"] for a, t in TUNING.items())


@pytest.mark.parametrize("arch", list(ARCHITECTURES))
def test_unfreeze_layers_fit_inside_the_base_model(arch):
    """No pedir descongelar más capas de las que tiene el modelo, y que quede al menos una congelada."""
    from app.ml.architectures import fine_tune_settings
    from app.ml.train import build_model

    _, base = build_model(arch, 3, weights=None)
    assert 1 <= fine_tune_settings(arch)[0] < len(base.layers)


@pytest.mark.parametrize("arch", list(ARCHITECTURES))
def test_architecture_builds_and_exports_all_formats(arch, tmp_path):
    import tensorflow as tf

    from app.ml.export import export_all
    from app.ml.train import build_model

    tf.keras.backend.clear_session()
    model, base = build_model(arch, 3, weights=None)
    assert model.input_shape == (None, 224, 224, 3) and model.output_shape == (None, 3)
    assert base.trainable is False
    samples = np.random.default_rng(0).uniform(0, 255, (4, 224, 224, 3)).astype("float32")
    results = export_all(model, tmp_path, samples)
    failed = {r["fmt"]: r["error"] for r in results if r["status"] != "ok"}
    assert not failed, failed
