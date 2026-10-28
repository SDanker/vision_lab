"""Datos tabulares (CSV/TSV): cada fila es un ejemplo y una columna es la categoría a predecir.

A diferencia de imágenes y audio, no se convierten en imágenes ni se entrenan redes de visión: se usan modelos de
scikit-learn (ver `ml/tabular.py`). El entrenamiento tiene su propio flujo (`ml/tabular_runner.py`); este módulo
sólo cubre la carga de archivos, la configuración efectiva del trabajo, el paquete exportado y la predicción.
"""
import shutil
from pathlib import Path

from ..storage import save_table_file, validate_table_bytes
from .base import InvalidInput, Modality, ModalityInfo

# ajustes de las redes de visión: no aplican a tablas (y así cambiarlos no marca el modelo como «modificado»)
_IMAGE_ONLY = ("batch_size", "learning_rate", "patience", "augmentation", "image_size", "clip_seconds",
               "unfreeze_layers", "fine_tune_lr_factor")


class TabularModality(Modality):
    info = ModalityInfo(
        id="tabular", label="Datos tabulares", icon="📊",
        description="Clasifica filas de una tabla (CSV): por ejemplo, predecir una categoría a partir de sus características.",
        unit="archivos CSV", unit_one="archivo CSV", extensions=("csv", "tsv"), accept=".csv,.tsv,text/csv",
        upload_help="CSV o TSV con una fila por ejemplo y los nombres de columna en la primera línea. "
                    "Si subes varios archivos, deben tener las mismas columnas.",
        preview="table", windowed=False, tabular=True,
        test_hint=" Completa los valores de una fila o sube un CSV con varias filas.",
    )
    media = {"csv": "text/csv"}
    sample_unit = "filas"

    def max_upload_mb(self, settings) -> int:
        return settings.max_tabular_upload_mb

    def validate(self, data: bytes, filename: str):
        return validate_table_bytes(data, filename)

    def save(self, project_id: int, v) -> str:
        return save_table_file(project_id, v)

    def item_fields(self, v) -> dict:
        return {"width": v.columns, "height": v.rows, "duration_s": None}  # columnas y filas del archivo

    def job_config(self, cfg: dict) -> dict:
        cfg = {k: v for k, v in cfg.items() if k not in _IMAGE_ONLY}
        return {**cfg, "modality": "tabular", "epochs": 1, "fine_tune_epochs": 0}

    def write_split(self, items, out_dir: Path, name: str, progress, cfg: dict, settings) -> int:
        raise NotImplementedError("Los datos tabulares se preparan en ml/tabular_runner.py.")

    def dataset_note(self, cfg: dict) -> str:
        return f" Columna a predecir: {cfg.get('tabular_target')}."

    def metadata(self, cfg: dict) -> dict:
        return {"modality": "tabular"}

    def export_assets(self, out_dir: Path) -> None:
        from ..ml import tabular_features
        from ..ml.templates import INFERENCE_TABULAR_PY

        # la misma conversión de filas usada al entrenar viaja con el modelo
        shutil.copy2(Path(tabular_features.__file__), out_dir / "tabular_features.py")
        (out_dir / "inference.py").write_text(INFERENCE_TABULAR_PY, encoding="utf-8")

    def predict(self, model_dir, labels, data, filename, ok, cfg) -> dict:
        import json

        from ..config import get_settings
        from ..ml import tabular
        from ..services import inference

        meta = json.loads((Path(model_dir) / "metadata.json").read_text(encoding="utf-8"))
        features = meta["tabular"]["features"]
        limit = get_settings().max_tabular_predict_rows
        try:
            rows = tabular.parse_input(data, filename, limit)
        except tabular.InvalidTable as exc:
            raise InvalidInput("INVALID_TABLE", str(exc)) from None
        names = {f["name"] for f in features}
        present = set().union(*(r.keys() for r in rows))
        if not (names & present):
            raise InvalidInput(
                "INVALID_TABLE",
                f"No hay ninguna de las columnas del modelo en la consulta. Se esperan: {', '.join(sorted(names)[:12])}…",
            )
        probs, engine = tabular.predict_proba(model_dir, features, rows, ok.get("onnx", False), ok.get("joblib", False))
        out = inference._result(labels, probs[0], engine)
        out["n_rows"] = len(rows)
        out["missing_fields"] = sorted(names - present)  # columnas ausentes: se completaron con la mediana / «sin dato»
        out["inputs"] = {f["name"]: rows[0].get(f["name"]) for f in features}
        out["rows"] = [
            {"row": i + 1, "prediction": labels[int(p.argmax())], "confidence": float(p.max()),
             "top": [{"label": labels[int(j)], "score": float(p[j])} for j in p.argsort()[::-1][:3]]}
            for i, p in enumerate(probs)
        ]
        return out

    def embed_file(self, model_dir, source, cfg):
        raise InvalidInput("NOT_SUPPORTED", "«Lo más parecido» no está disponible para datos tabulares.")
