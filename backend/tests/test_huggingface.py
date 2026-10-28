"""Importación de datasets desde Hugging Face. Sin red: el Hub se reemplaza por un repositorio falso en memoria."""
import io
from datetime import datetime
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from app import calendar_rules, models
from app.config import get_settings
from app.db import SessionLocal, utcnow
from app.main import app
from app.services import hf, vault
from tests.conftest import create_project, logged_in, make_user, png_bytes

TOKEN = "hf_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7"
SATURDAY = datetime(2026, 10, 24, 12, 0, tzinfo=ZoneInfo("America/Santiago"))


class FakeHub:
    """Repositorios falsos: {repo_id: {"private": bool, "gated": bool, "files": {ruta: bytes}}}."""

    def __init__(self):
        self.repos: dict[str, dict] = {}
        self.tokens_seen: list[str | None] = []
        self.on_download = None
        self.reported_size = None  # tamaño que se informa de cada archivo (para simular datasets que «mienten»)
        self.valid_tokens = {TOKEN: "ana-hf"}

    def add(self, repo_id, files, private=False, gated=False):
        self.repos[repo_id] = {"private": private, "gated": gated, "files": files}

    # ---- las funciones hub_* que se sustituyen
    def whoami(self, token):
        if token not in self.valid_tokens:
            raise hf.HfError("HF_TOKEN_INVALID", "Hugging Face rechazó el token.", 401)
        return self.valid_tokens[token]

    def _repo(self, repo_id, token):
        self.tokens_seen.append(token)
        repo = self.repos.get(repo_id)
        if repo is None or (repo["private"] and token is None):
            raise hf.HfError("HF_NOT_FOUND", "No existe o es privado.", 404)
        if repo["gated"] and token is None:
            raise hf.HfError("HF_GATED", "Dataset restringido.", 403)
        return repo

    def dataset_info(self, repo_id, token):
        repo = self._repo(repo_id, token)
        return {"id": repo_id, "sha": "c0ffee" * 6 + "abcd", "private": repo["private"], "gated": repo["gated"]}

    def list_files(self, repo_id, revision, token):
        return [(p, self.reported_size or len(b)) for p, b in self._repo(repo_id, token)["files"].items()]

    def parquet_info(self, repo_id, revision, token, path):
        data = self._repo(repo_id, token)["files"][path]
        return hf._parquet_info_from(pq.ParquetFile(io.BytesIO(data)))

    def sample(self, repo_id, revision, token, path, columns, max_rows):
        data = self._repo(repo_id, token)["files"][path]
        return hf._sample_from(pq.ParquetFile(io.BytesIO(data)), columns, max_rows, pa)

    def download(self, repo_id, revision, token, path, dest):
        data = self._repo(repo_id, token)["files"][path]
        if self.on_download:
            self.on_download(path)
        out = dest / path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        return out


def parquet_bytes(rows, label_names=("gato", "perro"), label_col="label", with_meta=True):
    """rows: [(png_bytes, label)] -> parquet con la misma forma que usa el Hub (image{bytes,path} + ClassLabel)."""
    import json

    image = pa.array([{"bytes": b, "path": f"{i}.png"} for i, (b, _) in enumerate(rows)],
                     pa.struct([("bytes", pa.binary()), ("path", pa.string())]))
    labels = pa.array([lab for _, lab in rows], pa.int64())
    table = pa.table({"image": image, label_col: labels})
    if with_meta:
        meta = {"info": {"features": {"image": {"_type": "Image"}, label_col: {"_type": "ClassLabel", "names": list(label_names)}}}}
        table = table.replace_schema_metadata({"huggingface": json.dumps(meta)})
    buf = io.BytesIO()
    pq.write_table(table, buf, row_group_size=4)
    return buf.getvalue()


@pytest.fixture
def hub(monkeypatch):
    h = FakeHub()
    monkeypatch.setattr(hf, "hub_whoami", h.whoami)
    monkeypatch.setattr(hf, "hub_dataset_info", h.dataset_info)
    monkeypatch.setattr(hf, "hub_list_files", h.list_files)
    monkeypatch.setattr(hf, "hub_parquet_info", h.parquet_info)
    monkeypatch.setattr(hf, "hub_sample", h.sample)
    monkeypatch.setattr(hf, "hub_download", h.download)
    monkeypatch.setattr(hf, "kick_import", hf.run_import)  # sin broker: la «cola» corre en el acto
    monkeypatch.setattr(hf.time, "sleep", lambda s: None)
    return h


def cats_dataset(per_class=8, kinds=(("gato", "red"), ("perro", "blue")), split="train", seed0=0, size=96):
    files = {}
    for ci, (name, kind) in enumerate(kinds):
        for i in range(per_class):
            files[f"{split}/{name}/{i}.png"] = png_bytes(kind, seed0 + ci * 1000 + i, size=size)
    return files


def start(client, project_id, source="ana/gatos", **extra):
    return client.post(f"/api/projects/{project_id}/hf/imports", json={"source": source, **extra})


def project_state(client, pid):
    p = client.get(f"/api/projects/{pid}").json()
    return {c["name"]: c["image_count"] for c in p["categories"]}, p["image_count"]


def connect(client, token=TOKEN):
    r = client.put("/api/integrations/huggingface", json={"token": token})
    assert r.status_code == 200, r.text
    return r.json()


def after_checkpoints(monkeypatch, n, action):
    """Ejecuta `action(runner)` justo después del n-ésimo guardado de progreso (sin hilos: determinista)."""
    real, count = hf._Runner.checkpoint, {"n": 0}

    def wrapped(self):
        real(self)
        count["n"] += 1
        if count["n"] == n:
            action(self)

    monkeypatch.setattr(hf._Runner, "checkpoint", wrapped)


# ------------------------------------------------------------------------------ dirección del dataset
@pytest.mark.parametrize("raw,expected", [
    ("ana/gatos", "ana/gatos"),
    ("  datasets/ana/gatos  ", "ana/gatos"),
    ("https://huggingface.co/datasets/ana/gatos", "ana/gatos"),
    ("https://huggingface.co/datasets/ana/gatos/tree/main/train", "ana/gatos"),
    ("https://huggingface.co/datasets/ana/gatos?row=3", "ana/gatos"),
    ("https://huggingface.co/datasets/mnist", "mnist"),
    ("https://huggingface.co/datasets/mnist/viewer", "mnist"),
])
def test_parse_repo_id_accepts_hub_addresses(raw, expected):
    assert hf.parse_repo_id(raw) == expected


@pytest.mark.parametrize("raw", [
    "", "https://evil.example.com/datasets/ana/gatos", "http://huggingface.co.evil.com/datasets/a/b",
    "ftp://huggingface.co/datasets/a/b", "../etc/passwd", "ana/gatos/extra/más", "ana gatos", "https://huggingface.co/ana/gatos",
])
def test_parse_repo_id_rejects_other_hosts_and_junk(raw):
    with pytest.raises(hf.HfError):
        hf.parse_repo_id(raw)


def _response(status):
    import requests

    r = requests.Response()
    r.status_code = status
    return r


@pytest.mark.parametrize("make,token,code,status", [
    # las excepciones reales de huggingface_hub, tal como las lanza la librería
    (lambda: __import__("huggingface_hub.utils", fromlist=["x"]).GatedRepoError("x", response=_response(403)), True, "HF_GATED", 403),
    (lambda: __import__("huggingface_hub.utils", fromlist=["x"]).GatedRepoError("x", response=_response(403)), False, "HF_GATED", 403),
    (lambda: __import__("huggingface_hub.utils", fromlist=["x"]).RepositoryNotFoundError("x", response=_response(404)), False, "HF_NOT_FOUND", 404),
    (lambda: __import__("huggingface_hub.utils", fromlist=["x"]).RevisionNotFoundError("x", response=_response(404)), True, "HF_NOT_FOUND", 404),
    (lambda: __import__("huggingface_hub.utils", fromlist=["x"]).HfHubHTTPError("x", response=_response(401)), True, "HF_TOKEN_INVALID", 401),
    (lambda: __import__("huggingface_hub.utils", fromlist=["x"]).HfHubHTTPError("x", response=_response(429)), True, "HF_RATE_LIMIT", 429),
    (lambda: __import__("huggingface_hub.utils", fromlist=["x"]).HfHubHTTPError("x", response=_response(503)), True, "HF_UNAVAILABLE", 502),
    # whoami() con un token inválido lanza un requests.HTTPError genérico
    (lambda: __import__("requests").exceptions.HTTPError("Invalid user token.", response=_response(401)), True, "HF_TOKEN_INVALID", 401),
    (lambda: __import__("requests").exceptions.HTTPError("Invalid user token. If you didn't pass a user token..."), True, "HF_TOKEN_INVALID", 401),
    (lambda: __import__("requests").exceptions.ConnectionError("sin red"), False, "HF_UNREACHABLE", 502),
    (lambda: __import__("requests").exceptions.Timeout("lento"), False, "HF_UNREACHABLE", 502),
])
def test_hub_errors_are_translated_to_clear_messages(make, token, code, status):
    err = hf._translate(make(), "ana/gatos", token)
    assert err is not None and (err.code, err.status) == (code, status)
    assert "ana/gatos" in err.message or code != "HF_GATED" or not token


def test_unknown_errors_are_not_swallowed():
    assert hf._translate(ValueError("otra cosa"), None, False) is None
    with pytest.raises(ValueError):
        with hf._guard("ana/gatos", None):
            raise ValueError("otra cosa")


def test_guard_turns_the_invalid_token_error_of_whoami_into_an_api_error(hub, monkeypatch):
    import requests

    def boom(token):
        with hf._guard(None, token):
            raise requests.exceptions.HTTPError("Invalid user token.", response=_response(401))

    monkeypatch.setattr(hf, "hub_whoami", boom)
    ana = logged_in("ana")
    r = ana.put("/api/integrations/huggingface", json={"token": "hf_" + "z" * 34})
    assert r.status_code == 401 and r.json()["code"] == "HF_TOKEN_INVALID"  # antes de la corrección era un 500


# ------------------------------------------------------------------------------ cuenta
def test_connect_stores_token_encrypted_and_never_returns_it(hub):
    ana = logged_in("ana")
    assert ana.get("/api/integrations/huggingface").json() == {"connected": False}
    r = ana.put("/api/integrations/huggingface", json={"token": TOKEN})
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    body = r.json()
    assert body["connected"] and body["username"] == "ana-hf" and body["hint"] == "hf_…" + TOKEN[-4:]
    assert TOKEN not in r.text and TOKEN not in ana.get("/api/integrations/huggingface").text
    with SessionLocal() as db:
        row = db.query(models.HfConnection).one()
        assert TOKEN not in row.token_enc and vault.decrypt(row.token_enc) == TOKEN
        audit = db.query(models.AuditLog).filter_by(action="hf.connected").one()
        assert TOKEN not in str(audit.detail)


def test_connect_rejects_bad_format_and_tokens_hugging_face_refuses(hub):
    ana = logged_in("ana")
    assert ana.put("/api/integrations/huggingface", json={"token": "no-es-un-token-de-hf"}).status_code == 422
    r = ana.put("/api/integrations/huggingface", json={"token": "hf_" + "z" * 34})
    assert r.status_code == 401 and r.json()["code"] == "HF_TOKEN_INVALID"
    with SessionLocal() as db:
        assert db.query(models.HfConnection).count() == 0


def test_reconnect_replaces_and_disconnect_deletes(hub):
    ana = logged_in("ana")
    connect(ana)
    other = "hf_" + "Q" * 34
    hub.valid_tokens[other] = "ana-otra"
    assert connect(ana, other)["username"] == "ana-otra"
    with SessionLocal() as db:
        assert db.query(models.HfConnection).count() == 1
    assert ana.delete("/api/integrations/huggingface").status_code == 204
    assert ana.get("/api/integrations/huggingface").json() == {"connected": False}
    assert ana.delete("/api/integrations/huggingface").status_code == 204  # idempotente


def test_each_user_has_their_own_connection(hub):
    ana, bob = logged_in("ana"), logged_in("bob")
    connect(ana)
    assert bob.get("/api/integrations/huggingface").json() == {"connected": False}


def test_api_tokens_cannot_touch_the_hugging_face_account(hub):
    ana = logged_in("ana")
    tok = ana.post("/api/tokens", json={"name": "lectura"}).json()["token"]
    bearer = TestClient(app, headers={"Authorization": f"Bearer {tok}"})
    assert bearer.get("/api/integrations/huggingface").json()["code"] == "SESSION_REQUIRED"
    r = bearer.put("/api/integrations/huggingface", json={"token": TOKEN})
    assert r.status_code in (403,) and r.json()["code"] in ("SESSION_REQUIRED", "TOKEN_READ_ONLY")


def test_anonymous_visitors_cannot_use_it(hub):
    anon = TestClient(app, headers={"X-Requested-With": "vision-lab"})
    assert anon.get("/api/integrations/huggingface").status_code == 401
    assert anon.put("/api/integrations/huggingface", json={"token": TOKEN}).status_code == 401
    assert anon.delete("/api/integrations/huggingface").status_code == 401


def test_vault_survives_adding_a_dedicated_key_later_and_flags_a_lost_one(monkeypatch):
    s = get_settings()
    old = vault.encrypt("hf_secreto")  # con la clave derivada de SECRET_KEY
    monkeypatch.setattr(s, "hf_encryption_key", Fernet.generate_key().decode())
    assert vault.decrypt(old) == "hf_secreto"  # la derivada sigue sirviendo para descifrar
    new = vault.encrypt("hf_otro")
    assert vault.decrypt(new) == "hf_otro"
    monkeypatch.setattr(s, "hf_encryption_key", Fernet.generate_key().decode())  # otra clave distinta: se perdió la anterior
    assert vault.decrypt(new) is None and vault.decrypt(old) == "hf_secreto"
    monkeypatch.setattr(s, "hf_encryption_key", "no-es-una-clave")
    with pytest.raises(RuntimeError):
        vault.encrypt("x")
    monkeypatch.setattr(s, "hf_encryption_key", "CHANGE_ME_HF_ENCRYPTION_KEY")  # plantilla sin editar = sin definir
    assert vault.decrypt(vault.encrypt("hf_y")) == "hf_y" and vault.decrypt(old) == "hf_secreto"


def test_token_that_can_no_longer_be_decrypted_asks_to_reconnect(hub, monkeypatch):
    ana = logged_in("ana")
    connect(ana)
    p = create_project(ana)
    hub.add("ana/gatos", cats_dataset(3))
    monkeypatch.setattr(vault, "decrypt", lambda _: None)
    assert ana.get("/api/integrations/huggingface").json()["needs_reconnect"] is True
    r = ana.post(f"/api/projects/{p['id']}/hf/inspect", json={"source": "ana/gatos"})
    assert r.status_code == 409 and r.json()["code"] == "HF_RECONNECT"


# ------------------------------------------------------------------------------ revisar
def test_inspect_folders_lists_categories_without_downloading(hub):
    ana = logged_in("ana")
    p = create_project(ana)
    files = cats_dataset(8)
    files["train/suelta.png"] = png_bytes("green", 5)  # sin categoría
    files["README.md"] = b"# hola"
    hub.add("ana/gatos", files)
    downloads = []
    hub.on_download = downloads.append
    r = ana.post(f"/api/projects/{p['id']}/hf/inspect", json={"source": "https://huggingface.co/datasets/ana/gatos"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["layout"] == "files" and b["items"] == 17 and b["can_import"] and b["repo_id"] == "ana/gatos"
    assert b["categories"] == [{"name": "gato", "count": 8}, {"name": "perro", "count": 8}]
    assert b["unlabeled"] == 1 and any("omitirán" in w for w in b["warnings"])
    assert downloads == []


def test_inspect_parquet_detects_columns_and_class_names(hub):
    ana = logged_in("ana")
    p = create_project(ana)
    rows = [(png_bytes("red", i), 0) for i in range(5)] + [(png_bytes("blue", i), 1) for i in range(5)]
    hub.add("ana/tabla", {"data/train-0.parquet": parquet_bytes(rows)})
    b = ana.post(f"/api/projects/{p['id']}/hf/inspect", json={"source": "ana/tabla"}).json()
    assert b["layout"] == "parquet" and b["items"] == 10 and b["image_column"] == "image"
    assert b["label_column"] == "label" and b["label_columns"] == ["label"] and b["class_names"] == ["gato", "perro"]
    assert b["category_count"] == 2 and b["can_import"]


def test_inspect_parquet_without_labels_needs_a_destination_category(hub):
    ana = logged_in("ana")
    p = create_project(ana)
    only_images = pa.table({"image": pa.array([{"bytes": png_bytes("red", 1), "path": "a.png"}], pa.struct([("bytes", pa.binary()), ("path", pa.string())]))})
    buf = io.BytesIO()
    pq.write_table(only_images, buf)
    hub.add("ana/sinlabel", {"d.parquet": buf.getvalue()})
    b = ana.post(f"/api/projects/{p['id']}/hf/inspect", json={"source": "ana/sinlabel"}).json()
    assert not b["can_import"] and "categoría" in b["blockers"][0]
    b = ana.post(f"/api/projects/{p['id']}/hf/inspect", json={"source": "ana/sinlabel", "default_category": "todas"}).json()
    assert b["can_import"]


def test_inspect_errors_are_clear(hub):
    ana = logged_in("ana")
    p = create_project(ana)
    hub.add("ana/privado", cats_dataset(2), private=True)
    hub.add("ana/restringido", cats_dataset(2), gated=True)
    hub.add("ana/solotexto", {"README.md": b"x", "datos.csv": b"a,b"})
    url = f"/api/projects/{p['id']}/hf/inspect"
    assert ana.post(url, json={"source": "ana/nada"}).json()["code"] == "HF_NOT_FOUND"
    assert ana.post(url, json={"source": "ana/privado"}).json()["code"] == "HF_NOT_FOUND"
    assert ana.post(url, json={"source": "ana/restringido"}).json()["code"] == "HF_GATED"
    r = ana.post(url, json={"source": "ana/solotexto"})
    assert r.status_code == 422 and r.json()["code"] == "HF_NO_IMAGES"
    assert ana.post(url, json={"source": "https://evil.example.com/datasets/a/b"}).json()["code"] == "INVALID_REPO"


def test_private_and_gated_datasets_work_with_the_users_own_token(hub):
    ana, bob = logged_in("ana"), logged_in("bob")
    p, pb = create_project(ana), create_project(bob)
    hub.add("ana/privado", cats_dataset(3), private=True)
    hub.add("ana/restringido", cats_dataset(3), gated=True)
    connect(ana)
    for repo in ("ana/privado", "ana/restringido"):
        assert ana.post(f"/api/projects/{p['id']}/hf/inspect", json={"source": repo}).status_code == 200
        # bob no conectó su cuenta: el token de ana no se usa para él
        assert bob.post(f"/api/projects/{pb['id']}/hf/inspect", json={"source": repo}).status_code in (403, 404)
    assert hub.tokens_seen.count(TOKEN) >= 2 and hub.tokens_seen.count(None) >= 2


def test_public_dataset_is_fetched_anonymously_when_no_account_is_connected(hub):
    ana = logged_in("ana")
    p = create_project(ana)
    hub.add("ana/gatos", cats_dataset(3))
    assert ana.post(f"/api/projects/{p['id']}/hf/inspect", json={"source": "ana/gatos"}).status_code == 200
    assert set(hub.tokens_seen) == {None}


# ------------------------------------------------------------------------------ importar (carpetas)
def test_import_folders_creates_categories_and_images(hub):
    ana = logged_in("ana")
    p = create_project(ana, categories=())
    hub.add("ana/gatos", cats_dataset(8))
    r = start(ana, p["id"])
    assert r.status_code == 201, r.text
    imp = ana.get(f"/api/projects/{p['id']}/hf/imports").json()[0]
    assert imp["status"] == "completed" and imp["percent"] == 100 and imp["imported"] == 16
    assert imp["duplicates"] == 0 and imp["rejected"] == 0 and imp["repo_id"] == "ana/gatos" and imp["error"] is None
    assert project_state(ana, p["id"]) == ({"gato": 8, "perro": 8}, 16)
    with SessionLocal() as db:
        assert db.query(models.Notification).filter_by(user_id=db.query(models.User).filter_by(username="ana").one().id, kind="hf.import_done").count() == 1
        assert {a.action for a in db.query(models.AuditLog)} >= {"hf.import_started", "hf.import_finished"}
    # los archivos son reales, con miniatura, y se sirven como cualquier imagen subida
    first = ana.get(f"/api/projects/{p['id']}/images").json()["items"][0]
    assert ana.get(f"/api/images/{first['id']}/file").status_code == 200 and ana.get(f"/api/images/{first['id']}/thumb").status_code == 200


def test_import_reuses_existing_categories_ignoring_case_and_skips_duplicates(hub):
    ana = logged_in("ana")
    p = create_project(ana, categories=("Gato",))
    hub.add("ana/gatos", cats_dataset(6))
    assert start(ana, p["id"]).status_code == 201
    names, total = project_state(ana, p["id"])
    assert names == {"Gato": 6, "perro": 6} and total == 12
    # la segunda importación es idéntica: todo se detecta como repetido
    assert start(ana, p["id"]).status_code == 201
    imp = ana.get(f"/api/projects/{p['id']}/hf/imports").json()[0]
    assert imp["imported"] == 0 and imp["duplicates"] == 12 and project_state(ana, p["id"])[1] == 12


def test_import_rejects_corrupt_files_but_keeps_the_rest(hub):
    ana = logged_in("ana")
    p = create_project(ana, categories=())
    files = cats_dataset(5)
    files["train/gato/roto.png"] = b"esto no es una imagen"
    files["train/gato/ejecutable.png"] = png_bytes("red", 99) + b"<?php system($_GET['x']); ?>"
    hub.add("ana/gatos", files)
    start(ana, p["id"])
    imp = ana.get(f"/api/projects/{p['id']}/hf/imports").json()[0]
    assert imp["status"] == "completed" and imp["imported"] == 10 and imp["rejected"] == 2
    assert {r["name"] for r in imp["rejected_sample"]} == {"roto.png", "ejecutable.png"}


def test_paths_that_could_escape_the_work_directory_are_ignored(hub):
    ana = logged_in("ana")
    p = create_project(ana, categories=())
    files = cats_dataset(3)
    files.update({"../fuera/gato/x.png": png_bytes("red", 900), "/abs/gato/y.png": png_bytes("red", 901),
                  "a/../../z.png": png_bytes("red", 902), "gato\\w.png": png_bytes("red", 903), ".oculta/gato/v.png": png_bytes("red", 904)})
    hub.add("ana/maligno", files)
    b = ana.post(f"/api/projects/{p['id']}/hf/inspect", json={"source": "ana/maligno"}).json()
    assert b["items"] == 6 and [c["name"] for c in b["categories"]] == ["gato", "perro"]
    start(ana, p["id"], source="ana/maligno")
    assert project_state(ana, p["id"]) == ({"gato": 3, "perro": 3}, 6)


def test_parquet_with_huge_row_groups_is_refused_instead_of_exhausting_memory(hub, monkeypatch):
    ana = logged_in("ana")
    p = create_project(ana, categories=())
    hub.add("ana/enorme", {"d.parquet": parquet_bytes([(png_bytes("red", i), 0) for i in range(6)])})
    monkeypatch.setattr(hf, "MAX_ROW_GROUP_BYTES", 100)
    start(ana, p["id"], source="ana/enorme")
    imp = ana.get(f"/api/projects/{p['id']}/hf/imports").json()[0]
    assert imp["status"] == "error" and "bloques" in imp["error"] and imp["imported"] == 0


def test_import_merges_splits_and_skips_unlabeled_images(hub):
    ana = logged_in("ana")
    p = create_project(ana, categories=())
    files = {**cats_dataset(4, split="train"), **cats_dataset(3, split="test", seed0=500)}
    files["train/huerfana.png"] = png_bytes("green", 77)
    hub.add("ana/gatos", files)
    start(ana, p["id"])
    imp = ana.get(f"/api/projects/{p['id']}/hf/imports").json()[0]
    assert imp["unlabeled"] == 1 and imp["imported"] == 14
    assert project_state(ana, p["id"])[0] == {"gato": 7, "perro": 7}


def test_import_unlabeled_images_go_to_the_chosen_category(hub):
    ana = logged_in("ana")
    p = create_project(ana, categories=())
    hub.add("ana/planas", {f"img{i}.png": png_bytes("red", i) for i in range(6)})
    assert start(ana, p["id"], source="ana/planas").status_code == 422  # sin carpetas ni categoría de destino
    assert start(ana, p["id"], source="ana/planas", default_category="  flores ").status_code == 201
    assert project_state(ana, p["id"]) == ({"flores": 6}, 6)


def test_import_downloads_with_the_connected_token(hub):
    ana = logged_in("ana")
    connect(ana)
    p = create_project(ana, categories=())
    hub.add("ana/privado", cats_dataset(3), private=True)
    assert start(ana, p["id"], source="ana/privado").status_code == 201
    assert project_state(ana, p["id"])[1] == 6 and set(hub.tokens_seen) == {TOKEN}
    with SessionLocal() as db:
        assert db.query(models.HfConnection).one().last_used_at is not None


# ------------------------------------------------------------------------------ importar (parquet)
def test_import_parquet_maps_class_names_and_dedupes(hub):
    ana = logged_in("ana")
    p = create_project(ana, categories=())
    rows = [(png_bytes("red", i), 0) for i in range(6)] + [(png_bytes("blue", i), 1) for i in range(6)] + [(png_bytes("red", 0), 0)]
    hub.add("ana/tabla", {"data/train-0.parquet": parquet_bytes(rows), "data/test-0.parquet": parquet_bytes([(png_bytes("green", i), 1) for i in range(3)])})
    assert start(ana, p["id"], source="ana/tabla").status_code == 201
    imp = ana.get(f"/api/projects/{p['id']}/hf/imports").json()[0]
    assert imp["status"] == "completed" and imp["total"] == 16 and imp["processed"] == 16
    assert imp["imported"] == 15 and imp["duplicates"] == 1
    assert project_state(ana, p["id"])[0] == {"gato": 6, "perro": 9}


def test_import_parquet_with_the_label_column_chosen_by_the_user(hub):
    ana = logged_in("ana")
    p = create_project(ana, categories=())
    image = pa.array([{"bytes": png_bytes("red", i), "path": None} for i in range(6)], pa.struct([("bytes", pa.binary()), ("path", pa.string())]))
    table = pa.table({"img": image, "fine_label": pa.array([0, 1, 0, 1, 0, 1], pa.int64()), "texto": pa.array(list("abcdef"))})
    buf = io.BytesIO()
    pq.write_table(table, buf)
    hub.add("ana/columnas", {"x.parquet": buf.getvalue()})
    b = ana.post(f"/api/projects/{p['id']}/hf/inspect", json={"source": "ana/columnas"}).json()
    assert b["image_columns"] == ["img"] and b["label_columns"] == ["fine_label"] and b["class_names"] is None
    assert start(ana, p["id"], source="ana/columnas", image_column="img", label_column="fine_label").status_code == 201
    assert project_state(ana, p["id"])[0] == {"0": 3, "1": 3}  # sin nombres de clase se usan los números


def test_import_parquet_with_string_labels_and_missing_labels(hub):
    ana = logged_in("ana")
    p = create_project(ana, categories=())
    image = pa.array([{"bytes": png_bytes("red", i), "path": f"f{i}.png"} for i in range(6)], pa.struct([("bytes", pa.binary()), ("path", pa.string())]))
    table = pa.table({"image": image, "label": pa.array(["rosa", "rosa", "rosa", "lirio", None, "lirio"])})
    buf = io.BytesIO()
    pq.write_table(table, buf)
    hub.add("ana/flores", {"x.parquet": buf.getvalue()})
    start(ana, p["id"], source="ana/flores")
    imp = ana.get(f"/api/projects/{p['id']}/hf/imports").json()[0]
    assert imp["imported"] == 5 and imp["unlabeled"] == 1
    assert project_state(ana, p["id"])[0] == {"rosa": 3, "lirio": 2}


# ------------------------------------------------------------------------------ límites y permisos
def test_blocked_when_bigger_than_the_import_limit_or_the_free_space(hub, monkeypatch):
    ana = logged_in("ana")
    p = create_project(ana, categories=())
    hub.add("ana/gatos", cats_dataset(4))
    monkeypatch.setattr(get_settings(), "hf_import_max_gb", 0.00001)  # ~10 KB
    r = start(ana, p["id"])
    assert r.status_code == 422 and r.json()["code"] == "IMPORT_BLOCKED" and "máximo por importación" in r.json()["detail"]
    insp = ana.post(f"/api/projects/{p['id']}/hf/inspect", json={"source": "ana/gatos"}).json()
    assert not insp["can_import"] and insp["blockers"]
    monkeypatch.setattr(get_settings(), "hf_import_max_gb", 5.0)
    with SessionLocal() as db:
        db.query(models.User).filter_by(username="ana").one().quota_mb = 0
        db.commit()
    r = start(ana, p["id"])
    assert r.status_code == 422 and "espacio libre" in r.json()["detail"]
    with SessionLocal() as db:
        assert db.query(models.HfImport).count() == 0


def test_quota_runs_out_midway_and_keeps_what_was_imported(hub):
    """El Hub informa tamaños menores que los reales: la revisión pasa, pero al guardar se agota la cuota."""
    ana = logged_in("ana")
    p = create_project(ana, categories=())
    hub.add("ana/gatos", cats_dataset(3, size=500))  # más de 0,5 MB por imagen
    hub.reported_size = 1
    with SessionLocal() as db:
        db.query(models.User).filter_by(username="ana").one().quota_mb = 1
        db.commit()
    assert start(ana, p["id"]).status_code == 201
    imp = ana.get(f"/api/projects/{p['id']}/hf/imports").json()[0]
    assert imp["status"] == "error" and "cuota" in imp["error"].lower()
    assert 1 <= imp["imported"] < 6 and project_state(ana, p["id"])[1] == imp["imported"]  # lo importado hasta ahí se conserva


def test_viewers_cannot_import_and_non_members_see_nothing(hub):
    ana, bob, carl = logged_in("ana"), logged_in("bob"), logged_in("carl")
    p = create_project(ana)
    ana.post(f"/api/projects/{p['id']}/members", json={"identifier": "bob", "role": "viewer"})
    hub.add("ana/gatos", cats_dataset(3))
    assert start(bob, p["id"]).status_code == 403
    assert bob.post(f"/api/projects/{p['id']}/hf/inspect", json={"source": "ana/gatos"}).status_code == 403
    assert start(carl, p["id"]).status_code == 404
    assert carl.get(f"/api/projects/{p['id']}/hf/imports").status_code == 404
    assert bob.get(f"/api/projects/{p['id']}/hf/imports").status_code == 200  # puede ver el historial


def test_editor_imports_with_their_own_account_not_the_owners(hub):
    ana, bob = logged_in("ana"), logged_in("bob")
    connect(ana)
    p = create_project(ana, categories=())
    ana.post(f"/api/projects/{p['id']}/members", json={"identifier": "bob", "role": "editor"})
    hub.add("ana/privado", cats_dataset(3), private=True)
    r = start(bob, p["id"], source="ana/privado")
    assert r.status_code in (403, 404)  # bob no conectó su cuenta: no hereda el acceso del propietario
    assert TOKEN not in hub.tokens_seen
    hub.add("ana/gatos", cats_dataset(3))
    assert start(bob, p["id"], source="ana/gatos").status_code == 201


def test_only_image_projects_can_import(hub):
    ana = logged_in("ana")
    p = ana.post("/api/projects", json={"name": "Sonidos", "modality": "audio", "categories": ["a"]})
    assert p.status_code == 201
    hub.add("ana/gatos", cats_dataset(3))
    r = start(ana, p.json()["id"])
    assert r.status_code == 409 and r.json()["code"] == "NOT_APPLICABLE"


def test_import_is_blocked_while_the_weekly_calendar_is_closed(hub):
    ana = logged_in("ana")
    p = create_project(ana, categories=())
    hub.add("ana/gatos", cats_dataset(3))
    calendar_rules.set_clock_override(SATURDAY)
    r = start(ana, p["id"])
    assert r.status_code == 423 and r.json()["code"] == "UPLOADS_LOCKED"


def test_calendar_closing_midway_stops_the_import_and_keeps_progress(hub, monkeypatch):
    ana = logged_in("ana")
    p = create_project(ana, categories=())
    hub.add("ana/gatos", cats_dataset(8))
    monkeypatch.setattr(hf, "CHECKPOINT_EVERY", 1)
    after_checkpoints(monkeypatch, 5, lambda runner: calendar_rules.set_clock_override(SATURDAY))
    start(ana, p["id"])
    imp = ana.get(f"/api/projects/{p['id']}/hf/imports").json()[0]
    assert imp["status"] == "error" and "bloquead" in imp["error"].lower()
    assert 4 <= imp["imported"] < 16 and project_state(ana, p["id"])[1] == imp["imported"]  # lo anterior al cierre se conserva


def test_only_one_active_import_per_user(hub, monkeypatch):
    ana = logged_in("ana")
    p = create_project(ana, categories=())
    hub.add("ana/gatos", cats_dataset(3))
    monkeypatch.setattr(hf, "kick_import", lambda _id: None)  # se queda «en cola»
    first = start(ana, p["id"])
    assert first.status_code == 201 and first.json()["status"] == "pending"
    r = start(ana, p["id"])
    assert r.status_code == 409 and r.json()["code"] == "IMPORT_IN_PROGRESS"


# ------------------------------------------------------------------------------ cancelar y recuperar
def test_cancel_a_queued_import_and_a_running_one(hub, monkeypatch):
    ana, bob = logged_in("ana"), logged_in("bob")
    p = create_project(ana, categories=())
    ana.post(f"/api/projects/{p['id']}/members", json={"identifier": "bob", "role": "editor"})
    hub.add("ana/gatos", cats_dataset(8))
    monkeypatch.setattr(hf, "kick_import", lambda _id: None)
    queued = start(ana, p["id"]).json()
    assert bob.delete(f"/api/hf/imports/{queued['id']}").status_code == 403  # un editor ajeno no la cancela
    r = ana.delete(f"/api/hf/imports/{queued['id']}")
    assert r.status_code == 200 and r.json()["status"] == "cancelled"
    hf.run_import(queued["id"])  # un worker que la recoja tarde no hace nada
    assert project_state(ana, p["id"])[1] == 0
    assert ana.delete(f"/api/hf/imports/{queued['id']}").status_code == 409

    # en curso: otra sesión (la petición del usuario) deja la marca y el worker la ve en su siguiente guardado
    monkeypatch.setattr(hf, "CHECKPOINT_EVERY", 1)
    running = start(ana, p["id"]).json()
    assert running["status"] == "pending"

    def user_clicks_cancel(runner):
        with SessionLocal() as other:
            other.get(models.HfImport, running["id"]).cancel_requested = True
            other.commit()

    after_checkpoints(monkeypatch, 4, user_clicks_cancel)
    hf.run_import(running["id"])
    imp = ana.get(f"/api/projects/{p['id']}/hf/imports").json()[0]
    assert imp["status"] == "cancelled" and 3 <= imp["imported"] <= 5  # se detiene en el guardado siguiente a la marca
    assert project_state(ana, p["id"])[1] == imp["imported"] < 16


def test_cancel_endpoint_marks_a_running_import(hub, monkeypatch):
    ana = logged_in("ana")
    p = create_project(ana, categories=())
    monkeypatch.setattr(hf, "kick_import", lambda _id: None)
    hub.add("ana/gatos", cats_dataset(3))
    imp = start(ana, p["id"]).json()
    with SessionLocal() as db:
        db.get(models.HfImport, imp["id"]).status = "running"
        db.commit()
    r = ana.delete(f"/api/hf/imports/{imp['id']}")
    assert r.status_code == 200 and r.json()["cancel_requested"] is True and r.json()["status"] == "running"
    assert logged_in("carl").delete(f"/api/hf/imports/{imp['id']}").status_code == 404


def test_stale_running_imports_are_marked_as_failed(hub):
    ana = logged_in("ana")
    p = create_project(ana)
    with SessionLocal() as db:
        uid = db.query(models.User).filter_by(username="ana").one().id
        old = utcnow().replace(year=2020)
        db.add(models.HfImport(project_id=p["id"], user_id=uid, repo_id="a/b", revision="x", status="running", options={}, heartbeat_at=old, started_at=old))
        db.add(models.HfImport(project_id=p["id"], user_id=uid, repo_id="a/c", revision="x", status="running", options={}, heartbeat_at=utcnow(), started_at=utcnow()))
        db.commit()
        assert hf.recover_stale(db) == 1
        states = {i.repo_id: i.status for i in db.query(models.HfImport)}
        assert states == {"a/b": "error", "a/c": "running"}


def test_unexpected_failure_is_reported_and_leaves_no_orphan_files(hub, monkeypatch):
    ana = logged_in("ana")
    p = create_project(ana, categories=())
    hub.add("ana/gatos", cats_dataset(8))
    boom = {"n": 0}
    real_save = hf.get_modality("image").save

    def flaky_save(project_id, v):
        boom["n"] += 1
        if boom["n"] == 5:
            raise MemoryError("sin memoria")
        return real_save(project_id, v)

    monkeypatch.setattr(hf.get_modality("image"), "save", flaky_save)
    start(ana, p["id"])
    imp = ana.get(f"/api/projects/{p['id']}/hf/imports").json()[0]
    assert imp["status"] == "error" and "MemoryError" in imp["error"] and "sin memoria" not in imp["error"]
    from app.storage import image_dir

    rows = project_state(ana, p["id"])[1]
    on_disk = len(list(image_dir(p["id"]).glob("*.png"))) if image_dir(p["id"]).exists() else 0
    assert on_disk == rows  # nada de archivos huérfanos sin fila en la base de datos


def test_token_never_shows_up_in_errors(hub):
    ana = logged_in("ana")
    connect(ana)
    p = create_project(ana, categories=())
    hub.add("ana/privado", cats_dataset(3), private=True)

    def fail(path):
        raise hf.HfError("HF_FILE_FAIL", f"fallo con {TOKEN}", 500)

    hub.on_download = fail
    start(ana, p["id"], source="ana/privado")
    imp = ana.get(f"/api/projects/{p['id']}/hf/imports").json()[0]
    assert imp["status"] == "error" and TOKEN not in str(imp)
