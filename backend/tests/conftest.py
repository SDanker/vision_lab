import io
import os
import shutil
import tempfile
from datetime import datetime
from zoneinfo import ZoneInfo

_TMP = tempfile.mkdtemp(prefix="vision-tests-")
os.environ.update(
    DATABASE_URL=f"sqlite:///{_TMP}/test.db",
    REDIS_URL="",
    SECRET_KEY="test-secret-key-with-more-than-32-characters",
    DATA_DIR=f"{_TMP}/data",
    MIN_IMAGES_PER_CLASS="6",
    TZ="America/Santiago",
)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from app import calendar_rules, models, ratelimit  # noqa: E402
from app.db import Base, SessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.security import hash_password  # noqa: E402
from app.services import similarity, training_queue as tq  # noqa: E402

TZ = ZoneInfo("America/Santiago")
WEDNESDAY = datetime(2026, 10, 21, 12, 0, tzinfo=TZ)
PASSWORD = "Password123"


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    shutil.rmtree(os.environ["DATA_DIR"], ignore_errors=True)
    ratelimit.reset_memory()
    calendar_rules.set_clock_override(WEDNESDAY)
    monkeypatch.setattr(tq, "kick_queue", lambda: None)
    monkeypatch.setattr(similarity, "kick_build", lambda model_id: None)  # sin broker en las pruebas; se llama a build_index directo
    yield
    calendar_rules.set_clock_override(None)


def new_client() -> TestClient:
    return TestClient(app, headers={"X-Requested-With": "vision-lab"})


@pytest.fixture
def client():
    return new_client()


@pytest.fixture
def db():
    with SessionLocal() as session:
        yield session


def make_user(username, status=models.APPROVED, role=models.ROLE_USER, password=PASSWORD, **kw):
    with SessionLocal() as db:
        u = models.User(
            first_name=kw.pop("first_name", "Test"), last_name=kw.pop("last_name", username.title()),
            email=f"{username}@example.com", username=username, password_hash=hash_password(password),
            status=status, role=role, **kw,
        )
        db.add(u)
        db.commit()
        return u.id


def login(client: TestClient, username: str, password: str = PASSWORD):
    return client.post("/api/auth/login", json={"identifier": username, "password": password})


def logged_in(username: str, **kw) -> TestClient:
    """Crea (si no existe) un usuario aprobado y devuelve un cliente con sesión iniciada."""
    with SessionLocal() as db:
        exists = db.query(models.User).filter_by(username=username).first()
    if not exists:
        make_user(username, **kw)
    c = new_client()
    assert login(c, username).status_code == 200
    return c


def png_bytes(kind: str = "red", seed: int = 0, size: int = 96, fmt: str = "PNG") -> bytes:
    rng = np.random.default_rng(seed)
    base = {"red": (200, 40, 40), "blue": (40, 40, 200), "green": (40, 200, 40)}[kind]
    arr = np.clip(np.array(base)[None, None, :] + rng.normal(0, 25, (size, size, 3)), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, fmt)
    return buf.getvalue()


def create_project(client: TestClient, name="Proyecto", categories=("rojo", "azul"), **extra):
    r = client.post("/api/projects", json={"name": name, "categories": list(categories), **extra})
    assert r.status_code == 201, r.text
    return r.json()


def upload(client: TestClient, project_id: int, category_id: int, files):
    return client.post(
        f"/api/projects/{project_id}/images",
        data={"category_id": str(category_id)},
        files=[("files", (n, d, m)) for n, d, m in files],
    )
