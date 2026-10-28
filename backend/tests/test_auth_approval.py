import pytest

from app import models, ratelimit
from app.cli import create_admin
from app.config import get_settings
from app.db import SessionLocal
from tests.conftest import PASSWORD, login, logged_in, make_user, new_client

REGISTER = {
    "first_name": "Ana", "last_name": "Pérez", "email": "ana@example.com", "username": "ana",
    "password": PASSWORD, "password_confirm": PASSWORD, "reason": "Clasificar piezas",
}


def register(client, **over):
    return client.post("/api/auth/register", json={**REGISTER, **over})


def admin_client():
    make_user("root", role=models.ROLE_ADMIN)
    c = new_client()
    assert login(c, "root").status_code == 200
    return c


def test_register_creates_pending_account_with_argon2id(client):
    r = register(client)
    assert r.status_code == 201
    assert "pendiente de aprobación" in r.json()["message"]
    with SessionLocal() as db:
        u = db.query(models.User).filter_by(username="ana").one()
        assert u.status == models.PENDING
        assert u.password_hash.startswith("$argon2id$")
        assert PASSWORD not in u.password_hash


def test_register_validations(client):
    assert register(client).status_code == 201
    assert register(client, username="otro").json()["code"] == "EMAIL_TAKEN"
    assert register(client, email="otra@example.com", username="ANA").json()["code"] == "USERNAME_TAKEN"
    assert register(client, email="x@example.com", username="x1", password_confirm="Distinta123").status_code == 422
    assert register(client, email="y@example.com", username="y1", password="corta1", password_confirm="corta1").status_code == 422
    assert register(client, email="z@example.com", username="z z").status_code == 422


def test_pending_user_can_only_see_status(client):
    register(client)
    r = login(client, "ana")
    assert r.status_code == 200
    me = client.get("/api/auth/me").json()
    assert me["status"] == "PENDING" and me["can_use_platform"] is False
    assert "pendiente de aprobación" in me["status_message"]
    for path in ("/api/projects", "/api/models", "/api/training/jobs", "/api/schedule", "/api/admin/dashboard"):
        resp = client.get(path)
        assert resp.status_code == 403, path
    assert client.post("/api/projects", json={"name": "x"}).status_code == 403


def test_admin_approves_and_user_gets_access_and_notification(client):
    admin = admin_client()
    register(client)
    pend = admin.get("/api/admin/registrations?status=PENDING").json()
    assert [u["username"] for u in pend] == ["ana"]
    # el admin recibió la notificación interna de la solicitud
    notes = admin.get("/api/notifications").json()
    assert any(n["kind"] == "registration.pending" for n in notes["items"])
    uid = pend[0]["id"]
    r = admin.post(f"/api/admin/users/{uid}/approve", json={})
    assert r.status_code == 200 and r.json()["status"] == "APPROVED" and r.json()["reviewed_by"] == "root"
    login(client, "ana")
    assert client.get("/api/projects").status_code == 200
    assert any(n["kind"] == "user.approved" for n in client.get("/api/notifications").json()["items"])


def test_reject_blocks_access_and_records_history(client):
    register(client)
    admin = admin_client()
    uid = admin.get("/api/admin/registrations").json()[0]["id"]
    assert admin.post(f"/api/admin/users/{uid}/reject", json={"note": "No cumple"}).status_code == 200
    login(client, "ana")
    assert client.get("/api/projects").status_code == 403
    assert "No cumple" in client.get("/api/auth/me").json()["status_message"]
    hist = admin.get(f"/api/admin/users/{uid}/history").json()
    assert [h["to_status"] for h in hist] == ["REJECTED", "PENDING"]
    assert hist[0]["actor"] == "root" and hist[0]["note"] == "No cumple"


def test_suspension_invalidates_active_sessions_and_reactivation(client):
    uid = make_user("bob")
    user = logged_in("bob")
    assert user.get("/api/projects").status_code == 200
    admin = admin_client()
    assert admin.post(f"/api/admin/users/{uid}/suspend", json={"note": "Abuso"}).status_code == 200
    r = user.get("/api/projects")  # la sesión previa ya no sirve
    assert r.status_code == 401
    assert login(new_client(), "bob").status_code == 403
    assert admin.post(f"/api/admin/users/{uid}/reactivate", json={}).status_code == 200
    # la cookie antigua sigue invalidada: se requiere un nuevo inicio de sesión
    assert user.get("/api/projects").status_code == 401
    assert login(user, "bob").status_code == 200
    assert user.get("/api/projects").status_code == 200


def test_non_admins_cannot_use_admin_endpoints_or_self_approve():
    pending_id = make_user("pen", status=models.PENDING)
    approved = logged_in("usr")
    pending = new_client()
    login(pending, "pen")
    for c in (approved, pending):
        assert c.get("/api/admin/dashboard").status_code == 403
        assert c.post(f"/api/admin/users/{pending_id}/approve", json={}).status_code == 403
        assert c.post("/api/admin/users/bulk-approve", json={"user_ids": [pending_id], "confirm": True}).status_code == 403
    with SessionLocal() as db:
        assert db.get(models.User, pending_id).status == models.PENDING


def test_admin_cannot_act_on_self_and_last_admin_is_protected():
    root = admin_client()
    with SessionLocal() as db:
        root_id = db.query(models.User).filter_by(username="root").one().id
    assert root.post(f"/api/admin/users/{root_id}/suspend", json={}).json()["code"] == "SELF_ACTION"
    assert root.delete(f"/api/admin/users/{root_id}").json()["code"] == "SELF_ACTION"
    assert root.patch(f"/api/admin/users/{root_id}", json={"role": "USER"}).json()["code"] == "SELF_ACTION"
    # con otro admin activo sí se puede gestionar a ese admin; el último siempre queda
    other_id = make_user("second", role=models.ROLE_ADMIN)
    assert root.post(f"/api/admin/users/{other_id}/suspend", json={}).status_code == 200
    with SessionLocal() as db:
        assert db.query(models.User).filter_by(role="ADMIN", status="APPROVED").count() == 1


def test_last_active_admin_cannot_be_removed_by_other_admin():
    from app.errors import ApiError
    from app.services.users import ensure_not_last_admin

    admin_id = make_user("solo", role=models.ROLE_ADMIN)
    with SessionLocal() as db:
        with pytest.raises(ApiError) as exc:
            ensure_not_last_admin(db, db.get(models.User, admin_id))
        assert exc.value.code == "LAST_ADMIN"


def test_bulk_approval_requires_confirmation():
    ids = [make_user(f"p{i}", status=models.PENDING) for i in range(3)]
    admin = admin_client()
    assert admin.post("/api/admin/users/bulk-approve", json={"user_ids": ids}).json()["code"] == "CONFIRMATION_REQUIRED"
    r = admin.post("/api/admin/users/bulk-approve", json={"user_ids": ids, "confirm": True})
    assert r.json()["approved"] == 3
    with SessionLocal() as db:
        assert all(db.get(models.User, i).status == models.APPROVED for i in ids)
        assert db.query(models.AuditLog).filter_by(action="user.approved").count() == 3


def test_login_rate_limit(client):
    register(client)
    codes = [login(client, "ana", "mala-clave1").status_code for _ in range(get_settings().login_max_attempts + 2)]
    assert codes[0] == 401 and codes[-1] == 429


def test_register_rate_limit(client):
    s = get_settings()
    codes = [register(client, email=f"u{i}@example.com", username=f"user{i}").status_code for i in range(s.register_max_attempts + 1)]
    assert codes[:-1] == [201] * s.register_max_attempts and codes[-1] == 429


def test_csrf_header_required():
    from fastapi.testclient import TestClient

    from app.main import app

    bare = TestClient(app)
    assert bare.post("/api/auth/login", json={"identifier": "a", "password": "b"}).json()["code"] == "CSRF"


def test_unauthenticated_requests_rejected(client):
    assert client.get("/api/projects").status_code == 401
    assert client.get("/api/admin/users").status_code == 401


def test_create_admin_command_and_no_default_credentials():
    with SessionLocal() as db:
        assert db.query(models.User).count() == 0  # sin usuarios predeterminados
    create_admin("jefe", "jefe@example.com", "Jefa", "Principal", "ClaveSegura123")
    c = new_client()
    r = login(c, "jefe", "ClaveSegura123")
    assert r.status_code == 200 and r.json()["role"] == "ADMIN" and r.json()["status"] == "APPROVED"
    with pytest.raises(SystemExit):
        create_admin("otro", "o@example.com", "O", "O", "corta")


def test_notifications_without_smtp_still_work(client):
    assert not get_settings().smtp_host
    admin = admin_client()
    register(client)
    assert admin.get("/api/notifications").json()["unread"] >= 1
    assert admin.post("/api/notifications/read-all").status_code == 200
    assert admin.get("/api/notifications").json()["unread"] == 0


def test_state_check_on_every_request_even_with_valid_cookie():
    uid = make_user("carl")
    c = logged_in("carl")
    with SessionLocal() as db:  # estado cambiado directamente en BD (sin pasar por la API)
        db.get(models.User, uid).status = models.REJECTED
        db.commit()
    assert c.get("/api/projects").status_code == 403
    assert c.get("/api/auth/me").json()["can_use_platform"] is False
    ratelimit.reset_memory()
