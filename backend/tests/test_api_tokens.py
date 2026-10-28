from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

from app import calendar_rules, models
from app.db import SessionLocal
from app.main import app
from app.security import hash_api_token
from tests.conftest import create_project, logged_in, make_user, png_bytes, upload


def bearer(token: str) -> TestClient:
    """Cliente sin cookies ni cabecera CSRF: sólo el token de API."""
    return TestClient(app, headers={"Authorization": f"Bearer {token}"})


def read_token(web: TestClient, name="lectura", **body) -> str:
    r = web.post("/api/tokens", json={"name": name, **body})
    assert r.status_code == 201, r.text
    return r.json()["token"]


def write_token(web: TestClient, admin: TestClient, name="escritura", days=None) -> str:
    r = web.post("/api/tokens/requests", json={"name": name, "reason": "automatizar la carga de datos"})
    assert r.status_code == 201, r.text
    rid = r.json()["id"]
    r = admin.post(f"/api/admin/token-requests/{rid}/approve", json={"expires_days": days} if days else {})
    assert r.status_code == 200, r.text
    r = web.post(f"/api/tokens/requests/{rid}/generate")
    assert r.status_code == 201, r.text
    return r.json()["token"]


def png(i, kind="red"):
    return (f"{kind}{i}.png", png_bytes(kind, i), "image/png")


# ------------------------------------------------------------------ lectura
def test_read_token_is_shown_once_and_only_its_hash_is_stored():
    ana = logged_in("ana")
    r = ana.post("/api/tokens", json={"name": "panel", "expires_days": 30})
    assert r.status_code == 201
    body = r.json()
    token = body["token"]
    assert token.startswith("vl_") and body["prefix"] == token[:11] and body["scope"] == "read"
    assert r.headers["cache-control"] == "no-store"
    listing = ana.get("/api/tokens").json()
    assert "token" not in listing["tokens"][0] and listing["tokens"][0]["state"] == "active"
    assert token not in ana.get("/api/tokens").text
    with SessionLocal() as db:
        row = db.query(models.ApiToken).one()
        assert row.token_hash == hash_api_token(token) and token not in (row.token_hash, row.prefix)
        assert row.expires_at is not None


def test_read_token_sees_own_and_shared_projects_but_nothing_else():
    ana, bob, carl = logged_in("ana"), logged_in("bob"), logged_in("carl")
    own = create_project(bob, name="De bob")
    shared = create_project(ana, name="De ana")
    ana.post(f"/api/projects/{shared['id']}/members", json={"identifier": "bob", "role": "viewer"})
    secret = create_project(carl, name="De carl")
    api = bearer(read_token(bob))
    assert {p["id"] for p in api.get("/api/projects").json()} == {own["id"], shared["id"]}
    assert api.get(f"/api/projects/{shared['id']}").json()["my_role"] == "viewer"
    assert api.get(f"/api/projects/{secret['id']}").status_code == 404  # sigue siendo privado
    assert api.get("/api/training/jobs").status_code == 200
    assert api.get("/api/auth/me").json()["username"] == "bob"


def test_read_token_can_follow_training_progress_like_a_shared_project():
    ana, bob = logged_in("ana"), logged_in("bob")
    p = create_project(ana, name="Progreso")
    ana.post(f"/api/projects/{p['id']}/members", json={"identifier": "bob", "role": "viewer"})
    with SessionLocal() as db:
        job = models.TrainingJob(project_id=p["id"], status="training", base_model="mobilenetv3", config={}, percent=42.0)
        db.add(job)
        db.commit()
        job_id = job.id
    api = bearer(read_token(bob))
    jobs = api.get(f"/api/training/jobs?project_id={p['id']}").json()
    assert [j["id"] for j in jobs] == [job_id] and jobs[0]["percent"] == 42.0
    assert api.get(f"/api/training/jobs/{job_id}").status_code == 200
    assert api.get(f"/api/training/jobs/{job_id}/logs").status_code == 200


def test_read_token_cannot_modify_anything():
    ana = logged_in("ana")
    p = create_project(ana, name="Mio")
    cats = {c["name"]: c["id"] for c in p["categories"]}
    api = bearer(read_token(ana))
    for call in (
        lambda: api.post("/api/projects", json={"name": "Nuevo", "categories": ["a", "b"]}),
        lambda: api.patch(f"/api/projects/{p['id']}", json={"description": "x"}),
        lambda: api.delete(f"/api/projects/{p['id']}"),
        lambda: upload(api, p["id"], cats["rojo"], [png(1)]),
        lambda: api.post("/api/notifications/read-all"),
    ):
        r = call()
        assert r.status_code == 403 and r.json()["code"] == "TOKEN_READ_ONLY", r.text
    assert ana.get(f"/api/projects/{p['id']}").status_code == 200  # nada cambió


# ---------------------------------------------------------- validez del token
def test_invalid_revoked_and_expired_tokens_are_rejected():
    ana = logged_in("ana")
    tok = read_token(ana)
    assert bearer(tok).get("/api/projects").status_code == 200
    for bad in ("vl_" + "x" * 43, "no-es-un-token", ""):
        r = bearer(bad).get("/api/projects")
        assert r.status_code == 401 and (r.json()["code"] == "INVALID_API_TOKEN" or bad == ""), r.text
    exp = read_token(ana, name="vence")
    with SessionLocal() as db:
        t = db.query(models.ApiToken).filter_by(name="vence").one()
        t.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        db.commit()
    assert bearer(exp).get("/api/projects").status_code == 401
    mine = next(t for t in ana.get("/api/tokens").json()["tokens"] if t["state"] == "active")
    assert ana.delete(f"/api/tokens/{mine['id']}").status_code == 204
    assert bearer(tok).get("/api/projects").status_code == 401
    states = {t["name"]: t["state"] for t in ana.get("/api/tokens").json()["tokens"]}
    assert states == {"lectura": "revoked", "vence": "expired"}


def test_token_wins_over_cookie_and_they_are_never_mixed():
    ana, bob = logged_in("ana"), logged_in("bob")
    bob_tok = read_token(bob)
    r = ana.get("/api/auth/me", headers={"Authorization": f"Bearer {bob_tok}"})  # ana trae su cookie
    assert r.json()["username"] == "bob"
    r = ana.get("/api/auth/me", headers={"Authorization": "Bearer vl_" + "y" * 43})
    assert r.status_code == 401  # un token inválido no cae de vuelta a la cookie


def test_token_usage_is_recorded_and_rate_limited(monkeypatch):
    ana = logged_in("ana")
    api = bearer(read_token(ana))
    assert api.get("/api/projects").status_code == 200
    with SessionLocal() as db:
        t = db.query(models.ApiToken).one()
        assert t.last_used_at is not None
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "api_token_rate_per_minute", 3)
    codes = [api.get("/api/projects").status_code for _ in range(5)]
    assert codes[:2] == [200, 200] and codes[-1] == 429


def test_token_failures_are_rate_limited_per_ip(monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "api_token_fail_max", 3)
    api = bearer("vl_" + "z" * 43)
    codes = [api.get("/api/projects").status_code for _ in range(5)]
    assert codes[:3] == [401, 401, 401] and codes[-1] == 429


def test_read_token_limit_per_user():
    from app.config import get_settings

    ana = logged_in("ana")
    for i in range(get_settings().api_max_read_tokens):
        read_token(ana, name=f"t{i}")
    r = ana.post("/api/tokens", json={"name": "uno más"})
    assert r.status_code == 409 and r.json()["code"] == "TOKEN_LIMIT"
    other = logged_in("bob")
    assert other.post("/api/tokens", json={"name": "ok"}).status_code == 201  # el límite es por usuario


def test_only_the_owner_can_see_and_revoke_a_token():
    ana, bob = logged_in("ana"), logged_in("bob")
    read_token(ana)
    tid = ana.get("/api/tokens").json()["tokens"][0]["id"]
    assert bob.get("/api/tokens").json()["tokens"] == []
    assert bob.delete(f"/api/tokens/{tid}").status_code == 404
    assert ana.delete(f"/api/tokens/{tid}").status_code == 204


# ------------------------------------------------------------------- escritura
def test_write_token_requires_admin_approval_and_is_generated_once():
    ana = logged_in("ana")
    admin = logged_in("root", role=models.ROLE_ADMIN)
    r = ana.post("/api/tokens/requests", json={"name": "cargador", "reason": "subir imágenes desde mi script"})
    assert r.status_code == 201 and r.json()["status"] == "pending" and r.json()["can_generate"] is False
    rid = r.json()["id"]
    # sin aprobación no hay token
    assert ana.post(f"/api/tokens/requests/{rid}/generate").json()["code"] == "REQUEST_NOT_APPROVED"
    # un usuario común no puede aprobar, ni siquiera la suya
    assert ana.post(f"/api/admin/token-requests/{rid}/approve", json={}).status_code == 403
    # el administrador la ve y la aprueba
    pend = admin.get("/api/admin/token-requests?status=pending").json()
    assert [(x["id"], x["owner"]["username"]) for x in pend] == [(rid, "ana")]
    assert admin.get("/api/admin/dashboard").json()["token_requests_pending"] == 1
    r = admin.post(f"/api/admin/token-requests/{rid}/approve", json={"expires_days": 30, "note": "ok"})
    assert r.status_code == 200 and r.json()["status"] == "approved" and r.json()["expires_days"] == 30
    assert admin.post(f"/api/admin/token-requests/{rid}/approve", json={}).json()["code"] == "INVALID_STATE"
    # el usuario genera el token una sola vez
    g = ana.post(f"/api/tokens/requests/{rid}/generate")
    assert g.status_code == 201 and g.json()["scope"] == "write" and g.json()["token"].startswith("vl_")
    assert ana.post(f"/api/tokens/requests/{rid}/generate").json()["code"] == "TOKEN_ALREADY_ISSUED"
    with SessionLocal() as db:
        t = db.query(models.ApiToken).one()
        assert t.scope == "write" and t.expires_at is not None
        assert 29 <= (t.expires_at - datetime.now(timezone.utc)).days <= 30
    assert admin.get("/api/admin/dashboard").json()["token_requests_pending"] == 0


def test_write_token_acts_within_the_users_privileges():
    ana, bob = logged_in("ana"), logged_in("bob")
    admin = logged_in("root", role=models.ROLE_ADMIN)
    shared = create_project(ana, name="Compartido")
    cats = {c["name"]: c["id"] for c in shared["categories"]}
    ana.post(f"/api/projects/{shared['id']}/members", json={"identifier": "bob", "role": "viewer"})
    secret = create_project(logged_in("carl"), name="Ajeno")
    api = bearer(write_token(bob, admin))  # sin cabecera CSRF: con Bearer no hace falta
    # puede crear lo suyo
    r = api.post("/api/projects", json={"name": "Propio", "categories": ["a", "b"]})
    assert r.status_code == 201 and r.json()["owner"] == "bob"
    # como lector del proyecto compartido sigue siendo lector
    r = upload(api, shared["id"], cats["rojo"], [png(1)])
    assert r.status_code == 403 and r.json()["code"] == "PROJECT_FORBIDDEN"
    assert api.patch(f"/api/projects/{shared['id']}", json={"config": {"epochs": 3}}).status_code == 403
    # y no ve ni toca proyectos ajenos
    assert api.get(f"/api/projects/{secret['id']}").status_code == 404
    assert api.delete(f"/api/projects/{secret['id']}").status_code == 404
    # si el propietario lo sube a editor, el mismo token gana ese permiso, ni más ni menos
    bob_id = ana.get(f"/api/projects/{shared['id']}/members").json()["members"][0]["user_id"]
    ana.patch(f"/api/projects/{shared['id']}/members/{bob_id}", json={"role": "editor"})
    assert upload(api, shared["id"], cats["rojo"], [png(1)]).status_code == 201
    assert api.delete(f"/api/projects/{shared['id']}").status_code == 403  # eliminar es sólo del propietario


def test_write_token_still_obeys_the_weekly_calendar():
    ana = logged_in("ana")
    admin = logged_in("root", role=models.ROLE_ADMIN)
    p = create_project(ana, name="Calendario")
    cats = {c["name"]: c["id"] for c in p["categories"]}
    api = bearer(write_token(ana, admin))
    assert upload(api, p["id"], cats["rojo"], [png(1)]).status_code == 201
    calendar_rules.set_clock_override(datetime(2026, 10, 24, 12, 0, tzinfo=ZoneInfo("America/Santiago")))  # sábado
    r = upload(api, p["id"], cats["rojo"], [png(2)])
    assert r.status_code == 423 and r.json()["code"] == "UPLOADS_LOCKED"


def test_tokens_never_reach_administration_or_token_management():
    admin = logged_in("root", role=models.ROLE_ADMIN)
    second = logged_in("root2", role=models.ROLE_ADMIN)
    user_tok = write_token(logged_in("dan"), admin)
    admin_tok = write_token(admin, second)  # un administrador con token de escritura (lo aprueba otro)
    api = bearer(admin_tok)
    assert api.get("/api/projects").status_code == 200  # sí puede usar la plataforma
    for r in (
        api.get("/api/admin/dashboard"),
        api.get("/api/admin/users"),
        api.post("/api/admin/token-requests/1/approve", json={}),
        api.get("/api/tokens"),
        api.post("/api/tokens", json={"name": "otro"}),
        api.post("/api/tokens/requests", json={"name": "x", "reason": "quiero más permisos"}),
    ):
        assert r.status_code == 403 and r.json()["code"] == "SESSION_REQUIRED", r.text
    assert bearer(user_tok).get("/api/admin/dashboard").status_code == 403
    assert admin.get("/api/admin/dashboard").status_code == 200  # la sesión web del administrador sigue igual


def test_admin_cannot_resolve_own_request_and_can_reject_with_a_reason():
    admin = logged_in("root", role=models.ROLE_ADMIN)
    second = logged_in("root2", role=models.ROLE_ADMIN)
    rid = admin.post("/api/tokens/requests", json={"name": "mio", "reason": "mi propia automatización"}).json()["id"]
    r = admin.post(f"/api/admin/token-requests/{rid}/approve", json={})
    assert r.status_code == 403 and r.json()["code"] == "SELF_ACTION"
    ana = logged_in("ana")
    aid = ana.post("/api/tokens/requests", json={"name": "x", "reason": "lo necesito para un script"}).json()["id"]
    r = second.post(f"/api/admin/token-requests/{aid}/reject", json={"note": "usa el token de lectura"})
    assert r.status_code == 200 and r.json()["status"] == "rejected"
    mine = ana.get("/api/tokens").json()["requests"][0]
    assert mine["status"] == "rejected" and mine["review_note"] == "usa el token de lectura" and not mine["can_generate"]
    assert ana.post(f"/api/tokens/requests/{aid}/generate").status_code == 409
    with SessionLocal() as db:
        n = db.query(models.Notification).filter_by(kind="apitoken.write_rejected").one()
        assert "usa el token de lectura" in n.body
        assert db.query(models.AuditLog).filter_by(action="apitoken.write_rejected").count() == 1


def test_only_one_write_token_or_open_request_per_user_and_cancel():
    ana = logged_in("ana")
    admin = logged_in("root", role=models.ROLE_ADMIN)
    body = {"name": "a", "reason": "automatizar cargas"}
    rid = ana.post("/api/tokens/requests", json=body).json()["id"]
    assert ana.post("/api/tokens/requests", json=body).json()["code"] == "TOKEN_LIMIT"  # ya hay una en curso
    assert ana.delete(f"/api/tokens/requests/{rid}").status_code == 204
    rid = ana.post("/api/tokens/requests", json=body).json()["id"]  # cancelada: puede pedir otra
    admin.post(f"/api/admin/token-requests/{rid}/approve", json={})
    ana.post(f"/api/tokens/requests/{rid}/generate")
    assert ana.post("/api/tokens/requests", json=body).json()["code"] == "TOKEN_LIMIT"  # ya tiene un token activo
    assert ana.delete(f"/api/tokens/requests/{rid}").status_code == 409  # ya generado: no se cancela
    tid = next(t["id"] for t in ana.get("/api/tokens").json()["tokens"] if t["scope"] == "write")
    ana.delete(f"/api/tokens/{tid}")
    assert ana.post("/api/tokens/requests", json=body).status_code == 201  # al revocarlo puede pedir otro


def test_request_input_is_validated():
    ana = logged_in("ana")
    assert ana.post("/api/tokens/requests", json={"name": "", "reason": "motivo válido"}).status_code == 422
    assert ana.post("/api/tokens/requests", json={"name": "x", "reason": "  "}).status_code == 422
    assert ana.post("/api/tokens", json={"name": "   "}).status_code == 422
    assert ana.post("/api/tokens", json={"name": "x", "expires_days": 0}).status_code == 422
    assert ana.post("/api/tokens", json={"name": "x", "expires_days": 5000}).status_code == 422


# ------------------------------------------------------------ revocación / cuentas
def test_suspending_a_user_revokes_all_their_tokens_for_good():
    ana = logged_in("ana")
    admin = logged_in("root", role=models.ROLE_ADMIN)
    tok = read_token(ana)
    wtok = write_token(ana, admin)
    ana_id = ana.get("/api/auth/me").json()["id"]
    assert admin.post(f"/api/admin/users/{ana_id}/suspend", json={"note": "prueba"}).status_code == 200
    assert bearer(tok).get("/api/projects").status_code == 401
    assert bearer(wtok).get("/api/projects").status_code == 401
    assert admin.post(f"/api/admin/users/{ana_id}/reactivate", json={}).status_code == 200
    assert bearer(tok).get("/api/projects").status_code == 401  # reactivar no resucita tokens
    assert ana.get("/api/auth/me").status_code == 401  # su sesión web también se invalidó al suspenderla
    again = logged_in("ana")
    assert again.post("/api/tokens", json={"name": "nuevo"}).status_code == 201  # pero puede crear otros


def test_pending_or_unapproved_users_cannot_use_tokens():
    make_user("nuevo", status=models.PENDING)
    ana = logged_in("ana")
    tok = read_token(ana)
    with SessionLocal() as db:
        u = db.query(models.User).filter_by(username="ana").one()
        u.status = models.PENDING
        db.commit()
    r = bearer(tok).get("/api/projects")
    assert r.status_code == 403 and r.json()["code"] == "ACCOUNT_NOT_APPROVED"


def test_admin_can_list_and_revoke_any_token_and_user_is_told():
    ana = logged_in("ana")
    admin = logged_in("root", role=models.ROLE_ADMIN)
    tok = read_token(ana, name="panel")
    rows = admin.get("/api/admin/api-tokens").json()
    assert len(rows) == 1 and rows[0]["owner"]["username"] == "ana" and "token" not in rows[0] and tok not in str(rows)
    assert admin.post(f"/api/admin/api-tokens/{rows[0]['id']}/revoke").json()["state"] == "revoked"
    assert bearer(tok).get("/api/projects").status_code == 401
    assert ana.get("/api/auth/me").status_code == 200  # la sesión web sigue intacta
    with SessionLocal() as db:
        assert db.query(models.Notification).filter_by(kind="apitoken.revoked").count() == 1
    assert ana.get("/api/admin/api-tokens").status_code == 403


def test_token_lifecycle_is_audited_without_leaking_the_secret():
    ana = logged_in("ana")
    admin = logged_in("root", role=models.ROLE_ADMIN)
    tok = read_token(ana)
    wtok = write_token(ana, admin)
    ana.delete(f"/api/tokens/{ana.get('/api/tokens').json()['tokens'][0]['id']}")
    with SessionLocal() as db:
        actions = [a.action for a in db.query(models.AuditLog).order_by(models.AuditLog.id)]
        blob = str([(a.detail, a.target_id) for a in db.query(models.AuditLog)])
    for expected in ("apitoken.created", "apitoken.write_requested", "apitoken.write_approved", "apitoken.revoked"):
        assert expected in actions
    assert tok not in blob and wtok not in blob


def test_deleting_a_user_removes_their_tokens_and_requests():
    ana = logged_in("ana")
    admin = logged_in("root", role=models.ROLE_ADMIN)
    tok = read_token(ana)
    ana.post("/api/tokens/requests", json={"name": "x", "reason": "motivo suficiente"})
    ana_id = ana.get("/api/auth/me").json()["id"]
    assert admin.delete(f"/api/admin/users/{ana_id}").status_code == 204
    with SessionLocal() as db:
        assert db.query(models.ApiToken).count() == 0 and db.query(models.ApiTokenRequest).count() == 0
    assert bearer(tok).get("/api/projects").status_code == 401
