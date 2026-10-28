"""Prueba de humo contra una plataforma en ejecución (a través de Nginx).

    docker compose exec -e SMOKE_ADMIN_USER=admin -e SMOKE_ADMIN_PASSWORD=... api python smoke_test.py

Crea un usuario temporal, lo aprueba, sube imágenes SINTÉTICAS de prueba, fuerza un entrenamiento
manual (como administrador), sigue el progreso por SSE y verifica descargas e inferencia.
Al terminar elimina el usuario temporal (y con él sus proyectos, imágenes y modelos).
"""
import io
import json
import os
import sys
import threading
import time
import uuid
import zipfile

import httpx
import numpy as np
from PIL import Image

BASE = os.getenv("SMOKE_URL", "http://web")
H = {"X-Requested-With": "vision-lab"}


def img(kind: str, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    base = {"rojo": (200, 40, 40), "azul": (40, 40, 200)}[kind]
    arr = np.clip(np.array(base)[None, None, :] + rng.normal(0, 25, (96, 96, 3)), 0, 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def check(cond: bool, msg: str):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        sys.exit(1)


def main():
    admin_user, admin_pass = os.environ["SMOKE_ADMIN_USER"], os.environ["SMOKE_ADMIN_PASSWORD"]
    suffix = uuid.uuid4().hex[:6]
    uname, pw = f"smoke{suffix}", "SmokeTest12345"
    anon, admin, user = (httpx.Client(base_url=BASE, headers=H, timeout=120) for _ in range(3))

    print("1. Registro y aprobación")
    r = anon.post("/api/auth/register", json={"first_name": "Smoke", "last_name": "Test", "email": f"{uname}@example.com",
                                              "username": uname, "password": pw, "password_confirm": pw})
    check(r.status_code == 201, "registro creado (PENDING)")
    check(user.post("/api/auth/login", json={"identifier": uname, "password": pw}).status_code == 200, "pendiente puede iniciar sesión")
    check(user.get("/api/projects").status_code == 403, "pendiente NO accede a proyectos")
    check(admin.post("/api/auth/login", json={"identifier": admin_user, "password": admin_pass}).status_code == 200, "admin inicia sesión")
    uid = next(u["id"] for u in admin.get("/api/admin/registrations?status=PENDING").json() if u["username"] == uname)
    check(admin.post(f"/api/admin/users/{uid}/approve", json={}).status_code == 200, "admin aprueba")
    check(user.get("/api/projects").status_code == 200, "usuario aprobado accede")

    try:
        print("2. Proyecto e imágenes")
        sched = user.get("/api/schedule").json()
        print(f"     calendario: abierto={sched['open']} ({sched['reason']})")
        p = user.post("/api/projects", json={"name": f"Humo {suffix}", "categories": ["rojo", "azul"],
                                              "config": {"epochs": 2, "batch_size": 8}}).json()
        cats = {c["name"]: c["id"] for c in p["categories"]}
        if sched["open"]:
            for name in cats:
                files = [("files", (f"{name}{i}.png", img(name, i + (0 if name == "rojo" else 1000)), "image/png")) for i in range(12)]
                r = user.post(f"/api/projects/{p['id']}/images", data={"category_id": str(cats[name])}, files=files)
                check(r.status_code == 201 and len(r.json()["uploaded"]) == 12, f"12 imágenes subidas a {name}")
        else:
            print("     (cargas bloqueadas por el calendario: se omite la subida; es el comportamiento correcto)")
            return admin.delete(f"/api/admin/users/{uid}")

        print("3. Entrenamiento manual + progreso en tiempo real (SSE)")
        events: list[dict] = []

        def listen():
            with httpx.Client(base_url=BASE, headers=H, timeout=None, cookies=user.cookies) as c:
                with c.stream("GET", "/api/training/stream") as s:
                    for line in s.iter_lines():
                        if line.startswith("data:"):
                            events.append(json.loads(line[5:]))
                        if len(events) > 400 or (events and events[-1]["job"]["status"] in ("completed", "error")):
                            return

        t = threading.Thread(target=listen, daemon=True)
        t.start()
        time.sleep(1)
        r = admin.post("/api/admin/training/enqueue", json={"project_ids": [p["id"]]})
        check(r.status_code == 200, "entrenamiento encolado")
        check(user.post(f"/api/projects/{p['id']}/images", data={"category_id": str(cats["rojo"])},
                        files=[("files", ("x.png", img("rojo", 99), "image/png"))]).status_code == 423, "cargas bloqueadas mientras entrena (423)")
        deadline, job = time.time() + 1500, None
        while time.time() < deadline:
            jobs = user.get(f"/api/training/jobs?project_id={p['id']}").json()
            job = jobs[0] if jobs else None
            if job and job["status"] in ("completed", "error", "cancelled"):
                break
            time.sleep(5)
        check(job is not None and job["status"] == "completed", f"entrenamiento completado ({job and job['status']}: {job and job['error']})")
        t.join(timeout=10)
        statuses = {e["job"]["status"] for e in events}
        check(len(events) >= 5 and {"training", "completed"} <= statuses, f"{len(events)} eventos SSE recibidos ({sorted(statuses)})")

        print("4. Modelo: descargas e inferencia")
        models = user.get(f"/api/projects/{p['id']}/models").json()
        m = models[0]
        check([e["status"] for e in m["exports"]] == ["ok"] * 5, "5 formatos validados")
        for fmt in ("keras", "h5", "savedmodel", "tflite", "onnx", "labels", "metadata", "metrics", "inference", "all"):
            check(user.get(f"/api/models/{m['id']}/download/{fmt}").status_code == 200, f"descarga {fmt}")
        z = zipfile.ZipFile(io.BytesIO(user.get(f"/api/models/{m['id']}/download/all").content))
        check({"model.onnx", "model.tflite", "model.keras", "model.h5", "saved_model.zip"} <= set(z.namelist()), "ZIP completo con los 5 formatos")
        r = user.post(f"/api/models/{m['id']}/predict", files={"file": ("t.png", img("azul", 4242), "image/png")}).json()
        check(r["prediction"] == "azul" and len(r["scores"]) == 2, f"inferencia correcta: {r['prediction']} ({r['confidence']:.2f}), v{r['model_version']}")

        print("5. Publicación y reapertura")
        time.sleep(3)
        print("     calendario:", user.get("/api/schedule").json()["reason"])
    finally:
        r = admin.delete(f"/api/admin/users/{uid}")
        print("limpieza: usuario temporal eliminado" if r.status_code == 204 else f"limpieza falló: {r.status_code} {r.text}")
    print("PRUEBA DE HUMO COMPLETADA")


if __name__ == "__main__":
    main()
