"""Comandos de administración.

    python -m app.cli create-admin --username admin --email admin@ejemplo.cl --first-name Ana --last-name Pérez
        (la contraseña se toma de ADMIN_PASSWORD o se pide por consola sin eco)
    python -m app.cli bootstrap-admin
        (usa ADMIN_USERNAME/ADMIN_EMAIL/ADMIN_PASSWORD; no hace nada si ya existe un administrador)
    python -m app.cli build-similarity [--model ID ...] [--now]
        (precalcula «lo más parecido»: por defecto el mejor modelo de cada proyecto, en segundo plano)
"""
import argparse
import getpass
import os
import sys

from sqlalchemy import func, select

from .config import check_secret, get_settings
from .db import SessionLocal
from .models import APPROVED, ROLE_ADMIN, User, UserStatusHistory
from .schemas import USERNAME_RE
from .security import hash_password, password_error


def create_admin(username: str, email: str, first: str, last: str, password: str) -> str:
    if not USERNAME_RE.match(username):
        raise SystemExit("Nombre de usuario inválido (3-32 caracteres: letras, números, . _ -).")
    err = password_error(password)
    if err:
        raise SystemExit(err)
    with SessionLocal() as db:
        if db.scalar(select(User.id).where(func.lower(User.username) == username.lower())) or db.scalar(
            select(User.id).where(func.lower(User.email) == email.lower())
        ):
            raise SystemExit("Ya existe un usuario con ese nombre de usuario o correo.")
        user = User(
            first_name=first, last_name=last, email=email.lower(), username=username.lower(),
            password_hash=hash_password(password), role=ROLE_ADMIN, status=APPROVED,
            review_note="Administrador inicial",
        )
        db.add(user)
        db.flush()
        db.add(UserStatusHistory(user_id=user.id, from_status=None, to_status=APPROVED, note="Administrador inicial"))
        db.commit()
    return f"Administrador '{username}' creado."


def build_similarity(model_ids: list[int] | None, now: bool) -> int:
    """Precalcula el índice de «lo más parecido» (por defecto, del mejor modelo de cada proyecto)."""
    from .models import ModelVersion
    from .services import similarity

    with SessionLocal() as db:
        if model_ids:
            ids = model_ids
        else:
            best: dict[int, ModelVersion] = {}
            for m in db.scalars(select(ModelVersion).order_by(ModelVersion.version)):
                acc = (m.metrics or {}).get("test_accuracy") or -1
                cur = best.get(m.project_id)
                if cur is None or acc >= ((cur.metrics or {}).get("test_accuracy") or -1):
                    best[m.project_id] = m
            ids = [m.id for m in best.values()]
    for mid in ids:
        if now:
            print(f"Modelo {mid}: calculando…", flush=True)
            print(f"Modelo {mid}: {similarity.build_index(mid)} vectores nuevos.")
        else:
            similarity.kick_build(mid)
            print(f"Modelo {mid}: encolado en el worker liviano.")
    return 0


def main(argv: list[str] | None = None) -> int:
    check_secret(get_settings())
    ap = argparse.ArgumentParser(prog="app.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create-admin")
    c.add_argument("--username", default=os.getenv("ADMIN_USERNAME"))
    c.add_argument("--email", default=os.getenv("ADMIN_EMAIL"))
    c.add_argument("--first-name", default=os.getenv("ADMIN_FIRST_NAME", "Administrador"))
    c.add_argument("--last-name", default=os.getenv("ADMIN_LAST_NAME", "Vision Lab"))
    sub.add_parser("bootstrap-admin")
    b = sub.add_parser("build-similarity")
    b.add_argument("--model", type=int, action="append", help="id de modelo (se puede repetir); sin él, el mejor de cada proyecto")
    b.add_argument("--now", action="store_true", help="calcula aquí y ahora en vez de encolarlo en el worker")
    args = ap.parse_args(argv)

    if args.cmd == "bootstrap-admin":
        with SessionLocal() as db:
            if db.scalar(select(User.id).where(User.role == ROLE_ADMIN).limit(1)):
                print("Ya existe un administrador; no se crea ninguno.")
                return 0
        user, email, password = os.getenv("ADMIN_USERNAME"), os.getenv("ADMIN_EMAIL"), os.getenv("ADMIN_PASSWORD")
        if not (user and email and password):
            print("Sin ADMIN_USERNAME/ADMIN_EMAIL/ADMIN_PASSWORD: omitiendo. Usa `python -m app.cli create-admin`.")
            return 0
        print(create_admin(user, email, os.getenv("ADMIN_FIRST_NAME", "Administrador"), os.getenv("ADMIN_LAST_NAME", "Vision Lab"), password))
        return 0

    if args.cmd == "build-similarity":
        return build_similarity(args.model, args.now)

    if not args.username or not args.email:
        print("Faltan --username y --email (o ADMIN_USERNAME / ADMIN_EMAIL).", file=sys.stderr)
        return 2
    password = os.getenv("ADMIN_PASSWORD")
    if not password:
        password = getpass.getpass("Contraseña del administrador: ")
        if password != getpass.getpass("Repite la contraseña: "):
            print("Las contraseñas no coinciden.", file=sys.stderr)
            return 2
    print(create_admin(args.username, args.email, args.first_name, args.last_name, password))
    return 0


if __name__ == "__main__":
    sys.exit(main())
