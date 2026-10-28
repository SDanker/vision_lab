#!/bin/sh
# Genera .env con secretos aleatorios a partir de .env.example (no sobrescribe un .env existente).
set -e
cd "$(dirname "$0")/.."
if [ -f .env ]; then
  echo ".env ya existe; no se modifica."
  exit 0
fi
rand() { head -c 48 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c "$1"; }
fernet() { head -c 32 /dev/urandom | base64 | tr '+/' '-_' | tr -d '\n'; }  # clave Fernet: 32 bytes en base64 url-safe
sed \
  -e "s/CHANGE_ME_POSTGRES_PASSWORD/$(rand 32)/" \
  -e "s/CHANGE_ME_SECRET_KEY_AT_LEAST_32_CHARS/$(rand 64)/" \
  -e "s/CHANGE_ME_HF_ENCRYPTION_KEY/$(fernet)/" \
  .env.example > .env
chmod 600 .env 2>/dev/null || true
echo ".env creado con secretos aleatorios."
echo "Siguiente paso: docker compose up -d --build"
echo "Luego crea el administrador: docker compose exec api python -m app.cli create-admin --username admin --email tu@correo.cl"
