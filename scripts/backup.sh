#!/bin/sh
# Respaldo completo: base de datos (pg_dump) + volumen de datos (imágenes, snapshots, modelos).
# Uso: ./scripts/backup.sh [directorio_destino]     (por defecto ./backups)
set -e
cd "$(dirname "$0")/.."
DEST="${1:-./backups}"
STAMP="$(date +%Y%m%d-%H%M%S)"
mkdir -p "$DEST"

echo "[1/2] Base de datos..."
docker compose exec -T db pg_dump -U vision -d vision --format=custom > "$DEST/db-$STAMP.dump"

echo "[2/2] Archivos (imágenes, modelos, checkpoints)..."
# El volumen se lee con un contenedor temporal; `work/` (temporales de entrenamiento) se excluye.
docker compose run --rm --no-deps -T --entrypoint sh api -c \
  "tar -C /data --exclude=work --exclude=keras -czf - ." > "$DEST/data-$STAMP.tar.gz"

echo "Respaldo listo en $DEST: db-$STAMP.dump, data-$STAMP.tar.gz"
echo "Recomendación: ejecútalo a diario por cron y copia los archivos fuera del servidor."
