#!/bin/sh
# Restaura un respaldo creado con backup.sh. DESTRUCTIVO: reemplaza la base de datos y los archivos actuales.
# Uso: ./scripts/restore.sh backups/db-AAAAMMDD-HHMMSS.dump backups/data-AAAAMMDD-HHMMSS.tar.gz
set -e
cd "$(dirname "$0")/.."
[ $# -eq 2 ] || { echo "Uso: $0 <db.dump> <data.tar.gz>"; exit 1; }
printf "Esto reemplazará TODOS los datos actuales. Escribe RESTAURAR para continuar: "
read -r answer
[ "$answer" = "RESTAURAR" ] || { echo "Cancelado."; exit 1; }

docker compose stop web worker-train worker-light beat api
docker compose up -d db
sleep 5
docker compose exec -T db psql -U vision -d postgres -c "DROP DATABASE IF EXISTS vision;" -c "CREATE DATABASE vision OWNER vision;"
docker compose exec -T db pg_restore -U vision -d vision --no-owner < "$1"
docker compose run --rm --no-deps -T --entrypoint sh api -c "find /data -mindepth 1 -maxdepth 1 ! -name keras -exec rm -rf {} + && tar -C /data -xzf -" < "$2"
docker compose up -d
echo "Restauración completada."
