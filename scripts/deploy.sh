#!/bin/sh
# Reconstruye y reinicia la plataforma SÓLO si no hay entrenamientos en cola o en proceso.
# Reiniciar worker-train corta el entrenamiento en curso (se recupera, pero pierde progreso).
#   ./scripts/deploy.sh            # despliegue seguro
#   ./scripts/deploy.sh --force    # despliega aunque haya entrenamientos (se reintentarán)
set -e
cd "$(dirname "$0")/.."

ACTIVE=$(docker compose exec -T db psql -U vision -d vision -tAc \
  "select count(*) from training_jobs where status in ('pending','preparing','training','validating','exporting')" 2>/dev/null || echo 0)
ACTIVE=$(echo "$ACTIVE" | tr -dc '0-9')
ACTIVE=${ACTIVE:-0}

if [ "$ACTIVE" -gt 0 ] && [ "$1" != "--force" ]; then
  echo "Hay $ACTIVE entrenamiento(s) en cola o en proceso. No se despliega para no interrumpirlos."
  docker compose exec -T db psql -U vision -d vision -c \
    "select j.id, p.name as proyecto, j.status, round(j.percent::numeric,0) as pct from training_jobs j join projects p on p.id=j.project_id where j.status in ('pending','preparing','training','validating','exporting') order by j.id"
  echo "Espera a que terminen, cancélalos desde Administración, o usa --force."
  exit 1
fi

echo "Sin entrenamientos activos: construyendo y reiniciando..."
docker compose up -d --build
docker compose ps
