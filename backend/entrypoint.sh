#!/bin/sh
set -e

case "$1" in
  api)
    alembic upgrade head
    python -m app.cli bootstrap-admin
    exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2 \
      --proxy-headers --forwarded-allow-ips='*'
    ;;
  worker-train)
    # Un único entrenamiento intensivo a la vez: pool "solo" (sin fork, compatible con TensorFlow)
    exec celery -A app.celery_app worker -Q training --pool=solo -n train@%h --loglevel=INFO
    ;;
  worker-light)
    exec celery -A app.celery_app worker -Q default --concurrency=2 -n light@%h --loglevel=INFO
    ;;
  beat)
    exec celery -A app.celery_app beat --schedule /tmp/celerybeat-schedule --loglevel=INFO
    ;;
  *)
    exec "$@"
    ;;
esac
