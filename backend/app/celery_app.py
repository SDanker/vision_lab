from celery import Celery
from celery.schedules import crontab

from .config import get_settings

s = get_settings()

celery_app = Celery("vision", broker=s.redis_url, include=["app.tasks"])
celery_app.conf.update(
    timezone=s.timezone,  # America/Santiago: el crontab respeta los cambios de horario
    enable_utc=True,
    task_default_queue="default",
    task_routes={"vision.process_queue": {"queue": "training"}},
    task_serializer="json",
    accept_content=["json"],
    broker_connection_retry_on_startup=True,
    worker_prefetch_multiplier=1,
    beat_schedule={
        "tick": {"task": "vision.tick", "schedule": 60.0},
        "weekly-training-start": {
            "task": "vision.weekly_start",
            "schedule": crontab(minute=5, hour=0, day_of_week="sat"),
        },
        "cleanup": {"task": "vision.cleanup", "schedule": crontab(minute=30, hour=3)},
    },
)
