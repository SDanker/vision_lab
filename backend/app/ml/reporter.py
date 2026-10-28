"""Persistencia y difusión del progreso de un trabajo (BD + Redis pub/sub)."""
import threading
import time

import psutil

from ..db import SessionLocal, utcnow
from ..models import TrainingJob, TrainingLog
from ..services.training_queue import publish_job_event


class Cancelled(Exception):
    pass


class Reporter:
    def __init__(self, job_id: int):
        self.job_id = job_id
        self.history: list[dict] = []
        self._last = 0.0
        self._last_cancel_check = 0.0
        self._proc = psutil.Process()
        self._proc.cpu_percent(None)
        self._ncpu = psutil.cpu_count() or 1
        self._stop = threading.Event()
        self._beat: threading.Thread | None = None
        self.train_started_at: float | None = None

    # ------------------------------------------------------------ heartbeat
    def start_heartbeat(self, every: float = 20.0) -> None:
        def loop():
            while not self._stop.wait(every):
                try:
                    self.update(force=True)
                except Exception:  # noqa: BLE001
                    pass

        self._beat = threading.Thread(target=loop, daemon=True)
        self._beat.start()

    def stop_heartbeat(self) -> None:
        self._stop.set()

    # --------------------------------------------------------------- updates
    def update(self, force: bool = False, min_interval: float = 1.5, **fields) -> None:
        now = time.time()
        if not force and now - self._last < min_interval:
            return
        self._last = now
        with SessionLocal() as db:
            job = db.get(TrainingJob, self.job_id)
            if job is None:
                return
            for k, v in fields.items():
                setattr(job, k, v)
            job.history = list(self.history)
            job.heartbeat_at = utcnow()
            job.cpu_percent = round(self._proc.cpu_percent(None) / self._ncpu, 1)
            job.ram_mb = round(self._proc.memory_info().rss / 1024 / 1024, 1)
            db.commit()
            publish_job_event(job)

    def log(self, message: str, level: str = "info") -> None:
        with SessionLocal() as db:
            db.add(TrainingLog(job_id=self.job_id, level=level, message=message[:4000]))
            db.commit()

    def check_cancel(self, force: bool = False) -> bool:
        now = time.time()
        if not force and now - self._last_cancel_check < 3:
            return False
        self._last_cancel_check = now
        with SessionLocal() as db:
            job = db.get(TrainingJob, self.job_id)
            return bool(job and job.cancel_requested)

    def raise_if_cancelled(self) -> None:
        if self.check_cancel(force=True):
            raise Cancelled()
