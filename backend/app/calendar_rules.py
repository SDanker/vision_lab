"""Calendario semanal obligatorio (America/Santiago).

* Lunes a viernes: se puede modificar el dataset (si no hay entrenamientos activos).
* Sábado 00:00 en adelante y domingo: bloqueado. El sábado a las 00:05 se crean
  snapshots y arranca la cola semanal.
* Lunes 00:00: se reabre sólo cuando terminaron los entrenamientos y se publicaron
  los resultados.
"""
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from .config import get_settings

TZ = ZoneInfo(get_settings().timezone)
TRAIN_START = time(0, 5)

_override: datetime | None = None  # sólo para pruebas


def set_clock_override(value: datetime | None) -> None:
    global _override
    _override = value


def now_local() -> datetime:
    if _override is not None:
        return _override.astimezone(TZ)
    return datetime.now(timezone.utc).astimezone(TZ)


def is_weekend_lock(local: datetime) -> bool:
    return local.weekday() >= 5  # sábado=5, domingo=6


def week_key_for(local: datetime) -> str | None:
    """Clave de la ejecución semanal si `local` está dentro de la ventana de
    entrenamiento (sábado >= 00:05 o domingo). Es la fecha del sábado."""
    wd = local.weekday()
    if wd == 5 and local.time() >= TRAIN_START:
        return local.date().isoformat()
    if wd == 6:
        return (local.date() - timedelta(days=1)).isoformat()
    return None


def _next_weekday_at(local: datetime, weekday: int, at: time) -> datetime:
    days = (weekday - local.weekday()) % 7
    candidate = datetime.combine(local.date() + timedelta(days=days), at, tzinfo=TZ)
    if candidate <= local:
        candidate += timedelta(days=7)
        candidate = datetime.combine(candidate.date(), at, tzinfo=TZ)
    return candidate


def window_state(training_active: bool, local: datetime | None = None) -> dict:
    local = local or now_local()
    if is_weekend_lock(local):
        open_, reason = False, "WEEKEND_LOCK"
        message = (
            "Fin de semana: las cargas y modificaciones están bloqueadas mientras se "
            "entrenan los modelos. Se reabren el lunes cuando termine el entrenamiento."
        )
    elif training_active:
        open_, reason = False, "TRAINING_ACTIVE"
        message = (
            "Hay entrenamientos activos o resultados sin publicar. Las cargas se "
            "reabrirán cuando finalicen."
        )
    else:
        open_, reason, message = True, "OPEN", "Las cargas y modificaciones están habilitadas."
    return {
        "open": open_,
        "reason": reason,
        "message": message,
        "now": local.isoformat(),
        "timezone": str(TZ),
        "next_lock_at": _next_weekday_at(local, 5, time(0, 0)).isoformat(),
        "next_training_start": _next_weekday_at(local, 5, TRAIN_START).isoformat(),
    }


def _hhmm(value: str) -> time:
    h, m = value.split(":")
    return time(int(h), int(m))


def night_window_end(local: datetime | None = None) -> datetime | None:
    """Fin de la ventana nocturna en curso (lunes a viernes, NIGHT_WINDOW_START–END), o None si no estamos en ella.

    La ventana no cruza la medianoche: por defecto 00:00–08:00 de cada día hábil.
    """
    local = local or now_local()
    s = get_settings()
    if not s.night_training_enabled or local.weekday() >= 5:
        return None
    start, end = _hhmm(s.night_window_start), _hhmm(s.night_window_end)
    if not (start <= local.time() < end):
        return None
    return datetime.combine(local.date(), end, tzinfo=TZ)


def night_key(local: datetime | None = None) -> str:
    """Identificador de la noche (fecha local)."""
    return (local or now_local()).date().isoformat()


def saturday_of(d: date) -> date:
    return d - timedelta(days=(d.weekday() - 5) % 7)


def next_training_start(local: datetime | None = None) -> datetime:
    """Próximo inicio del entrenamiento semanal: sábado 00:05 (si ya pasó, el sábado siguiente)."""
    return _next_weekday_at(local or now_local(), 5, TRAIN_START)


def describe_training_start(when: datetime) -> str:
    """«el sábado 31-10-2026 a las 00:05 (hora de Chile)»."""
    return f"el sábado {when:%d-%m-%Y} a las {when:%H:%M} (hora de Chile)"


def describe_next_execution(local: datetime | None = None) -> str:
    """Cuándo se entrenaría un proyecto marcado: el sábado y, si hay entrenamiento nocturno, antes en la noche (modelos cortos)."""
    text = describe_training_start(next_training_start(local))
    s = get_settings()
    if s.night_training_enabled:
        text += f"; los modelos cortos pueden entrenarse antes, de lunes a viernes entre las {s.night_window_start} y las {s.night_window_end}"
    return text
