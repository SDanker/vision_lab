"""Limitación de intentos: Redis si está disponible, memoria local como respaldo."""
import time
from collections import defaultdict

import redis

from .config import get_settings
from .errors import ApiError

_mem: dict[str, list[float]] = defaultdict(list)
_client: redis.Redis | None = None


def _redis() -> redis.Redis | None:
    global _client
    url = get_settings().redis_url
    if not url:
        return None
    if _client is None:
        _client = redis.Redis.from_url(url, socket_connect_timeout=1, socket_timeout=1)
    return _client


def reset_memory() -> None:
    _mem.clear()


def hit(key: str, limit: int, window: int) -> bool:
    """Registra un intento. Devuelve True si aún está dentro del límite."""
    r = _redis()
    if r is not None:
        try:
            k = f"rl:{key}"
            n = r.incr(k)
            if n == 1:
                r.expire(k, window)
            return n <= limit
        except redis.RedisError:
            pass
    now = time.time()
    bucket = [t for t in _mem[key] if now - t < window]
    bucket.append(now)
    _mem[key] = bucket
    return len(bucket) <= limit


def clear(key: str) -> None:
    r = _redis()
    if r is not None:
        try:
            r.delete(f"rl:{key}")
        except redis.RedisError:
            pass
    _mem.pop(key, None)


def enforce(key: str, limit: int, window: int, message: str) -> None:
    if not hit(key, limit, window):
        raise ApiError(429, "RATE_LIMITED", message)
