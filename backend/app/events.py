"""Publicación de eventos en tiempo real (Redis pub/sub -> SSE)."""
import json
import logging

import redis

from .config import get_settings

CHANNEL = "vision:events"
log = logging.getLogger(__name__)
_client: redis.Redis | None = None


def publish(payload: dict) -> None:
    global _client
    url = get_settings().redis_url
    if not url:
        return
    try:
        if _client is None:
            _client = redis.Redis.from_url(url, socket_connect_timeout=1, socket_timeout=2)
        _client.publish(CHANNEL, json.dumps(payload, default=str))
    except redis.RedisError as exc:  # el progreso también se persiste en BD
        log.debug("No se pudo publicar evento: %s", exc)
