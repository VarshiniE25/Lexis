"""
cache.py — Disk-based caching layer for LLM responses.
Prevents redundant API calls for identical prompts.
"""

from __future__ import annotations
import hashlib
import json
from typing import Any, Optional

import diskcache

from .config import CACHE_DIR, CACHE_TTL, CACHE_ENABLED
from .logger import get_logger

logger = get_logger(__name__)

# Module-level cache singleton
_cache: diskcache.Cache | None = None


def _get_cache() -> diskcache.Cache:
    global _cache
    if _cache is None:
        _cache = diskcache.Cache(str(CACHE_DIR), size_limit=500 * 1024 * 1024)  # 500 MB
    return _cache


def _make_key(prompt: str, model: str) -> str:
    """Create a stable cache key from prompt + model."""
    content = f"{model}:{prompt}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def cache_get(prompt: str, model: str) -> Optional[dict]:
    """
    Retrieve a cached LLM response.

    Returns:
        Parsed JSON dict if cache hit, None otherwise.
    """
    if not CACHE_ENABLED:
        return None

    key = _make_key(prompt, model)
    cache = _get_cache()

    try:
        value = cache.get(key)
        if value is not None:
            logger.debug(f"Cache HIT for key {key[:12]}...")
            return json.loads(value)
    except Exception as e:
        logger.warning(f"Cache read error: {e}")

    return None


def cache_set(prompt: str, model: str, response: Any) -> None:
    """
    Store an LLM response in cache.

    Args:
        prompt: The prompt that generated the response.
        model: The model name.
        response: The response object (will be JSON-serialized).
    """
    if not CACHE_ENABLED:
        return

    key = _make_key(prompt, model)
    cache = _get_cache()

    try:
        value = json.dumps(response, default=str)
        cache.set(key, value, expire=CACHE_TTL)
        logger.debug(f"Cache SET for key {key[:12]}...")
    except Exception as e:
        logger.warning(f"Cache write error: {e}")


def cache_clear() -> None:
    """Clear all cached entries."""
    cache = _get_cache()
    cache.clear()
    logger.info("Cache cleared")


def cache_stats() -> dict:
    """Return cache statistics."""
    cache = _get_cache()
    return {
        "size": len(cache),
        "volume_mb": round(cache.volume() / 1024 / 1024, 2),
    }
