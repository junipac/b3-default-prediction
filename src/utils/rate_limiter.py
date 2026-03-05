"""
Rate limiter token-bucket para controle de requisições por fonte.
Suporta uso síncrono e assíncrono.
"""

import time
import asyncio
import threading
from typing import Dict
from config.settings import RATE_LIMIT
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TokenBucket:
    """Token bucket thread-safe para rate limiting síncrono."""

    def __init__(self, rate_per_minute: int, burst: int = 0):
        self.rate = rate_per_minute / 60.0  # tokens por segundo
        self.capacity = burst or max(rate_per_minute // 10, 5)
        self.tokens = float(self.capacity)
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now

    def acquire(self, tokens: int = 1) -> float:
        """
        Aguarda até ter tokens suficientes.
        Retorna o tempo de espera em segundos.
        """
        with self._lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0

            deficit = tokens - self.tokens
            wait_time = deficit / self.rate
            self.tokens = 0.0

        time.sleep(wait_time)
        return wait_time


class AsyncTokenBucket:
    """Token bucket para rate limiting assíncrono."""

    def __init__(self, rate_per_minute: int, burst: int = 0):
        self.rate = rate_per_minute / 60.0
        self.capacity = burst or max(rate_per_minute // 10, 5)
        self.tokens = float(self.capacity)
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now

    async def acquire(self, tokens: int = 1) -> float:
        async with self._lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0

            deficit = tokens - self.tokens
            wait_time = deficit / self.rate
            self.tokens = 0.0

        await asyncio.sleep(wait_time)
        return wait_time


class RateLimiterRegistry:
    """Registro centralizado de rate limiters por fonte."""

    _sync_buckets: Dict[str, TokenBucket] = {}
    _async_buckets: Dict[str, AsyncTokenBucket] = {}
    _lock = threading.Lock()

    @classmethod
    def get_sync(cls, source: str) -> TokenBucket:
        with cls._lock:
            if source not in cls._sync_buckets:
                rpm = _get_rpm(source)
                cls._sync_buckets[source] = TokenBucket(rpm)
                logger.info(
                    "rate_limiter_created",
                    source=source,
                    rpm=rpm,
                    type="sync",
                )
            return cls._sync_buckets[source]

    @classmethod
    def get_async(cls, source: str) -> AsyncTokenBucket:
        if source not in cls._async_buckets:
            rpm = _get_rpm(source)
            cls._async_buckets[source] = AsyncTokenBucket(rpm)
            logger.info(
                "rate_limiter_created",
                source=source,
                rpm=rpm,
                type="async",
            )
        return cls._async_buckets[source]


def _get_rpm(source: str) -> int:
    mapping = {
        "b3": RATE_LIMIT["b3_requests_per_minute"],
        "cvm": RATE_LIMIT["cvm_requests_per_minute"],
        "diario_oficial": RATE_LIMIT["diario_oficial_requests_per_minute"],
    }
    return mapping.get(source, 30)
