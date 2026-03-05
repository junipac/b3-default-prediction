"""
Decoradores de retry com backoff exponencial e jitter.
Baseado em tenacity com logging estruturado de tentativas.
"""

import random
import asyncio
from functools import wraps
from typing import Callable, Type, Tuple, Optional
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_random,
    retry_if_exception_type,
    before_sleep_log,
    after_log,
    RetryError,
)
import requests
import aiohttp
import logging

from config.settings import RETRY
from src.utils.logging import get_logger

logger = get_logger(__name__)
_std_logger = logging.getLogger(__name__)

# Exceções que justificam retry
RETRYABLE_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
    aiohttp.ClientConnectionError,
    aiohttp.ServerTimeoutError,
    aiohttp.ClientPayloadError,
    IOError,
    OSError,
)

# Códigos HTTP que justificam retry
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504, 520, 521, 522, 524}


class HTTPRetryError(Exception):
    """Levantada quando todos os retries de uma requisição HTTP falham."""
    def __init__(self, url: str, status_code: int, attempts: int):
        self.url = url
        self.status_code = status_code
        self.attempts = attempts
        super().__init__(
            f"HTTP {status_code} após {attempts} tentativas: {url}"
        )


class BlockedIPError(Exception):
    """Levantada quando é detectado bloqueio de IP."""
    pass


def sync_retry(
    max_attempts: Optional[int] = None,
    initial_wait: Optional[float] = None,
    max_wait: Optional[float] = None,
    retryable_exceptions: Tuple[Type[Exception], ...] = RETRYABLE_EXCEPTIONS,
) -> Callable:
    """Decorator de retry para funções síncronas."""
    max_attempts = max_attempts or RETRY["max_attempts"]
    initial_wait = initial_wait or RETRY["initial_wait"]
    max_wait = max_wait or RETRY["max_wait"]

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            last_exception = None
            wait = initial_wait

            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    attempt += 1
                    last_exception = e
                    if attempt >= max_attempts:
                        break
                    jitter = random.uniform(0, wait * 0.1)
                    sleep_time = min(wait + jitter, max_wait)
                    logger.warning(
                        "retry_attempt",
                        function=func.__name__,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        sleep_seconds=round(sleep_time, 2),
                        error=str(e),
                    )
                    import time
                    time.sleep(sleep_time)
                    wait = min(wait * RETRY["multiplier"], max_wait)

            logger.error(
                "retry_exhausted",
                function=func.__name__,
                attempts=attempt,
                error=str(last_exception),
            )
            raise last_exception

        return wrapper
    return decorator


def async_retry(
    max_attempts: Optional[int] = None,
    initial_wait: Optional[float] = None,
    max_wait: Optional[float] = None,
    retryable_exceptions: Tuple[Type[Exception], ...] = RETRYABLE_EXCEPTIONS,
) -> Callable:
    """Decorator de retry para funções assíncronas."""
    max_attempts = max_attempts or RETRY["max_attempts"]
    initial_wait = initial_wait or RETRY["initial_wait"]
    max_wait = max_wait or RETRY["max_wait"]

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            attempt = 0
            last_exception = None
            wait = initial_wait

            while attempt < max_attempts:
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    attempt += 1
                    last_exception = e
                    if attempt >= max_attempts:
                        break
                    jitter = random.uniform(0, wait * 0.1)
                    sleep_time = min(wait + jitter, max_wait)
                    logger.warning(
                        "async_retry_attempt",
                        function=func.__name__,
                        attempt=attempt,
                        sleep_seconds=round(sleep_time, 2),
                        error=str(e),
                    )
                    await asyncio.sleep(sleep_time)
                    wait = min(wait * RETRY["multiplier"], max_wait)

            logger.error(
                "async_retry_exhausted",
                function=func.__name__,
                attempts=attempt,
                error=str(last_exception),
            )
            raise last_exception

        return wrapper
    return decorator


def is_blocked_response(response: requests.Response) -> bool:
    """Detecta se a resposta indica bloqueio de IP ou CAPTCHA."""
    blocked_indicators = [
        "captcha",
        "blocked",
        "access denied",
        "403 forbidden",
        "too many requests",
        "rate limit",
    ]
    content_lower = (response.text or "").lower()
    if response.status_code in {403, 429}:
        return True
    return any(indicator in content_lower for indicator in blocked_indicators)
