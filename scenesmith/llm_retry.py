"""Shared retry helpers for transient provider/transport failures.

Goal: long TaskSmith runs should not fail on a single dropped connection.

This module intentionally uses conservative heuristics (connection-ish errors only)
so we don't mask real logic or schema problems.
"""

from __future__ import annotations

import asyncio
import os
import random

from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

T = TypeVar("T")


def get_retry_settings() -> tuple[int, float]:
    """(max_attempts, base_backoff_s) from env, with safe defaults."""
    max_attempts = int(os.environ.get("TASKSMITH_PROVIDER_MAX_ATTEMPTS", "3"))
    base_backoff_s = float(os.environ.get("TASKSMITH_PROVIDER_RETRY_BACKOFF_S", "2.0"))
    if max_attempts < 1:
        max_attempts = 1
    if base_backoff_s < 0:
        base_backoff_s = 0.0
    return max_attempts, base_backoff_s


def is_retryable_llm_error(exc: BaseException) -> bool:
    name = type(exc).__name__
    msg = str(exc)
    msg_lower = msg.lower()

    # OpenAI python client
    if "APIConnectionError" in name:
        return True

    # httpx/httpcore transport
    if "RemoteProtocolError" in name:
        return True
    if "Connect" in name or "Connection" in name:
        # catches ConnectError, ConnectTimeout, ReadError, ReadTimeout, etc.
        return True
    if "Timeout" in name:
        return True

    # common message fragments
    if "Server disconnected" in msg:
        return True
    if "Connection error" in msg:
        return True
    if "timed out" in msg_lower:
        return True

    return False


async def aretry(
    fn: Callable[[], Awaitable[T]],
    *,
    op_name: str,
    max_attempts: int | None = None,
    base_backoff_s: float | None = None,
    logger: Any | None = None,
) -> T:
    """Retry an async operation on transient transport failures."""
    env_attempts, env_backoff = get_retry_settings()
    max_attempts = env_attempts if max_attempts is None else max_attempts
    base_backoff_s = env_backoff if base_backoff_s is None else base_backoff_s

    attempt = 0
    while True:
        attempt += 1
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001
            if (not is_retryable_llm_error(exc)) or attempt >= max_attempts:
                raise
            sleep_s = base_backoff_s * (2 ** (attempt - 1))
            sleep_s *= 1.0 + random.random() * 0.25  # jitter
            if logger is not None:
                try:
                    logger.warning(
                        "%s failed with %s (attempt %d/%d); retrying in %.1fs",
                        op_name,
                        type(exc).__name__,
                        attempt,
                        max_attempts,
                        sleep_s,
                    )
                except Exception:
                    pass
            await asyncio.sleep(sleep_s)


def retry_sync(
    fn: Callable[[], T],
    *,
    op_name: str,
    max_attempts: int | None = None,
    base_backoff_s: float | None = None,
    logger: Any | None = None,
) -> T:
    """Retry a sync operation on transient transport failures."""
    env_attempts, env_backoff = get_retry_settings()
    max_attempts = env_attempts if max_attempts is None else max_attempts
    base_backoff_s = env_backoff if base_backoff_s is None else base_backoff_s

    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if (not is_retryable_llm_error(exc)) or attempt >= max_attempts:
                raise
            sleep_s = base_backoff_s * (2 ** (attempt - 1))
            sleep_s *= 1.0 + random.random() * 0.25
            if logger is not None:
                try:
                    logger.warning(
                        "%s failed with %s (attempt %d/%d); retrying in %.1fs",
                        op_name,
                        type(exc).__name__,
                        attempt,
                        max_attempts,
                        sleep_s,
                    )
                except Exception:
                    pass
            if sleep_s > 0:
                import time

                time.sleep(sleep_s)
