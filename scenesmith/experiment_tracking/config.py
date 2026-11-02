"""Process-wide tracking switches.

The tracker is **opt-out**: it defaults to enabled so existing runs behave
unchanged. Two switches are exposed:

- ``tracking_enabled()`` — master switch. When false, every ``emit_*`` is a
  no-op and the provider returns the raw inner Model (no wrapping).
- ``agents_sdk_wrapping_enabled()`` — sub-switch. Only consulted when
  tracking is enabled; lets the user keep direct-VLM tracking while
  disabling the wrapped-Model path (useful if a future SDK upgrade breaks
  the wrapper).

Resolution order: explicit ``init_tracking_from_cfg(cfg)`` call wins,
then environment variable ``TASKSMITH_TRACKING_ENABLED`` (``0``/``false``
turns it off), then default ``True``.
"""

from __future__ import annotations

import logging
import os

from typing import Any

console_logger = logging.getLogger(__name__)


_ENV_ENABLED = "TASKSMITH_TRACKING_ENABLED"

_enabled: bool | None = None
_agents_sdk_wrapping: bool | None = None


def _env_says_disabled() -> bool:
    raw = os.environ.get(_ENV_ENABLED)
    if raw is None:
        return False
    return raw.strip().lower() in {"0", "false", "no", "off"}


def tracking_enabled() -> bool:
    if _env_says_disabled():
        return False
    if _enabled is None:
        return True
    return _enabled


def agents_sdk_wrapping_enabled() -> bool:
    if not tracking_enabled():
        return False
    if _agents_sdk_wrapping is None:
        return True
    return _agents_sdk_wrapping


def init_tracking_from_cfg(cfg: Any) -> None:
    """Prime the process-level cache from a Hydra/OmegaConf ``cfg``.

    Tolerates missing keys — absent fields keep the defaults.
    Call once at pipeline startup.
    """

    global _enabled, _agents_sdk_wrapping

    enabled = _select(cfg, "experiment.tracking.enabled")
    if enabled is not None:
        _enabled = bool(enabled)

    wrapping = _select(cfg, "experiment.tracking.agents_sdk_wrapping")
    if wrapping is not None:
        _agents_sdk_wrapping = bool(wrapping)

    console_logger.info(
        "tracking initialized: enabled=%s, agents_sdk_wrapping=%s",
        tracking_enabled(),
        agents_sdk_wrapping_enabled(),
    )


def _select(cfg: Any, dotted: str) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if cur is None:
            return None
        try:
            cur = cur[part] if isinstance(cur, dict) else getattr(cur, part)
        except (AttributeError, KeyError):
            return None
    return cur
