"""Runtime tracking context for experiment observability."""

from __future__ import annotations

import contextlib
import contextvars

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class TrackingContext:
    """Describes the active experiment/scene/stage scope for tracking writes."""

    experiment_dir: Path
    scene_dir: Path
    room_id: str | None = None
    stage_name: str | None = None
    agent_label: str | None = None
    agent_role: str | None = None
    agent_type: str | None = None
    subsystem: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        data = asdict(self)
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in data.items()
        }


_TRACKING_CONTEXT: contextvars.ContextVar[TrackingContext | None] = (
    contextvars.ContextVar("tracking_context", default=None)
)


def get_tracking_context() -> TrackingContext | None:
    """Return the current tracking context, if any."""

    return _TRACKING_CONTEXT.get()


@contextlib.contextmanager
def tracking_context(context: TrackingContext) -> Iterator[TrackingContext]:
    """Set the active tracking context for the duration of the context manager."""

    token = _TRACKING_CONTEXT.set(context)
    try:
        yield context
    finally:
        _TRACKING_CONTEXT.reset(token)


@contextlib.contextmanager
def extend_tracking_context(
    **updates: str | Path | None,
) -> Iterator[TrackingContext | None]:
    """Temporarily update fields on the existing tracking context."""

    current = get_tracking_context()
    if current is None:
        yield None
        return

    token = _TRACKING_CONTEXT.set(replace(current, **updates))
    try:
        yield _TRACKING_CONTEXT.get()
    finally:
        _TRACKING_CONTEXT.reset(token)
