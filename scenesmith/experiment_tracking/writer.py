"""Incremental JSONL/YAML writers for experiment tracking.

All public ``emit_*`` functions honour two contracts:

1. **Gated by config.** When ``tracking_enabled()`` is false, the function
   short-circuits and writes nothing.
2. **Never raise.** All filesystem and serialization errors are caught and
   logged at WARNING. A tracking bug must never abort the caller's run.
"""

from __future__ import annotations

import functools
import json
import logging

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from .config import tracking_enabled
from .context import TrackingContext, get_tracking_context
from .records import ArtifactRecord, LLMCallRecord, StageEvent, parse_agent_label

console_logger = logging.getLogger(__name__)


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _safe(fn: Callable) -> Callable:
    """Wrap an emitter so any error is logged at WARNING and swallowed."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not tracking_enabled():
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — never break the caller
            console_logger.warning(
                "tracking emission failed (%s): %s", fn.__name__, exc, exc_info=False
            )
            return None

    return wrapper


def _scene_id_from_path(scene_dir: Path) -> str:
    return scene_dir.name


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _context_paths(context: TrackingContext) -> tuple[Path, Path | None]:
    scene_tracking_dir = context.scene_dir / "run_analysis"
    room_tracking_dir = (
        context.scene_dir / f"room_{context.room_id}" / "run_analysis"
        if context.room_id is not None
        else None
    )
    return scene_tracking_dir, room_tracking_dir


@_safe
def emit_stage_event(
    *,
    stage_name: str,
    event_type: str,
    status: str,
    started_at: str | None = None,
    ended_at: str | None = None,
    duration_s: float | None = None,
    output_paths: list[str] | None = None,
    notes: str | None = None,
) -> None:
    context = get_tracking_context()
    if context is None:
        return

    scene_dir, room_dir = _context_paths(context)
    event = StageEvent(
        timestamp=utc_now_iso(),
        event_type=event_type,
        stage_name=stage_name,
        status=status,
        scene_id=_scene_id_from_path(context.scene_dir),
        room_id=context.room_id,
        started_at=started_at,
        ended_at=ended_at,
        duration_s=duration_s,
        output_paths=output_paths or [],
        notes=notes,
    )
    _append_jsonl(scene_dir / "events.jsonl", event.to_dict())
    if room_dir is not None:
        _append_jsonl(room_dir / "stage_events.jsonl", event.to_dict())


@_safe
def emit_artifact(
    *,
    artifact_type: str,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
    stage_name: str | None = None,
) -> None:
    context = get_tracking_context()
    if context is None:
        return

    scene_dir, room_dir = _context_paths(context)
    record = ArtifactRecord(
        timestamp=utc_now_iso(),
        artifact_type=artifact_type,
        path=str(path),
        scene_id=_scene_id_from_path(context.scene_dir),
        room_id=context.room_id,
        stage_name=stage_name or context.stage_name,
        metadata=metadata or {},
    )
    _append_jsonl(scene_dir / "artifacts.jsonl", record.to_dict())
    if room_dir is not None:
        _append_jsonl(room_dir / "artifacts.jsonl", record.to_dict())


@_safe
def emit_llm_call_record(
    *,
    source: str,
    provider_path: str,
    model: str | None,
    subsystem: str | None = None,
    agent_label: str | None = None,
    agent_role: str | None = None,
    agent_type: str | None = None,
    tool_call_target: str | None = None,
    tool_calls_in_response: list[str] | None = None,
    request_index: int | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    reasoning_tokens: int = 0,
    cached_tokens: int = 0,
    total_tokens: int = 0,
    num_images: int = 0,
    duration_s: float | None = None,
) -> None:
    """Emit one per-LLM-call record.

    Context fields (stage / agent_label) fall back to the active
    TrackingContext. If ``agent_label`` is present but ``agent_role`` /
    ``agent_type`` are not, they're parsed from the label.
    """

    context = get_tracking_context()
    if context is None:
        return

    effective_label = agent_label or context.agent_label
    if effective_label is not None and (agent_role is None or agent_type is None):
        parsed_role, parsed_type = parse_agent_label(effective_label)
        agent_role = agent_role or parsed_role or context.agent_role
        agent_type = agent_type or parsed_type or context.agent_type

    scene_dir, room_dir = _context_paths(context)
    record = LLMCallRecord(
        timestamp=utc_now_iso(),
        source=source,
        provider_path=provider_path,
        scene_id=_scene_id_from_path(context.scene_dir),
        room_id=context.room_id,
        stage_name=context.stage_name,
        agent_label=effective_label,
        agent_role=agent_role,
        agent_type=agent_type,
        subsystem=subsystem or context.subsystem,
        tool_call_target=tool_call_target,
        tool_calls_in_response=list(tool_calls_in_response or []),
        model=model,
        request_index=request_index,
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
        reasoning_tokens=int(reasoning_tokens),
        cached_tokens=int(cached_tokens),
        total_tokens=int(total_tokens),
        num_images=int(num_images),
        duration_s=duration_s,
    )
    _append_jsonl(scene_dir / "usage_records.jsonl", record.to_dict())
    if room_dir is not None:
        _append_jsonl(room_dir / "usage_records.jsonl", record.to_dict())


def write_scene_summary(scene_dir: Path, payload: dict[str, Any]) -> None:
    _write_yaml(scene_dir / "run_analysis" / "scene_summary.yaml", payload)


def write_scene_json(
    scene_dir: Path, name: str, payload: dict[str, Any] | list[dict[str, Any]]
) -> None:
    path = scene_dir / "run_analysis" / name
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
