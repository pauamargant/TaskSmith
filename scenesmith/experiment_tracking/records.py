"""Dataclasses used by experiment tracking.

The unit of observability is the **LLM API call**: one ``LLMCallRecord`` per
actual round-trip to an inference endpoint, whether issued by the Agents SDK
(via the tracked Model wrapper in ``scenesmith.provider``) or by direct
``VLMService`` calls.
"""

from __future__ import annotations

import re

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _normalize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_value(item) for key, item in value.items()}
    return value


# Regex parses canonical agent labels like "DESIGNER (FURNITURE)" or
# "CRITIC (INITIAL FLOOR PLAN)" into (role, type). Tolerates extra
# whitespace.
_AGENT_LABEL_RE = re.compile(r"^\s*(?P<role>[A-Z][A-Z_]+)\s*\((?P<type>[^)]+)\)\s*$")


def parse_agent_label(label: str | None) -> tuple[str | None, str | None]:
    """Extract ``(role, type)`` from a label like ``"DESIGNER (FURNITURE)"``.

    Returns ``(None, None)`` when the label is missing or doesn't match.
    Role is lowercased ("designer"); type is uppercased and underscored
    ("INITIAL_FLOOR_PLAN").
    """

    if not label:
        return (None, None)
    match = _AGENT_LABEL_RE.match(label)
    if not match:
        return (None, None)
    role = match.group("role").strip().lower()
    type_raw = match.group("type").strip().upper()
    type_norm = re.sub(r"\s+", "_", type_raw)
    return role, type_norm


@dataclass
class LLMCallRecord:
    """One LLM API call. Written to ``run_analysis/usage_records.jsonl``."""

    timestamp: str
    source: str  # "agents_sdk" | "vlm_direct"
    provider_path: str  # "agents_sdk" | "openai_direct"

    # Run context (from TrackingContext at emission time).
    scene_id: str | None = None
    room_id: str | None = None
    stage_name: str | None = None
    agent_label: str | None = None  # original "DESIGNER (FURNITURE)" form
    agent_role: str | None = None  # parsed from label, e.g. "designer"
    agent_type: str | None = None  # parsed from label, e.g. "FURNITURE"
    subsystem: str | None = None  # "physics_analysis" | "router_validation" |
    # "vlm_direct" for direct VLM, None for Agents SDK
    tool_call_target: str | None = None  # name of tool whose body issued this
    # call (rare; usually None)
    tool_calls_in_response: list[str] = field(default_factory=list)
    # Names of tools this response decided to invoke.

    # API call details.
    model: str | None = None
    request_index: int | None = None  # 0-based index within the agent run
    # (Agents SDK only; None for direct VLM)

    # Counts.
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    num_images: int = 0
    duration_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return _normalize_value(asdict(self))


@dataclass
class StageEvent:
    timestamp: str
    event_type: str
    stage_name: str
    status: str
    scene_id: str | None = None
    room_id: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    duration_s: float | None = None
    output_paths: list[str] = field(default_factory=list)
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _normalize_value(asdict(self))


@dataclass
class ArtifactRecord:
    timestamp: str
    artifact_type: str
    path: str
    scene_id: str | None = None
    room_id: str | None = None
    stage_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _normalize_value(asdict(self))
