"""Shared IO helpers for structured experiment tracking files."""

from __future__ import annotations

import json

from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def load_scene_and_room_jsonl(
    scene_dir: Path, relative_path: str
) -> list[dict[str, Any]]:
    """Load JSONL from `scene_dir/run_analysis/` plus every `room_*/run_analysis/`."""
    records: list[dict[str, Any]] = []
    records.extend(load_jsonl(scene_dir / "run_analysis" / relative_path))
    for room_dir in sorted(scene_dir.glob("room_*")):
        records.extend(load_jsonl(room_dir / "run_analysis" / relative_path))
    return records
