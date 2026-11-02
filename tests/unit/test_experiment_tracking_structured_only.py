from __future__ import annotations

import json

from pathlib import Path

from scenesmith.experiment_tracking.analyzer import aggregate, analyze_scene
from scenesmith.experiment_tracking.records import LLMCallRecord


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def test_analyze_scene_merges_room_usage_and_dedupes(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene_000"
    scene_usage_path = scene_dir / "run_analysis" / "usage_records.jsonl"
    scene_events_path = scene_dir / "run_analysis" / "events.jsonl"
    room_usage_path = (
        scene_dir / "room_kitchen" / "run_analysis" / "usage_records.jsonl"
    )

    row = {
        "timestamp": "2026-01-01T00:00:00.000Z",
        "source": "runtime_direct",
        "scene_id": "scene_000",
        "room_id": "kitchen",
        "stage_name": "furniture",
        "agent_label": None,
        "subsystem": "vlm_direct",
        "provider_path": "openai_direct",
        "model": "example-model",
        "input_tokens": 10,
        "output_tokens": 2,
        "reasoning_tokens": 0,
        "cached_tokens": 0,
        "total_tokens": 12,
        "num_images": 0,
        "duration_s": 1.25,
    }

    # Duplicate the same row in both scene-level and room-level usage files.
    _write_jsonl(scene_usage_path, [row])
    _write_jsonl(room_usage_path, [row])
    _write_jsonl(
        scene_events_path,
        [
            {
                "timestamp": "2026-01-01T00:00:00.000Z",
                "event_type": "completed",
                "stage_name": "furniture",
                "status": "completed",
                "scene_id": "scene_000",
                "room_id": None,
                "duration_s": 12.5,
                "output_paths": [],
                "notes": None,
            }
        ],
    )

    summary = analyze_scene(scene_dir)

    assert summary["num_calls"] == 1
    assert summary["totals"]["total_tokens"] == 12
    assert summary["totals"]["num_calls"] == 1
    assert summary["by_stage"]["furniture"]["total_tokens"] == 12
    assert summary["by_subsystem"]["vlm_direct"]["total_tokens"] == 12
    assert summary["stage_wall_time_s"]["furniture"] == 12.5

    aggregated_path = scene_dir / "run_analysis" / "aggregated.json"
    assert aggregated_path.exists()
    with open(aggregated_path, encoding="utf-8") as f:
        aggregated = json.load(f)
    assert (
        aggregated["by_stage_subsystem"]["furniture"]["vlm_direct"]["duration_s"]
        == 1.25
    )


def test_agents_sdk_calls_appear_in_by_subsystem() -> None:
    """Agents SDK records (subsystem=None) are bucketed under their source key."""
    agents_sdk_record = LLMCallRecord(
        timestamp="2026-01-01T00:00:00.000Z",
        source="agents_sdk",
        provider_path="agents_sdk",
        subsystem=None,
    )
    vlm_record = LLMCallRecord(
        timestamp="2026-01-01T00:00:01.000Z",
        source="vlm_direct",
        provider_path="openai_direct",
        subsystem="physics_analysis",
    )

    agg = aggregate([agents_sdk_record, vlm_record])

    assert agg.by_subsystem["agents_sdk"].num_calls == 1
    assert agg.by_subsystem["physics_analysis"].num_calls == 1
    assert agg.totals.num_calls == 2


def test_analyze_scene_allows_empty_structured_usage(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene_000"
    (scene_dir / "run_analysis").mkdir(parents=True, exist_ok=True)

    summary = analyze_scene(scene_dir)

    assert summary["num_calls"] == 0
    assert summary["totals"]["total_tokens"] == 0
    assert (scene_dir / "run_analysis" / "aggregated.json").exists()
