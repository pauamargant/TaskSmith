from __future__ import annotations

from pathlib import Path

from scenesmith.experiment_tracking.analysis_renders import (
    ANALYSIS_RENDER_BATCH,
    ANALYSIS_RENDER_SUBDIR,
    analysis_renders_enabled,
    discover_analysis_render_paths,
)
from scenesmith.experiment_tracking.analyzer import _discover_stage_media


def test_analysis_renders_enabled_defaults_true() -> None:
    assert analysis_renders_enabled({}) is True
    assert analysis_renders_enabled({"experiment": {"tracking": {}}}) is True


def test_analysis_renders_enabled_respects_config() -> None:
    cfg = {
        "experiment": {
            "tracking": {"analysis_renders": {"enabled": False}},
        }
    }
    assert analysis_renders_enabled(cfg) is False


def test_discover_analysis_render_paths(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene_000"
    batch_dir = (
        scene_dir
        / "room_kitchen"
        / "scene_renders"
        / ANALYSIS_RENDER_SUBDIR
        / ANALYSIS_RENDER_BATCH
    )
    batch_dir.mkdir(parents=True)
    (batch_dir / "0_top.png").write_bytes(b"png")
    (batch_dir / "0_side.png").write_bytes(b"png")

    found = discover_analysis_render_paths(scene_dir)
    assert list(found.keys()) == ["kitchen"]
    assert len(found["kitchen"]) == 2


def test_discover_stage_media_prefers_analysis_renders_for_final_scene(
    tmp_path: Path,
) -> None:
    scene_dir = tmp_path / "scene_000"
    analysis_batch = (
        scene_dir
        / "room_kitchen"
        / "scene_renders"
        / ANALYSIS_RENDER_SUBDIR
        / ANALYSIS_RENDER_BATCH
    )
    analysis_batch.mkdir(parents=True)
    analysis_png = analysis_batch / "0_top.png"
    analysis_png.write_bytes(b"analysis")

    furniture_batch = (
        scene_dir / "room_kitchen" / "scene_renders" / "furniture" / "renders_001"
    )
    furniture_batch.mkdir(parents=True)
    (furniture_batch / "0_top.png").write_bytes(b"furniture")

    media = _discover_stage_media(scene_dir)
    assert media["final_scene"] == [str(analysis_png)]
    assert media["analysis:kitchen"] == [str(analysis_png)]
    assert media["furniture:kitchen"] == [str(furniture_batch / "0_top.png")]
