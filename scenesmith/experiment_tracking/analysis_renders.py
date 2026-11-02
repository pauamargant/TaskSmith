"""Whole-room renders for post-run experiment analysis.

Produces multi-view PNGs with the same layout as furniture-stage ``observe_scene``
(top + side views + optional isometric) from each room's latest checkpoint.
Outputs live under ``room_*/scene_renders/experiment_analysis/renders_001/``.
"""

from __future__ import annotations

import json
import logging
import shutil

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from scenesmith.agent_utils.blender import BlenderServer
from scenesmith.agent_utils.house import RoomGeometry
from scenesmith.agent_utils.rendering_manager import RenderingManager
from scenesmith.agent_utils.room import RoomScene
from scenesmith.utils.logging import ConsoleLogger

console_logger = logging.getLogger(__name__)

ANALYSIS_RENDER_SUBDIR = "experiment_analysis"
ANALYSIS_RENDER_BATCH = "renders_001"

_CHECKPOINT_PRIORITY = (
    "final_scene",
    "scene_after_ceiling_objects",
    "scene_after_wall_objects",
    "scene_after_furniture",
)


def analysis_renders_enabled(cfg_dict: dict[str, Any]) -> bool:
    """Return whether whole-room analysis renders are enabled."""

    tracking = cfg_dict.get("experiment", {}).get("tracking", {}) or {}
    analysis_cfg = tracking.get("analysis_renders", {}) or {}
    if "enabled" in analysis_cfg:
        return bool(analysis_cfg["enabled"])
    return True


def _resolve_checkpoint_name(room_dir: Path) -> str | None:
    for name in _CHECKPOINT_PRIORITY:
        state_path = room_dir / "scene_states" / name / "scene_state.json"
        if state_path.exists():
            return name
    return None


def load_room_scene_from_checkpoint(
    room_dir: Path,
    *,
    checkpoint_name: str | None = None,
) -> RoomScene | None:
    """Load a ``RoomScene`` from a room checkpoint directory."""

    room_dir = room_dir.resolve()
    if checkpoint_name is None:
        checkpoint_name = _resolve_checkpoint_name(room_dir)
    if checkpoint_name is None:
        return None

    state_path = room_dir / "scene_states" / checkpoint_name / "scene_state.json"
    if not state_path.exists():
        return None

    with open(state_path, encoding="utf-8") as f:
        scene_state = json.load(f)

    room_geometry = RoomGeometry.from_dict(
        scene_state["room_geometry"], scene_dir=room_dir
    )
    room_id = room_dir.name.removeprefix("room_")
    room_scene = RoomScene(
        room_geometry=room_geometry,
        scene_dir=room_dir,
        room_id=room_id,
        text_description=scene_state.get("text_description", ""),
        action_log_path=room_dir / "action_log.json",
    )
    room_scene.restore_from_state_dict(scene_state)
    return room_scene


def discover_analysis_render_paths(scene_dir: Path) -> dict[str, list[str]]:
    """Return PNG paths for saved analysis renders, keyed by room id."""

    media: dict[str, list[str]] = {}
    for room_dir in sorted(scene_dir.glob("room_*")):
        room_id = room_dir.name.removeprefix("room_")
        batch_dir = (
            room_dir / "scene_renders" / ANALYSIS_RENDER_SUBDIR / ANALYSIS_RENDER_BATCH
        )
        if not batch_dir.exists():
            continue
        pngs = sorted(str(path) for path in batch_dir.glob("*.png"))
        if pngs:
            media[room_id] = pngs
    return media


def _flatten_analysis_paths(by_room: dict[str, list[str]]) -> list[str]:
    paths: list[str] = []
    for room_id in sorted(by_room):
        paths.extend(by_room[room_id])
    return paths


def render_rooms_for_analysis(
    scene_dir: Path,
    cfg_dict: dict[str, Any],
    *,
    rooms: dict[str, RoomScene] | None = None,
    render_gpu_id: int | None = None,
) -> dict[str, list[str]]:
    """Render whole-room views for experiment analysis.

    Args:
        scene_dir: Scene output directory (``scene_NNN/``).
        cfg_dict: Resolved Hydra config dictionary.
        rooms: Optional in-memory room scenes (preferred when available).
        render_gpu_id: Optional GPU index for BlenderServer isolation.

    Returns:
        Mapping of room id to rendered PNG paths.
    """

    scene_dir = scene_dir.resolve()
    if not analysis_renders_enabled(cfg_dict):
        return discover_analysis_render_paths(scene_dir)

    furniture_cfg = cfg_dict.get("furniture_agent", {}) or {}
    rendering_cfg = OmegaConf.create(furniture_cfg.get("rendering", {}) or {})
    quality_preset = furniture_cfg.get("image_quality_preset")

    room_entries: list[tuple[str, Path, RoomScene]] = []
    if rooms:
        for room_id, scene in sorted(rooms.items()):
            room_entries.append((room_id, scene.scene_dir, scene))
    else:
        for room_dir in sorted(scene_dir.glob("room_*")):
            room_id = room_dir.name.removeprefix("room_")
            scene = load_room_scene_from_checkpoint(room_dir)
            if scene is not None:
                room_entries.append((room_id, room_dir, scene))

    if not room_entries:
        console_logger.info(
            "Skipping experiment analysis renders: no room scenes found in %s",
            scene_dir,
        )
        return {}

    blender_server: BlenderServer | None = None
    rendered: dict[str, list[str]] = {}
    try:
        blender_server = BlenderServer(
            port_range=tuple(
                rendering_cfg.get("blender_server_port_range", [8000, 8050])
            ),
            server_startup_delay=float(rendering_cfg.get("server_startup_delay", 0.1)),
            port_cleanup_delay=float(rendering_cfg.get("port_cleanup_delay", 0.1)),
            gpu_id=render_gpu_id,
            log_file=scene_dir / "analysis_renders.log",
        )
        blender_server.start()
        blender_server.wait_until_ready()

        for room_id, room_dir, scene in room_entries:
            output_dir = (
                room_dir
                / "scene_renders"
                / ANALYSIS_RENDER_SUBDIR
                / ANALYSIS_RENDER_BATCH
            )
            if output_dir.exists():
                shutil.rmtree(output_dir)

            logger = ConsoleLogger(output_dir=room_dir)
            rendering_manager = RenderingManager(
                cfg=rendering_cfg,
                logger=logger,
                subdirectory=ANALYSIS_RENDER_SUBDIR,
                quality_preset=quality_preset,
            )
            rendering_manager.clear_cache()
            images_dir = rendering_manager.render_scene(
                scene=scene,
                blender_server=blender_server,
                rendering_mode="furniture",
                render_name=ANALYSIS_RENDER_BATCH,
            )
            pngs = sorted(str(path) for path in images_dir.glob("*.png"))
            rendered[room_id] = pngs
            console_logger.info(
                "Experiment analysis renders for room '%s': %d views -> %s",
                room_id,
                len(pngs),
                images_dir,
            )
    finally:
        if blender_server is not None and blender_server.is_running():
            blender_server.stop()

    return rendered


def first_analysis_render_paths(
    scene_dir: Path,
    *,
    limit: int = 4,
) -> list[str]:
    """Return up to ``limit`` analysis render PNGs across all rooms."""

    by_room = discover_analysis_render_paths(scene_dir)
    return _flatten_analysis_paths(by_room)[:limit]


def finalize_scene_analysis(
    scene_dir: Path,
    cfg_dict: dict[str, Any],
    *,
    rooms: dict[str, RoomScene] | None = None,
    render_gpu_id: int | None = None,
) -> dict[str, Any]:
    """Render whole-room views when needed, then run token/media analysis."""

    from .analyzer import analyze_scene

    scene_dir = scene_dir.resolve()
    if analysis_renders_enabled(cfg_dict):
        existing = discover_analysis_render_paths(scene_dir)
        if not existing:
            try:
                render_rooms_for_analysis(
                    scene_dir,
                    cfg_dict,
                    rooms=rooms,
                    render_gpu_id=render_gpu_id,
                )
            except Exception as exc:
                console_logger.warning(
                    "Experiment analysis renders failed (continuing with rollup): %s",
                    exc,
                )
    return analyze_scene(scene_dir)
