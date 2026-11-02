"""Package variation outputs for Drake and optionally Blender."""

from __future__ import annotations

import logging
import shutil

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig

from scenesmith.agent_utils.house import HouseLayout, HouseScene
from scenesmith.agent_utils.room import RoomScene

console_logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Controls which export formats are produced per variation."""

    drake: bool = True
    blender: bool = False


def export_variation(
    variation_scene_dir: Path,
    rooms: list[RoomScene],
    house_layout: HouseLayout,
    cfg: DictConfig,
    export_cfg: ExportConfig,
) -> None:
    """Assemble combined house and run requested exports for a variation.

    Args:
        variation_scene_dir: The variation's scene_NNN directory.
        rooms: Fully populated RoomScene objects for this variation.
        house_layout: HouseLayout from the base experiment.
        cfg: Full Hydra DictConfig (needed for Blender export).
        export_cfg: Which export formats to produce.
    """
    exports_dir = variation_scene_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    # Assemble combined_house/house.dmd.yaml from the modified rooms.
    variation_layout = deepcopy(house_layout)
    variation_layout.house_dir = variation_scene_dir
    rooms_by_id = {room.room_id: room for room in rooms}
    house_scene = HouseScene(layout=variation_layout, rooms=rooms_by_id)
    combined_dir = house_scene.assemble(cfg=cfg, output_name="combined_house")
    console_logger.debug(f"Assembled combined house: {combined_dir}")

    # Drake — DMD yaml is already at combined_house/house.dmd.yaml.
    if export_cfg.drake:
        dmd_src = combined_dir / "house.dmd.yaml"
        if dmd_src.exists():
            shutil.copy(dmd_src, exports_dir / "scene.dmd.yaml")
            console_logger.debug(f"Drake export: {exports_dir / 'scene.dmd.yaml'}")

    # Blender — optional, only if server is available.
    if export_cfg.blender:
        _export_blender(combined_dir, exports_dir / "blender", cfg)


def _export_blender(combined_dir: Path, output_dir: Path, cfg: DictConfig) -> None:
    """Run Blender export if a Blender server is available."""
    blend_src = combined_dir / "house.blend"
    if not blend_src.exists():
        console_logger.debug("Blender export skipped: house.blend not found")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(blend_src, output_dir / "house.blend")
    console_logger.debug(f"Blender export: {output_dir / 'house.blend'}")
