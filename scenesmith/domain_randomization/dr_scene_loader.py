"""Load ``RoomScene`` + ``ManipulandTools`` from a finished base scene dir.

Used by DR v2 Subflow tests and the orchestrator's v2 path. The base scene
must have a final-stage room state under
``<base>/room_*/scene_states/final_scene/scene_state.json`` (the layout
``IndoorSceneGenerationExperiment`` produces).

For Step 6 / N=1 we anchor the subflow on the furniture object with the
most support surfaces. Multi-furniture mutation is a Step 8 concern.
"""

from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig

from scenesmith.agent_utils.asset_manager import AssetManager
from scenesmith.agent_utils.house import RoomGeometry
from scenesmith.agent_utils.room import AgentType, ObjectType, RoomScene
from scenesmith.manipuland_agents.tools.manipuland_tools import ManipulandTools
from scenesmith.utils.logging import BaseLogger


@dataclass
class LoadedRoom:
    scene: RoomScene
    manipuland_tools: ManipulandTools
    asset_manager: AssetManager
    base_state_dict: dict
    room_dir: Path
    primary_furniture_id: str


def _final_state_path(base_dir: Path) -> Path:
    finals = sorted(base_dir.rglob("scene_states/final_scene/scene_state.json"))
    if not finals:
        raise FileNotFoundError(f"no final scene_state.json under {base_dir}")
    return finals[0]


def _primary_furniture_id(scene: RoomScene) -> str:
    """Pick the furniture object that most manipulands sit on top of.

    Rationale: ManipulandTools' move/remove ops are scoped to one
    furniture id at a time. For DR v2 at N=1 we want the furniture whose
    surfaces the existing manipulands already live on, so the Designer's
    moves stay on the right surface set. Falls back to most-support-
    surfaces when no manipulands provide a geometric signal.
    """
    furniture = [
        obj
        for obj in scene.objects.values()
        if obj.object_type == ObjectType.FURNITURE and obj.support_surfaces
    ]
    if not furniture:
        raise RuntimeError("scene has no furniture with support surfaces")

    manipulands = [
        obj
        for obj in scene.objects.values()
        if obj.object_type == ObjectType.MANIPULAND
    ]
    counts: dict[str, int] = {}
    for furn in furniture:
        top_z = float(furn.bbox_max[2])
        on_top = 0
        for m in manipulands:
            mz = float(m.transform.translation()[2])
            if top_z - 0.05 <= mz <= top_z + 0.5:
                on_top += 1
        counts[str(furn.object_id)] = on_top

    best_id = max(counts, key=lambda k: counts[k])
    if counts[best_id] == 0:
        # No manipulands sit on any furniture — fall back to most surfaces.
        best = max(furniture, key=lambda o: len(o.support_surfaces))
        return str(best.object_id)
    return best_id


def load_room_for_dr(
    base_dir: Path,
    *,
    cfg: DictConfig,
    logger: BaseLogger,
) -> LoadedRoom:
    """Reconstruct RoomScene + ManipulandTools from a finished base scene."""
    base_dir = base_dir.resolve()
    state_path = _final_state_path(base_dir)
    state_dict = json.loads(state_path.read_text())
    room_dir = state_path.parent.parent.parent

    room_geometry = RoomGeometry.from_dict(
        state_dict["room_geometry"], scene_dir=room_dir
    )
    scene = RoomScene(
        room_geometry=room_geometry,
        scene_dir=room_dir,
        room_id=room_dir.name,
        text_description=state_dict.get("text_description", ""),
        action_log_path=room_dir / "dr_subflow_action_log.json",
    )
    scene.restore_from_state_dict(state_dict)

    primary_id = _primary_furniture_id(scene)
    primary = scene.get_object(primary_id)
    support_surfaces = {
        str(surface.surface_id): surface for surface in primary.support_surfaces
    }

    # AssetManager with no blender / collision servers — checkpoint-load mode,
    # safe as long as we only call mutations that don't generate new assets.
    asset_manager = AssetManager(
        logger=logger,
        blender_server=None,
        collision_client=None,
        cfg=cfg,
        agent_type=AgentType.MANIPULAND,
    )
    manipuland_tools = ManipulandTools(
        scene=scene,
        asset_manager=asset_manager,
        cfg=cfg,
        current_furniture_id=primary_id,
        support_surfaces=support_surfaces,
    )

    return LoadedRoom(
        scene=scene,
        manipuland_tools=manipuland_tools,
        asset_manager=asset_manager,
        base_state_dict=state_dict,
        room_dir=room_dir,
        primary_furniture_id=primary_id,
    )
