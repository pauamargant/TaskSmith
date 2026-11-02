"""Programmatic floor plan for tabletop-only scenes.

Bypasses the LLM floor plan agent entirely: produces a minimal flat floor
(no walls, no ceiling) so downstream stages only need to place a table and
then populate it with manipulands.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from pathlib import Path

import numpy as np

from pydrake.all import RigidTransform

from scenesmith.agent_utils.house import HouseLayout, RoomGeometry, RoomSpec
from scenesmith.agent_utils.room import ObjectType, SceneObject

# Appended to the user prompt so the furniture agent places exactly one table.
_TABLETOP_FURNITURE_CONSTRAINT = """\

TABLETOP SCENE CONSTRAINT: This is a minimalist, wall-free open surface \
(no room enclosure). Place exactly ONE table in the centre of the floor \
area — no chairs, shelves, or any other furniture. The table should be \
large enough to hold multiple small objects on its top surface."""


def build_tabletop_house_layout(
    prompt: str,
    scene_dir: Path,
    room_width: float = 4.0,
    room_length: float = 4.0,
) -> HouseLayout:
    """Build a minimal HouseLayout for tabletop scene generation.

    Creates a single flat floor (no walls, no ceiling). The RoomSpec prompt
    instructs the furniture agent to place exactly one table; the rest of the
    pipeline (manipuland stage) populates the tabletop normally.

    Args:
        prompt: Original user prompt (scene / task description).
        scene_dir: Root output directory for this scene.
        room_width: Floor width in metres (y-axis).
        room_length: Floor length in metres (x-axis).

    Returns:
        HouseLayout ready to serialise as house_layout.json.
    """
    floor_plan_dir = scene_dir / "floor_plans" / "tabletop"
    floor_plan_dir.mkdir(parents=True, exist_ok=True)

    sdf_path = floor_plan_dir / "room_geometry.sdf"
    _write_floor_sdf(sdf_path, room_width, room_length)
    sdf_tree = ET.parse(sdf_path)

    floor_obj = _build_floor_object(room_width, room_length)

    room_geometry = RoomGeometry(
        sdf_tree=sdf_tree,
        sdf_path=sdf_path,
        walls=[],
        floor=floor_obj,
        wall_normals={},
        width=room_width,
        length=room_length,
        # Large sentinel so physics-validation wall-height checks always pass.
        wall_height=10.0,
        wall_thickness=0.0,
        openings=[],
    )

    room_prompt = prompt + _TABLETOP_FURNITURE_CONSTRAINT

    room_spec = RoomSpec(
        room_id="tabletop",
        room_type="tabletop",
        prompt=room_prompt,
        position=(0.0, 0.0),
        width=room_width,
        length=room_length,
    )

    return HouseLayout(
        wall_height=10.0,
        house_prompt=prompt,
        room_specs=[room_spec],
        room_geometries={"tabletop": room_geometry},
        house_dir=scene_dir,
        placement_valid=True,
        connectivity_valid=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_floor_sdf(sdf_path: Path, width: float, length: float) -> None:
    """Write a minimal SDF file: just a flat floor box, no walls."""
    t = 0.1  # floor thickness (metres)
    content = (
        '<?xml version="1.0"?>\n'
        '<sdf version="1.8">\n'
        '  <model name="room_geometry">\n'
        '    <link name="room_geometry_body_link">\n'
        f'      <visual name="floor_visual">\n'
        f"        <pose>0 0 {-t/2:.4f} 0 0 0</pose>\n"
        f"        <geometry><box><size>{width:.4f} {length:.4f} {t:.4f}</size></box></geometry>\n"
        "        <material><diffuse>0.85 0.80 0.70 1</diffuse></material>\n"
        "      </visual>\n"
        f'      <collision name="floor_collision">\n'
        f"        <pose>0 0 {-t/2:.4f} 0 0 0</pose>\n"
        f"        <geometry><box><size>{width:.4f} {length:.4f} {t:.4f}</size></box></geometry>\n"
        "      </collision>\n"
        "    </link>\n"
        "  </model>\n"
        "</sdf>\n"
    )
    sdf_path.write_text(content)


def _build_floor_object(room_width: float, room_length: float) -> SceneObject:
    """Return a minimal floor SceneObject for RoomGeometry.floor."""
    t = 0.1
    hw = room_width / 2
    hl = room_length / 2
    return SceneObject(
        object_id="floor_tabletop",
        object_type=ObjectType.FLOOR,
        name="Floor",
        description="Open flat floor surface",
        transform=RigidTransform(),
        geometry_path=None,
        sdf_path=None,
        support_surfaces=[],
        bbox_min=np.array([-hw, -hl, -t]),
        bbox_max=np.array([hw, hl, 0.0]),
        immutable=True,
    )
