"""Shared AABB collision and bounds validation for programmatic ops."""

from __future__ import annotations

import numpy as np

from scenesmith.agent_utils.room import ObjectType, RoomScene, SceneObject


def get_world_aabb(obj: SceneObject) -> tuple[np.ndarray, np.ndarray]:
    """Return world-frame axis-aligned bounding box for an object.

    Uses the object's translation + stored bbox_min/max offsets.
    This is an approximation: it ignores rotation, so it over-estimates
    the occupied volume but is cheap and conservative.

    Returns:
        (world_min, world_max) each as np.ndarray of shape (3,).
    """
    translation = np.array(obj.transform.translation())
    # bbox_min/max are in object-local frame, centred at origin.
    world_min = translation + np.array(obj.bbox_min)
    world_max = translation + np.array(obj.bbox_max)
    return world_min, world_max


def aabb_overlap(
    min_a: np.ndarray,
    max_a: np.ndarray,
    min_b: np.ndarray,
    max_b: np.ndarray,
    margin: float = 0.0,
) -> bool:
    """Return True if two AABBs overlap (with optional margin)."""
    return bool(np.all(min_a - margin < max_b) and np.all(min_b - margin < max_a))


def _is_structural_object(obj: SceneObject) -> bool:
    return obj.object_type in {ObjectType.WALL, ObjectType.FLOOR}


def check_collisions(
    scene: RoomScene,
    ignore_ids: set[str] | None = None,
    *,
    ignore_structural: bool = True,
) -> list[str]:
    """Check for AABB overlaps between non-structural objects in the scene.

    Args:
        scene: Scene to check.
        ignore_ids: Object IDs to skip entirely.
        ignore_structural: If True, skip walls and floor. Structural objects are
            expected to overlap or tightly bound room contents in this cheap AABB
            check, and including them creates noisy DR validation warnings.

    Returns:
        List of collision messages. Empty means no collisions.
    """
    ignore_ids = ignore_ids or set()
    objects = [
        obj
        for obj in scene.objects.values()
        if obj.object_id not in ignore_ids
        and not (ignore_structural and _is_structural_object(obj))
    ]

    violations = []
    for i, obj_a in enumerate(objects):
        min_a, max_a = get_world_aabb(obj_a)
        for obj_b in objects[i + 1 :]:
            min_b, max_b = get_world_aabb(obj_b)
            if aabb_overlap(min_a, max_a, min_b, max_b):
                violations.append(
                    f"Collision: {obj_a.object_id} ({obj_a.name}) overlaps "
                    f"{obj_b.object_id} ({obj_b.name})"
                )
    return violations


def check_room_bounds(scene: RoomScene) -> list[str]:
    """Check that all objects stay within the room floor polygon bounds.

    Uses a simple XY bounding box check against the room geometry.

    Returns:
        List of violation messages. Empty means all objects are in bounds.
    """
    if scene.room_geometry is None:
        return []

    rg = scene.room_geometry
    room_min_x = (
        min(w.transform.translation()[0] for w in rg.walls) if rg.walls else 0.0
    )
    room_max_x = (
        max(w.transform.translation()[0] for w in rg.walls) if rg.walls else 10.0
    )
    room_min_y = (
        min(w.transform.translation()[1] for w in rg.walls) if rg.walls else 0.0
    )
    room_max_y = (
        max(w.transform.translation()[1] for w in rg.walls) if rg.walls else 10.0
    )

    violations = []
    for obj in scene.objects.values():
        world_min, world_max = get_world_aabb(obj)
        cx = (world_min[0] + world_max[0]) / 2
        cy = (world_min[1] + world_max[1]) / 2
        if cx < room_min_x or cx > room_max_x or cy < room_min_y or cy > room_max_y:
            violations.append(
                f"Out of bounds: {obj.object_id} ({obj.name}) centre "
                f"({cx:.2f}, {cy:.2f}) outside room "
                f"[{room_min_x:.2f},{room_max_x:.2f}] x [{room_min_y:.2f},{room_max_y:.2f}]"
            )
    return violations
