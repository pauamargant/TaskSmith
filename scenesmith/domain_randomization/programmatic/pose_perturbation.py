"""Pose perturbation: Gaussian noise on XY translation and yaw rotation."""

from __future__ import annotations

import numpy as np

from pydrake.math import RigidTransform, RotationMatrix

from scenesmith.agent_utils.room import ObjectType, RoomScene
from scenesmith.domain_randomization.programmatic.base import BaseRandomizationOp
from scenesmith.domain_randomization.programmatic.collision_checker import (
    check_collisions,
    check_room_bounds,
)
from scenesmith.domain_randomization.programmatic.ops_utils import _resolve_types

_OBJECT_TYPE_MAP: dict[str, ObjectType] = {t.value: t for t in ObjectType}


class PosePerturbation(BaseRandomizationOp):
    """Apply Gaussian noise to object XY position and yaw rotation.

    Params:
        translation_std_m (float): Std-dev for XY displacement in metres. Default 0.05.
        rotation_std_deg (float): Std-dev for yaw rotation in degrees. Default 5.0.
        target_object_types (list[str]): Object type strings to perturb.
            Defaults to all non-wall, non-floor types.
        seed (int | None): Random seed for reproducibility.
    """

    def apply(self, scene: RoomScene, params: dict, locked_ids: set[str]) -> RoomScene:
        rng = np.random.default_rng(params.get("seed"))
        translation_std = float(params.get("translation_std_m", 0.05))
        rotation_std = float(params.get("rotation_std_deg", 5.0))
        target_types = _resolve_types(params.get("target_object_types", []))

        for obj in scene.objects.values():
            if obj.object_id in locked_ids:
                continue
            if target_types and obj.object_type not in target_types:
                continue

            t = np.array(obj.transform.translation())
            dx, dy = rng.normal(0.0, translation_std, size=2)
            t[0] += dx
            t[1] += dy

            # Apply yaw perturbation around Z axis.
            dyaw = float(rng.normal(0.0, np.deg2rad(rotation_std)))
            current_rot = obj.transform.rotation()
            delta_rot = RotationMatrix.MakeZRotation(dyaw)
            new_rot = delta_rot @ current_rot

            obj.transform = RigidTransform(new_rot, t)

        return scene

    def validate(self, scene: RoomScene) -> list[str]:
        return check_collisions(scene) + check_room_bounds(scene)
