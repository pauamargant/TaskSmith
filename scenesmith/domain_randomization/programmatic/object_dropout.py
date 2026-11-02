"""Object dropout: remove a random fraction of non-critical objects."""

from __future__ import annotations

import logging
import random

from scenesmith.agent_utils.room import RoomScene
from scenesmith.domain_randomization.programmatic.base import BaseRandomizationOp
from scenesmith.domain_randomization.programmatic.ops_utils import _resolve_types

console_logger = logging.getLogger(__name__)


class ObjectDropout(BaseRandomizationOp):
    """Remove a random percentage of objects by type.

    Task-critical objects (locked_ids) are never removed.

    Params:
        rate (float): Fraction of eligible objects to remove. Default 0.3.
        target_object_types (list[str]): Object type strings to consider for removal.
            Default: ["manipuland"].
        min_count (int): Minimum objects of each targeted type to keep. Default 1.
        preferred_ids (list[str]): Object IDs to prefer for removal (e.g. clutter_ids
            from rl_artifacts). When provided, these are removed first before random
            selection fills the remainder up to `rate`.
        seed (int | None): Random seed for reproducibility.
    """

    def apply(self, scene: RoomScene, params: dict, locked_ids: set[str]) -> RoomScene:
        rate = float(params.get("rate", 0.3))
        target_types = _resolve_types(params.get("target_object_types", ["manipuland"]))
        min_count = int(params.get("min_count", 1))
        preferred_ids: list[str] = params.get("preferred_ids", [])
        seed = params.get("seed")
        rng = random.Random(seed)

        eligible = [
            obj
            for obj in scene.objects.values()
            if obj.object_id not in locked_ids
            and (not target_types or obj.object_type in target_types)
        ]

        if not eligible:
            return scene

        n_remove = max(0, int(len(eligible) * rate))
        n_keep_min = max(0, len(eligible) - max(n_remove, len(eligible) - min_count))
        n_remove = min(n_remove, len(eligible) - n_keep_min)

        if n_remove == 0:
            return scene

        # Prioritise preferred_ids (e.g. clutter objects from rl_artifacts).
        preferred = [o for o in eligible if o.object_id in set(preferred_ids)]
        others = [o for o in eligible if o.object_id not in set(preferred_ids)]

        to_remove = preferred[:n_remove]
        remaining_slots = n_remove - len(to_remove)
        if remaining_slots > 0:
            rng.shuffle(others)
            to_remove.extend(others[:remaining_slots])

        for obj in to_remove:
            console_logger.debug(
                f"ObjectDropout: removing {obj.object_id} ({obj.name})"
            )
            del scene.objects[obj.object_id]

        return scene

    def validate(self, scene: RoomScene) -> list[str]:
        return []
