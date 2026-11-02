"""Deterministic regeneration of rl_artifacts.json after scene mutation."""

from __future__ import annotations

import copy


def regenerate_rl_artifacts(
    base_artifacts: dict,
    new_scene_state: dict,
    task_critical_ids: list[str],
) -> dict:
    """Recompute clutter_ids and per-task-object position ranges from the mutated scene.

    Args:
        base_artifacts: Parsed rl_artifacts.json from the base scene (template for keys).
        new_scene_state: Parsed scene_state.json after DR mutation.
        task_critical_ids: IDs that remain task objects; other manipulands become clutter.

    Returns:
        New dict with the same top-level keys as base_artifacts, with updated
        domain_randomization clutter_ids and task_object_variants position ranges.

    Raises:
        ValueError: If a task-critical id is missing from new_scene_state.
    """
    result = copy.deepcopy(base_artifacts)
    task_critical_set = set(task_critical_ids)
    objects = new_scene_state.get("objects", {})
    if not isinstance(objects, dict):
        raise ValueError("new_scene_state.objects must be a dict keyed by object id")

    for object_id in task_critical_ids:
        if object_id not in objects:
            raise ValueError(
                f"task-critical id {object_id!r} missing from mutated scene"
            )

    manipuland_ids = sorted(
        obj.get("object_id", oid)
        for oid, obj in objects.items()
        if obj.get("object_type") == "manipuland"
    )
    clutter_ids = sorted(oid for oid in manipuland_ids if oid not in task_critical_set)

    base_dr = base_artifacts.get("domain_randomization", {})
    dr = result.setdefault("domain_randomization", {})
    dr["clutter_ids"] = clutter_ids

    base_variants = {v["object_id"]: v for v in base_dr.get("task_object_variants", [])}
    updated_variants = []
    for variant in dr.get("task_object_variants", []):
        object_id = variant["object_id"]
        updated = copy.deepcopy(variant)
        base_variant = base_variants.get(object_id, variant)
        eps = base_variant.get("position_range_xy_on_surface")
        if eps and object_id in objects:
            updated["position_range_xy_on_surface"] = list(eps)
        updated_variants.append(updated)
    dr["task_object_variants"] = updated_variants

    return result
