"""LLM-directed scene editing for domain randomization.

A single structured-output LLM call interprets a natural language variation
goal and returns targeted edit operations (move, remove, scale). The LLM sees
a compact scene summary rather than the full pipeline context.

Used by the ``llm_directed`` strategy in DomainRandomizationOrchestrator. For
agentic multi-turn DR (planner/designer/critic loop), use
``mode="agentic"`` + DRSubflowAgent instead.
"""

from __future__ import annotations

import asyncio
import logging
import math

from typing import Annotated, Literal, Union

import numpy as np

from pydantic import BaseModel, Field
from pydrake.math import RigidTransform, RotationMatrix

from scenesmith.agent_utils.room import RoomScene
from scenesmith.agent_utils.sdk_vlm_runner import run_structured_vlm_call_messages
from scenesmith.domain_randomization.programmatic.collision_checker import (
    check_collisions,
    check_room_bounds,
)

logger = logging.getLogger(__name__)

# ── Edit operation schemas ─────────────────────────────────────────────────────


class MoveEdit(BaseModel):
    op: Literal["move"] = "move"
    object_id: str = Field(description="Exact object_id from the scene list")
    delta_x_m: float = Field(description="X displacement in metres (positive = right)")
    delta_y_m: float = Field(
        description="Y displacement in metres (positive = forward)"
    )
    delta_yaw_deg: float = Field(
        default=0.0, description="Yaw rotation delta in degrees"
    )


class RemoveEdit(BaseModel):
    op: Literal["remove"] = "remove"
    object_id: str = Field(description="Exact object_id from the scene list")


class ScaleEdit(BaseModel):
    op: Literal["scale"] = "scale"
    object_id: str = Field(description="Exact object_id from the scene list")
    scale_factor: float = Field(
        description="Multiplier applied to current scale (e.g. 0.9 = 10% smaller). Must be in [0.7, 1.3]."
    )


SceneEdit = Annotated[
    Union[MoveEdit, RemoveEdit, ScaleEdit],
    Field(discriminator="op"),
]


class SceneEditPlan(BaseModel):
    reasoning: str = Field(description="Brief explanation of the chosen edits and why")
    edits: list[SceneEdit] = Field(
        description="Ordered list of edit operations to apply"
    )


# ── Scene summary builder ──────────────────────────────────────────────────────


def _build_scene_summary(scene: RoomScene, locked_ids: set[str]) -> str:
    lines: list[str] = []

    rg = scene.room_geometry
    if rg is not None and rg.walls:
        room_min_x = min(w.transform.translation()[0] for w in rg.walls)
        room_max_x = max(w.transform.translation()[0] for w in rg.walls)
        room_min_y = min(w.transform.translation()[1] for w in rg.walls)
        room_max_y = max(w.transform.translation()[1] for w in rg.walls)
        lines.append(
            f"Room bounds: X [{room_min_x:.2f}, {room_max_x:.2f}] m, "
            f"Y [{room_min_y:.2f}, {room_max_y:.2f}] m"
        )

    if scene.text_description:
        lines.append(f"Scene: {scene.text_description[:200]}")

    lines.append("")
    lines.append("Objects (object_id | type | name | XY position [m]):")
    for obj_id, obj in scene.objects.items():
        t = obj.transform.translation()
        lock = (
            " [LOCKED — do not move, scale, or remove]" if obj_id in locked_ids else ""
        )
        lines.append(
            f"  {obj_id} | {obj.object_type.value} | {obj.name}"
            f" | ({t[0]:.2f}, {t[1]:.2f}){lock}"
        )

    return "\n".join(lines)


# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a scene layout editor for a robotics simulation environment.
Your task is to produce a list of targeted edit operations to modify an indoor scene
so that it matches a described variation goal.

Rules:
- NEVER move, scale, or remove objects marked [LOCKED].
- After applying edits, all objects must remain within the given room bounds.
- Typical translation deltas: ±0.05 – 0.50 m; use up to ±1.0 m only for deliberate repositioning.
- Scale multipliers must be in [0.7, 1.3].
- Only use object_ids that appear in the scene list — do not invent new IDs.
- Prefer few, targeted edits over many small perturbations.
- If no safe edit can achieve the goal, return an empty edits list and explain why in reasoning.
"""


# ── LLM call ──────────────────────────────────────────────────────────────────


async def generate_scene_edits(
    scene: RoomScene,
    description: str,
    locked_ids: set[str],
    cfg: dict,
) -> SceneEditPlan:
    """Call the LLM once with a structured-output schema and return edit ops.

    Args:
        scene: Current room scene.
        description: Natural language variation goal (from prompt_injection or spec description).
        locked_ids: Object IDs the LLM must not touch.
        cfg: Full resolved Hydra config as a plain dict.
    """
    scene_summary = _build_scene_summary(scene, locked_ids)
    user_input = f"Scene state:\n{scene_summary}\n\nVariation goal: {description}"

    provider = cfg.get("provider", {})
    capabilities = provider.get("capabilities", {})
    thinking_mode = capabilities.get("thinking_control_mode", "openai")

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    return await asyncio.to_thread(
        run_structured_vlm_call_messages,
        model=provider["model"],
        messages=messages,
        output_type=SceneEditPlan,
        # Force thinking off only on binary-mode providers (Qwen/Gemma via vLLM).
        # OpenAI-mode providers handle reasoning natively.
        reasoning_effort="minimal" if thinking_mode == "binary" else None,
        verbosity=None,
        subsystem="llm_scene_editor",
        agent_name="llm_scene_editor",
    )


# ── Apply edits ────────────────────────────────────────────────────────────────


def apply_scene_edits(
    scene: RoomScene,
    plan: SceneEditPlan,
    locked_ids: set[str],
) -> tuple[RoomScene, list[str]]:
    """Apply a SceneEditPlan to a RoomScene in-place.

    Locked-object edits and missing object_ids are silently skipped (same
    behaviour as programmatic ops). Collision/bounds violations are logged
    but do not abort.

    Returns:
        (modified scene, list of skipped-edit reason strings)
    """
    skipped: list[str] = []

    for edit in plan.edits:
        obj_id = edit.object_id

        if obj_id in locked_ids:
            skipped.append(f"skip {edit.op} on '{obj_id}': locked")
            continue
        if obj_id not in scene.objects:
            skipped.append(f"skip {edit.op} on '{obj_id}': not in scene")
            continue

        obj = scene.objects[obj_id]

        if isinstance(edit, MoveEdit):
            t = np.array(obj.transform.translation())
            t[0] += edit.delta_x_m
            t[1] += edit.delta_y_m
            if edit.delta_yaw_deg != 0.0:
                delta_rot = RotationMatrix.MakeZRotation(
                    math.radians(edit.delta_yaw_deg)
                )
                new_rot = delta_rot @ obj.transform.rotation()
            else:
                new_rot = obj.transform.rotation()
            obj.transform = RigidTransform(new_rot, t)

        elif isinstance(edit, RemoveEdit):
            del scene.objects[obj_id]

        elif isinstance(edit, ScaleEdit):
            factor = max(0.7, min(1.3, edit.scale_factor))
            obj.apply_scale(factor)

    violations = check_collisions(scene) + check_room_bounds(scene)
    if violations:
        logger.warning(
            "llm_directed edits produced %d violation(s): %s",
            len(violations),
            violations[:3],
        )

    return scene, skipped
