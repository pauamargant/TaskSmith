"""RL Artifact Exporter for grounding task requirements to final scene objects.

Single-shot SDK call that reviews the completed scene and task requirements,
then produces a grounded rl_artifacts.json with actual object IDs, positions,
success conditions, and domain randomization specs.
"""

import base64
import json
import logging

from pathlib import Path

from agents.agent_output import AgentOutputSchema
from pydantic import BaseModel, Field

from scenesmith.agent_utils.house import HouseScene
from scenesmith.agent_utils.room import ObjectType
from scenesmith.agent_utils.sdk_vlm_runner import run_structured_vlm_call
from scenesmith.prompts import prompt_registry
from scenesmith.prompts.registry import TaskAnalysisAgentPrompts
from scenesmith.task_analysis_agent.task_requirements import TaskRequirements

console_logger = logging.getLogger(__name__)

# Object types included in RL artifacts (task-relevant only).
_TASK_RELEVANT_TYPES = {ObjectType.FURNITURE, ObjectType.MANIPULAND}


class _TaskObjectSchema(BaseModel):
    object_id: str
    name: str
    position_xyz: list[float]
    bbox_dims: list[float] | None = None
    sdf_path: str | None = None
    parent_surface: str | None = None
    is_initial_object: bool = False
    is_goal_location: bool = False


class _SuccessConditionsSchema(BaseModel):
    description: str = ""
    goal_objects: list[str] = Field(default_factory=list)
    goal_location: str | None = None
    tolerance_m: float = 0.1


class _TaskObjectVariantSchema(BaseModel):
    object_id: str
    can_swap_with: list[str] = Field(default_factory=list)
    randomizable_properties: list[str] = Field(default_factory=list)
    position_range_xy_on_surface: list[float] = Field(default_factory=list)


class _FurnitureMovableSchema(BaseModel):
    object_id: str
    can_reposition: bool = False
    reposition_range_xy: list[float] = Field(default_factory=list)
    can_swap_with: list[str] = Field(default_factory=list)
    constraint: str = ""


class _DomainRandomizationArtifactSchema(BaseModel):
    task_object_variants: list[_TaskObjectVariantSchema] = Field(default_factory=list)
    furniture_movable: list[_FurnitureMovableSchema] = Field(default_factory=list)
    clutter_ids: list[str] = Field(default_factory=list)
    scene_level: dict = Field(default_factory=dict)


class _VerificationSchema(BaseModel):
    all_required_objects_present: bool = True
    missing_objects: list[str] = Field(default_factory=list)
    notes: str = ""


class _RLArtifactsSchema(BaseModel):
    """Pydantic mirror of rl_artifacts.json used as SDK output_type.

    initial_state is left as a generic dict (object_id keys are dynamic).
    """

    task_objects: list[_TaskObjectSchema] = Field(default_factory=list)
    success_conditions: _SuccessConditionsSchema = Field(
        default_factory=_SuccessConditionsSchema
    )
    initial_state: dict = Field(default_factory=dict)
    domain_randomization: _DomainRandomizationArtifactSchema = Field(
        default_factory=_DomainRandomizationArtifactSchema
    )
    verification: _VerificationSchema = Field(default_factory=_VerificationSchema)


def _normalize_domain_randomization(artifacts: dict) -> None:
    """Normalize domain_randomization fields in-place so the written file is type-consistent.

    The LLM sometimes returns clutter_ids as [{"object_id": "..."}] instead of the
    documented ["..."], causing _extract_object_ids in orchestrator.py to do post-hoc
    normalization on every read. Fixing at write time means the file is always canonical.
    """
    dr = artifacts.get("domain_randomization")
    if not isinstance(dr, dict):
        return

    raw = dr.get("clutter_ids")
    if not isinstance(raw, list):
        return

    normalized: list[str] = []
    for entry in raw:
        if isinstance(entry, str) and entry:
            normalized.append(entry)
        elif isinstance(entry, dict):
            oid = entry.get("object_id")
            if isinstance(oid, str) and oid:
                normalized.append(oid)
    dr["clutter_ids"] = normalized


class RLArtifactAgent:
    """SDK agent that reviews completed scene and produces grounded RL artifacts."""

    def __init__(self, cfg: dict) -> None:
        from scenesmith.provider import resolve_model_name

        openai_cfg = cfg.get("task_analysis_agent", {}).get("openai", {})
        self.model = resolve_model_name(cfg, agent_section="task_analysis_agent")
        self.reasoning_effort = openai_cfg.get("reasoning_effort", "high")
        self.verbosity = openai_cfg.get("verbosity", "medium")

    def export(
        self,
        house_scene: HouseScene,
        task_requirements: TaskRequirements,
        output_path: Path,
        render_paths: list[Path] | None = None,
    ) -> dict:
        """Produce rl_artifacts.json from completed scene + task requirements.

        Args:
            house_scene: Fully populated HouseScene.
            task_requirements: TaskRequirements from the task analysis step.
            output_path: Path to write rl_artifacts.json.
            render_paths: Optional top-down renders for visual grounding (up to 4).

        Returns:
            Parsed artifact dict (also written to output_path).
        """
        scene_objects = self._extract_scene_objects(house_scene)

        rendered = prompt_registry.get_prompt(
            TaskAnalysisAgentPrompts.RL_ARTIFACT_AGENT,
            task_requirements_json=json.dumps(task_requirements.to_dict(), indent=2),
            scene_objects_json=json.dumps(scene_objects, indent=2),
        )

        images_b64: list[str] = []
        if render_paths:
            for rp in render_paths[:4]:
                rp_path = Path(rp)
                if rp_path.exists():
                    images_b64.append(base64.b64encode(rp_path.read_bytes()).decode())

        console_logger.info("Running RL artifact export...")
        response = run_structured_vlm_call(
            model=self.model,
            instructions="",
            user_text=rendered,
            images_b64=images_b64,
            output_type=AgentOutputSchema(_RLArtifactsSchema, strict_json_schema=False),
            reasoning_effort=self.reasoning_effort,
            verbosity=self.verbosity,
            subsystem="rl_artifact_export",
            agent_name="rl_artifact_agent",
            timeout_s=1800,
        )
        artifacts = response.model_dump()
        _normalize_domain_randomization(artifacts)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(artifacts, f, indent=2)
        console_logger.info("RL artifacts written to %s", output_path)
        return artifacts

    def _extract_scene_objects(self, house_scene: HouseScene) -> list[dict]:
        """Flatten task-relevant objects from all rooms into a serializable list.

        Only includes FURNITURE and MANIPULAND objects to keep VLM context manageable.
        """
        objects = []
        for room_id, room_scene in house_scene.rooms.items():
            for obj_id, obj in room_scene.objects.items():
                if obj.object_type not in _TASK_RELEVANT_TYPES:
                    continue
                position = obj.transform.translation().tolist()
                bbox_dims = None
                if obj.bbox_min is not None and obj.bbox_max is not None:
                    bbox_dims = (obj.bbox_max - obj.bbox_min).tolist()
                parent_surface = (
                    obj.placement_info.parent_surface_id
                    if obj.placement_info is not None
                    else None
                )
                objects.append(
                    {
                        "object_id": obj_id,
                        "name": obj.name,
                        "room_id": room_id,
                        "object_type": obj.object_type.value,
                        "position_xyz": position,
                        "bbox_dims": bbox_dims,
                        "sdf_path": str(obj.sdf_path) if obj.sdf_path else None,
                        "parent_surface": parent_surface,
                        "metadata": dict(obj.metadata),
                    }
                )
        return objects
