"""Reward Spec Agent: generates dense reward function and curriculum spec from RL artifacts.

Single-shot SDK call — no multi-turn agent loop. Follows the same pattern as TaskAnalysisAgent.
"""

import json
import logging

from pydantic import BaseModel, Field

from scenesmith.agent_utils.sdk_vlm_runner import run_structured_vlm_call_messages
from scenesmith.prompts import prompt_registry
from scenesmith.prompts.registry import RewardSpecAgentPrompts
from scenesmith.reward_spec_agent.reward_spec import (
    END_EFFECTOR,
    CurriculumSpec,
    RewardSpec,
)
from scenesmith.task_analysis_agent.task_requirements import TaskRequirements

console_logger = logging.getLogger(__name__)


class _RewardTermSchema(BaseModel):
    name: str
    type: str
    weight: float
    shaping: str = "linear"
    active_in_stages: list[int] = Field(default_factory=list)
    from_object_id: str | None = None
    to_object_id: str | None = None


class _RewardSpecSchema(BaseModel):
    reward_terms: list[_RewardTermSchema] = Field(default_factory=list)
    notes: str = ""


class _CurriculumStageSchema(BaseModel):
    id: int
    name: str
    active_task_object_ids: list[str] = Field(default_factory=list)
    dr_scale: float = 0.0
    clutter_enabled: bool = False
    tolerance_m: float = 0.1
    transition_success_rate: float = 0.8
    transition_window_episodes: int = 50
    min_episodes_in_stage: int = 100


class _CurriculumSpecSchema(BaseModel):
    stages: list[_CurriculumStageSchema] = Field(default_factory=list)
    notes: str = ""


class _RewardSpecResponseSchema(BaseModel):
    """Top-level Pydantic mirror used as SDK output_type.

    Wraps both halves of the RewardSpecAgent's output so a single structured
    call returns the full pair. We then re-validate via the dataclass
    from_dict methods so cross-validation (object IDs × stage IDs) still runs.
    """

    reward_spec: _RewardSpecSchema
    curriculum_spec: _CurriculumSpecSchema


class RewardSpecAgent:
    """Single-shot SDK call to generate reward_spec.yaml + curriculum_spec.yaml.

    Reads rl_artifacts.json and task_requirements.json, calls the model once,
    validates the output, and returns (RewardSpec, CurriculumSpec).

    Parse order inside generate() is intentional: curriculum_spec is parsed
    first so its stage IDs can be used to cross-validate reward_spec
    active_in_stages. Reversing this order silently skips the validation.
    """

    def __init__(self, cfg: dict) -> None:
        from scenesmith.provider import resolve_model_name

        openai_cfg = cfg.get("reward_spec_agent", {}).get("openai", {})
        self.model = resolve_model_name(cfg, agent_section="reward_spec_agent")
        self.reasoning_effort = openai_cfg.get("reasoning_effort", "high")
        self.verbosity = openai_cfg.get("verbosity", "medium")

        validation_cfg = cfg.get("reward_spec_agent", {}).get("validation", {})
        self.strict_validation = bool(validation_cfg.get("strict", True))

    def generate(
        self,
        rl_artifacts: dict,
        task_requirements: TaskRequirements,
    ) -> tuple[RewardSpec, CurriculumSpec]:
        """Generate reward spec and curriculum spec from grounded RL artifacts.

        Args:
            rl_artifacts: Parsed rl_artifacts.json dict.
            task_requirements: TaskRequirements from task analysis step.

        Returns:
            (RewardSpec, CurriculumSpec) tuple, both validated.

        Raises:
            ValueError: If the LLM output fails object ID or stage ID validation.
        """
        rendered = prompt_registry.get_prompt(
            RewardSpecAgentPrompts.REWARD_SPEC_AGENT,
            rl_artifacts_json=json.dumps(rl_artifacts, indent=2),
            task_requirements_json=json.dumps(task_requirements.to_dict(), indent=2),
        )

        console_logger.info(
            "Generating reward spec for task: %s", task_requirements.task_summary
        )

        response = run_structured_vlm_call_messages(
            model=self.model,
            messages=[{"role": "user", "content": rendered}],
            output_type=_RewardSpecResponseSchema,
            reasoning_effort=self.reasoning_effort,
            verbosity=self.verbosity,
            subsystem="reward_spec",
            agent_name="reward_spec_agent",
        )

        data = response.model_dump()

        # Extract known object IDs for validation (non-goal objects are also valid refs).
        known_object_ids: set[str] = {
            obj["object_id"] for obj in rl_artifacts.get("task_objects", [])
        }
        goal_location = (rl_artifacts.get("success_conditions") or {}).get(
            "goal_location"
        )
        if isinstance(goal_location, str) and goal_location:
            known_object_ids.add(goal_location)

        # Parse curriculum FIRST to get valid stage IDs for reward_spec validation.
        curriculum_spec = CurriculumSpec.from_dict(data["curriculum_spec"])
        valid_stage_ids = curriculum_spec.stage_ids()

        # Parse reward spec with full cross-validation.
        try:
            reward_spec = RewardSpec.from_dict(
                data["reward_spec"],
                known_object_ids=known_object_ids,
                valid_stage_ids=valid_stage_ids,
            )
        except ValueError as e:
            if self.strict_validation:
                raise

            console_logger.warning(
                "RewardSpec validation failed; continuing with best-effort filtered spec: %s",
                e,
            )
            unvalidated = RewardSpec.from_dict(data["reward_spec"])

            filtered_terms = []
            for term in unvalidated.reward_terms:
                if (
                    term.from_object_id is not None
                    and term.from_object_id != END_EFFECTOR
                    and term.from_object_id not in known_object_ids
                ):
                    continue
                if (
                    term.to_object_id is not None
                    and term.to_object_id != END_EFFECTOR
                    and term.to_object_id not in known_object_ids
                ):
                    continue

                if term.active_in_stages:
                    cleaned = [
                        sid for sid in term.active_in_stages if sid in valid_stage_ids
                    ]
                    if not cleaned:
                        continue
                    term.active_in_stages = cleaned

                filtered_terms.append(term)

            dropped = len(unvalidated.reward_terms) - len(filtered_terms)
            if dropped:
                console_logger.warning("Dropped %d invalid reward term(s).", dropped)

            reward_spec = RewardSpec(
                reward_terms=filtered_terms, notes=unvalidated.notes
            )

        console_logger.info(
            "Reward spec generated: %d terms, %d curriculum stages",
            len(reward_spec.reward_terms),
            len(curriculum_spec.stages),
        )

        return reward_spec, curriculum_spec
