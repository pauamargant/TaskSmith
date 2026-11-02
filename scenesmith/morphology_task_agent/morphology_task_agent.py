"""Morphology Task Agent: selects the best RL task from a TaskSmith scene.

Single-shot SDK call, no multi-turn agent loop. Follows the same pattern as
RewardSpecAgent. Reads rl_artifacts.json, reward_spec.yaml (optional), and
curriculum_spec.yaml (optional) to produce morphology_task_config.json.
"""

import json
import logging

import yaml

from pydantic import BaseModel, Field

from scenesmith.agent_utils.sdk_vlm_runner import run_structured_vlm_call_messages
from scenesmith.morphology_task_agent.morphology_task_config import MorphologyTaskConfig
from scenesmith.prompts import prompt_registry
from scenesmith.prompts.registry import MorphologyTaskAgentPrompts

console_logger = logging.getLogger(__name__)


class _RewardConfigSchema(BaseModel):
    reach_weight: float = 2.0
    contact_bonus_weight: float = 1.0
    lift_weight: float = 4.0
    lift_target_height_m: float = 0.20
    grasp_proximity_threshold_m: float = 0.10
    drop_penalty: float = -10.0
    action_smoothness_weight: float = -0.001


class _MorphologyCurriculumStageSchema(BaseModel):
    id: int
    name: str
    active_phases: list[str] = Field(default_factory=list)
    transition_success_rate: float = 0.7
    transition_window_episodes: int = 20
    min_episodes_in_stage: int = 50


class _MorphologyTaskConfigSchema(BaseModel):
    """Pydantic mirror of MorphologyTaskConfig used as the SDK output_type.

    Kept private to this module; the persisted/imported representation
    everywhere else stays the dataclass MorphologyTaskConfig.
    """

    selected_object_id: str
    selected_object_name: str
    task_phases: list[str] = Field(
        default_factory=lambda: ["approach", "grasp", "lift"]
    )
    curriculum_stages: list[_MorphologyCurriculumStageSchema] = Field(
        default_factory=list
    )
    reward_config: _RewardConfigSchema = Field(default_factory=_RewardConfigSchema)
    hand_init_offset_above_object_m: float = 0.25
    rationale: str = ""


class MorphologyTaskAgent:
    """Single-shot VLM call to select the best task for morphological hand training.

    Reads rl_artifacts.json (required), reward_spec.yaml (optional), and
    curriculum_spec.yaml (optional). Returns a MorphologyTaskConfig with exactly
    one target object, a list of task phases, and reward weights.

    The agent prefers objects with good graspability (cylindrical, appropriately
    sized) over flat or oversized objects, using bbox_dims and object name as
    heuristics. When reward_spec.yaml is provided it is passed as context so the
    agent can set reward weights consistently with the scene's reward function;
    when absent the agent uses sensible defaults.
    """

    def __init__(self, cfg: dict) -> None:
        from scenesmith.provider import resolve_model_name

        openai_cfg = cfg.get("morphology_task_agent", {}).get("openai", {})
        self.model = resolve_model_name(cfg, agent_section="morphology_task_agent")
        self.reasoning_effort = openai_cfg.get("reasoning_effort", "high")
        self.verbosity = openai_cfg.get("verbosity", "medium")

    def generate(
        self,
        rl_artifacts: dict,
        reward_spec: dict | None = None,
        curriculum_spec: dict | None = None,
    ) -> MorphologyTaskConfig:
        """Select the best morphological training task from scene artifacts.

        Args:
            rl_artifacts: Parsed rl_artifacts.json dict (required).
            reward_spec: Parsed reward_spec.yaml dict (optional). When provided,
                passed as context so the agent can align reward weights with the
                scene's reward function. When absent, defaults are used.
            curriculum_spec: Parsed curriculum_spec.yaml dict (optional). Passed
                for context so the agent understands stage structure.

        Returns:
            MorphologyTaskConfig with exactly one target object and task phases.

        Raises:
            ValueError: If the LLM output is missing required fields or references
                an object_id not present in rl_artifacts.task_objects.
        """
        rendered = prompt_registry.get_prompt(
            MorphologyTaskAgentPrompts.MORPHOLOGY_TASK_AGENT,
            rl_artifacts_json=json.dumps(rl_artifacts, indent=2),
            reward_spec_yaml=(
                yaml.dump(reward_spec, default_flow_style=False)
                if reward_spec is not None
                else "(not available — use default reward weights)"
            ),
            curriculum_spec_yaml=(
                yaml.dump(curriculum_spec, default_flow_style=False)
                if curriculum_spec is not None
                else "(not available)"
            ),
        )

        task_objects = rl_artifacts.get("task_objects", [])
        initial_objects = [o for o in task_objects if o.get("is_initial_object")]
        console_logger.info(
            "Selecting morphology task from %d initial object(s): %s",
            len(initial_objects),
            [o["object_id"] for o in initial_objects],
        )

        response = run_structured_vlm_call_messages(
            model=self.model,
            messages=[{"role": "user", "content": rendered}],
            output_type=_MorphologyTaskConfigSchema,
            reasoning_effort=self.reasoning_effort,
            verbosity=self.verbosity,
            subsystem="morphology_task",
            agent_name="morphology_task_agent",
        )

        config = MorphologyTaskConfig.from_dict(response.model_dump())

        # Validate: selected_object_id must exist in rl_artifacts.
        known_ids = {o["object_id"] for o in task_objects}
        if config.selected_object_id not in known_ids:
            raise ValueError(
                f"MorphologyTaskAgent selected '{config.selected_object_id}' which is "
                f"not in rl_artifacts task_objects. Known IDs: {sorted(known_ids)}"
            )

        # Validate: task_phases must be a non-empty subset of allowed values.
        allowed_phases = {"approach", "grasp", "lift", "transport"}
        invalid = set(config.task_phases) - allowed_phases
        if invalid:
            raise ValueError(
                f"MorphologyTaskAgent returned invalid task_phases: {invalid}. "
                f"Allowed: {allowed_phases}"
            )
        if not config.task_phases:
            raise ValueError("MorphologyTaskAgent returned empty task_phases list.")

        # Validate: curriculum_stages must have one stage per task_phase, in order.
        if config.curriculum_stages:
            if len(config.curriculum_stages) != len(config.task_phases):
                raise ValueError(
                    f"curriculum_stages has {len(config.curriculum_stages)} entries but "
                    f"task_phases has {len(config.task_phases)}. Must match 1:1."
                )
            for i, stage in enumerate(config.curriculum_stages):
                expected_phases = config.task_phases[: i + 1]
                if stage.active_phases != expected_phases:
                    raise ValueError(
                        f"Stage {i} active_phases {stage.active_phases} must be cumulative "
                        f"prefix of task_phases: {expected_phases}"
                    )
                if stage.min_episodes_in_stage < 2 * stage.transition_window_episodes:
                    console_logger.warning(
                        "Stage %d '%s': min_episodes_in_stage (%d) < 2 × window (%d). "
                        "Risk of lucky-streak transitions.",
                        stage.id,
                        stage.name,
                        stage.min_episodes_in_stage,
                        stage.transition_window_episodes,
                    )

        console_logger.info(
            "Selected object: %s (%s), phases: %s, curriculum: %d stages",
            config.selected_object_id,
            config.selected_object_name,
            config.task_phases,
            len(config.curriculum_stages),
        )
        console_logger.info("Rationale: %s", config.rationale)

        return config
