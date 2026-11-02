"""Task Analysis Agent for extracting RL task requirements from task descriptions.

Single-shot SDK call that parses an RL task prompt and returns a structured
TaskRequirements object. Follows the same pattern as SceneAnalyzer.
"""

import logging

from agents.agent_output import AgentOutputSchema
from pydantic import BaseModel, Field

from scenesmith.agent_utils.sdk_vlm_runner import run_structured_vlm_call_messages
from scenesmith.prompts import prompt_registry
from scenesmith.prompts.registry import TaskAnalysisAgentPrompts
from scenesmith.task_analysis_agent.task_requirements import TaskRequirements

console_logger = logging.getLogger(__name__)


class _RequiredObjectSchema(BaseModel):
    name: str
    is_task_object: bool = False
    placement_stage: str = "furniture"
    properties: dict = Field(default_factory=dict)
    placement_hint: str = ""


class _DomainRandomizationSchema(BaseModel):
    task_object_variants: list[dict] = Field(default_factory=list)
    furniture_movable: list[dict] = Field(default_factory=list)
    clutter_objects: list[dict] = Field(default_factory=list)
    scene_level: dict = Field(default_factory=dict)
    notes: str = ""


class _TaskRequirementsSchema(BaseModel):
    """Pydantic mirror of TaskRequirements used as SDK output_type.

    raw_prompt is filled in by the caller (not the model) so it's omitted
    here — the model has no reason to echo the prompt back.
    """

    task_summary: str = ""
    task_type: str = "other"
    scene_description: str = ""
    required_objects: list[_RequiredObjectSchema] = Field(default_factory=list)
    task_objects: list[str] = Field(default_factory=list)
    spatial_constraints: list[str] = Field(default_factory=list)
    scene_requirements: dict = Field(default_factory=dict)
    rl_metadata: dict = Field(default_factory=dict)
    domain_randomization: _DomainRandomizationSchema = Field(
        default_factory=_DomainRandomizationSchema
    )


class TaskAnalysisAgent:
    """Single-shot SDK call to extract task requirements from an RL task prompt."""

    def __init__(self, cfg: dict) -> None:
        from scenesmith.provider import resolve_model_name

        openai_cfg = cfg.get("task_analysis_agent", {}).get("openai", {})
        self.model = resolve_model_name(cfg, agent_section="task_analysis_agent")
        self.reasoning_effort = openai_cfg.get("reasoning_effort", "high")
        self.verbosity = openai_cfg.get("verbosity", "medium")

    def analyze(self, task_prompt: str) -> TaskRequirements:
        """Run task analysis and return structured TaskRequirements.

        Args:
            task_prompt: The RL task description to analyze.

        Returns:
            TaskRequirements with extracted scene and training requirements.
        """
        rendered = prompt_registry.get_prompt(
            TaskAnalysisAgentPrompts.ANALYZER_AGENT,
            task_prompt=task_prompt,
        )
        console_logger.info("Running task analysis for prompt: %s", task_prompt[:120])
        response = run_structured_vlm_call_messages(
            model=self.model,
            messages=[{"role": "user", "content": rendered}],
            output_type=AgentOutputSchema(
                _TaskRequirementsSchema, strict_json_schema=False
            ),
            reasoning_effort=self.reasoning_effort,
            verbosity=self.verbosity,
            subsystem="task_analysis",
            agent_name="task_analysis_agent",
            timeout_s=1800,
        )
        data = response.model_dump()
        data["raw_prompt"] = task_prompt
        requirements = TaskRequirements.from_dict(data)
        console_logger.info("Task analysis complete: %s", requirements.task_summary)
        return requirements
