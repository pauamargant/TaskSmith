"""DR v2 Director: single structured-output call producing N VariationGoals."""

from __future__ import annotations

import json
import logging

from dataclasses import dataclass
from pathlib import Path

from agents.agent_output import AgentOutputSchema
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field

from scenesmith.agent_utils.sdk_vlm_runner import run_structured_vlm_call_messages
from scenesmith.domain_randomization.variation_spec import VariationGoal
from scenesmith.prompts import prompt_registry
from scenesmith.prompts.registry import DRSubflowPrompts

console_logger = logging.getLogger(__name__)


@dataclass
class DRSceneContext:
    """Raw scene + RL artifacts loaded from a finished base scene directory."""

    scene_state: dict
    task_requirements: dict
    rl_artifacts: dict
    reward_spec_text: str
    curriculum_spec_text: str


class _VariationGoalSchema(BaseModel):
    variation_id: str
    description: str
    soft_constraints: dict = Field(default_factory=dict)
    task_critical_object_ids: list[str] = Field(default_factory=list)
    difficulty: float
    seed: int


class _VariationGoalsResponseSchema(BaseModel):
    goals: list[_VariationGoalSchema]


def load_dr_scene_context(base_scene_dir: Path) -> DRSceneContext:
    """Load the artifacts the Director inlines into its prompt."""
    base_scene_dir = base_scene_dir.resolve()
    final_paths = sorted(
        base_scene_dir.rglob("scene_states/final_scene/scene_state.json")
    )
    if not final_paths:
        raise FileNotFoundError(f"No final scene state found under {base_scene_dir}")

    scene_state = json.loads(final_paths[0].read_text())
    task_requirements = json.loads(
        (base_scene_dir / "task_requirements.json").read_text()
    )
    rl_artifacts = json.loads((base_scene_dir / "rl_artifacts.json").read_text())

    reward_path = base_scene_dir / "reward_spec.yaml"
    curriculum_path = base_scene_dir / "curriculum_spec.yaml"
    reward_spec_text = reward_path.read_text() if reward_path.exists() else ""
    curriculum_spec_text = (
        curriculum_path.read_text() if curriculum_path.exists() else ""
    )

    return DRSceneContext(
        scene_state=scene_state,
        task_requirements=task_requirements,
        rl_artifacts=rl_artifacts,
        reward_spec_text=reward_spec_text,
        curriculum_spec_text=curriculum_spec_text,
    )


def _expected_task_critical_ids(rl_artifacts: dict) -> list[str]:
    variants = rl_artifacts.get("domain_randomization", {}).get(
        "task_object_variants", []
    )
    return [v["object_id"] for v in variants]


class DRDirectorAgent:
    """Single-turn structured-output Director for DR v2."""

    def __init__(
        self,
        cfg: DictConfig,
        scene_context: DRSceneContext,
        *,
        prompt_registry=prompt_registry,
    ) -> None:
        self._cfg = cfg
        self._sc = scene_context
        self._pr = prompt_registry

        # Centralized resolver — falls back through cfg.openai.model →
        # cfg.provider.model → PROVIDER_MODEL env → sentinel (logs a
        # WARNING if it reaches the sentinel).
        from scenesmith.provider import resolve_model_name

        self._model = resolve_model_name(cfg)
        dr_openai = (
            OmegaConf.to_container(
                (cfg.get("domain_randomization") or {}).get("openai") or {},
                resolve=True,
            )
            or {}
        )
        self._reasoning_effort = dr_openai.get("reasoning_effort", "medium")
        self._verbosity = dr_openai.get("verbosity", "medium")

    def _render_prompt(self, num_variations: int, user_description: str) -> str:
        return self._pr.get_prompt(
            DRSubflowPrompts.DIRECTOR,
            scene_state=json.dumps(self._sc.scene_state, indent=2),
            task_requirements=json.dumps(self._sc.task_requirements, indent=2),
            rl_artifacts=json.dumps(self._sc.rl_artifacts, indent=2),
            reward_spec=self._sc.reward_spec_text,
            curriculum_spec=self._sc.curriculum_spec_text,
            user_description=user_description,
            num_variations=num_variations,
        )

    def _call_structured(self, rendered: str) -> _VariationGoalsResponseSchema:
        return run_structured_vlm_call_messages(
            model=self._model,
            messages=[{"role": "user", "content": rendered}],
            output_type=AgentOutputSchema(
                _VariationGoalsResponseSchema, strict_json_schema=False
            ),
            reasoning_effort=self._reasoning_effort,
            verbosity=self._verbosity,
            subsystem="dr_director",
            agent_name="dr_director",
        )

    def _parse_freeform_json(self, text: str) -> list[VariationGoal]:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "goals" in parsed:
            items = parsed["goals"]
        elif isinstance(parsed, list):
            items = parsed
        else:
            raise ValueError('Director output must be a JSON list or {"goals": [...]}')
        return [VariationGoal.from_dict(item) for item in items]

    async def propose_goals(
        self,
        num_variations: int,
        user_description: str,
        *,
        self_critique: bool = False,
    ) -> list[VariationGoal]:
        """Propose variation goals via one structured LLM call.

        Args:
            num_variations: How many goals to emit.
            user_description: Free-text guidance from the caller.
            self_critique: When True, runs one additional LLM call that
                reviews the initial goals on three criteria: specificity
                (each goal must name >=2 concrete manipuland ids),
                diversity (qualitatively different layouts), and
                difficulty spread (covers [0,1]); rewrites any goal
                that fails. Adds one LLM call per ``propose_goals`` if
                enabled. Off by default for backward compatibility.
        """
        rendered = self._render_prompt(num_variations, user_description)
        console_logger.info(
            "DR Director proposing %d goals (model=%s, self_critique=%s)",
            num_variations,
            self._model,
            self_critique,
        )

        try:
            response = self._call_structured(rendered)
            goals = [
                VariationGoal.from_dict(goal.model_dump()) for goal in response.goals
            ]
        except Exception as first_exc:
            console_logger.warning(
                "Structured Director call failed (%s); retrying with JSON-only nudge",
                first_exc,
            )
            retry_prompt = (
                rendered + "\n\nYour previous output was not valid JSON. "
                'Return ONLY a JSON object {"goals": [...]} with no markdown fences.'
            )
            try:
                response = self._call_structured(retry_prompt)
                goals = [
                    VariationGoal.from_dict(goal.model_dump())
                    for goal in response.goals
                ]
            except json.JSONDecodeError as decode_exc:
                raise decode_exc from first_exc
            except Exception as retry_exc:
                raise retry_exc from first_exc

        expected_ids = _expected_task_critical_ids(self._sc.rl_artifacts)
        for goal in goals:
            if goal.task_critical_object_ids != expected_ids:
                goal.task_critical_object_ids = list(expected_ids)

        if self_critique:
            goals = self._self_critique_goals(
                goals, num_variations=num_variations, user_description=user_description
            )
            for goal in goals:
                if goal.task_critical_object_ids != expected_ids:
                    goal.task_critical_object_ids = list(expected_ids)

        return goals

    def _self_critique_goals(
        self,
        initial_goals: list[VariationGoal],
        *,
        num_variations: int,
        user_description: str,
    ) -> list[VariationGoal]:
        """Run one structured-output review pass over the initial goals.

        The model sees the original Director context plus the JSON-dumped
        initial goals and returns a revised set. We trust the LLM to
        echo unchanged goals it considers acceptable.
        """
        original_prompt = self._render_prompt(num_variations, user_description)
        initial_json = json.dumps(
            [g.to_dict() for g in initial_goals], indent=2, default=str
        )
        critique_prompt = (
            f"{original_prompt}\n\n"
            "## Self-critique pass\n"
            "You just produced the following VariationGoal list:\n\n"
            f"{initial_json}\n\n"
            "Review your own output against three criteria:\n"
            "  1. SPECIFICITY — each `description` MUST name at least two\n"
            "     concrete manipuland ids from clutter_ids or\n"
            "     task_object_variants[*].object_id. Vague language like\n"
            "     'various clutter' or 'some items' is bland and must be\n"
            "     rewritten with concrete object references.\n"
            "  2. DIVERSITY — every pair of goals MUST describe a\n"
            "     qualitatively different arrangement. If two goals say\n"
            "     essentially the same thing in different words, rewrite\n"
            "     one with a meaningfully different intent.\n"
            "  3. DIFFICULTY SPREAD — the `difficulty` values across the\n"
            "     list must include at least one easy (<= 0.3), one hard\n"
            "     (>= 0.7), and one mid (0.4-0.6). Adjust difficulty\n"
            "     values if they cluster in one band.\n\n"
            "Emit exactly the same number of goals — same variation_ids\n"
            "where possible — but rewrite descriptions/difficulty for any\n"
            "goal that fails any criterion. Echo unchanged goals you\n"
            "consider acceptable. Return the same JSON shape\n"
            '({"goals": [...]}) with no markdown fences.'
        )

        try:
            response = self._call_structured(critique_prompt)
            revised = [
                VariationGoal.from_dict(goal.model_dump()) for goal in response.goals
            ]
        except Exception as exc:  # noqa: BLE001 — best-effort polish
            console_logger.warning(
                "DR Director self-critique pass failed (%s); keeping initial goals.",
                exc,
            )
            return initial_goals

        if len(revised) != len(initial_goals):
            console_logger.warning(
                "DR Director self-critique returned %d goals (expected %d); "
                "keeping initial set instead of partial.",
                len(revised),
                len(initial_goals),
            )
            return initial_goals

        changes = sum(
            1
            for old, new in zip(initial_goals, revised)
            if old.description != new.description or old.difficulty != new.difficulty
        )
        console_logger.info(
            "DR Director self-critique applied %d/%d revisions.",
            changes,
            len(revised),
        )
        return revised
