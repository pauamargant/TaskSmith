"""DR v2 Subflow Critic — single-turn structured-output critique.

Standalone path: used by ``scripts/test_dr_critic.py`` and any non-agentic
caller that wants a one-shot critique. ``DRSubflowAgent`` uses its own
Agents-SDK Critic agent (see ``stateful_subflow.py``) but shares this
prompt + schema shape.
"""

from __future__ import annotations

import json
import logging

from dataclasses import dataclass
from typing import Any

from agents.agent_output import AgentOutputSchema
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field

from scenesmith.agent_utils.scoring import CategoryScore
from scenesmith.agent_utils.sdk_vlm_runner import run_structured_vlm_call_messages
from scenesmith.domain_randomization.subflow.critique import DRVariationCritique
from scenesmith.domain_randomization.variation_spec import VariationGoal
from scenesmith.prompts import prompt_registry
from scenesmith.prompts.registry import DRSubflowPrompts

console_logger = logging.getLogger(__name__)


def summarize_scene_for_critic(scene: dict) -> str:
    """Compact scene representation for prompt inlining.

    Keeps only the fields the Critic needs to judge invariance / position
    drift: id, object_type, asset_id, transform translation, and 2D placement
    position (if recorded by manipuland placement).
    """
    objects = scene.get("objects", {})
    if not isinstance(objects, dict):
        raise ValueError("scene.objects must be a dict keyed by id for summarization")

    rows: list[dict[str, Any]] = []
    for oid in sorted(objects):
        obj = objects[oid]
        transform = obj.get("transform") or {}
        placement = obj.get("placement_info") or {}
        rows.append(
            {
                "id": oid,
                "type": obj.get("object_type"),
                "asset": obj.get("asset_id"),
                "translation": transform.get("translation"),
                "position_2d": placement.get("position_2d"),
            }
        )
    return json.dumps(rows, indent=2)


@dataclass
class CriticContext:
    """Inputs the Critic needs to score one variation."""

    goal: VariationGoal
    base_scene: dict
    current_scene: dict
    sibling_descriptions: list[str]


class _CategoryScoreSchema(BaseModel):
    name: str
    grade: int = Field(ge=0, le=10)
    comment: str


class _DRVariationCritiqueSchema(BaseModel):
    """Pydantic mirror of ``DRVariationCritique`` for structured output."""

    critique: str
    goal_alignment: _CategoryScoreSchema
    feasibility: _CategoryScoreSchema
    task_validity: _CategoryScoreSchema
    diversity_vs_siblings: _CategoryScoreSchema
    manipuland_invariance: _CategoryScoreSchema


def _schema_to_dataclass(parsed: _DRVariationCritiqueSchema) -> DRVariationCritique:
    def _cat(field: _CategoryScoreSchema) -> CategoryScore:
        return CategoryScore(name=field.name, grade=field.grade, comment=field.comment)

    return DRVariationCritique(
        critique=parsed.critique,
        goal_alignment=_cat(parsed.goal_alignment),
        feasibility=_cat(parsed.feasibility),
        task_validity=_cat(parsed.task_validity),
        diversity_vs_siblings=_cat(parsed.diversity_vs_siblings),
        manipuland_invariance=_cat(parsed.manipuland_invariance),
    )


class DRSubflowCriticStandalone:
    """Synchronous one-shot structured-output Critic for offline/test use.

    Not used in the live planner/designer/critic loop (the in-loop Critic is an
    Agents-SDK ``Agent`` in ``stateful_subflow.py``). Useful for
    ``scripts/test_dr_critic.py`` and any non-agentic caller that wants a single
    critique without running the full subflow. Makes a single synchronous LLM
    call with retry on JSON failure.
    """

    def __init__(
        self,
        cfg: DictConfig,
        *,
        prompt_registry_=prompt_registry,
    ) -> None:
        self._cfg = cfg
        self._pr = prompt_registry_
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

    def _render(self, ctx: CriticContext) -> str:
        return self._pr.get_prompt(
            DRSubflowPrompts.CRITIC,
            goal=json.dumps(ctx.goal.to_dict(), indent=2),
            base_scene_summary=summarize_scene_for_critic(ctx.base_scene),
            current_scene_summary=summarize_scene_for_critic(ctx.current_scene),
            sibling_descriptions=json.dumps(ctx.sibling_descriptions, indent=2),
        )

    def _call_structured(self, rendered: str) -> _DRVariationCritiqueSchema:
        return run_structured_vlm_call_messages(
            model=self._model,
            messages=[{"role": "user", "content": rendered}],
            output_type=AgentOutputSchema(
                _DRVariationCritiqueSchema, strict_json_schema=False
            ),
            reasoning_effort=self._reasoning_effort,
            verbosity=self._verbosity,
            subsystem="dr_subflow_critic",
            agent_name="dr_subflow_critic",
        )

    def critique(self, ctx: CriticContext) -> DRVariationCritique:
        """Score one mutated scene against its variation goal."""
        rendered = self._render(ctx)
        console_logger.info(
            "DR Subflow Critic scoring variation_id=%s (model=%s)",
            ctx.goal.variation_id,
            self._model,
        )
        try:
            parsed = self._call_structured(rendered)
        except Exception as first_exc:
            console_logger.warning(
                "Structured Critic call failed (%s); retrying with JSON-only nudge",
                first_exc,
            )
            retry = (
                rendered
                + "\n\nYour previous output was not valid JSON. Return ONLY a JSON "
                "object matching the DRVariationCritique shape with no markdown fences."
            )
            parsed = self._call_structured(retry)
        return _schema_to_dataclass(parsed)
