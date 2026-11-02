"""Domain Randomization Orchestrator.

A multi-phase agentic orchestrator that:
  1. Analyzes the base scene and RL/task artifacts
  2. Plans a set of N varied scenes (mix of programmatic and agent re-run strategies)
  3. Critiques the plan for diversity and task validity
  4. Executes all variations in parallel
  5. Updates reward specs and packages exports per variation
"""

from __future__ import annotations

import asyncio
import json
import logging

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from agents import Agent, Runner, SQLiteSession, function_tool
from omegaconf import DictConfig, OmegaConf

from scenesmith.agent_utils.house import HouseLayout, RoomGeometry
from scenesmith.agent_utils.room import RoomScene
from scenesmith.domain_randomization.director import (
    DRDirectorAgent,
    load_dr_scene_context,
)
from scenesmith.domain_randomization.dr_scene_loader import load_room_for_dr
from scenesmith.domain_randomization.export import ExportConfig, export_variation
from scenesmith.domain_randomization.llm_editor import (
    apply_scene_edits,
    generate_scene_edits,
)
from scenesmith.domain_randomization.programmatic.asset_substitution import (
    AssetSubstitution,
)
from scenesmith.domain_randomization.programmatic.object_dropout import ObjectDropout
from scenesmith.domain_randomization.programmatic.pose_perturbation import (
    PosePerturbation,
)
from scenesmith.domain_randomization.programmatic.scale_jitter import ScaleJitter
from scenesmith.domain_randomization.reward_spec_updater import (
    classify_reward_spec_impact,
    update_reward_spec,
)
from scenesmith.domain_randomization.rl_artifacts_updater import regenerate_rl_artifacts
from scenesmith.domain_randomization.scaffolding import (
    get_room_scene_dirs,
    scaffold_variation_v2,
)
from scenesmith.domain_randomization.subflow.dr_subflow_cfg import build_dr_subflow_cfg
from scenesmith.domain_randomization.subflow.stateful_subflow import (
    DRSubflowAgent,
    DRSubflowRejected,
)
from scenesmith.domain_randomization.variation_spec import (
    OrchestrationPlan,
    ProgrammaticOpSpec,
    VariationGoal,
    VariationSpec,
    VariationStatus,
)
from scenesmith.llm_retry import aretry
from scenesmith.provider import make_chat_completions_model
from scenesmith.utils.logging import BaseLogger, ConsoleLogger

console_logger = logging.getLogger(__name__)

_OP_REGISTRY = {
    "pose_perturbation": PosePerturbation(),
    "scale_jitter": ScaleJitter(),
    "object_dropout": ObjectDropout(),
    "asset_substitution": AssetSubstitution(),
}


def _extract_object_ids(entries: object) -> list[str]:
    """Normalize a list of object-id entries into plain IDs.

    Supports either:
    - ["id_a", "id_b"]
    - [{"object_id": "id_a"}, {"object_id": "id_b"}]
    """
    if not isinstance(entries, list):
        return []

    object_ids: list[str] = []
    for entry in entries:
        if isinstance(entry, str):
            if entry:
                object_ids.append(entry)
        elif isinstance(entry, dict):
            object_id = entry.get("object_id")
            if isinstance(object_id, str) and object_id:
                object_ids.append(object_id)
    return object_ids


@dataclass
class SceneContext:
    """Analyzed context from the base scene and RL artifacts."""

    scene_summary: str
    task_type: str = "unknown"
    task_critical_object_ids: list[str] = field(default_factory=list)
    clutter_ids: list[str] = field(default_factory=list)
    furniture_movable_ids: list[str] = field(default_factory=list)
    position_bounds: dict[str, list[float]] = field(default_factory=dict)
    curriculum_axis: str = ""
    reward_invariants: str = ""
    has_rl_artifacts: bool = False


class DomainRandomizationOrchestrator:
    """Orchestrates domain randomization for a completed base scene.

    Phases:
      1. Analyze — load base scene + RL artifacts, build SceneContext
      2. Plan — LLM planning agent proposes N VariationSpecs
      3. Critique — LLM critic reviews plan for diversity + task validity
      4. Execute — run all variations in parallel (programmatic or agent re-run)
      5. Reward/export — update reward specs and package exports
    """

    def __init__(
        self,
        base_experiment_dir: Path,
        num_variations: int,
        cfg: DictConfig,
        logger: BaseLogger,
        description: str | None = None,
        export_cfg: ExportConfig | None = None,
    ) -> None:
        self.base_dir = Path(base_experiment_dir)
        self.num_variations = num_variations
        self.cfg = cfg
        self.logger = logger
        self.description = description
        self.export_cfg = export_cfg or ExportConfig()

        # Locate base scene directory (scene_000, scene_001, etc.)
        self._base_scene_dir = self._find_scene_dir(self.base_dir)

        # Output directory for this randomization run.
        timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")
        output_root = Path(
            cfg.get("domain_randomization", {}).get("output_dir")
            or self.base_dir / "randomizations" / timestamp
        )
        output_root.mkdir(parents=True, exist_ok=True)
        self._output_dir = output_root

        # Pending plan (populated by planning agent tools).
        self._pending_variations: list[VariationSpec] = []
        self._context: SceneContext | None = None

        # Centralized resolver — falls back through cfg.openai.model →
        # cfg.provider.model → PROVIDER_MODEL env → sentinel (logs a
        # WARNING if it reaches the sentinel).
        from scenesmith.provider import resolve_model_name

        self._model = make_chat_completions_model(resolve_model_name(cfg))

    async def run(
        self, existing_plan: OrchestrationPlan | None = None
    ) -> OrchestrationPlan:
        """Run the full orchestration pipeline and return the completed plan."""
        mode = str(
            self.cfg.get("domain_randomization", {}).get("mode", "deterministic")
        )
        console_logger.info(
            f"Starting domain randomization: {self.num_variations} variations "
            f"from {self._base_scene_dir.name} (mode={mode})"
        )

        if mode == "agentic":
            if existing_plan is not None:
                raise NotImplementedError(
                    "mode='agentic' does not yet support resume from an existing plan"
                )
            return await self._run_v2()

        # Phase 1: Analyze
        context = self._analyze()
        self._context = context

        if existing_plan is None:
            # Phase 2: Plan (multi-turn LLM with tools)
            reasoning = await self._plan(context)

            # Phase 3: Critique (single-turn structured output, max 2 revision rounds)
            variations = await self._critique_and_finalize(reasoning)

            # Build the OrchestrationPlan.
            plan = OrchestrationPlan(
                created_at=datetime.utcnow().isoformat(),
                base_experiment_dir=str(self.base_dir),
                description=self.description,
                num_variations=self.num_variations,
                orchestrator_reasoning=reasoning,
                variations={
                    v.variation_id: VariationStatus(spec=v) for v in variations
                },
            )
            plan.save(self._output_dir / "orchestration_plan.json")
            console_logger.info(
                f"Plan saved: {self._output_dir / 'orchestration_plan.json'}"
            )
        else:
            plan = self._prepare_plan_for_resume(existing_plan)
            plan.save(self._output_dir / "orchestration_plan.json")
            console_logger.info(
                f"Resuming existing plan: {self._output_dir / 'orchestration_plan.json'}"
            )

        # Phase 4: Execute
        await self._execute(plan)

        console_logger.info("Domain randomization complete.")
        return plan

    # ------------------------------------------------------------------ #
    # Phase 1 — Analyze                                                    #
    # ------------------------------------------------------------------ #

    def _analyze(self) -> SceneContext:
        """Load base scene + RL artifacts and build SceneContext."""
        scene_dir = self._base_scene_dir

        # Scene summary from final scene state.
        object_summary = self._summarize_base_scene(scene_dir)

        context = SceneContext(scene_summary=object_summary)

        # Task requirements.
        task_req_path = scene_dir / "task_requirements.json"
        if task_req_path.exists():
            from scenesmith.task_analysis_agent.task_requirements import (
                TaskRequirements,
            )

            with open(task_req_path) as f:
                tr_data = json.load(f)
            try:
                tr = TaskRequirements.from_dict(tr_data)
                context.task_type = tr.task_type
                context.task_critical_object_ids = [
                    obj.name for obj in tr.required_objects if obj.is_task_object
                ]
            except Exception as e:
                console_logger.warning(f"Could not parse task_requirements: {e}")

        # RL artifacts.
        rl_path = scene_dir / "rl_artifacts.json"
        if rl_path.exists():
            with open(rl_path) as f:
                rl = json.load(f)
            dr = rl.get("domain_randomization", {})
            context.clutter_ids = _extract_object_ids(dr.get("clutter_ids", []))
            context.furniture_movable_ids = _extract_object_ids(
                dr.get("furniture_movable", [])
            )
            for variant in dr.get("task_object_variants", []):
                oid = variant.get("object_id", "")
                bounds = variant.get("position_range_xy_on_surface", [])
                if oid and bounds:
                    context.position_bounds[oid] = bounds
            context.has_rl_artifacts = True

        # Curriculum.
        curriculum_path = scene_dir / "curriculum_spec.yaml"
        if curriculum_path.exists():
            import yaml

            with open(curriculum_path) as f:
                curriculum_data = yaml.safe_load(f)
            context.curriculum_axis = json.dumps(curriculum_data, indent=2)[:500]

        # Reward invariants summary.
        reward_path = scene_dir / "reward_spec.yaml"
        if reward_path.exists():
            import yaml

            with open(reward_path) as f:
                reward_data = yaml.safe_load(f)
            terms = reward_data.get("reward_terms", [])
            refs = [
                f"{t.get('from_object_id')} → {t.get('to_object_id')}"
                for t in terms
                if t.get("from_object_id") or t.get("to_object_id")
            ]
            context.reward_invariants = "; ".join(refs[:5])

        console_logger.info(
            f"Scene context: {object_summary[:100]}... "
            f"task_critical={context.task_critical_object_ids} "
            f"clutter_ids={context.clutter_ids[:3]}"
        )
        return context

    def _summarize_base_scene(self, scene_dir: Path) -> str:
        """Build a concise text summary of the base scene object inventory."""
        final_state_paths = list(
            scene_dir.rglob("scene_states/final_scene/scene_state.json")
        )
        if not final_state_paths:
            final_state_paths = list(scene_dir.rglob("scene_state.json"))
        if not final_state_paths:
            return "Base scene state not found."

        # Domain randomization currently summarizes the first room only.
        with open(final_state_paths[0]) as f:
            state = json.load(f)

        counts: dict[str, int] = {}
        for obj in state.get("objects", {}).values():
            otype = obj.get("object_type", "unknown")
            counts[otype] = counts.get(otype, 0) + 1

        summary_parts = [f"{count} {otype}" for otype, count in sorted(counts.items())]
        desc = state.get("text_description", "")
        return f"{desc} | Objects: {', '.join(summary_parts)}"

    # ------------------------------------------------------------------ #
    # Phase 2 — Plan                                                       #
    # ------------------------------------------------------------------ #

    def _make_planner_model_settings(self):
        """Build ModelSettings for the v1 planner agent.

        Applies provider-level sampling params (temperature/top_k etc. from
        the active preset) and a max_tokens cap, without an explicit reasoning
        effort — thinking mode defaults to whatever the vLLM server applies
        when enable_thinking is absent (off on Qwen3/Gemma4 by default).
        """
        from agents import ModelSettings

        from scenesmith.provider_validation import (
            merge_chat_template_kwargs_into_extra_body,
            split_provider_sampling,
        )

        kwargs: dict = {"max_tokens": 16384}
        native_sampling, extra_sampling = split_provider_sampling()
        for key, value in native_sampling.items():
            kwargs.setdefault(key, value)
        if extra_sampling:
            extra_body = dict(kwargs.get("extra_body") or {})
            for key, value in extra_sampling.items():
                extra_body.setdefault(key, value)
            kwargs["extra_body"] = extra_body
        extra_body = dict(kwargs.get("extra_body") or {})
        merge_chat_template_kwargs_into_extra_body(extra_body)
        if extra_body:
            kwargs["extra_body"] = extra_body
        return ModelSettings(**kwargs)

    async def _plan(self, context: SceneContext) -> str:
        """Run the multi-turn planning agent and return its reasoning summary."""
        self._pending_variations = []

        system_prompt = self._build_planner_system_prompt(context)
        session = SQLiteSession(str(self._output_dir / "orchestrator_planner.db"))

        planner_tools = self._create_planner_tools()
        planner = Agent(
            name="domain_randomization_planner",
            model=self._model,
            tools=planner_tools,
            instructions=system_prompt,
            model_settings=self._make_planner_model_settings(),
        )

        user_message = (
            f"Plan {self.num_variations} scene variations."
            + (f" User guidance: {self.description}" if self.description else "")
            + "\n\nCall propose_variation for each variation, then call finalize_plan."
        )

        result = await aretry(
            lambda: Runner.run(
                starting_agent=planner,
                input=user_message,
                session=session,
                # Generous turn budget so Qwen3 with enable_thinking=true
                # doesn't hit MaxTurnsExceeded mid-planning or during
                # critique-feedback revisions. Each propose_variation
                # tool call takes 5-10 model turns of chain-of-thought +
                # the tool response on Qwen; the original N*3+10 and the
                # first-bump N*6+20 both crashed on N=3 (plan phase OK
                # at 38 turns, revision phase blew the same budget).
                # Coefficient 10 + buffer of 30 → 60 turns for N=3,
                # enough headroom for revision + tool error retries.
                max_turns=self.num_variations * 10 + 30,
            ),
            op_name="dr_plan",
            logger=console_logger,
        )

        return result.final_output or "No reasoning provided."

    def _build_planner_system_prompt(self, ctx: SceneContext) -> str:
        available_ops = (
            "pose_perturbation, scale_jitter, object_dropout, asset_substitution"
        )
        return f"""You are a domain randomization planner for an RL training scene.

## Base Scene
{ctx.scene_summary}

## Task Context
- Task type: {ctx.task_type}
- Task-critical objects (DO NOT modify): {ctx.task_critical_object_ids}
- Safe objects for dropout (clutter): {ctx.clutter_ids}
- Reward invariants (must be preserved): {ctx.reward_invariants}

## Curriculum Axis
{ctx.curriculum_axis or "No curriculum available."}

## Available Strategies

**programmatic** (fast, deterministic):
  ops: {available_ops}
  Pass ops_json with the list of operation objects.

**llm_directed** (one LLM call, semantics-aware):
  The LLM interprets `prompt_injection` as a natural language goal and produces
  targeted move / remove / scale edits. Use this for goals that require spatial
  reasoning (e.g. "clear a path", "arrange objects for breakfast", "spread items
  further apart"). No ops_json needed — set prompt_injection to the goal.

## Your Goal
Propose {self.num_variations} varied scene specifications.
- Choose between programmatic and llm_directed per variation
- Vary difficulty if curriculum is available (0=easy, 1=hard)
- NEVER modify task-critical objects: {ctx.task_critical_object_ids}
- Use clutter_ids as preferred targets for object_dropout: {ctx.clutter_ids}
- Ensure variations are actually distinct from each other

Call propose_variation() for each variation, then finalize_plan().
For programmatic ops, pass them via `ops_json` as a JSON-encoded list.
For llm_directed, set prompt_injection to the natural language variation goal.
"""

    def _create_planner_tools(self) -> list:
        ctx = self._context

        @function_tool
        def propose_variation(
            variation_id: str,
            description: str,
            strategy: str,
            ops_json: str | None = None,
            prompt_injection: str | None = None,
            difficulty: float | None = None,
        ) -> str:
            """Propose a single scene variation. Call once per variation."""
            try:
                op_specs = None
                if ops_json:
                    ops = json.loads(ops_json)
                    if not isinstance(ops, list):
                        raise ValueError(
                            "ops_json must decode to a list of op objects."
                        )
                    op_specs = [
                        ProgrammaticOpSpec(
                            op_type=op["op_type"],
                            target_object_types=op.get("target_object_types", []),
                            params=op.get("params", {}),
                        )
                        for op in ops
                    ]
                spec = VariationSpec(
                    variation_id=variation_id,
                    description=description,
                    strategy=strategy,
                    ops=op_specs,
                    prompt_injection=prompt_injection,
                    task_critical_object_ids=(
                        ctx.task_critical_object_ids if ctx else []
                    ),
                    difficulty=difficulty,
                )
                self._pending_variations.append(spec)
                return f"Proposed: {variation_id} ({strategy})"
            except Exception as e:
                return f"Error proposing {variation_id}: {e}"

        @function_tool
        def finalize_plan() -> str:
            """Signal that all variations have been proposed. Call once at the end."""
            return f"Plan finalized with {len(self._pending_variations)} variations."

        @function_tool
        def get_task_critical_objects() -> str:
            """Return the list of task-critical object IDs that must not be modified."""
            return json.dumps(ctx.task_critical_object_ids if ctx else [])

        @function_tool
        def get_clutter_objects() -> str:
            """Return safe dropout target object IDs (clutter from rl_artifacts)."""
            return json.dumps(ctx.clutter_ids if ctx else [])

        return [
            propose_variation,
            finalize_plan,
            get_task_critical_objects,
            get_clutter_objects,
        ]

    # ------------------------------------------------------------------ #
    # Phase 3 — Critique                                                   #
    # ------------------------------------------------------------------ #

    async def _critique_and_finalize(
        self, planner_reasoning: str
    ) -> list[VariationSpec]:
        """Review the proposed plan and optionally request revisions (max 2 rounds)."""
        max_rounds = 2
        variations = list(self._pending_variations)

        for round_idx in range(max_rounds):
            critique = await self._run_critic(variations)
            console_logger.info(
                f"Critique round {round_idx + 1}: "
                f"diversity={critique.get('diversity_score', 0):.2f} "
                f"validity={critique.get('task_validity_score', 0):.2f} "
                f"approved={critique.get('approved', False)}"
            )

            if critique.get("approved", True):
                break

            issues = critique.get("issues", [])
            if issues:
                console_logger.info(f"Critique issues: {issues}")
                # Request one revision round from the planner.
                self._pending_variations = list(variations)
                revision_prompt = (
                    f"The plan has issues: {'; '.join(issues[:3])}. "
                    f"Please revise the plan."
                )
                await self._plan_revision(variations, revision_prompt)
                variations = list(self._pending_variations)

        # Defensive truncation: even when the prompt is unambiguous, an
        # over-eager planner can still propose > N variations. Keep the
        # first N (insertion order); the plan dict downstream is keyed
        # by id and would silently drop overflows anyway on key
        # collision, which is worse than failing loud here.
        if len(variations) > self.num_variations:
            console_logger.warning(
                "Planner produced %d > num_variations=%d variations after "
                "revision; truncating to the first %d.",
                len(variations),
                self.num_variations,
                self.num_variations,
            )
            variations = variations[: self.num_variations]
        return variations

    async def _run_critic(self, variations: list[VariationSpec]) -> dict:
        """Run the critic LLM and return a critique dict."""
        if self._context is None:
            return {
                "approved": True,
                "diversity_score": 1.0,
                "task_validity_score": 1.0,
                "coverage_score": 1.0,
                "issues": [],
            }

        plan_summary = json.dumps(
            [
                {
                    "id": v.variation_id,
                    "description": v.description,
                    "strategy": v.strategy,
                    "difficulty": v.difficulty,
                }
                for v in variations
            ],
            indent=2,
        )

        critic_prompt = f"""Review this domain randomization plan for a scene.

Task-critical objects (must not be modified): {self._context.task_critical_object_ids}
Reward invariants: {self._context.reward_invariants}
User description: {self.description or "none"}

Plan ({len(variations)} variations):
{plan_summary}

Evaluate:
1. Are variations distinct from each other? (diversity_score 0-1)
2. Do all respect task constraints? (task_validity_score 0-1)
3. Does the plan cover the user description? (coverage_score 0-1)
4. List any critical issues.
5. Should the plan be approved (True/False)?

Respond with a JSON object with keys:
diversity_score, task_validity_score, coverage_score, issues (list), approved (bool)"""

        model = self._model
        critic = Agent(
            name="domain_randomization_critic",
            model=model,
            instructions="You are a domain randomization plan critic.",
        )
        result = await aretry(
            lambda: Runner.run(
                starting_agent=critic,
                input=critic_prompt,
                max_turns=2,
            ),
            op_name="dr_plan_critique",
            logger=console_logger,
        )
        output = result.final_output or "{}"
        try:
            # Strip markdown code blocks if present.
            clean = output.strip()
            if clean.startswith("```"):
                clean = "\n".join(clean.split("\n")[1:-1])
            return json.loads(clean)
        except json.JSONDecodeError:
            console_logger.warning(
                "DR critic response could not be parsed as JSON — rejecting plan. "
                "Raw output: %s",
                output[:500],
            )
            return {
                "approved": False,
                "diversity_score": 0.0,
                "task_validity_score": 0.0,
                "coverage_score": 0.0,
                "issues": ["Critic response was not valid JSON; plan rejected."],
            }

    async def _plan_revision(
        self, current_variations: list[VariationSpec], feedback: str
    ) -> None:
        """Ask the planner to add/replace variations based on critique feedback."""
        self._pending_variations = []

        system_prompt = self._build_planner_system_prompt(self._context)
        existing_ids = {v.variation_id for v in current_variations}
        planner_tools = self._create_planner_tools()
        planner = Agent(
            name="domain_randomization_planner_revision",
            model=self._model,
            tools=planner_tools,
            instructions=system_prompt,
        )
        # IMPORTANT: the revision is a REPLACEMENT, not an extension. Past
        # bug: the prompt said "can reuse good ones" and the LLM responded
        # with 3 fresh-ID variants alongside the prior 3, so the final
        # plan.variations dict ended up with 6 entries (each ID is a
        # distinct dict key — no implicit dedup). Make the contract
        # explicit + truncate defensively in _critique_and_finalize.
        prior_ids = ", ".join(v.variation_id for v in current_variations) or "—"
        revision_message = (
            f"Critique feedback: {feedback}\n\n"
            f"You proposed {len(current_variations)} variations previously: "
            f"{prior_ids}. These are REPLACED by your next proposals — "
            f"do NOT re-call propose_variation for any of them, do NOT "
            f"create '<old_id>_v2'-style siblings, and do NOT append. "
            f"Call propose_variation EXACTLY {self.num_variations} times "
            f"with fresh, distinct ids that address the critique, then "
            f"call finalize_plan."
        )
        await aretry(
            lambda: Runner.run(
                starting_agent=planner,
                input=revision_message,
                # Same generous turn budget as the initial plan call — see
                # the comment in _plan() for the rationale.
                max_turns=self.num_variations * 10 + 30,
            ),
            op_name="dr_plan_revision",
            logger=console_logger,
        )

    # ------------------------------------------------------------------ #
    # Phase 4 — Execute                                                    #
    # ------------------------------------------------------------------ #

    async def _execute(self, plan: OrchestrationPlan) -> None:
        """Run all non-completed variations in parallel."""
        pending = [
            vs for vs in plan.variations.values() if vs.status in ("pending", "failed")
        ]
        console_logger.info(f"Executing {len(pending)} variations in parallel...")

        await asyncio.gather(
            *[self._run_single_variation(vs, plan) for vs in pending],
            return_exceptions=True,
        )

    async def _run_single_variation(
        self, var_status: VariationStatus, plan: OrchestrationPlan
    ) -> None:
        """Execute one variation: scaffold → modify → save → reward spec → export."""
        spec = var_status.spec
        vid = spec.variation_id
        plan.update_status(vid, "in_progress")
        plan.save(self._output_dir / "orchestration_plan.json")

        try:
            variation_dir = self._output_dir / vid
            variation_scene_dir = variation_dir / self._base_scene_dir.name

            # Flat-copy the finished scene tree (same path as v2). The old
            # resume_stage="manipuland" route loaded scene_after_ceiling_objects,
            # which has zero manipulands — v1 ops were silently mutating an
            # empty-of-manipulands scene. Loading final_scene gives us the real
            # post-pipeline state with all manipulands placed.
            scaffold_variation_v2(
                base_scene_dir=self._base_scene_dir,
                variation_dir=variation_scene_dir,
                overwrite=True,
            )

            # 2. Load scene from the final_scene checkpoint (post-manipuland).
            room_dirs = get_room_scene_dirs(variation_scene_dir)
            if not room_dirs:
                raise FileNotFoundError(
                    f"No room_* directories found in {variation_scene_dir}"
                )
            state_path = (
                room_dirs[0] / "scene_states" / "final_scene" / "scene_state.json"
            )
            with open(state_path) as f:
                state_dict = json.load(f)
            # Week 1 D3 intentionally operates on the first room only.
            room_dir = room_dirs[0]
            room_geometry = RoomGeometry.from_dict(
                state_dict["room_geometry"], scene_dir=room_dir
            )
            scene = RoomScene(
                room_geometry=room_geometry,
                scene_dir=room_dir,
                room_id=room_dir.name,
                text_description=state_dict.get("text_description", ""),
                action_log_path=room_dir / "action_log.json",
            )
            scene.restore_from_state_dict(state_dict)

            # 3. Execute strategy. v1 supports two strategies now:
            #   - programmatic: deterministic op pipeline
            #   - llm_directed: single structured-output LLM edit
            # agent_rerun and hybrid were removed; DR v2 mode='agentic' (DRSubflowAgent) supersedes them.
            locked_ids = set(spec.task_critical_object_ids)

            if spec.strategy == "programmatic":
                scene = self._apply_programmatic_ops(scene, spec, locked_ids)
            elif spec.strategy == "llm_directed":
                goal = spec.prompt_injection or spec.description
                edit_plan = await generate_scene_edits(
                    scene=scene,
                    description=goal,
                    locked_ids=locked_ids,
                    cfg=OmegaConf.to_container(self.cfg, resolve=True),
                )
                console_logger.info(
                    f"Variation {vid} LLM edit reasoning: {edit_plan.reasoning}"
                )
                scene, skipped = apply_scene_edits(scene, edit_plan, locked_ids)
                if skipped:
                    console_logger.warning(f"Variation {vid} skipped edits: {skipped}")
            else:
                raise ValueError(
                    f"Variation {vid}: strategy {spec.strategy!r} is no longer "
                    "supported. Use 'programmatic', 'llm_directed', or switch to "
                    "domain_randomization.mode='agentic' for the v2 subflow path."
                )

            # 4. Save scene state
            final_state_dir = room_dir / "scene_states" / "final_scene"
            final_state_dir.mkdir(parents=True, exist_ok=True)
            state_path = final_state_dir / "scene_state.json"
            with open(state_path, "w") as f:
                json.dump(scene.to_state_dict(), f, indent=2)

            # 5. Reward spec update
            ctx = self._context
            impact = classify_reward_spec_impact(
                spec=spec,
                task_critical_ids=locked_ids,
            )
            rl_artifacts = self._load_rl_artifacts()
            task_requirements = self._load_task_requirements()
            await update_reward_spec(
                impact=impact,
                variation_scene_dir=variation_scene_dir,
                base_scene_dir=self._base_scene_dir,
                new_scene=scene,
                task_critical_ids=locked_ids,
                cfg=OmegaConf.to_container(self.cfg, resolve=True),
                rl_artifacts=rl_artifacts,
                task_requirements=task_requirements,
            )

            # 6. Export
            house_layout = self._load_house_layout()
            export_variation(
                variation_scene_dir=variation_scene_dir,
                rooms=[scene],
                house_layout=house_layout,
                cfg=self.cfg,
                export_cfg=self.export_cfg,
            )

            plan.update_status(vid, "completed")
            console_logger.info(f"Variation {vid} completed.")

        except Exception as e:
            console_logger.error(f"Variation {vid} failed: {e}", exc_info=True)
            plan.update_status(vid, "failed", error=str(e))

        plan.save(self._output_dir / "orchestration_plan.json")

    # ------------------------------------------------------------------ #
    # DR v2 — agentic mode (Director -> single subflow)                   #
    # ------------------------------------------------------------------ #

    async def _run_v2(self) -> OrchestrationPlan:
        """Agentic-mode entry point. Director proposes N VariationGoals;
        each is satisfied by a DRSubflowAgent. Subflows run concurrently
        via ``asyncio.gather`` with a single asyncio.Lock guarding all
        ``plan.update_status`` + ``plan.save`` sequences (the
        ``OrchestrationPlan.variations`` dict is the shared mutable state
        and the JSON file is written atomically via ``Path.replace``,
        but the read-modify-write pair must serialize)."""
        context = self._analyze()
        self._context = context

        scene_ctx = load_dr_scene_context(self._base_scene_dir)
        director = DRDirectorAgent(self.cfg, scene_ctx)
        user_description = (
            self.description or "vary clutter arrangement on the table surface"
        )
        console_logger.info(
            "DR v2 Director proposing %d goals (description=%r)",
            self.num_variations,
            user_description,
        )
        dr_cfg = self.cfg.get("domain_randomization", {}) or {}
        self_critique = bool((dr_cfg.get("director") or {}).get("self_critique", False))
        goals: list[VariationGoal] = await director.propose_goals(
            num_variations=self.num_variations,
            user_description=user_description,
            self_critique=self_critique,
        )

        # Stash goals by id so _run_subflow_v2 can resolve them from the
        # OrchestrationPlan (which stores VariationSpec placeholders).
        self._v2_goals: dict[str, VariationGoal] = {g.variation_id: g for g in goals}

        # Build placeholder VariationSpecs (strategy="llm_directed" — the
        # closest existing slot) so OrchestrationPlan load/save still works.
        variations = {
            g.variation_id: VariationStatus(
                spec=VariationSpec(
                    variation_id=g.variation_id,
                    description=g.description,
                    strategy="llm_directed",
                    prompt_injection=g.description,
                    task_critical_object_ids=list(g.task_critical_object_ids),
                    difficulty=g.difficulty,
                )
            )
            for g in goals
        }
        plan = OrchestrationPlan(
            created_at=datetime.utcnow().isoformat(),
            base_experiment_dir=str(self.base_dir),
            description=self.description,
            num_variations=self.num_variations,
            orchestrator_reasoning=(
                "DR v2 agentic mode: Director proposed VariationGoals; each "
                "is satisfied by a DRSubflowAgent."
            ),
            variations=variations,
        )
        plan.save(self._output_dir / "orchestration_plan.json")

        # Pre-compose the DR subflow cfg once — all variations share it.
        self._v2_subflow_cfg = build_dr_subflow_cfg(self.cfg)

        # Single lock serializes plan mutations + saves across concurrent
        # subflows. Allocated here (not in __init__) so each `run()` call
        # gets a fresh lock bound to the active event loop.
        self._v2_plan_lock = asyncio.Lock()

        if len(goals) == 1:
            # Skip gather wrapping for the N=1 fast path — keeps the
            # asyncio cancellation semantics identical to the pre-Step-8
            # serial loop and makes single-goal stack traces clearer.
            await self._run_subflow_v2(plan, goals[0].variation_id)
        else:
            console_logger.info(
                "DR v2 running %d subflows concurrently via asyncio.gather",
                len(goals),
            )
            results = await asyncio.gather(
                *(self._run_subflow_v2(plan, g.variation_id) for g in goals),
                return_exceptions=True,
            )
            # Surface gather-level errors that escaped _run_subflow_v2's own
            # except branch — should be rare, but log loudly if they happen.
            for goal, result in zip(goals, results):
                if isinstance(result, BaseException):
                    console_logger.error(
                        "DR v2 subflow %s raised past its except branch: %r",
                        goal.variation_id,
                        result,
                    )

            # Optional wave 2: redo subflows whose wave-1 accepted scenes
            # are too similar to a sibling (text descriptions can rhyme
            # while the actual scene states cluster). Off by default.
            diversity_cfg = dr_cfg.get("diversity") or {}
            if diversity_cfg.get("two_wave") and len(goals) >= 2:
                await self._run_v2_wave_two(plan, goals, diversity_cfg)

        console_logger.info("DR v2 domain randomization complete.")
        return plan

    async def _run_v2_wave_two(
        self,
        plan: OrchestrationPlan,
        goals: list[VariationGoal],
        diversity_cfg: dict,
    ) -> None:
        """Redo wave-1 subflows whose accepted scenes are too similar.

        Loads each accepted variant's final scene_state.json, computes
        pairwise diversity distance via subflow.tools._diversity_distance
        (same metric the Critic's check uses), and re-runs the lower-
        priority side of any pair below ``diversity_cfg.min_distance``.
        Wave-2 invocations get the wave-1 accepted snapshots as
        ``sibling_snapshots`` so the Critic actually scores diversity
        against real layouts.
        """
        from scenesmith.domain_randomization.subflow.tools import (
            _diversity_distance,
            _diversity_features,
        )

        accepted_snapshots: dict[str, dict] = {}
        for vid in [g.variation_id for g in goals]:
            status = plan.variations.get(vid)
            if status is None or status.status != "completed":
                continue
            scene_path = self._output_dir / f"variant_{vid}" / self._base_scene_dir.name
            finals = sorted(
                scene_path.rglob("scene_states/final_scene/scene_state.json")
            )
            if not finals:
                continue
            try:
                accepted_snapshots[vid] = json.loads(finals[0].read_text())
            except Exception as exc:  # noqa: BLE001 — keep wave 2 best-effort
                console_logger.warning(
                    "DR v2 wave 2: failed to load %s: %s", finals[0], exc
                )

        if len(accepted_snapshots) < 2:
            console_logger.info(
                "DR v2 wave 2: fewer than 2 accepted variants — skipping"
            )
            return

        threshold = float(diversity_cfg.get("min_distance", 0.05))
        features = {
            vid: _diversity_features(scene) for vid, scene in accepted_snapshots.items()
        }
        redo: set[str] = set()
        items = sorted(accepted_snapshots.keys())
        for i, a in enumerate(items):
            for b in items[i + 1 :]:
                if _diversity_distance(features[a], features[b]) <= threshold:
                    # Pick the lexicographically-larger id to redo so the
                    # choice is deterministic across runs.
                    redo.add(max(a, b))
        if not redo:
            console_logger.info(
                "DR v2 wave 2: all %d pairs above threshold %.3f — skipping",
                len(accepted_snapshots) * (len(accepted_snapshots) - 1) // 2,
                threshold,
            )
            return

        console_logger.info(
            "DR v2 wave 2: re-running %d variants for diversity (threshold=%.3f): %s",
            len(redo),
            threshold,
            sorted(redo),
        )
        await asyncio.gather(
            *(
                self._run_subflow_v2(
                    plan,
                    vid,
                    sibling_snapshots=[
                        s
                        for other_vid, s in accepted_snapshots.items()
                        if other_vid != vid
                    ],
                )
                for vid in sorted(redo)
            ),
            return_exceptions=True,
        )

    async def _v2_set_status(
        self,
        plan: OrchestrationPlan,
        vid: str,
        status: str,
        error: str | None = None,
    ) -> None:
        """Mutate + persist plan.json under the shared lock."""
        async with self._v2_plan_lock:
            plan.update_status(vid, status, error=error)
            plan.save(self._output_dir / "orchestration_plan.json")

    async def _run_subflow_v2(
        self,
        plan: OrchestrationPlan,
        vid: str,
        *,
        sibling_snapshots: list[dict] | None = None,
    ) -> None:
        """One v2 variation: scaffold (flat) -> DRSubflowAgent -> save +
        regen rl_artifacts -> update reward spec -> export.

        Safe to invoke concurrently via ``asyncio.gather`` — all plan
        mutations route through ``_v2_set_status`` which serializes on a
        shared lock, the on-disk plan.json write is atomic
        (``Path.replace``), and per-variation logger + scaffold dirs are
        disjoint paths.

        ``sibling_snapshots`` is optional and used by wave-2 diversity
        redos — pass the wave-1 accepted scene-state dicts so the
        Critic's check_diversity_vs_siblings tool sees real layouts.
        Wave-1 invocations pass None and the Critic falls back to a
        trivially-true diversity result.
        """
        goal = self._v2_goals[vid]
        await self._v2_set_status(plan, vid, "in_progress")

        try:
            variation_dir = self._output_dir / f"variant_{vid}"
            variation_scene_dir = variation_dir / self._base_scene_dir.name
            scaffold_variation_v2(
                base_scene_dir=self._base_scene_dir,
                variation_dir=variation_scene_dir,
                overwrite=True,
            )

            variation_logger = self._make_variation_logger(variation_scene_dir)
            room = load_room_for_dr(
                variation_scene_dir,
                cfg=self._v2_subflow_cfg,
                logger=variation_logger,
            )
            base_snapshot = json.loads(json.dumps(room.base_state_dict))
            base_artifacts_path = variation_scene_dir / "rl_artifacts.json"
            base_artifacts = json.loads(base_artifacts_path.read_text())

            subflow = DRSubflowAgent(
                self._v2_subflow_cfg,
                variation_logger,
                goal=goal,
                scene=room.scene,
                manipuland_tools=room.manipuland_tools,
                base_scene_snapshot=base_snapshot,
                rl_artifacts=base_artifacts,
                sibling_descriptions=[
                    other.description
                    for other_vid, other in self._v2_goals.items()
                    if other_vid != vid
                ],
                sibling_snapshots=sibling_snapshots,
            )
            try:
                final_state = await subflow.process_variation()
            except DRSubflowRejected as rejected:
                await self._v2_set_status(
                    plan,
                    vid,
                    "failed",
                    error=f"subflow rejected: {rejected.reason}",
                )
                return

            # Save final scene state into the room's final_scene checkpoint
            # path (the layout downstream tools expect).
            final_state_path = room.room_dir / "scene_state.json"
            final_state_path.write_text(json.dumps(final_state, indent=2))
            checkpoint_path = (
                room.room_dir / "scene_states" / "final_scene" / "scene_state.json"
            )
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text(json.dumps(final_state, indent=2))

            # Regenerate rl_artifacts from the mutated scene + reward spec
            # update + export.
            new_artifacts = regenerate_rl_artifacts(
                base_artifacts=base_artifacts,
                new_scene_state=final_state,
                task_critical_ids=list(goal.task_critical_object_ids),
            )
            base_artifacts_path.write_text(json.dumps(new_artifacts, indent=2))

            locked_set = set(goal.task_critical_object_ids)
            impact = classify_reward_spec_impact(
                spec=plan.variations[vid].spec,
                task_critical_ids=locked_set,
            )
            await update_reward_spec(
                impact=impact,
                variation_scene_dir=variation_scene_dir,
                base_scene_dir=self._base_scene_dir,
                new_scene=room.scene,
                task_critical_ids=locked_set,
                cfg=OmegaConf.to_container(self.cfg, resolve=True),
                rl_artifacts=new_artifacts,
                task_requirements=self._load_task_requirements(),
            )

            house_layout = self._load_house_layout()
            export_variation(
                variation_scene_dir=variation_scene_dir,
                rooms=[room.scene],
                house_layout=house_layout,
                cfg=self.cfg,
                export_cfg=self.export_cfg,
            )

            await self._v2_set_status(plan, vid, "completed")
            console_logger.info("DR v2 variation %s completed.", vid)
        except Exception as exc:  # noqa: BLE001 — surface and continue
            console_logger.error(
                "DR v2 variation %s failed: %s", vid, exc, exc_info=True
            )
            await self._v2_set_status(plan, vid, "failed", error=str(exc))

    def _apply_programmatic_ops(
        self,
        scene: RoomScene,
        spec: VariationSpec,
        locked_ids: set[str],
    ) -> RoomScene:
        """Apply all programmatic ops, logging only violations the op introduced.

        ``op.validate`` returns the full set of issues in the current
        scene, which on the trusted base scenes includes pre-existing
        collisions from the original generation pass (e.g.
        ``dining_table_0 overlaps side_chair_0``). Logging those every
        time hides the signal we actually care about — issues the op
        itself created. Capture the pre-op baseline and set-diff after.
        """
        if not spec.ops:
            return scene
        for op_spec in spec.ops:
            op = _OP_REGISTRY.get(op_spec.op_type)
            if op is None:
                console_logger.warning(f"Unknown op type: {op_spec.op_type}")
                continue
            pre_violations = set(op.validate(scene))
            scene = op.apply(scene, op_spec.params, locked_ids)
            post_violations = op.validate(scene)
            new_violations = [v for v in post_violations if v not in pre_violations]
            if new_violations:
                console_logger.warning(
                    f"Op {op_spec.op_type} introduced violations: "
                    f"{new_violations[:3]}"
                )
            elif post_violations:
                console_logger.debug(
                    f"Op {op_spec.op_type}: {len(post_violations)} pre-existing "
                    f"violations carried over (no new issues introduced)."
                )
        return scene

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _find_scene_dir(self, experiment_dir: Path) -> Path:
        """Find the scene_NNN directory inside the experiment directory."""
        scene_dirs = sorted(experiment_dir.glob("scene_*"))
        if not scene_dirs:
            raise FileNotFoundError(f"No scene_* directory found in {experiment_dir}")
        return scene_dirs[0]

    def _load_house_layout(self) -> HouseLayout:
        layout_path = self._base_scene_dir / "house_layout.json"
        if not layout_path.exists():
            raise FileNotFoundError(f"house_layout.json not found: {layout_path}")
        with open(layout_path) as f:
            data = json.load(f)
        return HouseLayout.from_dict(data, house_dir=self._base_scene_dir)

    def _load_rl_artifacts(self) -> dict | None:
        path = self._base_scene_dir / "rl_artifacts.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    def _load_task_requirements(self) -> Any | None:
        path = self._base_scene_dir / "task_requirements.json"
        if path.exists():
            try:
                from scenesmith.task_analysis_agent.task_requirements import (
                    TaskRequirements,
                )

                with open(path) as f:
                    return TaskRequirements.from_dict(json.load(f))
            except Exception:
                return None
        return None

    def _make_variation_logger(self, variation_scene_dir: Path) -> BaseLogger:
        return ConsoleLogger(output_dir=variation_scene_dir)

    def _prepare_plan_for_resume(self, plan: OrchestrationPlan) -> OrchestrationPlan:
        """Normalize a loaded plan so interrupted variations can be retried."""
        if plan.base_experiment_dir != str(self.base_dir):
            raise ValueError(
                "Loaded orchestration plan does not match the requested "
                f"base_experiment_dir: {plan.base_experiment_dir} != {self.base_dir}"
            )

        if plan.num_variations != self.num_variations:
            console_logger.warning(
                "Loaded plan num_variations=%s overrides requested num_variations=%s",
                plan.num_variations,
                self.num_variations,
            )
            self.num_variations = plan.num_variations

        if self.description is None:
            self.description = plan.description

        for variation in plan.variations.values():
            if variation.status == "in_progress":
                variation.status = "pending"
                variation.completed_at = None

        return plan
