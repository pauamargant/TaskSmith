"""``DRSubflowAgent``: DR v2 per-variation planner / designer / critic loop.

Mirrors ``StatefulManipulandAgent`` shape via ``BaseStatefulAgent`` so we
reuse session creation, retry-with-backoff, planner-tool plumbing, and
checkpoint state. Overrides the parts of the base class that assume a
populated ``rendering_manager`` / Blender render dir / physics context:

- ``_request_critique_impl``: replaced with a no-render, no-physics
  version that calls the Critic Agent on a compact text scene summary and
  parses its structured output through the same pydantic mirror used by
  the standalone critic.
- ``_create_critic_agent``: built directly (the base helper forces
  ``tool_choice="observe_scene"`` which we don't have a render path for).
- ``_finalize_scene_and_scores``: no-op (no render dir to copy).

The Designer surface is: move, remove, and (optionally) swap_asset — see
``DRDesignerTools.as_function_tools`` and the ``enable_swap_asset`` parameter.
"""

from __future__ import annotations

import copy
import json
import logging
import time

from pathlib import Path
from typing import Any

from agents import Agent, function_tool
from agents.tool import FunctionTool

from scenesmith.agent_utils.base_stateful_agent import (
    BaseStatefulAgent,
    log_agent_usage,
)
from scenesmith.agent_utils.placement_noise import PlacementNoiseMode
from scenesmith.agent_utils.room import AgentType, RoomScene
from scenesmith.domain_randomization.subflow.critic import (
    _DRVariationCritiqueSchema,
    _schema_to_dataclass,
    summarize_scene_for_critic,
)
from scenesmith.domain_randomization.subflow.critique import DRVariationCritique
from scenesmith.domain_randomization.subflow.tools import (
    DRDesignerTools,
    check_diversity_vs_siblings,
    check_locked_ids_untouched,
    check_manipuland_invariance,
    check_position_bounds,
)
from scenesmith.domain_randomization.variation_spec import VariationGoal
from scenesmith.manipuland_agents.tools.manipuland_tools import ManipulandTools
from scenesmith.prompts.registry import DRSubflowPrompts

console_logger = logging.getLogger(__name__)


class DRSubflowRejected(Exception):
    """Raised when the subflow terminates without accepting the variation."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class DRSubflowAgent(BaseStatefulAgent):
    """Planner / Designer / Critic loop for one ``VariationGoal``.

    Construction expects an already-loaded ``RoomScene`` and a matching
    ``ManipulandTools`` (the on-disk surfaces / current furniture context
    that move/remove ops validate against). Use
    ``scenesmith.domain_randomization.dr_scene_loader.load_room_for_dr``
    to build both from a finished base scene directory.
    """

    _is_placement_agent = False

    @property
    def agent_type(self) -> AgentType:
        return AgentType.DR_SUBFLOW

    def __init__(
        self,
        cfg,
        logger,
        *,
        goal: VariationGoal,
        scene: RoomScene,
        manipuland_tools: ManipulandTools,
        base_scene_snapshot: dict,
        rl_artifacts: dict,
        sibling_descriptions: list[str] | None = None,
        sibling_snapshots: list[dict] | None = None,
        enable_swap_asset: bool = True,
        swap_asset_blender_server: Any = None,
        swap_asset_collision_client: Any = None,
    ) -> None:
        super().__init__(cfg=cfg, logger=logger)
        self.goal = goal
        self.scene = scene
        self._manipuland_tools = manipuland_tools
        self.base_snapshot = copy.deepcopy(base_scene_snapshot)
        self._rl_artifacts = rl_artifacts
        self.sibling_descriptions: list[str] = list(sibling_descriptions or [])
        self._enable_swap_asset = enable_swap_asset
        self._swap_asset_blender_server = swap_asset_blender_server
        self._swap_asset_collision_client = swap_asset_collision_client
        # Concrete sibling scene_state dicts for the Critic's
        # check_diversity_vs_siblings tool. Wave-1 subflows get [] (their
        # siblings haven't completed yet); wave-2 redos get the wave-1
        # accepted snapshots so the Critic can actually score diversity
        # against real layouts instead of just goal-description text.
        self.sibling_snapshots: list[dict] = list(sibling_snapshots or [])

        self._locked: set[str] = set(goal.task_critical_object_ids)
        self._designer_tools_obj: DRDesignerTools | None = None
        self._variation_accepted = False
        self._give_up_reason: str | None = None
        self.placement_style = "n/a"  # base critique impl references this attribute

        self.designer: Agent | None = None
        self.critic: Agent | None = None
        self.planner: Agent | None = None
        self.designer_session = None
        self.critic_session = None
        self._latest_critique: DRVariationCritique | None = None

    # ------------------------------------------------------------------
    # Hooks the base class requires (abstract or override-friendly).
    # ------------------------------------------------------------------

    def _get_final_scores_directory(self) -> Path:
        return self.logger.output_dir / "dr_subflow_final"

    def _get_critique_prompt_enum(self) -> Any:
        return DRSubflowPrompts.CRITIC_RUNNER_INSTRUCTION

    def _get_initial_design_prompt_enum(self) -> Any:
        return DRSubflowPrompts.DESIGNER

    def _get_initial_design_prompt_kwargs(self) -> dict:
        return {
            "goal": self._goal_json(),
            "locked_ids": self._locked_ids_json(),
            "planner_request": (
                "Make the initial mutation batch toward the variation goal. "
                "You may move identity-locked task-critical ids within their "
                "declared position bounds, and move/remove non-locked manipulands."
            ),
            "tool_list": self._tool_list_text(),
            "movable_ids": self._movable_ids_json(),
            "surface_ids": self._surface_ids_json(),
            "pose_bounds": self._pose_bounds_json(),
        }

    def _get_design_change_prompt_enum(self) -> Any:
        return DRSubflowPrompts.DESIGNER

    def _set_placement_noise_profile(self, mode: PlacementNoiseMode) -> None:
        # DR subflow doesn't place new objects — no noise profile to switch.
        del mode

    def _get_extra_critique_kwargs(self) -> dict[str, Any]:
        # Used by base _request_critique_impl, but we override that whole
        # method below; this stays here only so callers (e.g. the override
        # itself) can grab the kwargs for the dr_subflow critic runner.
        current = self.scene.to_state_dict()
        current_summary = summarize_scene_for_critic(current)

        # Surface AABB collisions in the scene summary so the Critic's
        # feasibility grade reflects real overlaps without us having to
        # edit the critic prompt template.
        cc_cfg = self._collision_check_cfg()
        if (
            cc_cfg.get("enabled", False)
            and cc_cfg.get("surface_to_critic", True)
            and self._designer_tools_obj is not None
        ):
            pairs = self._designer_tools_obj._collision_pairs()
            if pairs:
                pair_list = sorted([sorted(list(p)) for p in pairs])
                current_summary = (
                    f"{current_summary}\n\n[collisions] {len(pair_list)} AABB "
                    f"overlap(s) detected: {pair_list}"
                )

        return {
            "goal": self._goal_json(),
            "base_scene_summary": summarize_scene_for_critic(self.base_snapshot),
            "current_scene_summary": current_summary,
            "sibling_descriptions": json.dumps(self.sibling_descriptions, indent=2),
        }

    async def _finalize_scene_and_scores(self) -> None:
        # No render dir to copy and no scores.yaml to dump for DR subflow.
        return None

    # ------------------------------------------------------------------
    # Prompt-render helpers.
    # ------------------------------------------------------------------

    def _goal_json(self) -> str:
        return json.dumps(self.goal.to_dict(), indent=2)

    def _locked_ids_json(self) -> str:
        return json.dumps(sorted(self._locked), indent=2)

    def _tool_list_text(self) -> str:
        tools = ["move_manipuland", "remove_manipuland"]
        if self._enable_swap_asset:
            tools.append("swap_asset")
        return ", ".join(tools)

    def _movable_ids_json(self) -> str:
        movable = sorted(
            oid
            for oid, obj in self.scene.objects.items()
            if obj.object_type.value == "manipuland" and str(oid) not in self._locked
        )
        return json.dumps([str(x) for x in movable], indent=2)

    def _surface_ids_json(self) -> str:
        return json.dumps(
            sorted(self._manipuland_tools.support_surfaces.keys()), indent=2
        )

    def _pose_bounds_json(self) -> str:
        return json.dumps(self._build_pose_bounds(), indent=2)

    def _scene_summary_text(self) -> str:
        return summarize_scene_for_critic(self.scene.to_state_dict())

    # ------------------------------------------------------------------
    # Designer / Critic / Planner construction.
    # ------------------------------------------------------------------

    def _build_pose_bounds(self) -> dict[str, dict]:
        """Translate rl_artifacts.task_object_variants into per-id pose
        bounds the Designer's move tool enforces. Each entry carries the
        nominal surface-local (x, z) drawn from the base scene plus the
        declared ``position_range_xy_on_surface`` half-widths."""
        variants = self._rl_artifacts.get("domain_randomization", {}).get(
            "task_object_variants", []
        )
        base_objects = self.base_snapshot.get("objects", {})
        bounds: dict[str, dict] = {}
        for variant in variants:
            oid = variant.get("object_id")
            if not oid:
                continue
            bounds_xy = variant.get("position_range_xy_on_surface") or [0.0, 0.0]
            placement = (base_objects.get(oid) or {}).get("placement_info") or {}
            nominal = placement.get("position_2d") or [0.0, 0.0]
            bounds[str(oid)] = {
                "nominal_position_2d": [float(nominal[0]), float(nominal[1])],
                "max_dxdz": [float(bounds_xy[0]), float(bounds_xy[1])],
            }
        return bounds

    def _create_designer_tools(self) -> list[FunctionTool]:
        cc_cfg = self._collision_check_cfg()
        self._designer_tools_obj = DRDesignerTools(
            self._manipuland_tools,
            locked_ids=self._locked,
            pose_bounds=self._build_pose_bounds(),
            collision_check_enabled=bool(
                cc_cfg.get("enabled", False)
                and cc_cfg.get("feedback_to_designer", True)
            ),
            collision_margin_m=float(cc_cfg.get("margin_m", 0.0)),
            collision_ignore_within_task_critical=bool(
                cc_cfg.get("ignore_pairs_within_task_critical", False)
            ),
        )
        return self._designer_tools_obj.as_function_tools(
            enable_swap_asset=self._enable_swap_asset,
            swap_asset_blender_server=self._swap_asset_blender_server,
            swap_asset_collision_client=self._swap_asset_collision_client,
        )

    def _collision_check_cfg(self) -> dict:
        """Read subflow.collision_check block from the orchestrator cfg.

        Returns an empty dict if the block is missing (older configs);
        the caller should treat that as "feature disabled".
        """
        cfg = self.cfg
        # OmegaConf or plain dict — support both shapes.
        try:
            subflow = (
                cfg.domain_randomization.subflow
                if hasattr(cfg, "domain_randomization")
                else cfg.get("domain_randomization", {}).get("subflow", {})
            )
            cc = (
                subflow.collision_check
                if hasattr(subflow, "collision_check")
                else subflow.get("collision_check", {})
            )
            # Materialize to a plain dict for .get() ergonomics.
            from omegaconf import OmegaConf

            return OmegaConf.to_container(cc, resolve=True) if cc else {}
        except Exception:
            return {}

    def _create_critic_tools(self) -> list[FunctionTool]:
        agent = self

        @function_tool
        async def check_locked_ids_untouched_tool() -> str:
            """Verify task-critical (locked) ids' transforms match the base scene."""
            current = agent.scene.to_state_dict()
            result = check_locked_ids_untouched(
                agent.base_snapshot, current, agent._locked
            )
            return json.dumps({"passed": result.passed, "detail": result.detail})

        @function_tool
        async def check_position_bounds_tool() -> str:
            """Verify task-critical ids stayed within position_range_xy_on_surface."""
            current = agent.scene.to_state_dict()
            result = check_position_bounds(
                agent.base_snapshot, current, agent._rl_artifacts
            )
            return json.dumps({"passed": result.passed, "detail": result.detail})

        @function_tool
        async def check_manipuland_invariance_tool() -> str:
            """Verify no furniture / wall / ceiling object changed vs base."""
            current = agent.scene.to_state_dict()
            result = check_manipuland_invariance(agent.base_snapshot, current)
            return json.dumps({"passed": result.passed, "detail": result.detail})

        @function_tool
        async def check_diversity_vs_siblings_tool() -> str:
            """Verify the current manipuland layout differs from sibling variations."""
            current = agent.scene.to_state_dict()
            # Wave-1 subflows run before any sibling completes, so
            # ``sibling_snapshots`` is [] and the check returns True
            # ("no siblings to compare"). Wave-2 redos receive the
            # accepted wave-1 scenes as concrete snapshots and the
            # check measures real layout distance.
            result = check_diversity_vs_siblings(current, agent.sibling_snapshots)
            return json.dumps({"passed": result.passed, "detail": result.detail})

        return [
            check_locked_ids_untouched_tool,
            check_position_bounds_tool,
            check_manipuland_invariance_tool,
            check_diversity_vs_siblings_tool,
        ]

    def _create_critic_agent_dr(self, tools: list[FunctionTool]) -> Agent:
        """Build the Critic Agent directly (bypassing the base helper that
        forces observe_scene as the first tool call; we don't have a
        render path)."""
        critic_config = self.cfg.agents.critic_agent
        rendered = self.prompt_registry.get_prompt(
            DRSubflowPrompts.CRITIC,
            **self._get_extra_critique_kwargs(),
        )
        return Agent(
            name=critic_config.name,
            model=self._make_model(),
            tools=tools,
            instructions=rendered,
            output_type=_DRVariationCritiqueSchema,
            model_settings=self._get_model_settings(settings_key="critic"),
        )

    def _create_designer_agent_dr(self, tools: list[FunctionTool]) -> Agent:
        return super()._create_designer_agent(
            tools=tools,
            prompt_enum=DRSubflowPrompts.DESIGNER,
            goal=self._goal_json(),
            locked_ids=self._locked_ids_json(),
            planner_request="Awaiting planner instructions.",
            tool_list=self._tool_list_text(),
            movable_ids=self._movable_ids_json(),
            surface_ids=self._surface_ids_json(),
            pose_bounds=self._pose_bounds_json(),
        )

    def _create_planner_agent_dr(self, tools: list[FunctionTool]) -> Agent:
        cfg = self.cfg
        return super()._create_planner_agent(
            tools=tools,
            prompt_enum=DRSubflowPrompts.PLANNER,
            goal=self._goal_json(),
            scene_summary=self._scene_summary_text(),
            sibling_descriptions=json.dumps(self.sibling_descriptions, indent=2),
            max_critique_rounds=cfg.max_critique_rounds,
            early_finish_min_score=cfg.early_finish_min_score,
            reset_single_category_threshold=cfg.reset_single_category_threshold,
            reset_total_sum_threshold=cfg.reset_total_sum_threshold,
        )

    def _create_planner_tools(self) -> list[FunctionTool]:
        tools = super()._create_planner_tools()
        agent = self

        @function_tool
        async def accept_variation() -> str:
            """Accept the current scene state as satisfying the variation goal.

            Call this once the Critic's last DRVariationCritique meets the
            early_finish_min_score threshold across every category. Do not
            call further tools after this; return a short text summary.

            When subflow.collision_check.block_accept is true, this tool
            refuses to accept while AABB collisions remain in the scene;
            the planner must issue more designer changes to clear them.
            """
            cc_cfg = agent._collision_check_cfg()
            block = bool(
                cc_cfg.get("enabled", False) and cc_cfg.get("block_accept", True)
            )
            if block and agent._designer_tools_obj is not None:
                pairs = agent._designer_tools_obj._collision_pairs()
                if pairs:
                    pair_list = sorted([sorted(list(p)) for p in pairs])
                    console_logger.info(
                        "[DR planner] accept_variation REFUSED — %d collisions remain: %s",
                        len(pair_list),
                        pair_list,
                    )
                    return (
                        f"Refused: {len(pair_list)} AABB collisions remain in the scene "
                        f"({pair_list}). Issue another request_design_change to resolve "
                        f"them before calling accept_variation again."
                    )
            agent._variation_accepted = True
            console_logger.info("[DR planner] accept_variation()")
            return "Variation accepted. Stop calling tools and return a short summary."

        @function_tool
        async def give_up(reason: str) -> str:
            """Give up on the variation goal when it cannot be satisfied.

            Use this when the locked-id constraint or feasibility makes the
            goal unreachable from the current scene. Provide a one-sentence
            reason. Do not call further tools."""
            agent._give_up_reason = reason
            console_logger.info("[DR planner] give_up: %s", reason)
            return (
                f"Giving up: {reason}. Stop calling tools and return a short summary."
            )

        tools.extend([accept_variation, give_up])
        return tools

    # ------------------------------------------------------------------
    # Visibility overrides: log planner→designer/critic round structure.
    # The base impls only emit ``Tool called: X`` headers; without these
    # overrides a looped run produces no diagnostic stdout beyond
    # trace-export skips.
    # ------------------------------------------------------------------

    async def _request_initial_design_impl(self) -> str:  # type: ignore[override]
        console_logger.info("[DR planner -> designer] request_initial_design()")
        result = await super()._request_initial_design_impl()
        console_logger.info(
            "[DR designer <-] initial response: %s",
            (result or "")[:200].replace("\n", " "),
        )
        return result

    async def _request_design_change_impl(  # type: ignore[override]
        self, instruction: str
    ) -> str:
        # Don't call super() — the base impl renders the design-change
        # prompt with only ``instruction``, but our designer.yaml takes
        # goal/locked_ids/planner_request/tool_list. Render directly and
        # drive the designer Runner ourselves.
        # (Every change-request raised a strict template-vars mismatch
        # without the full render path, so we bypass super() here.)
        console_logger.info(
            "[DR planner -> designer] request_design_change: %s",
            instruction[:200].replace("\n", " "),
        )
        full_instruction = self.prompt_registry.get_prompt(
            DRSubflowPrompts.DESIGNER,
            goal=self._goal_json(),
            locked_ids=self._locked_ids_json(),
            planner_request=instruction,
            tool_list=self._tool_list_text(),
            movable_ids=self._movable_ids_json(),
            surface_ids=self._surface_ids_json(),
            pose_bounds=self._pose_bounds_json(),
        )
        start = time.perf_counter()
        result = await self._runner_run_with_retries(
            starting_agent=self.designer,
            input=full_instruction,
            session=self.designer_session,
            max_turns=self.cfg.agents.designer_agent.max_turns,
            run_config=self._create_run_config(),
            op_name="dr_subflow_designer_change",
        )
        log_agent_usage(
            result=result,
            agent_name="DR_SUBFLOW_DESIGNER (CHANGE)",
            duration_s=time.perf_counter() - start,
        )
        text = result.final_output or ""
        console_logger.info(
            "[DR designer <-] change response: %s", text[:200].replace("\n", " ")
        )
        return text

    # ------------------------------------------------------------------
    # Override the critique loop: no render, no physics, no scores.yaml.
    # ------------------------------------------------------------------

    async def _request_critique_impl(  # type: ignore[override]
        self,
        update_checkpoint: bool = True,
        instruction: str | None = None,
        _retry_depth: int = 0,
    ) -> str:
        console_logger.info("Tool called: request_critique (DR subflow)")
        rendered = self.prompt_registry.get_prompt(
            DRSubflowPrompts.CRITIC_RUNNER_INSTRUCTION,
            **self._get_extra_critique_kwargs(),
        )
        if instruction:
            rendered = f"{rendered}\n\n# CRITIC OVERRIDE\n{instruction}\n"

        start = time.perf_counter()
        result = await self._runner_run_with_retries(
            starting_agent=self.critic,
            input=rendered,
            session=self.critic_session,
            max_turns=self.cfg.agents.critic_agent.max_turns,
            run_config=self._create_run_config(),
            op_name="dr_subflow_critic",
        )
        log_agent_usage(
            result=result,
            agent_name="DR_SUBFLOW_CRITIC",
            duration_s=time.perf_counter() - start,
        )

        parsed = result.final_output_as(_DRVariationCritiqueSchema)
        critique = _schema_to_dataclass(parsed)
        self._latest_critique = critique

        # Lightweight checkpoint bookkeeping (no render dirs).
        if update_checkpoint:
            self.previous_scene_checkpoint = self.scene_checkpoint
            self.previous_checkpoint_scores = self.checkpoint_scores
            self.scene_checkpoint = copy.deepcopy(self.scene.to_state_dict())
            self.checkpoint_scores = critique
        self.previous_scores = critique

        score_summary = ", ".join(f"{s.name}={s.grade}" for s in critique.get_scores())
        console_logger.info(
            "[DR critic] grades: %s | critique: %s",
            score_summary,
            critique.critique[:200].replace("\n", " "),
        )
        return f"{critique.critique}\n\n[scores] {score_summary}"

    # ------------------------------------------------------------------
    # Entry point.
    # ------------------------------------------------------------------

    async def process_variation(self) -> dict:
        """Run the planner→designer→critic loop and return the mutated scene dict.

        Raises:
            DRSubflowRejected: if the planner called ``give_up`` or finished
                without calling ``accept_variation``.
        """
        self._variation_accepted = False
        self._give_up_reason = None
        self._latest_critique = None

        # Set up sessions. Leave scene_checkpoint / previous_scene_checkpoint
        # at the base-class default of None — the base reset tool also gates
        # on previous_checkpoint_scores (None until the first critique
        # populates it via the critique impl below), so pre-priming the
        # scene side is theater that misleads the planner.
        self.designer_session, self.critic_session = self._create_sessions(
            session_prefix=f"{self.goal.variation_id}_"
        )

        # Build agents in dependency order: designer first (planner tools call
        # designer/critic via runner), then critic, then planner.
        designer_tools = self._create_designer_tools()
        critic_tools = self._create_critic_tools()
        self.designer = self._create_designer_agent_dr(designer_tools)
        self.critic = self._create_critic_agent_dr(critic_tools)
        planner_tools = self._create_planner_tools()
        self.planner = self._create_planner_agent_dr(planner_tools)

        runner_input = self.prompt_registry.get_prompt(
            DRSubflowPrompts.PLANNER_RUNNER_INSTRUCTION,
            goal_description=self.goal.description,
        )

        start = time.perf_counter()
        result = await self._runner_run_with_retries(
            starting_agent=self.planner,
            input=runner_input,
            session=None,  # planner is single-turn-style; no persistent session
            max_turns=self.cfg.agents.planner_agent.max_turns,
            run_config=self._create_run_config(),
            op_name="dr_subflow_planner",
        )
        log_agent_usage(
            result=result,
            agent_name="DR_SUBFLOW_PLANNER",
            duration_s=time.perf_counter() - start,
        )

        if self._give_up_reason:
            raise DRSubflowRejected(self._give_up_reason)
        if not self._variation_accepted:
            raise DRSubflowRejected(
                "planner finished without accept_variation or give_up"
            )

        return self.scene.to_state_dict()
