"""Build the agent-shaped DictConfig that ``DRSubflowAgent`` consumes.

The subflow reuses the manipuland-agent yaml shape (sessions, retries,
critique cap, reset thresholds, openai model + reasoning effort) and
layers DR-specific overrides on top: planner/designer/critic max_turns,
the dr-subflow prompt enum names, and tighter early-finish thresholds
appropriate for "good enough at N=1".
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

# Per-round caps tuned for the DR subflow at N=1. Generous values from
# StatefulManipulandAgent (planner=30, designer=80, critic=20) made the
# total LLM call count explode on the first live run (40+ calls in 17 min
# with no convergence). Tighter caps here cap total subflow cost and
# surface unproductive loops earlier; the production loop can relax these
# once the planner prompt is iterated.
#
# planner.max_turns was bumped from 12 to 20 after the collision-feedback
# accept-gate was added: dense/cluttered goals now require multiple
# corrective rounds (move → re-render → critic → request_design_change)
# before all AABB overlaps clear. 12 was too tight — a high-density scene
# run hit "Max turns (12) exceeded" mid-revision.
# 20 gives the planner ~6 extra correction rounds.
_DR_OVERRIDES = {
    "agents": {
        "planner_agent": {
            "name": "dr_subflow_planner",
            "max_turns": 20,
            "prompt": "PLANNER",
        },
        "designer_agent": {
            "name": "dr_subflow_designer",
            "max_turns": 10,
            "prompt": "DESIGNER",
        },
        "critic_agent": {
            "name": "dr_subflow_critic",
            "max_turns": 6,
            "prompt": "CRITIC",
        },
    },
    "max_critique_rounds": 3,
    "early_finish_min_score": 7,
    "reset_single_category_threshold": 3,
    "reset_total_sum_threshold": 8,
}


def build_dr_subflow_cfg(root_cfg: DictConfig) -> DictConfig:
    """Compose root cfg + manipuland base + DR subflow overrides.

    Merging order matches Hydra's defaults-list semantics: manipuland
    base + stateful first (for openai, sessions, api_timeout, etc.), then
    root cfg (for provider model + service_tier), then the in-line DR
    overrides (for planner/designer/critic shape and thresholds).
    """
    repo = Path(__file__).resolve().parents[3]
    base = OmegaConf.load(repo / "configurations/manipuland_agent/_base_agent.yaml")
    stateful = OmegaConf.load(
        repo / "configurations/manipuland_agent/base_manipuland_agent.yaml"
    )
    agent_cfg = OmegaConf.merge(base, stateful)
    merged = OmegaConf.merge(agent_cfg, root_cfg)
    # Strict-struct mode propagates from root_cfg; disable so we can layer
    # DR-specific keys (planner/designer/critic names, thresholds) on top
    # without OmegaConf rejecting the "unknown" keys.
    OmegaConf.set_struct(merged, False)
    merged = OmegaConf.merge(merged, OmegaConf.create(_DR_OVERRIDES))
    return merged
