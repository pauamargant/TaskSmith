"""DR v2 subflow Critic structured output types."""

from __future__ import annotations

from dataclasses import dataclass

from scenesmith.agent_utils.scoring import CategoryScore, CritiqueWithScores


@dataclass
class DRVariationCritique(CritiqueWithScores):
    """Subflow Critic output. Same 5-category shape as design §3.2."""

    goal_alignment: CategoryScore
    feasibility: CategoryScore
    task_validity: CategoryScore
    diversity_vs_siblings: CategoryScore
    manipuland_invariance: CategoryScore

    def get_scores(self) -> list[CategoryScore]:
        return [
            self.goal_alignment,
            self.feasibility,
            self.task_validity,
            self.diversity_vs_siblings,
            self.manipuland_invariance,
        ]
