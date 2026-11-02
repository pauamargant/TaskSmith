"""MorphologyTaskConfig dataclass.

Produced by MorphologyTaskAgent. Saved as morphology_task_config.json in the
scene output directory. Consumed by SceneSmithHandEnv to configure the RL task
without hardcoding any scene-specific values.
"""

from __future__ import annotations

import json
import logging

from dataclasses import dataclass, field
from pathlib import Path

console_logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """Reward weights and thresholds for the morphological hand training task."""

    reach_weight: float = 2.0
    """Exponential reach reward weight (EEF → object distance)."""

    contact_bonus_weight: float = 1.0
    """Bonus applied when EEF is within grasp_proximity_threshold_m of the object."""

    lift_weight: float = 4.0
    """Linear lift reward weight (object height above initial Z)."""

    lift_target_height_m: float = 0.20
    """Target lift height in metres above the object's initial position."""

    grasp_proximity_threshold_m: float = 0.10
    """Distance threshold in metres for the contact proxy bonus."""

    drop_penalty: float = -10.0
    """Penalty applied when the object falls below initial Z minus 0.05 m. Must be negative."""

    action_smoothness_weight: float = -0.001
    """Penalty weight for action magnitude (encourages smooth control). Must be negative."""

    def to_dict(self) -> dict:
        return {
            "reach_weight": self.reach_weight,
            "contact_bonus_weight": self.contact_bonus_weight,
            "lift_weight": self.lift_weight,
            "lift_target_height_m": self.lift_target_height_m,
            "grasp_proximity_threshold_m": self.grasp_proximity_threshold_m,
            "drop_penalty": self.drop_penalty,
            "action_smoothness_weight": self.action_smoothness_weight,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RewardConfig":
        return cls(
            reach_weight=float(data.get("reach_weight", 2.0)),
            contact_bonus_weight=float(data.get("contact_bonus_weight", 1.0)),
            lift_weight=float(data.get("lift_weight", 4.0)),
            lift_target_height_m=float(data.get("lift_target_height_m", 0.20)),
            grasp_proximity_threshold_m=float(
                data.get("grasp_proximity_threshold_m", 0.10)
            ),
            drop_penalty=float(data.get("drop_penalty", -10.0)),
            action_smoothness_weight=float(
                data.get("action_smoothness_weight", -0.001)
            ),
        )


@dataclass
class MorphologyCurriculumStage:
    """A single stage in the morphological hand training curriculum.

    Each stage enables a cumulative subset of task phases. Stages are ordered
    from simplest (approach only) to hardest (all phases active).
    """

    id: int
    """Stage index (0-based)."""

    name: str
    """Human-readable stage name, e.g. 'approach_only' or 'grasp'."""

    active_phases: list[str]
    """Cumulative phases active in this stage, e.g. ['approach', 'grasp'].
    All reward terms for these phases contribute to the reward signal."""

    transition_success_rate: float
    """Rolling success rate (over transition_window_episodes) required to advance."""

    transition_window_episodes: int
    """Number of recent episodes over which success rate is computed."""

    min_episodes_in_stage: int
    """Minimum episodes before any transition check. Should be >= 2 * transition_window_episodes."""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "active_phases": self.active_phases,
            "transition_success_rate": self.transition_success_rate,
            "transition_window_episodes": self.transition_window_episodes,
            "min_episodes_in_stage": self.min_episodes_in_stage,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MorphologyCurriculumStage":
        return cls(
            id=int(data["id"]),
            name=data["name"],
            active_phases=data.get("active_phases", []),
            transition_success_rate=float(data.get("transition_success_rate", 0.7)),
            transition_window_episodes=int(data.get("transition_window_episodes", 20)),
            min_episodes_in_stage=int(data.get("min_episodes_in_stage", 50)),
        )


@dataclass
class MorphologyTaskConfig:
    """Task configuration for morphological hand RL training.

    Selects a single target object from the scene, defines the task phases
    (approach, grasp, lift, transport), and specifies a phase curriculum that
    progressively enables reward terms as the agent masters each phase.
    """

    selected_object_id: str
    """object_id from rl_artifacts.task_objects (e.g. 'coffee_mug_0')."""

    selected_object_name: str
    """Human-readable object name (e.g. 'coffee mug')."""

    task_phases: list[str] = field(
        default_factory=lambda: ["approach", "grasp", "lift"]
    )
    """Full ordered list of task phases. Subset of: approach, grasp, lift, transport.
    Defines the final stage of the curriculum."""

    curriculum_stages: list[MorphologyCurriculumStage] = field(default_factory=list)
    """Phase curriculum: one stage per task phase, each enabling one more reward term.
    Stage 0 = approach only; last stage = all task_phases active."""

    reward_config: RewardConfig = field(default_factory=RewardConfig)
    """Reward weights and thresholds for the RL training environment."""

    hand_init_offset_above_object_m: float = 0.25
    """How far above the selected object (Z axis) to spawn the hand at episode start."""

    rationale: str = ""
    """LLM reasoning: why this object was selected and other candidates rejected."""

    def to_dict(self) -> dict:
        return {
            "selected_object_id": self.selected_object_id,
            "selected_object_name": self.selected_object_name,
            "task_phases": self.task_phases,
            "curriculum_stages": [s.to_dict() for s in self.curriculum_stages],
            "reward_config": self.reward_config.to_dict(),
            "hand_init_offset_above_object_m": self.hand_init_offset_above_object_m,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MorphologyTaskConfig":
        return cls(
            selected_object_id=data["selected_object_id"],
            selected_object_name=data["selected_object_name"],
            task_phases=data.get("task_phases", ["approach", "grasp", "lift"]),
            curriculum_stages=[
                MorphologyCurriculumStage.from_dict(s)
                for s in data.get("curriculum_stages", [])
            ],
            reward_config=RewardConfig.from_dict(data.get("reward_config", {})),
            hand_init_offset_above_object_m=float(
                data.get("hand_init_offset_above_object_m", 0.25)
            ),
            rationale=data.get("rationale", ""),
        )

    def save(self, path: Path) -> None:
        """Save to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        console_logger.info("Morphology task config saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "MorphologyTaskConfig":
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)
