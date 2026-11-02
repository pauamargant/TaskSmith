"""Reward spec and curriculum spec dataclasses.

Produced by RewardSpecAgent from rl_artifacts.json + task_requirements.json.
Saved as reward_spec.yaml and curriculum_spec.yaml in the scene output directory.
"""

from __future__ import annotations

import logging

from dataclasses import dataclass, field
from pathlib import Path

import yaml

console_logger = logging.getLogger(__name__)

# Sentinel for robot end-effector in from_object_id / to_object_id fields.
END_EFFECTOR = "end_effector"

_VALID_REWARD_TYPES = frozenset(
    {
        "reach",
        "grasp_contact",
        "lift",
        "transport",
        "success_bonus",
        "drop_penalty",
        "action_smoothness",
    }
)
_VALID_SHAPING = frozenset({"linear", "exp"})


@dataclass
class RewardTerm:
    """A single component of the dense reward function.

    object_id fields use the following conventions:
    - "end_effector"  robot end-effector (not a scene object)
    - "<object_id>"   exact object_id string from rl_artifacts.task_objects
    - None            term has no source/target entity (e.g. action_smoothness)
    """

    name: str
    """Unique identifier for this reward term, e.g. 'reach_mug'."""

    type: str
    """Term type: reach | grasp_contact | lift | transport | success_bonus |
    drop_penalty | action_smoothness."""

    weight: float
    """Scalar multiplier. Negative for penalty terms."""

    shaping: str
    """Reward shaping function: 'linear' or 'exp'."""

    active_in_stages: list[int]
    """Stage IDs (from curriculum) in which this term is active. Empty = all stages."""

    from_object_id: str | None
    """Source entity. Use END_EFFECTOR sentinel for robot EEF, None for untethered terms."""

    to_object_id: str | None
    """Target entity object_id from rl_artifacts. None for untethered terms."""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.type,
            "weight": self.weight,
            "shaping": self.shaping,
            "active_in_stages": self.active_in_stages,
            "from_object_id": self.from_object_id,
            "to_object_id": self.to_object_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RewardTerm":
        term_type = data["type"]
        shaping = data.get("shaping", "linear")
        if term_type not in _VALID_REWARD_TYPES:
            console_logger.warning(
                "RewardTerm '%s': unrecognised type %r (expected one of %s)",
                data.get("name", "?"),
                term_type,
                sorted(_VALID_REWARD_TYPES),
            )
        if shaping not in _VALID_SHAPING:
            console_logger.warning(
                "RewardTerm '%s': unrecognised shaping %r (expected one of %s)",
                data.get("name", "?"),
                shaping,
                sorted(_VALID_SHAPING),
            )
        return cls(
            name=data["name"],
            type=term_type,
            weight=float(data["weight"]),
            shaping=shaping,
            active_in_stages=data.get("active_in_stages", []),
            from_object_id=data.get("from_object_id"),
            to_object_id=data.get("to_object_id"),
        )


@dataclass
class RewardSpec:
    """Dense reward function specification.

    References object IDs from rl_artifacts.json. Does NOT duplicate
    success_conditions — consumers should read rl_artifacts.json directly
    for that field (single source of truth).
    """

    reward_terms: list[RewardTerm]
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "notes": self.notes,
            "reward_terms": [t.to_dict() for t in self.reward_terms],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict,
        known_object_ids: set[str] | None = None,
        valid_stage_ids: set[int] | None = None,
    ) -> "RewardSpec":
        """Deserialize and optionally validate object ID and stage ID references.

        Args:
            data: Raw dict from JSON/YAML.
            known_object_ids: Set of valid object IDs from rl_artifacts.task_objects.
                If provided, every non-null, non-END_EFFECTOR object_id in reward
                terms is validated against this set.
            valid_stage_ids: Set of stage IDs from the parsed CurriculumSpec. If
                provided, all active_in_stages values are validated against this set.
                Parse curriculum_spec FIRST and pass its stage IDs here.
        """
        terms = [RewardTerm.from_dict(t) for t in data.get("reward_terms", [])]

        if known_object_ids is not None or valid_stage_ids is not None:
            errors = []
            for term in terms:
                for field_name, oid in [
                    ("from_object_id", term.from_object_id),
                    ("to_object_id", term.to_object_id),
                ]:
                    if oid is None or oid == END_EFFECTOR:
                        continue
                    if known_object_ids is not None and oid not in known_object_ids:
                        errors.append(
                            f"Term '{term.name}': {field_name}='{oid}' not in "
                            f"rl_artifacts task_objects. Known IDs: {sorted(known_object_ids)}"
                        )
                if valid_stage_ids is not None:
                    for sid in term.active_in_stages:
                        if sid not in valid_stage_ids:
                            errors.append(
                                f"Term '{term.name}': active_in_stages contains "
                                f"stage ID {sid} not in curriculum. "
                                f"Valid stage IDs: {sorted(valid_stage_ids)}"
                            )
            if errors:
                raise ValueError(
                    "RewardSpec validation failed:\n"
                    + "\n".join(f"  - {e}" for e in errors)
                )

        return cls(
            reward_terms=terms,
            notes=data.get("notes", ""),
        )

    def save(self, path: Path) -> None:
        """Save to YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=True)
        console_logger.info("Reward spec saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "RewardSpec":
        """Load from YAML file (no validation — use from_dict for that)."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


@dataclass
class CurriculumStage:
    """A single stage in the RL training curriculum."""

    id: int
    """Stage index (0-based). Used in reward term active_in_stages lists."""

    name: str
    """Human-readable stage name, e.g. 'single_object_no_dr'."""

    active_task_object_ids: list[str]
    """Subset of rl_artifacts task_objects active in this stage."""

    dr_scale: float
    """Domain randomization scale 0.0–1.0. Multiplies position_range_xy_on_surface
    from rl_artifacts.domain_randomization. 0.0 = no randomization."""

    clutter_enabled: bool
    """Whether clutter_ids from rl_artifacts.domain_randomization are added."""

    tolerance_m: float
    """Success distance tolerance in meters (may be tighter than rl_artifacts default)."""

    transition_success_rate: float
    """Success rate threshold over the last transition_window_episodes to advance."""

    transition_window_episodes: int
    """Rolling window size for computing success rate."""

    min_episodes_in_stage: int
    """Minimum episodes to spend in this stage before any transition check.
    Should be >= 2 * transition_window_episodes to prevent lucky-streak transitions."""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "active_task_object_ids": self.active_task_object_ids,
            "dr_scale": self.dr_scale,
            "clutter_enabled": self.clutter_enabled,
            "tolerance_m": self.tolerance_m,
            "transition_success_rate": self.transition_success_rate,
            "transition_window_episodes": self.transition_window_episodes,
            "min_episodes_in_stage": self.min_episodes_in_stage,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CurriculumStage":
        return cls(
            id=data["id"],
            name=data["name"],
            active_task_object_ids=data.get("active_task_object_ids", []),
            dr_scale=float(data.get("dr_scale", 0.0)),
            clutter_enabled=bool(data.get("clutter_enabled", False)),
            tolerance_m=float(data.get("tolerance_m", 0.1)),
            transition_success_rate=float(data.get("transition_success_rate", 0.8)),
            transition_window_episodes=int(data.get("transition_window_episodes", 50)),
            min_episodes_in_stage=int(data.get("min_episodes_in_stage", 100)),
        )


@dataclass
class CurriculumSpec:
    """RL training curriculum specification.

    Stages are ordered by difficulty. Each stage is learnable given mastery
    of the previous stage. Transitions trigger when success_rate over the last
    transition_window_episodes exceeds transition_success_rate AND the agent
    has spent at least min_episodes_in_stage in the current stage.
    """

    stages: list[CurriculumStage] = field(default_factory=list)
    notes: str = ""

    def stage_ids(self) -> set[int]:
        """Return the set of all stage IDs."""
        return {s.id for s in self.stages}

    def to_dict(self) -> dict:
        return {
            "notes": self.notes,
            "stages": [s.to_dict() for s in self.stages],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CurriculumSpec":
        stages = [CurriculumStage.from_dict(s) for s in data.get("stages", [])]
        return cls(
            stages=stages,
            notes=data.get("notes", ""),
        )

    def save(self, path: Path) -> None:
        """Save to YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=True)
        console_logger.info("Curriculum spec saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "CurriculumSpec":
        """Load from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
