"""Data structures for domain randomization variation specifications."""

from __future__ import annotations

import json
import logging

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

_logger = logging.getLogger(__name__)

SUPPORTED_STRATEGIES = {"programmatic", "llm_directed"}


@dataclass
class ProgrammaticOpSpec:
    """Specification for a single programmatic randomization operation."""

    op_type: Literal[
        "pose_perturbation",
        "scale_jitter",
        "object_dropout",
        "asset_substitution",
    ]
    target_object_types: list[str]
    params: dict = field(default_factory=dict)


@dataclass
class VariationSpec:
    """Full specification for a single scene variation.

    Two v1 strategies are supported: ``programmatic`` (deterministic op
    pipeline) and ``llm_directed`` (single structured LLM edit). The
    historical ``agent_rerun`` / ``hybrid`` paths plus their
    ``resume_stage`` / ``cascade_downstream`` machinery were removed in favor
    of DR v2 ``mode="agentic"`` (DRSubflowAgent), which supersedes them.
    """

    variation_id: str
    description: str
    strategy: Literal["programmatic", "llm_directed"]

    # Programmatic ops list (only used when strategy == "programmatic").
    ops: list[ProgrammaticOpSpec] | None = None

    # Natural-language prompt for the llm_directed strategy.
    prompt_injection: str | None = None

    # Task-aware constraints
    task_critical_object_ids: list[str] = field(default_factory=list)

    # Curriculum ordering (0 = easiest, 1 = hardest)
    difficulty: float | None = None

    def __post_init__(self) -> None:
        if self.strategy not in SUPPORTED_STRATEGIES:
            if self.ops:
                _logger.warning(
                    "VariationSpec %s: strategy %r not recognised; "
                    "auto-assigning 'programmatic' because ops are present.",
                    self.variation_id,
                    self.strategy,
                )
                self.strategy = "programmatic"
            else:
                raise ValueError(
                    f"VariationSpec {self.variation_id}: unsupported strategy "
                    f"'{self.strategy}'. Supported: {sorted(SUPPORTED_STRATEGIES)}."
                )


@dataclass
class VariationStatus:
    """Runtime status for a single variation."""

    spec: VariationSpec
    status: Literal["pending", "in_progress", "completed", "failed"] = "pending"
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        spec_dict = {
            "variation_id": self.spec.variation_id,
            "description": self.spec.description,
            "strategy": self.spec.strategy,
            "prompt_injection": self.spec.prompt_injection,
            "task_critical_object_ids": self.spec.task_critical_object_ids,
            "difficulty": self.spec.difficulty,
            "ops": (
                [
                    {
                        "op_type": op.op_type,
                        "target_object_types": op.target_object_types,
                        "params": op.params,
                    }
                    for op in self.spec.ops
                ]
                if self.spec.ops
                else None
            ),
        }
        return {
            "spec": spec_dict,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }


@dataclass
class OrchestrationPlan:
    """Full plan produced by the orchestrator, with per-variation status tracking."""

    created_at: str
    base_experiment_dir: str
    description: str | None
    num_variations: int
    orchestrator_reasoning: str
    variations: dict[str, VariationStatus] = field(default_factory=dict)

    def update_status(
        self,
        variation_id: str,
        status: Literal["pending", "in_progress", "completed", "failed"],
        error: str | None = None,
    ) -> None:
        vs = self.variations[variation_id]
        vs.status = status
        if status == "in_progress":
            vs.started_at = datetime.utcnow().isoformat()
        elif status in ("completed", "failed"):
            vs.completed_at = datetime.utcnow().isoformat()
        if error is not None:
            vs.error = error

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        data = {
            "created_at": self.created_at,
            "base_experiment_dir": self.base_experiment_dir,
            "description": self.description,
            "num_variations": self.num_variations,
            "orchestrator_reasoning": self.orchestrator_reasoning,
            "variations": {k: v.to_dict() for k, v in self.variations.items()},
        }
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "OrchestrationPlan":
        data = json.loads(path.read_text())
        variations: dict[str, VariationStatus] = {}
        for vid, vdata in data["variations"].items():
            spec_data = vdata["spec"]
            ops = None
            if spec_data.get("ops"):
                ops = [
                    ProgrammaticOpSpec(
                        op_type=op["op_type"],
                        target_object_types=op["target_object_types"],
                        params=op.get("params", {}),
                    )
                    for op in spec_data["ops"]
                ]
            # Tolerate legacy plan.json files that still carry the
            # historical resume_stage / cascade_downstream fields — they're
            # ignored after the Step 10 cleanup. Map legacy agent_rerun /
            # hybrid strategies to llm_directed so old plans still load,
            # but log a warning so callers know the runtime semantics changed.
            legacy_strategy = spec_data["strategy"]
            strategy = legacy_strategy
            if legacy_strategy in ("agent_rerun", "hybrid"):
                _logger.warning(
                    "OrchestrationPlan %s: strategy %r is no longer supported; "
                    "mapping to 'llm_directed' on load.",
                    spec_data["variation_id"],
                    legacy_strategy,
                )
                strategy = "llm_directed"
            spec = VariationSpec(
                variation_id=spec_data["variation_id"],
                description=spec_data["description"],
                strategy=strategy,
                ops=ops,
                prompt_injection=spec_data.get("prompt_injection"),
                task_critical_object_ids=spec_data.get("task_critical_object_ids", []),
                difficulty=spec_data.get("difficulty"),
            )
            variations[vid] = VariationStatus(
                spec=spec,
                status=vdata["status"],
                started_at=vdata.get("started_at"),
                completed_at=vdata.get("completed_at"),
                error=vdata.get("error"),
            )
        return cls(
            created_at=data["created_at"],
            base_experiment_dir=data["base_experiment_dir"],
            description=data.get("description"),
            num_variations=data["num_variations"],
            orchestrator_reasoning=data["orchestrator_reasoning"],
            variations=variations,
        )


@dataclass
class VariationGoal:
    """Top-level goal emitted by the Director for a single DR variation.

    Carries NL intent + soft constraints + locked IDs; does NOT prescribe ops.
    The Subflow Planner decides how to satisfy the goal; the Critic judges alignment.
    """

    variation_id: str
    description: str
    soft_constraints: dict[str, Any]
    task_critical_object_ids: list[str]
    difficulty: float
    seed: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.difficulty <= 1.0:
            raise ValueError(
                f"VariationGoal {self.variation_id}: difficulty must be in [0, 1], "
                f"got {self.difficulty}"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VariationGoal":
        return cls(**d)
