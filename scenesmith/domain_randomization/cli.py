"""CLI entry point for the domain randomization orchestrator.

Usage (Hydra):
    python -m scenesmith.domain_randomization.cli \
        +name=dr_batch_01 \
        +domain_randomization.base_experiment_dir=/path/to/experiment \
        domain_randomization.num_variations=15 \
        '+domain_randomization.description=vary clutter on desk'

    python -m scenesmith.domain_randomization.cli \
        +name=dr_batch_01 \
        +domain_randomization.base_experiment_dir=/path/to/experiment \
        +domain_randomization.description_file=/path/to/instructions.txt

Note: base_experiment_dir and description_file are null in the base config and
must be added with '+'. Keys with existing defaults (num_variations, mode,
exports.*) do not need '+'.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from pathlib import Path

import hydra

from omegaconf import DictConfig, OmegaConf

from scenesmith.domain_randomization.export import ExportConfig
from scenesmith.domain_randomization.orchestrator import DomainRandomizationOrchestrator
from scenesmith.domain_randomization.variation_spec import OrchestrationPlan
from scenesmith.experiment_tracking import init_tracking_from_cfg
from scenesmith.provider_validation import initialize_provider_runtime
from scenesmith.utils.logging import ConsoleLogger
from scenesmith.utils.omegaconf import register_resolvers

console_logger = logging.getLogger(__name__)


@hydra.main(
    config_path="../../configurations",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Hydra entry point."""
    register_resolvers()
    OmegaConf.resolve(cfg)
    init_tracking_from_cfg(cfg)
    initialize_provider_runtime(cfg)

    dr_cfg = cfg.get("domain_randomization", {})

    base_dir = dr_cfg.get("base_experiment_dir")
    if base_dir is None:
        console_logger.error(
            "domain_randomization.base_experiment_dir must be set. "
            "Pass it with +domain_randomization.base_experiment_dir=/path/to/experiment"
        )
        sys.exit(1)

    num_variations = int(dr_cfg.get("num_variations", 10))
    description = _load_description(dr_cfg)

    exports = dr_cfg.get("exports", {})
    export_cfg = ExportConfig(
        drake=exports.get("drake", True),
        blender=exports.get("blender", False),
    )

    logger = ConsoleLogger(output_dir=Path(base_dir))
    existing_plan = _load_existing_plan(Path(base_dir), dr_cfg)

    if (
        existing_plan is not None
        and str(dr_cfg.get("mode", "deterministic")) == "agentic"
    ):
        console_logger.error(
            "domain_randomization.mode='agentic' does not support resume from an "
            "existing plan. To start a fresh agentic run, either omit "
            "domain_randomization.output_dir (a timestamped directory is created "
            "automatically) or point it at a new empty directory."
        )
        sys.exit(1)

    orchestrator = DomainRandomizationOrchestrator(
        base_experiment_dir=Path(base_dir),
        num_variations=num_variations,
        cfg=cfg,
        logger=logger,
        description=description,
        export_cfg=export_cfg,
    )

    plan = asyncio.run(orchestrator.run(existing_plan=existing_plan))

    completed = sum(1 for v in plan.variations.values() if v.status == "completed")
    failed = sum(1 for v in plan.variations.values() if v.status == "failed")
    console_logger.info(
        f"Done: {completed}/{len(plan.variations)} completed, {failed} failed. "
        f"Output: {orchestrator._output_dir}"
    )


def _load_existing_plan(
    base_experiment_dir: Path, dr_cfg: DictConfig | dict
) -> OrchestrationPlan | None:
    """Load an existing orchestration plan when output_dir points at a prior run."""
    output_dir = dr_cfg.get("output_dir")
    if not output_dir:
        return None

    plan_path = Path(output_dir) / "orchestration_plan.json"
    if not plan_path.exists():
        return None

    plan = OrchestrationPlan.load(plan_path)
    console_logger.info("Loaded existing orchestration plan for resume: %s", plan_path)

    if plan.base_experiment_dir != str(base_experiment_dir):
        raise ValueError(
            "Loaded orchestration plan base_experiment_dir does not match "
            f"requested base directory: {plan.base_experiment_dir} != {base_experiment_dir}"
        )

    return plan


def _load_description(dr_cfg: DictConfig | dict) -> str | None:
    """Load the variation guidance string from the config's ``description`` field or a text file."""
    description_file = dr_cfg.get("description_file")
    if description_file:
        description_path = Path(description_file)
        description = description_path.read_text().strip()
        if not description:
            raise ValueError(
                f"domain_randomization.description_file is empty: {description_path}"
            )
        return description

    return dr_cfg.get("description")


if __name__ == "__main__":
    main()
