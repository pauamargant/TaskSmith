#!/usr/bin/env python3
"""Standalone script: generate reward_spec.yaml + curriculum_spec.yaml from an existing scene.

Usage:
    python scripts/generate_reward_spec.py <scene_dir> [--model MODEL]

Example:
    python scripts/generate_reward_spec.py outputs/latest-run/scene_000
"""

import argparse
import json
import logging
import os
import sys

from pathlib import Path

from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reward and curriculum specs."
    )
    parser.add_argument("scene_dir", type=Path, help="Path to scene_000/ directory")
    parser.add_argument(
        "--model",
        default=os.environ.get("PROVIDER_MODEL", "openai/gpt-oss-120b"),
        help="LLM model ID (default: $PROVIDER_MODEL or openai/gpt-oss-120b)",
    )
    parser.add_argument(
        "--service-tier",
        default=None,
        choices=[None, "default", "flex", "priority"],
        help="API service tier",
    )
    args = parser.parse_args()

    scene_dir: Path = args.scene_dir.resolve()

    rl_artifacts_path = scene_dir / "rl_artifacts.json"
    task_req_path = scene_dir / "task_requirements.json"

    for p in [rl_artifacts_path, task_req_path]:
        if not p.exists():
            log.error("Missing required file: %s", p)
            sys.exit(1)

    cfg = OmegaConf.create(
        {
            "provider": {
                "model": args.model,
            },
            "openai": {
                "service_tier": args.service_tier,
            },
            "reward_spec_agent": {
                "openai": {
                    "model": args.model,
                    "service_tier": args.service_tier,
                }
            },
        }
    )

    from scenesmith.provider_validation import initialize_provider_runtime
    from scenesmith.reward_spec_agent.reward_spec_agent import RewardSpecAgent
    from scenesmith.task_analysis_agent.task_requirements import TaskRequirements

    initialize_provider_runtime(cfg)

    task_requirements = TaskRequirements.load(task_req_path)
    rl_artifacts = json.loads(rl_artifacts_path.read_text())

    log.info("Task: %s", task_requirements.task_summary)
    log.info("Model: %s", args.model)

    agent = RewardSpecAgent(cfg=cfg)
    reward_spec, curriculum_spec = agent.generate(
        rl_artifacts=rl_artifacts,
        task_requirements=task_requirements,
    )

    reward_spec.save(scene_dir / "reward_spec.yaml")
    curriculum_spec.save(scene_dir / "curriculum_spec.yaml")

    log.info("Written: %s", scene_dir / "reward_spec.yaml")
    log.info("Written: %s", scene_dir / "curriculum_spec.yaml")
    log.info(
        "  %d reward terms, %d curriculum stages",
        len(reward_spec.reward_terms),
        len(curriculum_spec.stages),
    )


if __name__ == "__main__":
    main()
