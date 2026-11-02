#!/usr/bin/env python3
"""Standalone script: generate morphology_task_config.json from an existing scene.

Reads rl_artifacts.json (required), reward_spec.yaml (optional), and
curriculum_spec.yaml (optional) from a scene directory. Writes
morphology_task_config.json to the same directory.

Usage:
    python scripts/generate_morphology_task_config.py <scene_dir> [--model MODEL]

Example:
    python scripts/generate_morphology_task_config.py outputs/latest-run/scene_000
"""

import argparse
import json
import logging
import os
import sys

from pathlib import Path

import yaml

from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the best morphological training task from a TaskSmith scene."
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
    if not rl_artifacts_path.exists():
        log.error("Missing required file: %s", rl_artifacts_path)
        sys.exit(1)

    rl_artifacts = json.loads(rl_artifacts_path.read_text())

    # Load optional reward_spec and curriculum_spec if present.
    reward_spec = None
    reward_spec_path = scene_dir / "reward_spec.yaml"
    if reward_spec_path.exists():
        with open(reward_spec_path) as f:
            reward_spec = yaml.safe_load(f)
        log.info("Using reward_spec.yaml: %s", reward_spec_path)
    else:
        log.info("reward_spec.yaml not found — agent will use default reward weights")

    curriculum_spec = None
    curriculum_spec_path = scene_dir / "curriculum_spec.yaml"
    if curriculum_spec_path.exists():
        with open(curriculum_spec_path) as f:
            curriculum_spec = yaml.safe_load(f)
        log.info("Using curriculum_spec.yaml: %s", curriculum_spec_path)
    else:
        log.info("curriculum_spec.yaml not found — skipping")

    from scenesmith.morphology_task_agent.morphology_task_agent import (
        MorphologyTaskAgent,
    )
    from scenesmith.provider_validation import initialize_provider_runtime

    cfg = OmegaConf.create(
        {
            "provider": {
                "model": args.model,
            },
            "openai": {
                "service_tier": args.service_tier,
            },
            "morphology_task_agent": {
                "openai": {
                    "model": args.model,
                    "service_tier": args.service_tier,
                }
            },
        }
    )
    initialize_provider_runtime(cfg)

    initial_objects = [
        o["object_id"]
        for o in rl_artifacts.get("task_objects", [])
        if o.get("is_initial_object")
    ]
    log.info(
        "Scene has %d initial object(s): %s", len(initial_objects), initial_objects
    )
    log.info("Model: %s", args.model)

    agent = MorphologyTaskAgent(cfg=cfg)
    config = agent.generate(
        rl_artifacts=rl_artifacts,
        reward_spec=reward_spec,
        curriculum_spec=curriculum_spec,
    )

    output_path = scene_dir / "morphology_task_config.json"
    config.save(output_path)

    log.info("Written: %s", output_path)
    log.info(
        "  Selected object: %s (%s)",
        config.selected_object_id,
        config.selected_object_name,
    )
    log.info("  Task phases: %s", config.task_phases)
    log.info(
        "  Rationale: %s",
        (
            config.rationale[:120] + "..."
            if len(config.rationale) > 120
            else config.rationale
        ),
    )


if __name__ == "__main__":
    main()
