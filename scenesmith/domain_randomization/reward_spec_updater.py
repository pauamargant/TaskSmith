"""Classifies and updates reward specs after scene variations are generated."""

from __future__ import annotations

import logging
import shutil

from pathlib import Path
from typing import Literal

from scenesmith.agent_utils.room import RoomScene
from scenesmith.domain_randomization.variation_spec import VariationSpec

console_logger = logging.getLogger(__name__)

RewardSpecImpact = Literal["unchanged", "patch", "regenerate"]


def classify_reward_spec_impact(
    spec: VariationSpec,
    task_critical_ids: set[str],
) -> RewardSpecImpact:
    """Decide how the reward spec needs to change for this variation.

    Rules:
    - asset_substitution on task-critical objects → structural change → regenerate
    - pose_perturbation / scale_jitter on task-critical objects → patch positions
    - everything else (including llm_directed and DR v2 mode='agentic' subflows)
      → unchanged; the caller is responsible for rewriting the reward spec from
      the mutated scene if positions changed

    Args:
        spec: The variation specification.
        task_critical_ids: IDs of task-critical objects.

    Returns:
        One of "unchanged", "patch", or "regenerate".
    """
    if spec.ops:
        for op in spec.ops:
            if op.op_type in ("asset_substitution",) and task_critical_ids:
                return "regenerate"
            if (
                op.op_type in ("pose_perturbation", "scale_jitter")
                and task_critical_ids
            ):
                return "patch"

    return "unchanged"


async def update_reward_spec(
    impact: RewardSpecImpact,
    variation_scene_dir: Path,
    base_scene_dir: Path,
    new_scene: RoomScene,
    task_critical_ids: set[str],
    cfg: "DictConfig | None" = None,
    rl_artifacts: dict | None = None,
    task_requirements: "TaskRequirements | None" = None,
) -> None:
    """Apply the appropriate reward spec update strategy for a variation.

    Args:
        impact: Classification from classify_reward_spec_impact.
        variation_scene_dir: The variation's scene directory.
        base_scene_dir: The base scene directory to copy from.
        new_scene: The modified RoomScene (used for patching positions).
        task_critical_ids: IDs of task-critical objects.
        cfg: Hydra DictConfig (needed for regenerate strategy).
        rl_artifacts: Parsed rl_artifacts.json (needed for regenerate strategy).
        task_requirements: TaskRequirements (needed for regenerate strategy).
    """
    base_reward_spec = base_scene_dir / "reward_spec.yaml"
    base_curriculum = base_scene_dir / "curriculum_spec.yaml"

    if impact == "unchanged":
        _copy_if_exists(base_reward_spec, variation_scene_dir / "reward_spec.yaml")
        _copy_if_exists(base_curriculum, variation_scene_dir / "curriculum_spec.yaml")
        console_logger.debug("reward_spec: copied base (unchanged)")

    elif impact == "patch":
        if base_reward_spec.exists():
            _patch_reward_spec_positions(
                base_reward_spec=base_reward_spec,
                output_path=variation_scene_dir / "reward_spec.yaml",
                scene=new_scene,
                task_critical_ids=task_critical_ids,
            )
        _copy_if_exists(base_curriculum, variation_scene_dir / "curriculum_spec.yaml")
        console_logger.debug("reward_spec: patched positions")

    elif impact == "regenerate":
        if cfg is None or rl_artifacts is None or task_requirements is None:
            console_logger.warning(
                "reward_spec: regenerate requested but cfg/rl_artifacts/"
                "task_requirements not provided — copying base instead"
            )
            _copy_if_exists(base_reward_spec, variation_scene_dir / "reward_spec.yaml")
            _copy_if_exists(
                base_curriculum, variation_scene_dir / "curriculum_spec.yaml"
            )
            return

        from scenesmith.reward_spec_agent.reward_spec_agent import RewardSpecAgent

        agent = RewardSpecAgent(cfg=cfg)
        reward_spec, curriculum_spec = agent.generate(
            rl_artifacts=rl_artifacts,
            task_requirements=task_requirements,
        )
        reward_spec.save(variation_scene_dir / "reward_spec.yaml")
        curriculum_spec.save(variation_scene_dir / "curriculum_spec.yaml")
        console_logger.debug("reward_spec: regenerated via RewardSpecAgent")


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy(src, dst)


def _patch_reward_spec_positions(
    base_reward_spec: Path,
    output_path: Path,
    scene: RoomScene,
    task_critical_ids: set[str],
) -> None:
    """Update position_xyz in reward spec to match new scene object positions.

    Reads the base reward spec YAML, finds position references for task-critical
    objects, updates them to the new world positions, and writes to output_path.
    """
    import yaml

    with open(base_reward_spec) as f:
        spec_data = yaml.safe_load(f)

    object_positions = {
        obj.object_id: [round(float(v), 4) for v in obj.transform.translation()]
        for obj in scene.objects.values()
        if obj.object_id in task_critical_ids
    }

    def patch_recursive(node: object) -> object:
        if isinstance(node, dict):
            if "object_id" in node and node["object_id"] in object_positions:
                node = dict(node)
                node["position_xyz"] = object_positions[node["object_id"]]
            return {k: patch_recursive(v) for k, v in node.items()}
        if isinstance(node, list):
            return [patch_recursive(item) for item in node]
        return node

    patched = patch_recursive(spec_data)
    with open(output_path, "w") as f:
        yaml.dump(patched, f, default_flow_style=False, sort_keys=False)
