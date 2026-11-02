import asyncio
import contextlib
import csv
import faulthandler
import json
import logging
import os
import shutil
import time
import uuid

from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Callable, Iterator

import requests
import yaml

from agents import custom_span, trace
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict

from scenesmith.agent_utils.articulated_retrieval_server import (
    ArticulatedRetrievalServer,
)
from scenesmith.agent_utils.geometry_generation_server import GeometryGenerationServer
from scenesmith.agent_utils.house import HouseLayout, HouseScene, RoomGeometry
from scenesmith.agent_utils.hssd_retrieval_server import HssdRetrievalServer
from scenesmith.agent_utils.materials_retrieval_server import MaterialsRetrievalServer
from scenesmith.agent_utils.objaverse_retrieval_server import ObjaverseRetrievalServer
from scenesmith.agent_utils.physical_feasibility import (
    apply_physical_feasibility_postprocessing,
)
from scenesmith.agent_utils.room import AgentType, ObjectType, RoomScene
from scenesmith.agent_utils.sceneeval_exporter import (
    SceneEvalExportConfig,
    SceneEvalExporter,
)
from scenesmith.ceiling_agents.stateful_ceiling_agent import StatefulCeilingAgent
from scenesmith.experiment_tracking import (
    TrackingContext,
    analyze_scene,
    emit_artifact,
    emit_stage_event,
    tracking_context,
    utc_now_iso,
)
from scenesmith.experiment_tracking.analysis_renders import (
    finalize_scene_analysis,
    first_analysis_render_paths,
    render_rooms_for_analysis,
)
from scenesmith.experiment_tracking.context import (
    extend_tracking_context,
    get_tracking_context,
)
from scenesmith.experiment_tracking.live_wandb import live_wandb_tracking
from scenesmith.experiments.base_experiment import BaseExperiment
from scenesmith.floor_plan_agents.stateful_floor_plan_agent import (
    StatefulFloorPlanAgent,
)
from scenesmith.furniture_agents.stateful_furniture_agent import StatefulFurnitureAgent
from scenesmith.manipuland_agents.stateful_manipuland_agent import (
    StatefulManipulandAgent,
)
from scenesmith.morphology_task_agent.morphology_task_agent import MorphologyTaskAgent
from scenesmith.reward_spec_agent.reward_spec_agent import RewardSpecAgent
from scenesmith.task_analysis_agent.rl_artifact_exporter import RLArtifactAgent
from scenesmith.task_analysis_agent.task_analysis_agent import TaskAnalysisAgent
from scenesmith.task_analysis_agent.task_requirements import TaskRequirements
from scenesmith.utils.logging import ConsoleLogger, FileLoggingContext
from scenesmith.utils.network_utils import find_available_port, is_port_available
from scenesmith.utils.parallel import run_parallel_isolated
from scenesmith.utils.print_utils import bold_green, yellow
from scenesmith.wall_agents.stateful_wall_agent import StatefulWallAgent

console_logger = logging.getLogger(__name__)

# Room-level pipeline stages (subset of AgentType values, excludes floor_plan and dr_subflow).
# DR_SUBFLOW is an AgentType enum member used for ordering/context but is never dispatched
# as a runnable stage in the main scene generation pipeline.
_ROOM_STAGES: list[str] = [
    agent.value
    for agent in AgentType
    if agent not in (AgentType.FLOOR_PLAN, AgentType.DR_SUBFLOW)
]

# Full pipeline stages in execution order: room stages + post-processing.
# DR_SUBFLOW is excluded — it is not a valid value for start_stage, stop_stage, or skip_stages.
PIPELINE_STAGES: list[str] = [
    agent.value for agent in AgentType if agent != AgentType.DR_SUBFLOW
] + [
    "reward_spec",
    "morphology_task",
]

# Indices of post-processing stages — computed once to avoid repeated .index() calls.
_REWARD_SPEC_IDX: int = PIPELINE_STAGES.index("reward_spec")
_MORPHOLOGY_TASK_IDX: int = PIPELINE_STAGES.index("morphology_task")

RESUME_MANIFEST_NAME = "resume_manifest.json"

# Stage dependencies for resume from checkpoint.
# Maps start_stage to the checkpoint it needs from the previous stage.
STAGE_CHECKPOINTS = {
    "floor_plan": None,
    "furniture": None,
    "wall_mounted": "scene_after_furniture",
    "ceiling_mounted": "scene_after_wall_objects",
    "manipuland": "scene_after_ceiling_objects",
    "reward_spec": None,  # reads rl_artifacts.json + task_requirements.json from disk
    "morphology_task": None,  # reads scene-level artifacts directly from disk
}

# Maps start_stage to the asset directories it needs from previous stages.
STAGE_ASSET_DIRS = {
    "floor_plan": [],
    "furniture": [],
    "wall_mounted": ["furniture"],
    "ceiling_mounted": ["furniture", "wall_mounted"],
    "manipuland": ["furniture", "wall_mounted", "ceiling_mounted"],
    "reward_spec": [],  # no room-level assets needed
    "morphology_task": [],  # no room-level assets needed
}

_ROOM_STAGE_RETAINED_ASSET_DIRS: dict[str, set[str]] = {
    "furniture": set(),
    "wall_mounted": {"furniture"},
    "ceiling_mounted": {"furniture", "wall_mounted"},
    "manipuland": {"furniture", "wall_mounted", "ceiling_mounted"},
}

_ROOM_STAGE_RETAINED_CHECKPOINT_DIRS: dict[str, set[str]] = {
    "furniture": set(),
    "wall_mounted": {"scene_after_furniture"},
    "ceiling_mounted": {"scene_after_wall_objects"},
    "manipuland": {"scene_after_ceiling_objects"},
}

_SCENE_LEVEL_DOWNSTREAM_OUTPUTS = (
    "rl_artifacts.json",
    "reward_spec.yaml",
    "curriculum_spec.yaml",
    "morphology_task_config.json",
)


def _get_retrieval_gpu_device() -> str | None:
    """Get GPU device for retrieval servers.

    If multiple GPUs available (as seen by PyTorch), returns the last
    logical GPU index to avoid competing with Blender and geometry
    generation (which use lower-indexed GPUs).

    This respects CUDA_VISIBLE_DEVICES - PyTorch remaps physical GPUs
    to logical indices 0, 1, 2, ... so we use the last logical index.

    Falls back to "cpu" if the GPU architecture is not supported by the
    installed PyTorch build (e.g. RTX 5080 / sm_120 on torch <2.8).

    Returns:
        Device string like "cuda:7", "cpu", or None (single GPU, use default).
    """
    try:
        # Import torch inside function to avoid CUDA initialization before
        # ProcessPoolExecutor forks workers (fork-after-CUDA causes corruption).
        import torch

        if not torch.cuda.is_available():
            return None

        # Check if the GPU architecture is supported by this PyTorch build.
        # Compute capability is returned as (major, minor); convert to sm_XY.
        major, minor = torch.cuda.get_device_capability(0)
        sm_arch = f"sm_{major}{minor}"
        if sm_arch not in torch.cuda.get_arch_list():
            console_logger.warning(
                f"GPU architecture {sm_arch} is not supported by the installed "
                f"PyTorch build (supported: {torch.cuda.get_arch_list()}). "
                "Falling back to CPU for CLIP retrieval servers."
            )
            return "cpu"

        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            # Use the last logical GPU for retrieval servers.
            return f"cuda:{gpu_count - 1}"
    except ImportError:
        pass
    return None


class RenderGPUAllocator:
    """Round-robin GPU allocator for distributing Blender rendering.

    Assigns GPUs in round-robin order for BlenderServer instances. This enables
    parallel scene generation without GPU memory exhaustion by spreading the
    rendering load across multiple GPUs.

    Thread-safe for concurrent allocation from multiple workers.
    """

    def __init__(self) -> None:
        self._gpus = self._detect_gpus()
        self._counter = 0
        self._lock = Lock()
        console_logger.info(f"RenderGPUAllocator initialized with GPUs: {self._gpus}")

    def _detect_gpus(self) -> list[int]:
        """Detect available GPU indices, respecting CUDA_VISIBLE_DEVICES."""
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible:
            # Parse comma-separated GPU indices from CUDA_VISIBLE_DEVICES.
            try:
                return [int(x.strip()) for x in cuda_visible.split(",")]
            except ValueError:
                console_logger.warning(
                    f"Failed to parse CUDA_VISIBLE_DEVICES='{cuda_visible}', "
                    "falling back to device file detection"
                )

        # Detect from /dev/nvidia* device files.
        gpus = []
        for i in range(16):
            if Path(f"/dev/nvidia{i}").exists():
                gpus.append(i)
        return gpus if gpus else [0]  # Default to GPU 0 if none detected.

    def allocate(self) -> int:
        """Get next GPU in round-robin order.

        Returns:
            GPU device index for BlenderServer.
        """
        with self._lock:
            gpu = self._gpus[self._counter % len(self._gpus)]
            self._counter += 1
            return gpu

    @property
    def available_gpus(self) -> list[int]:
        """Get list of available GPU indices."""
        return self._gpus.copy()


def _reset_inherited_sdk_state() -> None:
    """Reset OpenAI Agents SDK state inherited via fork.

    After fork(), the child inherits corrupted SDK state:
    1. Active trace/span ContextVars - makes workers think they're in parent's trace
    2. BatchTraceProcessor with orphaned threading.Lock and dead background thread
    3. BackendSpanExporter with corrupted httpx.Client connections

    We clear all of these so workers start fresh. Workers can reinitialize
    tracing if needed.

    Must be called at the start of each worker function.
    """
    from agents.tracing import scope

    # Clear any inherited trace/span context so workers start fresh.
    scope._current_trace.set(None)
    scope._current_span.set(None)

    # Clear the corrupted processor from the provider's processor list.
    # After fork(), the BatchTraceProcessor has orphaned locks and dead background thread.
    # The provider holds a reference to it via _multi_processor._processors.
    # We clear that list so traces won't try to use the corrupted processor.
    # Traces will still work, just won't be exported (which is fine for subprocesses).
    try:
        from agents.tracing import setup as tracing_setup

        provider = tracing_setup.GLOBAL_TRACE_PROVIDER
        if provider and hasattr(provider, "_multi_processor"):
            provider._multi_processor.set_processors([])
    except Exception:
        pass  # Best effort - don't crash on reset failure.


def _load_prompts_from_csv(csv_path: str) -> list[tuple[int, str]]:
    """Load scene prompts from CSV file.

    Args:
        csv_path: Path to CSV file with columns: scene_index, prompt.

    Returns:
        List of (scene_id, prompt) tuples.

    Raises:
        FileNotFoundError: If CSV file does not exist.
        ValueError: If CSV has invalid format or data.
    """
    prompts_with_ids = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row.
        # Start at 2 (after header).
        for row_num, row in enumerate(reader, start=2):
            if len(row) < 2:
                raise ValueError(f"CSV row {row_num} has fewer than 2 columns: {row}")
            try:
                scene_id = int(row[0])
            except ValueError:
                raise ValueError(
                    f"CSV row {row_num}: scene_index '{row[0]}' is not a valid integer"
                )
            prompt = row[1]
            prompts_with_ids.append((scene_id, prompt))
    return prompts_with_ids


def _export_scene_blend_file(
    scene: RoomScene, scene_dir: Path, cfg_dict: dict, name: str = "final_scene"
) -> None:
    """Export scene to a .blend file.

    Args:
        scene: The scene to export.
        scene_dir: Base directory for scene outputs.
        cfg_dict: Configuration dictionary.
        name: Name for the scene state subdirectory.
    """
    from scenesmith.agent_utils.rendering import save_scene_as_blend

    blend_output_path = scene_dir / "scene_states" / name / "scene.blend"
    try:
        rendering_cfg = cfg_dict.get("furniture_agent", {}).get("rendering", {})
        save_scene_as_blend(
            scene=scene,
            output_path=blend_output_path,
            blender_server_host=rendering_cfg.get("blender_server_host", "127.0.0.1"),
            blender_server_port_range=tuple(
                rendering_cfg.get("blender_server_port_range", [8000, 8050])
            ),
            server_startup_delay=rendering_cfg.get("server_startup_delay", 0.1),
            port_cleanup_delay=rendering_cfg.get("port_cleanup_delay", 0.1),
        )
    except Exception as e:
        console_logger.error(f"Failed to export .blend file: {e}")


def _fix_paths_in_json_file(
    json_path: Path, new_room_dir: Path, new_scene_dir: Path | None = None
) -> None:
    """Fix absolute paths in a JSON file to point to new directories.

    Scans JSON for any string values containing absolute paths and rebases them:
    - Room-level paths (generated_assets/, scene_renders/) → new_room_dir
    - Scene-level paths (room_geometry/, floor_plans/) → new_scene_dir

    Args:
        json_path: Path to JSON file to fix.
        new_room_dir: New room directory for room-level paths.
        new_scene_dir: New scene directory for scene-level paths.
                       If None, defaults to parent of new_room_dir.
    """
    if not json_path.exists():
        return

    if new_scene_dir is None:
        new_scene_dir = new_room_dir.parent

    with open(json_path) as f:
        data = json.load(f)

    def fix_path(value: str) -> str:
        """Fix a single path string if it's an absolute path."""
        if not value.startswith("/"):
            return value  # Already relative, no fix needed.

        # Room-level paths (relative to room directory).
        room_markers = ["generated_assets/", "scene_renders/", "scene_states/"]
        for marker in room_markers:
            if marker in value:
                rel_path = value.split(marker, 1)[1]
                return str(new_room_dir / marker.rstrip("/") / rel_path)

        # Scene-level paths (relative to scene directory).
        scene_markers = ["room_geometry/", "floor_plans/"]
        for marker in scene_markers:
            if marker in value:
                rel_path = value.split(marker, 1)[1]
                return str(new_scene_dir / marker.rstrip("/") / rel_path)

        return value  # Unknown pattern, leave as-is.

    def fix_paths_recursive(obj):
        """Recursively fix paths in a nested structure."""
        if isinstance(obj, dict):
            return {k: fix_paths_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [fix_paths_recursive(item) for item in obj]
        elif isinstance(obj, str):
            return fix_path(obj)
        return obj

    fixed_data = fix_paths_recursive(data)

    with open(json_path, "w") as f:
        json.dump(fixed_data, f, indent=2)

    console_logger.debug(f"Fixed paths in {json_path}")


def _fix_paths_in_yaml_file(
    yaml_path: Path, new_room_dir: Path, new_scene_dir: Path | None = None
) -> None:
    """Fix absolute paths in YAML file (e.g., scene.dmd.yaml Drake directives).

    Handles file:// URIs used in Drake model directives.

    Args:
        yaml_path: Path to YAML file to fix.
        new_room_dir: New room directory for room-level paths.
        new_scene_dir: New scene directory for scene-level paths.
                       If None, defaults to parent of new_room_dir.
    """
    import re

    if not yaml_path.exists():
        return

    if new_scene_dir is None:
        new_scene_dir = new_room_dir.parent

    content = yaml_path.read_text()

    def replace_path(match: re.Match) -> str:
        """Replace a file:// URI with the correct new path."""
        old_path = match.group(1)
        # Determine if room-level or scene-level path.
        if "/generated_assets/" in old_path or "/scene_renders/" in old_path:
            # Room-level: extract relative part after room_*/.
            rel_match = re.search(r"room_[^/]+/(.+)$", old_path)
            if rel_match:
                return f"file://{new_room_dir / rel_match.group(1)}"
        elif "/room_geometry/" in old_path or "/floor_plans/" in old_path:
            # Scene-level: extract relative part after scene_*/.
            rel_match = re.search(r"scene_\d+/(.+)$", old_path)
            if rel_match:
                return f"file://{new_scene_dir / rel_match.group(1)}"
        return match.group(0)

    new_content = re.sub(r"file://(/[^\s\"']+)", replace_path, content)
    yaml_path.write_text(new_content)
    console_logger.debug(f"Fixed paths in {yaml_path}")


def _delete_path(
    path: Path,
    *,
    base_dir: Path,
    deleted_paths: list[str],
) -> None:
    """Delete a file or directory and record its relative path once."""
    if not path.exists():
        return

    rel_path = str(path.relative_to(base_dir))
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()

    if rel_path not in deleted_paths:
        deleted_paths.append(rel_path)


def _invalidated_stages_for_resume(start_stage: str) -> list[str]:
    """Return broad pipeline stages invalidated by a resumed run."""
    start_idx = PIPELINE_STAGES.index(start_stage)
    return PIPELINE_STAGES[start_idx:]


def _remove_inherited_transient_outputs(
    scene_dir: Path,
    *,
    deleted_paths: list[str],
) -> None:
    """Remove copied runtime artifacts that should never be inherited as fresh output."""
    for pattern in ("*.db", "*.log"):
        for path in scene_dir.rglob(pattern):
            _delete_path(path, base_dir=scene_dir, deleted_paths=deleted_paths)

    for room_dir in scene_dir.iterdir():
        if not room_dir.is_dir() or not room_dir.name.startswith("room_"):
            continue
        _delete_path(
            room_dir / "action_log.json",
            base_dir=scene_dir,
            deleted_paths=deleted_paths,
        )
        _delete_path(
            room_dir / "scene_renders",
            base_dir=scene_dir,
            deleted_paths=deleted_paths,
        )
        _delete_path(
            room_dir / "simulation",
            base_dir=scene_dir,
            deleted_paths=deleted_paths,
        )


def _remove_scene_level_downstream_outputs(
    scene_dir: Path,
    *,
    deleted_paths: list[str],
) -> None:
    """Remove outputs that depend on the assembled final scene."""
    for path in scene_dir.glob("combined_house*"):
        _delete_path(path, base_dir=scene_dir, deleted_paths=deleted_paths)
    for name in _SCENE_LEVEL_DOWNSTREAM_OUTPUTS:
        _delete_path(scene_dir / name, base_dir=scene_dir, deleted_paths=deleted_paths)


def _invalidate_room_stage_outputs(
    scene_dir: Path,
    *,
    start_stage: str,
    deleted_paths: list[str],
) -> list[str]:
    """Remove stale room outputs at and after the resumed room stage."""
    retained_assets = _ROOM_STAGE_RETAINED_ASSET_DIRS[start_stage]
    retained_checkpoints = _ROOM_STAGE_RETAINED_CHECKPOINT_DIRS[start_stage]
    retained_paths: list[str] = []

    for room_dir in scene_dir.iterdir():
        if not room_dir.is_dir() or not room_dir.name.startswith("room_"):
            continue

        generated_assets_dir = room_dir / "generated_assets"
        if generated_assets_dir.exists():
            for asset_dir in generated_assets_dir.iterdir():
                if not asset_dir.is_dir():
                    continue
                if asset_dir.name in retained_assets:
                    retained_paths.append(str(asset_dir.relative_to(scene_dir)))
                    continue
                _delete_path(
                    asset_dir,
                    base_dir=scene_dir,
                    deleted_paths=deleted_paths,
                )

        scene_states_dir = room_dir / "scene_states"
        if scene_states_dir.exists():
            for state_dir in scene_states_dir.iterdir():
                if not state_dir.is_dir():
                    continue
                if state_dir.name in retained_checkpoints:
                    retained_paths.append(str(state_dir.relative_to(scene_dir)))
                    continue
                _delete_path(
                    state_dir,
                    base_dir=scene_dir,
                    deleted_paths=deleted_paths,
                )

    return retained_paths


def _rewrite_copied_resume_paths(
    source_scene_dir: Path,
    target_scene_dir: Path,
) -> list[str]:
    """Rewrite copied absolute paths to the new scene directory.

    Returns:
        Warning strings for files that still mention the source directory after
        rewrite.
    """
    warnings: list[str] = []

    for room_dir in target_scene_dir.iterdir():
        if not room_dir.is_dir() or not room_dir.name.startswith("room_"):
            continue

        for json_path in (room_dir / "scene_states").glob("*/scene_state.json"):
            _fix_paths_in_json_file(
                json_path=json_path,
                new_room_dir=room_dir,
                new_scene_dir=target_scene_dir,
            )
            if str(source_scene_dir) in json_path.read_text():
                warnings.append(
                    f"Path rewrite incomplete for {json_path.relative_to(target_scene_dir)}"
                )

        for yaml_path in (room_dir / "scene_states").glob("*/scene.dmd.yaml"):
            _fix_paths_in_yaml_file(
                yaml_path=yaml_path,
                new_room_dir=room_dir,
                new_scene_dir=target_scene_dir,
            )
            if str(source_scene_dir) in yaml_path.read_text():
                warnings.append(
                    f"Path rewrite incomplete for {yaml_path.relative_to(target_scene_dir)}"
                )

        for asset_registry in room_dir.glob("generated_assets/*/asset_registry.json"):
            _fix_paths_in_json_file(
                json_path=asset_registry,
                new_room_dir=room_dir,
                new_scene_dir=target_scene_dir,
            )
            if str(source_scene_dir) in asset_registry.read_text():
                warnings.append(
                    "Path rewrite incomplete for "
                    f"{asset_registry.relative_to(target_scene_dir)}"
                )

    return warnings


def _write_resume_manifest(
    manifest_path: Path,
    *,
    source_scene_dir: Path,
    target_scene_dir: Path,
    start_stage: str,
    stop_stage: str | None,
    skip_stages: list[str],
    copied_at: str,
    invalidated_stages: list[str],
    deleted_paths: list[str],
    retained_paths: list[str],
    warnings: list[str],
) -> None:
    """Write structured provenance for a resumed scene."""
    manifest = {
        "mode": "full_copy_resume",
        "source_experiment_dir": str(source_scene_dir.parent.parent),
        "source_scene_dir": str(source_scene_dir),
        "target_scene_dir": str(target_scene_dir),
        "start_stage": start_stage,
        "stop_stage": stop_stage,
        "skip_stages": skip_stages,
        "copied_at": copied_at,
        "invalidated_stages": invalidated_stages,
        "deleted_paths": deleted_paths,
        "retained_paths": retained_paths,
        "warnings": warnings,
        "notes": (
            f"Resumed from {source_scene_dir} into a self-contained copy. "
            f"Outputs from {start_stage} onward were invalidated before rerun."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))


def _copy_checkpoint_for_stage(
    source_scene_dir: Path,
    target_scene_dir: Path,
    start_stage: str,
    stop_stage: str | None = None,
    skip_stages: list[str] | None = None,
) -> None:
    """Create a self-contained resumed scene by full-copying then invalidating stale outputs.

    Args:
        source_scene_dir: Path to source scene directory.
        target_scene_dir: Path to target scene directory.
        start_stage: Stage to resume from (determines what to invalidate).
        stop_stage: Optional stage to stop after in the new resumed run.
        skip_stages: Optional list of stages skipped in the new resumed run.
    """
    if not source_scene_dir.exists():
        raise FileNotFoundError(
            f"Source scene directory not found: {source_scene_dir}. "
            f"Ensure resume_from_path points to an experiment with this scene."
        )

    if skip_stages is None:
        skip_stages = []

    console_logger.info(
        "Preparing self-contained resume for %s from %s",
        start_stage,
        source_scene_dir,
    )

    # Remove target if it exists (Hydra may have created it).
    if target_scene_dir.exists():
        shutil.rmtree(target_scene_dir)

    shutil.copytree(source_scene_dir, target_scene_dir)

    deleted_paths: list[str] = []
    retained_paths: list[str] = [
        "house_layout.json",
        "room_geometry/",
        "floor_plans/",
    ]
    if (target_scene_dir / "materials").exists():
        retained_paths.append("materials/")
    if (target_scene_dir / "task_requirements.json").exists():
        retained_paths.append("task_requirements.json")

    _remove_inherited_transient_outputs(
        target_scene_dir,
        deleted_paths=deleted_paths,
    )

    if start_stage in _ROOM_STAGES:
        retained_paths.extend(
            _invalidate_room_stage_outputs(
                target_scene_dir,
                start_stage=start_stage,
                deleted_paths=deleted_paths,
            )
        )
        _remove_scene_level_downstream_outputs(
            target_scene_dir,
            deleted_paths=deleted_paths,
        )
    elif start_stage == "reward_spec":
        for name in (
            "reward_spec.yaml",
            "curriculum_spec.yaml",
            "morphology_task_config.json",
        ):
            _delete_path(
                target_scene_dir / name,
                base_dir=target_scene_dir,
                deleted_paths=deleted_paths,
            )
        for name in ("rl_artifacts.json", "task_requirements.json"):
            if (target_scene_dir / name).exists():
                retained_paths.append(name)
    elif start_stage == "morphology_task":
        _delete_path(
            target_scene_dir / "morphology_task_config.json",
            base_dir=target_scene_dir,
            deleted_paths=deleted_paths,
        )
        if not (target_scene_dir / "rl_artifacts.json").exists():
            raise RuntimeError(
                "morphology_task resume requires rl_artifacts.json in the source scene "
                f"but it was not found under: {target_scene_dir}\n"
                "Run the full pipeline through at least the reward_spec stage first."
            )
        retained_paths.extend(
            [
                name
                for name in (
                    "rl_artifacts.json",
                    "reward_spec.yaml",
                    "curriculum_spec.yaml",
                    "task_requirements.json",
                )
                if (target_scene_dir / name).exists()
            ]
        )

    retained_paths = sorted(dict.fromkeys(retained_paths))
    deleted_paths = sorted(dict.fromkeys(deleted_paths))
    warnings = _rewrite_copied_resume_paths(source_scene_dir, target_scene_dir)
    invalidated_stages = _invalidated_stages_for_resume(start_stage)
    copied_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    _write_resume_manifest(
        manifest_path=target_scene_dir / RESUME_MANIFEST_NAME,
        source_scene_dir=source_scene_dir,
        target_scene_dir=target_scene_dir,
        start_stage=start_stage,
        stop_stage=stop_stage,
        skip_stages=skip_stages,
        copied_at=copied_at,
        invalidated_stages=invalidated_stages,
        deleted_paths=deleted_paths,
        retained_paths=retained_paths,
        warnings=warnings,
    )

    console_logger.info(
        "Prepared self-contained resume for %s. Deleted %d stale paths; retained %d path anchors.",
        start_stage,
        len(deleted_paths),
        len(retained_paths),
    )
    if warnings:
        for warning in warnings:
            console_logger.warning(warning)
        raise RuntimeError(
            f"Resume path rewrite incomplete for {len(warnings)} file(s) — "
            "see warnings above. Aborting to prevent stale-path failures at runtime."
        )


def _load_optional_yaml(path: Path) -> dict | None:
    """Load a YAML file when present, otherwise return None."""
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def _tracking_scene_dir() -> Path | None:
    context = get_tracking_context()
    return context.scene_dir if context is not None else None


def _record_stage_started(stage_name: str, *, notes: str | None = None) -> str:
    started_at = utc_now_iso()
    emit_stage_event(
        stage_name=stage_name,
        event_type="stage_started",
        status="running",
        started_at=started_at,
        notes=notes,
    )
    return started_at


def _record_stage_completed(
    stage_name: str,
    *,
    started_at: str,
    start_time_s: float,
    output_paths: list[str] | None = None,
    notes: str | None = None,
    status: str = "completed",
) -> None:
    ended_at = utc_now_iso()
    emit_stage_event(
        stage_name=stage_name,
        event_type="stage_completed",
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        duration_s=time.time() - start_time_s,
        output_paths=output_paths or [],
        notes=notes,
    )
    scene_dir = _tracking_scene_dir()
    if scene_dir is not None:
        analyze_scene(scene_dir)


def _record_stage_failed(
    stage_name: str,
    *,
    started_at: str,
    start_time_s: float,
    error: Exception,
) -> None:
    emit_stage_event(
        stage_name=stage_name,
        event_type="stage_failed",
        status="failed",
        started_at=started_at,
        ended_at=utc_now_iso(),
        duration_s=time.time() - start_time_s,
        notes=str(error),
    )


@contextlib.contextmanager
def _tracked_stage(
    stage_name: str,
    *,
    output_paths: list[str] | Callable[[], list[str] | None] | None = None,
    notes: str | None = None,
) -> Iterator[None]:
    with extend_tracking_context(stage_name=stage_name):
        started_at = _record_stage_started(stage_name, notes=notes)
        start_time_s = time.time()
        try:
            yield
        except Exception as exc:
            _record_stage_failed(
                stage_name,
                started_at=started_at,
                start_time_s=start_time_s,
                error=exc,
            )
            raise
        else:
            resolved_outputs = (
                output_paths() if callable(output_paths) else output_paths
            )
            _record_stage_completed(
                stage_name,
                started_at=started_at,
                start_time_s=start_time_s,
                output_paths=resolved_outputs,
            )


def _generate_morphology_task_config(
    scene_dir: Path,
    cfg_dict: dict,
    rl_artifacts_data: dict | None = None,
) -> None:
    """Generate morphology_task_config.json from scene-level artifacts."""
    rl_artifacts_path = scene_dir / "rl_artifacts.json"
    if rl_artifacts_data is None:
        if not rl_artifacts_path.exists():
            raise RuntimeError(
                "morphology_task stage requires rl_artifacts.json in scene_dir. "
                "Run the full pipeline first or use resume_from_path."
            )
        rl_artifacts_data = json.loads(rl_artifacts_path.read_text())

    reward_spec = _load_optional_yaml(scene_dir / "reward_spec.yaml")
    curriculum_spec = _load_optional_yaml(scene_dir / "curriculum_spec.yaml")

    console_logger.info("Generating morphology task config...")
    morphology_agent = MorphologyTaskAgent(cfg=cfg_dict)
    morphology_task_config = morphology_agent.generate(
        rl_artifacts=rl_artifacts_data,
        reward_spec=reward_spec,
        curriculum_spec=curriculum_spec,
    )
    morphology_task_config.save(scene_dir / "morphology_task_config.json")
    console_logger.info("Morphology task config saved to morphology_task_config.json")


def _generate_room(
    room_id: str,
    room_prompt: str,
    room_geometry: RoomGeometry,
    room_dir: Path,
    logger: ConsoleLogger,
    cfg_dict: dict,
    start_stage: str = "furniture",
    stop_stage: str = "manipuland",
    skip_stages: list[str] | None = None,
    house_layout: HouseLayout | None = None,
    render_gpu_id: int | None = None,
    rl_task_context: str = "",
) -> RoomScene:
    """Generate a single room with furniture, wall/ceiling objects, and manipulands.

    This is the core room generation function used by both single-room and
    multi-room (house) modes. It receives a pre-generated RoomGeometry from the
    HouseLayout and handles furniture, wall object, ceiling object, and
    manipuland placement.

    The room geometry is generated at the house level (by the floor plan generator)
    and passed in here. This ensures consistent handling for both single-room
    and multi-room modes.

    Pipeline stages run in order: furniture → wall_mounted → ceiling_mounted → manipuland
    (floor_plan stage is handled at house level before calling this function)

    State is always saved after each stage for resumability:
    - After furniture: scene_after_furniture.json
    - After wall_mounted: scene_after_wall_objects.json
    - After ceiling_mounted: scene_after_ceiling_objects.json
    - After manipuland: scene_after_manipulands.json (via final_scene logging)

    Args:
        room_id: Unique identifier for the room (e.g., "main", "living_room").
        room_prompt: Text description for the room.
        room_geometry: Pre-generated RoomGeometry from HouseLayout.
        room_dir: Directory for room outputs (e.g., scene_000/room_main/).
        logger: Logger instance for saving outputs.
        cfg_dict: Configuration dictionary.
        start_stage: Stage to start from ("furniture", "wall_mounted",
            "ceiling_mounted", or "manipuland").
        stop_stage: Stage to stop after ("furniture", "wall_mounted",
            "ceiling_mounted", or "manipuland").
        house_layout: Optional HouseLayout for door/window export in SceneEval.
        render_gpu_id: GPU device ID for Blender rendering. When set, uses
            bubblewrap to isolate the BlenderServer to this GPU.

    Returns:
        RoomScene with furniture, wall/ceiling objects, and (optionally) manipulands.
    """
    room_start_time = time.time()
    if skip_stages is None:
        skip_stages = []

    # Create scene and add walls and floor from room geometry.
    scene = RoomScene(
        room_geometry=room_geometry,
        scene_dir=room_dir,
        room_id=room_id,
        text_description=room_prompt,
        action_log_path=room_dir / "action_log.json",
    )
    for wall in room_geometry.walls:
        scene.add_object(wall)
    # Note: Floor is NOT added to scene.objects to avoid duplicate
    # collision geometry (room_geometry.sdf already contains floor).
    # Floor remains accessible via scene.room_geometry.floor for
    # manipuland placement queries.

    # Get stage index for comparison (room stages exclude floor_plan).
    # ["furniture", "wall_mounted", "ceiling_mounted", "manipuland"]
    room_stages = _ROOM_STAGES
    start_idx = room_stages.index(start_stage) if start_stage in room_stages else 0

    # Load projection config (needed for furniture and final post-processing).
    projection_cfg = cfg_dict["experiment"]["projection"]

    # Furniture stage.
    if start_idx <= 0:  # Run furniture if starting from furniture or earlier.
        with extend_tracking_context(stage_name="furniture"):
            stage_started_at = _record_stage_started("furniture")
            stage_start_time = time.time()
            try:
                with custom_span("furniture_placement"):
                    console_logger.info("Adding furniture to scene")
                    start_time = time.time()
                    furniture_agent = BaseExperiment.build_furniture_agent(
                        cfg_dict=cfg_dict,
                        compatible_agents=(
                            IndoorSceneGenerationExperiment.compatible_furniture_agents
                        ),
                        logger=logger,
                        render_gpu_id=render_gpu_id,
                        rl_task_context=rl_task_context,
                    )
                    try:
                        asyncio.run(furniture_agent.add_furniture(scene=scene))
                    finally:
                        # Always cleanup server subprocesses.
                        furniture_agent.cleanup()
                    end_time = time.time()
                    console_logger.info(
                        f"Furniture added to room {room_id} in "
                        f"{timedelta(seconds=end_time - start_time)}"
                    )

                # Furniture post-processing (projection + simulation).
                if projection_cfg["enabled"] and projection_cfg["furniture"]["enabled"]:
                    furniture_cfg = projection_cfg["furniture"]
                    sim_cfg = projection_cfg["simulation"]

                    # Log pre-projection state for debugging.
                    logger.log_scene(scene=scene, name="furniture_only_pre_projection")
                    emit_artifact(
                        artifact_type="scene_state",
                        path=room_dir
                        / "scene_states"
                        / "furniture_only_pre_projection"
                        / "scene_state.json",
                        stage_name="furniture",
                    )

                    console_logger.info(
                        "Running furniture post-processing (projection + simulation)"
                    )
                    start_time = time.time()

                    # Determine HTML output path for simulation.
                    furniture_sim_html_path = None
                    if sim_cfg.get("save_html", False):
                        furniture_sim_html_path = (
                            logger.output_dir
                            / "simulation"
                            / "furniture_simulation.html"
                        )

                    # Get fallen furniture config from physics_validation.
                    physics_val_cfg = cfg_dict["furniture_agent"]["physics_validation"]
                    scene, projection_success, removed_ids = (
                        apply_physical_feasibility_postprocessing(
                            scene=scene,
                            weld_furniture=False,
                            projection_enabled=True,
                            projection_influence_distance=furniture_cfg[
                                "influence_distance"
                            ],
                            projection_solver_name=furniture_cfg["solver_name"],
                            projection_iteration_limit=furniture_cfg["iteration_limit"],
                            projection_time_limit_s=furniture_cfg["time_limit_s"],
                            projection_xy_only=furniture_cfg["xy_only"],
                            projection_fix_rotation=furniture_cfg["fix_rotation"],
                            simulation_enabled=sim_cfg["enabled"],
                            simulation_time_s=sim_cfg["simulation_time_s"],
                            simulation_time_step_s=sim_cfg["time_step_s"],
                            simulation_timeout_s=sim_cfg["timeout_s"],
                            simulation_html_path=furniture_sim_html_path,
                            remove_fallen_furniture=physics_val_cfg[
                                "remove_fallen_furniture"
                            ],
                            fallen_tilt_threshold_degrees=physics_val_cfg[
                                "fallen_tilt_threshold_degrees"
                            ],
                        )
                    )
                    end_time = time.time()
                    if removed_ids:
                        console_logger.info(
                            f"Removed {len(removed_ids)} fallen furniture item(s) during "
                            f"simulation: {removed_ids}"
                        )
                    if not projection_success:
                        console_logger.error("Furniture projection solver failed")
                    else:
                        console_logger.info(
                            f"Furniture post-processing completed for room {room_id} in "
                            f"{end_time - start_time:.2f} seconds"
                        )

                # Always save state after furniture stage (unconditional for resumability).
                logger.log_scene(scene=scene, name="scene_after_furniture")
                _export_scene_blend_file(
                    scene=scene,
                    scene_dir=room_dir,
                    cfg_dict=cfg_dict,
                    name="scene_after_furniture",
                )
                emit_artifact(
                    artifact_type="scene_state",
                    path=room_dir
                    / "scene_states"
                    / "scene_after_furniture"
                    / "scene_state.json",
                    stage_name="furniture",
                )
                console_logger.info(
                    "Saved furniture checkpoint (scene_after_furniture)"
                )
                _record_stage_completed(
                    "furniture",
                    started_at=stage_started_at,
                    start_time_s=stage_start_time,
                    output_paths=[
                        str(room_dir / "scene_states" / "scene_after_furniture"),
                        str(room_dir / "action_log.json"),
                    ],
                )
            except Exception as exc:
                _record_stage_failed(
                    "furniture",
                    started_at=stage_started_at,
                    start_time_s=stage_start_time,
                    error=exc,
                )
                raise
    elif start_idx == 1:
        # Starting from wall_objects - load scene from saved furniture state.
        console_logger.info("Loading scene from saved furniture state for wall_objects")
        furniture_state_path = (
            room_dir / "scene_states" / "scene_after_furniture" / "scene_state.json"
        )
        if not furniture_state_path.exists():
            raise FileNotFoundError(
                f"Cannot start from 'wall_objects' stage: furniture state not found at "
                f"{furniture_state_path}. Run with start_stage='furniture' first."
            )
        with open(furniture_state_path) as f:
            furniture_state = json.load(f)
        scene.restore_from_state_dict(furniture_state)
        console_logger.info(
            f"Loaded {len(scene.objects)} objects from furniture checkpoint"
        )

    # Check if we should stop after furniture stage.
    if stop_stage == "furniture":
        console_logger.info("Stopping after furniture stage as configured")
        return scene

    # Wall objects stage.
    if start_idx <= 1:  # Run wall_objects if starting from wall_objects or earlier.
        with extend_tracking_context(stage_name="wall_mounted"):
            stage_started_at = _record_stage_started("wall_mounted")
            stage_start_time = time.time()
            try:
                if "wall_mounted" not in skip_stages:
                    with custom_span("wall_object_placement"):
                        console_logger.info("Adding wall-mounted objects to scene")
                        start_time = time.time()

                        # Load house_layout from parent directory (saved during floor plan stage).
                        house_layout_path = room_dir.parent / "house_layout.json"
                        if not house_layout_path.exists():
                            raise FileNotFoundError(
                                f"Cannot run wall_objects stage: house_layout.json not found at "
                                f"{house_layout_path}. This should have been saved during floor "
                                f"plan generation."
                            )
                        with open(house_layout_path) as f:
                            house_layout_dict = json.load(f)
                        house_layout = HouseLayout.from_dict(
                            house_layout_dict, house_dir=room_dir.parent
                        )

                        wall_agent = BaseExperiment.build_wall_agent(
                            cfg_dict=cfg_dict,
                            compatible_agents=IndoorSceneGenerationExperiment.compatible_wall_agents,
                            logger=logger,
                            house_layout=house_layout,
                            ceiling_height=room_geometry.wall_height,
                            wall_thickness=room_geometry.wall_thickness,
                            render_gpu_id=render_gpu_id,
                            rl_task_context=rl_task_context,
                        )
                        try:
                            asyncio.run(wall_agent.add_wall_objects(scene=scene))
                        finally:
                            # Always cleanup server subprocesses.
                            wall_agent.cleanup()
                        end_time = time.time()
                        console_logger.info(
                            f"Wall objects added to room {room_id} in "
                            f"{timedelta(seconds=end_time - start_time)}"
                        )
                else:
                    console_logger.info("Skipping wall_mounted stage as configured")

                # Always save checkpoint (even if skipped) for downstream resumability.
                logger.log_scene(scene=scene, name="scene_after_wall_objects")
                _export_scene_blend_file(
                    scene=scene,
                    scene_dir=room_dir,
                    cfg_dict=cfg_dict,
                    name="scene_after_wall_objects",
                )
                emit_artifact(
                    artifact_type="scene_state",
                    path=room_dir
                    / "scene_states"
                    / "scene_after_wall_objects"
                    / "scene_state.json",
                    stage_name="wall_mounted",
                )
                console_logger.info(
                    "Saved wall_objects checkpoint (scene_after_wall_objects)"
                )
                _record_stage_completed(
                    "wall_mounted",
                    started_at=stage_started_at,
                    start_time_s=stage_start_time,
                    output_paths=[
                        str(room_dir / "scene_states" / "scene_after_wall_objects"),
                        str(room_dir / "action_log.json"),
                    ],
                    status="skipped" if "wall_mounted" in skip_stages else "completed",
                )
            except Exception as exc:
                _record_stage_failed(
                    "wall_mounted",
                    started_at=stage_started_at,
                    start_time_s=stage_start_time,
                    error=exc,
                )
                raise
    elif start_idx == 2:
        # Starting from ceiling_mounted - load scene from saved wall_objects state.
        console_logger.info("Loading scene from saved wall_objects state for ceiling")
        wall_objects_state_path = (
            room_dir / "scene_states" / "scene_after_wall_objects" / "scene_state.json"
        )
        if not wall_objects_state_path.exists():
            raise FileNotFoundError(
                f"Cannot start from 'ceiling_mounted' stage: wall_objects state not "
                f"found at {wall_objects_state_path}. Run with "
                f"start_stage='wall_mounted' first."
            )
        with open(wall_objects_state_path) as f:
            wall_objects_state = json.load(f)
        scene.restore_from_state_dict(wall_objects_state)
        console_logger.info(
            f"Loaded {len(scene.objects)} objects from wall_objects checkpoint"
        )

    # Check if we should stop after wall_mounted stage.
    if stop_stage == AgentType.WALL_MOUNTED.value:
        console_logger.info("Stopping after wall_mounted stage as configured")
        return scene

    # Ceiling objects stage.
    if start_idx <= 2:  # Run ceiling if starting from ceiling or earlier.
        with extend_tracking_context(stage_name="ceiling_mounted"):
            stage_started_at = _record_stage_started("ceiling_mounted")
            stage_start_time = time.time()
            try:
                if "ceiling_mounted" not in skip_stages:
                    with custom_span("ceiling_object_placement"):
                        console_logger.info("Adding ceiling-mounted objects to scene")
                        start_time = time.time()

                        ceiling_agent = BaseExperiment.build_ceiling_agent(
                            cfg_dict=cfg_dict,
                            compatible_agents=(
                                IndoorSceneGenerationExperiment.compatible_ceiling_agents
                            ),
                            logger=logger,
                            ceiling_height=room_geometry.wall_height,
                            render_gpu_id=render_gpu_id,
                            rl_task_context=rl_task_context,
                        )
                        try:
                            asyncio.run(ceiling_agent.add_ceiling_objects(scene=scene))
                        finally:
                            # Always cleanup server subprocesses.
                            ceiling_agent.cleanup()
                        end_time = time.time()
                        console_logger.info(
                            f"Ceiling objects added to room {room_id} in "
                            f"{timedelta(seconds=end_time - start_time)}"
                        )
                else:
                    console_logger.info("Skipping ceiling_mounted stage as configured")

                # Always save checkpoint (even if skipped) for downstream resumability.
                logger.log_scene(scene=scene, name="scene_after_ceiling_objects")
                _export_scene_blend_file(
                    scene=scene,
                    scene_dir=room_dir,
                    cfg_dict=cfg_dict,
                    name="scene_after_ceiling_objects",
                )
                emit_artifact(
                    artifact_type="scene_state",
                    path=room_dir
                    / "scene_states"
                    / "scene_after_ceiling_objects"
                    / "scene_state.json",
                    stage_name="ceiling_mounted",
                )
                console_logger.info(
                    "Saved ceiling_objects checkpoint (scene_after_ceiling_objects)"
                )
                _record_stage_completed(
                    "ceiling_mounted",
                    started_at=stage_started_at,
                    start_time_s=stage_start_time,
                    output_paths=[
                        str(room_dir / "scene_states" / "scene_after_ceiling_objects"),
                        str(room_dir / "action_log.json"),
                    ],
                    status=(
                        "skipped" if "ceiling_mounted" in skip_stages else "completed"
                    ),
                )
            except Exception as exc:
                _record_stage_failed(
                    "ceiling_mounted",
                    started_at=stage_started_at,
                    start_time_s=stage_start_time,
                    error=exc,
                )
                raise
    else:
        # Starting from manipulands - load scene from saved ceiling_objects state.
        console_logger.info("Loading scene from saved ceiling_objects state")
        ceiling_objects_state_path = (
            room_dir
            / "scene_states"
            / "scene_after_ceiling_objects"
            / "scene_state.json"
        )
        if not ceiling_objects_state_path.exists():
            raise FileNotFoundError(
                f"Cannot start from 'manipuland' stage: ceiling_objects state not "
                f"found at {ceiling_objects_state_path}. Run with "
                f"start_stage='ceiling_mounted' first."
            )
        with open(ceiling_objects_state_path) as f:
            ceiling_objects_state = json.load(f)
        scene.restore_from_state_dict(ceiling_objects_state)
        console_logger.info(
            f"Loaded {len(scene.objects)} objects from ceiling_objects checkpoint"
        )

    # Check if we should stop after ceiling_mounted stage.
    if stop_stage == AgentType.CEILING_MOUNTED.value:
        console_logger.info("Stopping after ceiling_mounted stage as configured")
        return scene

    with extend_tracking_context(stage_name="manipuland"):
        stage_started_at = _record_stage_started("manipuland")
        stage_start_time = time.time()
        try:
            if "manipuland" not in skip_stages:
                with custom_span("manipuland_placement"):
                    console_logger.info("Adding manipulands to scene")
                    start_time = time.time()
                    manipuland_agent = BaseExperiment.build_manipuland_agent(
                        cfg_dict=cfg_dict,
                        compatible_agents=(
                            IndoorSceneGenerationExperiment.compatible_manipuland_agents
                        ),
                        logger=logger,
                        render_gpu_id=render_gpu_id,
                        rl_task_context=rl_task_context,
                    )
                    asyncio.run(manipuland_agent.add_manipulands(scene=scene))
                    end_time = time.time()
                    console_logger.info(
                        f"Manipulands added to room {room_id} in "
                        f"{timedelta(seconds=end_time - start_time)}"
                    )

                # Final post-processing (projection + simulation).
                if projection_cfg["enabled"] and projection_cfg["final"]["enabled"]:
                    final_cfg = projection_cfg["final"]
                    sim_cfg = projection_cfg["simulation"]

                    # Log pre-projection state for debugging.
                    logger.log_scene(scene=scene, name="final_scene_pre_projection")
                    emit_artifact(
                        artifact_type="scene_state",
                        path=room_dir
                        / "scene_states"
                        / "final_scene_pre_projection"
                        / "scene_state.json",
                        stage_name="manipuland",
                    )

                    console_logger.info(
                        "Running final post-processing (projection + simulation)"
                    )
                    start_time = time.time()

                    # Determine HTML output path for simulation.
                    final_sim_html_path = None
                    if sim_cfg.get("save_html", False):
                        final_sim_html_path = (
                            logger.output_dir / "simulation" / "final_simulation.html"
                        )

                    # Final post-processing: weld_furniture=True means only manipulands move.
                    # Fallen furniture removal is not needed here (furniture is welded).
                    # Get fallen manipuland config from manipuland_agent physics_validation.
                    manipuland_physics_cfg = cfg_dict["manipuland_agent"][
                        "physics_validation"
                    ]
                    scene, projection_success, removed_ids = (
                        apply_physical_feasibility_postprocessing(
                            scene=scene,
                            weld_furniture=True,
                            projection_enabled=True,
                            projection_influence_distance=final_cfg[
                                "influence_distance"
                            ],
                            projection_solver_name=final_cfg["solver_name"],
                            projection_iteration_limit=final_cfg["iteration_limit"],
                            projection_time_limit_s=final_cfg["time_limit_s"],
                            projection_xy_only=final_cfg["xy_only"],
                            projection_fix_rotation=final_cfg["fix_rotation"],
                            simulation_enabled=sim_cfg["enabled"],
                            simulation_time_s=sim_cfg["simulation_time_s"],
                            simulation_time_step_s=sim_cfg["time_step_s"],
                            simulation_timeout_s=sim_cfg["timeout_s"],
                            simulation_html_path=final_sim_html_path,
                            remove_fallen_furniture=False,
                            remove_fallen_manipulands=manipuland_physics_cfg[
                                "remove_fallen_manipulands"
                            ],
                            fallen_manipuland_floor_z=manipuland_physics_cfg[
                                "fallen_manipuland_floor_z"
                            ],
                            fallen_manipuland_near_floor_z=manipuland_physics_cfg[
                                "fallen_manipuland_near_floor_z"
                            ],
                            fallen_manipuland_z_displacement=manipuland_physics_cfg[
                                "fallen_manipuland_z_displacement"
                            ],
                        )
                    )
                    end_time = time.time()
                    if removed_ids:
                        console_logger.info(
                            f"Removed {len(removed_ids)} fallen manipuland(s) during "
                            f"final simulation: {removed_ids}"
                        )
                    if not projection_success:
                        console_logger.error("Final projection solver failed")
                    else:
                        console_logger.info(
                            f"Final post-processing completed for room {room_id} in "
                            f"{end_time - start_time:.2f} seconds"
                        )

            else:
                console_logger.info("Skipping manipuland stage as configured")
            # Log and export final scene.
            logger.log_scene(scene=scene, name="final_scene")
            _export_scene_blend_file(
                scene=scene, scene_dir=room_dir, cfg_dict=cfg_dict, name="final_scene"
            )
            emit_artifact(
                artifact_type="scene_state",
                path=room_dir / "scene_states" / "final_scene" / "scene_state.json",
                stage_name="manipuland",
            )

            # Export to SceneEval format if enabled.
            sceneeval_cfg = cfg_dict["experiment"]["sceneeval_export"]
            if sceneeval_cfg["enabled"]:
                export_config = SceneEvalExportConfig(
                    asset_id_prefix=sceneeval_cfg["asset_id_prefix"]
                )
                exporter = SceneEvalExporter(
                    scene=scene,
                    scene_dir=room_dir,
                    config=export_config,
                    house_layout=house_layout,
                )
                exporter.export()

            _record_stage_completed(
                "manipuland",
                started_at=stage_started_at,
                start_time_s=stage_start_time,
                output_paths=[
                    str(room_dir / "scene_states" / "final_scene"),
                    str(room_dir / "action_log.json"),
                ],
                status="skipped" if "manipuland" in skip_stages else "completed",
            )
        except Exception as exc:
            _record_stage_failed(
                "manipuland",
                started_at=stage_started_at,
                start_time_s=stage_start_time,
                error=exc,
            )
            raise

    console_logger.info(
        f"Room {room_id} generation completed successfully in "
        f"{timedelta(seconds=time.time() - room_start_time)}"
    )

    return scene


def _run_sequential_room_generation(
    house_layout: HouseLayout,
    logger: ConsoleLogger,
    cfg_dict: dict,
    start_stage: str,
    stop_stage: str,
    render_gpu_id: int | None = None,
    rl_task_context: str = "",
) -> dict[str, RoomScene]:
    """Generate rooms sequentially (existing behavior).

    Args:
        house_layout: HouseLayout containing room specs and geometries.
        logger: Logger for output routing.
        cfg_dict: Configuration dictionary.
        start_stage: Stage to start from.
        stop_stage: Stage to stop after.
        render_gpu_id: GPU device ID for Blender rendering. When set, uses
            bubblewrap to isolate the BlenderServer to this GPU.

    Returns:
        Dictionary mapping room_id to RoomScene.
    """
    rooms: dict[str, RoomScene] = {}
    for room_id in house_layout.room_ids:
        room_spec = house_layout.get_room_spec(room_id)
        room_geometry = house_layout.get_room_geometry(room_id)
        if room_geometry is None:
            raise RuntimeError(f"Room geometry not generated for room '{room_id}'")

        with custom_span(f"room_{room_id}_generation"):
            with logger.room_context(room_id) as room_dir:
                with extend_tracking_context(room_id=room_id):
                    console_logger.info(
                        f"Generating room '{room_id}': {room_spec.prompt}"
                    )
                    room_scene = _generate_room(
                        room_id=room_id,
                        room_prompt=room_spec.prompt,
                        room_geometry=room_geometry,
                        room_dir=room_dir,
                        logger=logger,
                        cfg_dict=cfg_dict,
                        start_stage=start_stage,
                        stop_stage=stop_stage,
                        skip_stages=cfg_dict["experiment"]["pipeline"].get(
                            "skip_stages", []
                        ),
                        house_layout=house_layout,
                        render_gpu_id=render_gpu_id,
                        rl_task_context=rl_task_context,
                    )
                    rooms[room_id] = room_scene
    return rooms


def _generate_floor_plan_worker(
    prompt: str,
    scene_dir: str,
    cfg_dict: dict,
    experiment_run_id: str | None,
    render_gpu_id: int | None = None,
    rl_task_context: str = "",
) -> None:
    """Run floor plan generation in isolated subprocess.

    This function runs in a separate process to ensure all fork-unsafe state
    (SQLiteSession locks, tracing threads) is destroyed when the subprocess
    exits, before we fork room workers.

    Args:
        prompt: Scene description prompt.
        scene_dir: Path to scene output directory (as string).
        cfg_dict: Configuration dictionary.
        experiment_run_id: Unique ID for this experiment run.
        render_gpu_id: GPU device ID for Blender rendering. When set, uses
            bubblewrap to isolate the BlenderServer to this GPU.
    """
    # Reset any SDK state inherited via fork (defense in depth).
    _reset_inherited_sdk_state()

    faulthandler.enable()

    scene_path = Path(scene_dir)
    logger = ConsoleLogger(output_dir=scene_path)
    scene_tracking = TrackingContext(
        experiment_dir=scene_path.parent,
        scene_dir=scene_path,
    )

    # Use FileLoggingContext to capture floor plan logs to scene.log.
    log_path = scene_path / "scene.log"
    with tracking_context(scene_tracking):
        with FileLoggingContext(log_file_path=log_path, suppress_stdout=True):
            console_logger.info(f"Floor plan worker started for scene: {scene_dir}")

            # Create trace metadata for this floor plan generation.
            trace_metadata = {"scene_dir": scene_dir, "prompt": prompt}
            if experiment_run_id:
                trace_metadata["experiment_run_id"] = experiment_run_id

            with trace(workflow_name="floor_plan_generation", metadata=trace_metadata):
                with custom_span("floor_plan_generation"):
                    with extend_tracking_context(stage_name="floor_plan"):
                        stage_started_at = _record_stage_started("floor_plan")
                        stage_start_time = time.time()

                        # Tabletop bypass: skip LLM floor plan entirely and build
                        # a minimal flat-floor HouseLayout programmatically.
                        if cfg_dict.get("experiment", {}).get("tabletop_mode", False):
                            from scenesmith.tabletop.floor_plan import (
                                build_tabletop_house_layout,
                            )

                            room_size = cfg_dict["experiment"].get(
                                "tabletop_room_size", [4.0, 4.0]
                            )
                            house_layout = build_tabletop_house_layout(
                                prompt=prompt,
                                scene_dir=scene_path,
                                room_width=float(room_size[0]),
                                room_length=float(room_size[1]),
                            )
                            house_layout_path = scene_path / "house_layout.json"
                            with open(house_layout_path, "w") as f:
                                json.dump(
                                    house_layout.to_dict(scene_dir=scene_path),
                                    f,
                                    indent=2,
                                )
                            emit_artifact(
                                artifact_type="house_layout",
                                path=house_layout_path,
                                stage_name="floor_plan",
                            )
                            console_logger.info(
                                f"Tabletop mode: saved minimal house layout to "
                                f"{house_layout_path}"
                            )
                            _record_stage_completed(
                                "floor_plan",
                                started_at=stage_started_at,
                                start_time_s=stage_start_time,
                                output_paths=[str(house_layout_path)],
                            )
                            return

                        floor_plan_agent = BaseExperiment.build_floor_plan_agent(
                            cfg_dict=cfg_dict,
                            compatible_agents=(
                                IndoorSceneGenerationExperiment.compatible_floor_plan_agents
                            ),
                            logger=logger,
                            render_gpu_id=render_gpu_id,
                            rl_task_context=rl_task_context,
                        )
                        try:
                            house_layout = asyncio.run(
                                floor_plan_agent.generate_house_layout(
                                    prompt=prompt,
                                    output_dir=scene_path / "floor_plans",
                                )
                            )
                        except Exception as exc:
                            _record_stage_failed(
                                "floor_plan",
                                started_at=stage_started_at,
                                start_time_s=stage_start_time,
                                error=exc,
                            )
                            raise
                        finally:
                            floor_plan_agent.cleanup()

                        # Save to disk for parent to load.
                        house_layout_path = scene_path / "house_layout.json"
                        with open(house_layout_path, "w") as f:
                            json.dump(
                                house_layout.to_dict(scene_dir=scene_path), f, indent=2
                            )
                        emit_artifact(
                            artifact_type="house_layout",
                            path=house_layout_path,
                            stage_name="floor_plan",
                        )
                        console_logger.info(
                            f"Saved house layout to {house_layout_path}"
                        )
                        _record_stage_completed(
                            "floor_plan",
                            started_at=stage_started_at,
                            start_time_s=stage_start_time,
                            output_paths=[
                                str(house_layout_path),
                                str(scene_path / "floor_plans"),
                            ],
                        )


def _generate_room_worker(
    room_id: str,
    room_prompt: str,
    room_geometry_dict: dict,
    room_dir: str,
    cfg_dict: dict,
    start_stage: str,
    stop_stage: str,
    scene_id: int,
    experiment_run_id: str | None = None,
    house_layout_dict: dict | None = None,
    render_gpu_id: int | None = None,
    rl_task_context: str = "",
) -> dict:
    """Worker function for parallel room generation.

    Runs in a subprocess. All args must be picklable (no Path, no complex objects).

    Note on tracing: Room traces are INDEPENDENT from parent scene trace because
    ProcessPoolExecutor creates separate processes. We include scene_id in metadata
    to enable correlation via trace queries.

    Args:
        room_id: Unique identifier for the room.
        room_prompt: Text description for the room.
        room_geometry_dict: Serialized RoomGeometry dictionary.
        room_dir: Path to room output directory (as string).
        cfg_dict: Configuration dictionary.
        start_stage: Stage to start from.
        stop_stage: Stage to stop after.
        scene_id: Parent scene ID for trace correlation.
        experiment_run_id: Unique ID for this experiment run.
        house_layout_dict: Optional serialized HouseLayout for door/window export.
        render_gpu_id: GPU device ID for Blender rendering. When set, uses
            bubblewrap to isolate the BlenderServer to this GPU.

    Returns:
        Dict containing scene_state and metadata for reconstruction.
    """
    # Reset any SDK state inherited via fork (defense in depth).
    _reset_inherited_sdk_state()

    room_dir_path = Path(room_dir)

    faulthandler.enable()

    log_path = room_dir_path / "room.log"

    # Create logger for this room (logs to file, not stdout).
    room_logger = ConsoleLogger(output_dir=room_dir_path)

    # Reconstruct RoomGeometry from serialized dict.
    room_geometry = RoomGeometry.from_dict(room_geometry_dict, scene_dir=room_dir_path)

    # Reconstruct HouseLayout from serialized dict (if provided).
    house_layout = None
    if house_layout_dict:
        house_layout = HouseLayout.from_dict(
            house_layout_dict, house_dir=room_dir_path.parent
        )

    # Use FileLoggingContext to capture logs to room.log.
    room_tracking = TrackingContext(
        experiment_dir=room_dir_path.parent.parent,
        scene_dir=room_dir_path.parent,
        room_id=room_id,
    )
    with tracking_context(room_tracking):
        with FileLoggingContext(log_file_path=log_path, suppress_stdout=True):
            console_logger.info(
                f"Worker started for room '{room_id}' with room prompt '{room_prompt}'"
            )

            # Create trace metadata for this room.
            trace_metadata = {
                "room_id": room_id,
                "parent_scene_id": f"scene_{scene_id:03d}",
                "experiment_name": cfg_dict["name"],
                "room_dir": str(room_dir_path),
                "room_prompt": room_prompt,
            }
            if experiment_run_id:
                trace_metadata["experiment_run_id"] = experiment_run_id

            with trace(
                workflow_name=f"scene_{scene_id:03d}_room_{room_id}",
                metadata=trace_metadata,
            ):
                room_scene = _generate_room(
                    room_id=room_id,
                    room_prompt=room_prompt,
                    room_geometry=room_geometry,
                    room_dir=room_dir_path,
                    logger=room_logger,
                    cfg_dict=cfg_dict,
                    start_stage=start_stage,
                    stop_stage=stop_stage,
                    skip_stages=cfg_dict["experiment"]["pipeline"].get(
                        "skip_stages", []
                    ),
                    house_layout=house_layout,
                    render_gpu_id=render_gpu_id,
                    rl_task_context=rl_task_context,
                )

            console_logger.info(f"Worker completed for room '{room_id}'")

    # Return serializable result for cross-process transfer.
    return {
        "scene_state": room_scene.to_state_dict(),
        "room_id": room_scene.room_id,
        "text_description": room_scene.text_description,
    }


def _reconstruct_room_scene(worker_result: dict, scene_dir: Path) -> RoomScene:
    """Reconstruct RoomScene from worker result dict.

    Args:
        worker_result: Dict containing scene_state from worker.
        scene_dir: Path to room directory for path resolution.

    Returns:
        Reconstructed RoomScene.
    """
    scene_state = worker_result["scene_state"]

    # Reconstruct RoomGeometry first (needed for RoomScene constructor).
    room_geometry = RoomGeometry.from_dict(
        scene_state["room_geometry"], scene_dir=scene_dir
    )

    # Create RoomScene with required fields.
    room_scene = RoomScene(
        room_geometry=room_geometry,
        scene_dir=scene_dir,
        room_id=worker_result["room_id"],
        text_description=worker_result.get("text_description", ""),
        action_log_path=scene_dir / "action_log.json",
    )

    # Restore objects and other state.
    room_scene.restore_from_state_dict(scene_state)

    return room_scene


def _run_parallel_room_generation(
    house_layout: HouseLayout,
    output_dir: Path,
    cfg_dict: dict,
    start_stage: str,
    stop_stage: str,
    max_workers: int,
    scene_id: int,
    experiment_run_id: str | None = None,
    render_gpu_id: int | None = None,
    rl_task_context: str = "",
) -> dict[str, RoomScene]:
    """Generate rooms in parallel with fault tolerance.

    Uses isolated processes per room instead of a shared executor pool.
    This ensures that if one room crashes, other rooms continue running.

    Args:
        house_layout: HouseLayout containing room specs and geometries.
        output_dir: Base output directory for the scene.
        cfg_dict: Configuration dictionary.
        start_stage: Stage to start from.
        stop_stage: Stage to stop after.
        max_workers: Maximum number of concurrent room processes.
        scene_id: Scene identifier for trace correlation.
        experiment_run_id: Unique ID for this experiment run.
        render_gpu_id: GPU device ID for Blender rendering. When set, uses
            bubblewrap to isolate the BlenderServer to this GPU.

    Returns:
        Dictionary mapping room_id to RoomScene.

    Raises:
        RuntimeError: If any room generation fails.
    """
    console_logger.info("Running room generation in parallel")

    # Build task list.
    tasks: list[tuple[str, Callable, dict]] = []
    room_dirs: dict[str, Path] = {}
    for room_id in house_layout.room_ids:
        room_spec = house_layout.get_room_spec(room_id)
        room_geometry = house_layout.get_room_geometry(room_id)
        if room_geometry is None:
            raise RuntimeError(f"Room geometry not generated for room '{room_id}'")

        # Create room directory (must exist before worker starts).
        room_dir = output_dir / f"room_{room_id}"
        room_dir.mkdir(parents=True, exist_ok=True)
        room_dirs[room_id] = room_dir

        console_logger.info(f"Queued room '{room_id}' (logs → {room_dir / 'room.log'})")

        kwargs = {
            "room_id": room_id,
            "room_prompt": room_spec.prompt,
            "room_geometry_dict": room_geometry.to_dict(scene_dir=room_dir),
            "room_dir": str(room_dir),
            "cfg_dict": cfg_dict,
            "start_stage": start_stage,
            "stop_stage": stop_stage,
            "scene_id": scene_id,
            "experiment_run_id": experiment_run_id,
            "house_layout_dict": house_layout.to_dict(scene_dir=output_dir),
            "render_gpu_id": render_gpu_id,
            "rl_task_context": rl_task_context,
        }
        tasks.append((room_id, _generate_room_worker, kwargs))

    # Run with fault tolerance and get return values.
    results = run_parallel_isolated(
        tasks=tasks, max_workers=max_workers, return_values=True
    )

    # Reconstruct RoomScenes from worker results.
    rooms: dict[str, RoomScene] = {}
    failures: list[tuple[str, str]] = []
    for room_id, (success, result_or_error) in results.items():
        room_dir = room_dirs[room_id]
        if success:
            rooms[room_id] = _reconstruct_room_scene(
                worker_result=result_or_error, scene_dir=room_dir
            )
            console_logger.info(f"Room '{room_id}' completed successfully")
        else:
            console_logger.error(f"Room '{room_id}' failed: {result_or_error}")
            failures.append((room_id, result_or_error))

    if failures:
        error_msg = "; ".join(f"Room '{rid}': {err}" for rid, err in failures)
        raise RuntimeError(f"Room generation failures: {error_msg}")

    return rooms


class IndoorSceneGenerationExperiment(BaseExperiment):
    """An experiment that generates indoor scenes."""

    compatible_floor_plan_agents = {
        "stateful_floor_plan_agent": StatefulFloorPlanAgent,
    }
    compatible_furniture_agents = {
        "stateful_furniture_agent": StatefulFurnitureAgent,
        "tabletop_furniture_agent": StatefulFurnitureAgent,
    }
    compatible_manipuland_agents = {
        "stateful_manipuland_agent": StatefulManipulandAgent,
        "tabletop_manipuland_agent": StatefulManipulandAgent,
    }
    compatible_wall_agents = {
        "stateful_wall_agent": StatefulWallAgent,
    }
    compatible_ceiling_agents = {
        "stateful_ceiling_agent": StatefulCeilingAgent,
    }

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg=cfg)
        self.geometry_server: GeometryGenerationServer | None = None
        self.hssd_server: HssdRetrievalServer | None = None
        self.objaverse_server: ObjaverseRetrievalServer | None = None
        self.articulated_server: ArticulatedRetrievalServer | None = None
        self.materials_server: MaterialsRetrievalServer | None = None
        self._reuse_external_servers: set[str] = set()

    def _server_health_ok(
        self, *, host: str, port: int, timeout_s: float = 1.0
    ) -> bool:
        try:
            response = requests.get(
                f"http://{host}:{port}/health",
                timeout=timeout_s,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _resolve_server_port(
        self,
        *,
        server_label: str,
        host: str,
        preferred_port: int,
        port_range_size: int = 200,
    ) -> tuple[int, str]:
        """Resolve the port for a server.

        If the preferred port is free, use it. If it is occupied (possibly by
        another live run), find a free port in the fallback range rather than
        killing the occupant — killing would crash a concurrently running
        experiment that owns the port.

        Returns:
            (port, mode) where mode is "free" or "reassigned".
        """
        if is_port_available(host, preferred_port):
            return preferred_port, "free"

        # Port is occupied — could be another live run, so do NOT kill it.
        # Find a free alternate port immediately.
        start_port = preferred_port + 1
        end_port = preferred_port + port_range_size
        selected = find_available_port(host=host, port_range=(start_port, end_port))
        if selected is None:
            raise RuntimeError(
                f"{server_label} port {host}:{preferred_port} is occupied and "
                f"no free ports found in [{start_port}, {end_port}]. "
                "Run scripts/kill_tasksmith_zombies.sh and retry."
            )

        console_logger.warning(
            f"{server_label}: preferred port {host}:{preferred_port} occupied; "
            f"using {host}:{selected} instead"
        )
        return selected, "reassigned"

    def __del__(self):
        """Ensure servers are stopped when experiment is destroyed."""
        if self.geometry_server and self.geometry_server.is_running():
            console_logger.warning("Stopping geometry server in destructor")
            try:
                self.geometry_server.stop()
            except Exception as e:
                console_logger.error(
                    f"Failed to stop geometry server in destructor: {e}"
                )

        if self.hssd_server and self.hssd_server.is_running():
            console_logger.warning("Stopping HSSD server in destructor")
            try:
                self.hssd_server.stop()
            except Exception as e:
                console_logger.error(f"Failed to stop HSSD server in destructor: {e}")

        if self.articulated_server and self.articulated_server.is_running():
            console_logger.warning("Stopping articulated server in destructor")
            try:
                self.articulated_server.stop()
            except Exception as e:
                console_logger.error(
                    f"Failed to stop articulated server in destructor: {e}"
                )

        if self.materials_server and self.materials_server.is_running():
            console_logger.warning("Stopping materials server in destructor")
            try:
                self.materials_server.stop()
            except Exception as e:
                console_logger.error(
                    f"Failed to stop materials server in destructor: {e}"
                )

    def _start_geometry_server(self) -> None:
        """Start geometry generation server (if general_asset_source == 'generated')."""
        # Only start if at least one agent uses generated strategy.
        furniture_uses_generated = (
            self.cfg.furniture_agent.asset_manager.general_asset_source == "generated"
        )
        manipuland_uses_generated = (
            self.cfg.manipuland_agent.asset_manager.general_asset_source == "generated"
        )

        if not (furniture_uses_generated or manipuland_uses_generated):
            return

        # Get server configuration from experiment config.
        server_config = self.cfg.experiment.geometry_generation_server
        # Re-check at start time to avoid races with other concurrent runs.
        port, mode = self._resolve_server_port(
            server_label="Geometry generation server",
            host=str(server_config.host),
            preferred_port=int(server_config.port),
        )
        if mode == "reuse":
            self._reuse_external_servers.add("geometry")
            return
        if mode == "reassigned":
            with open_dict(self.cfg):
                self.cfg.experiment.geometry_generation_server.port = port
            server_config = self.cfg.experiment.geometry_generation_server

        # Determine backend - use furniture agent config (they should match).
        backend = self.cfg.furniture_agent.asset_manager.get("backend", "hunyuan3d")

        # Prepare SAM3D config if using SAM3D backend.
        sam3d_config = None
        if backend == "sam3d":
            sam3d_cfg = self.cfg.furniture_agent.asset_manager.sam3d
            sam3d_config = {
                "sam3_checkpoint": str(sam3d_cfg.sam3_checkpoint),
                "sam3d_checkpoint": str(sam3d_cfg.sam3d_checkpoint),
            }

        console_logger.info(
            f"Starting geometry generation server ({backend}) on "
            f"{server_config.host}:{server_config.port}"
        )

        self.geometry_server = GeometryGenerationServer(
            host=server_config.host,
            port=server_config.port,
            backend=backend,
            sam3d_config=sam3d_config,
            log_file=self.output_dir / "experiment.log",
        )

        self.geometry_server.start()
        self.geometry_server.wait_until_ready(timeout_s=30.0)
        console_logger.info("Geometry generation server ready")

    def _stop_geometry_server(self) -> None:
        """Stop the geometry generation server."""
        if "geometry" in self._reuse_external_servers:
            return
        if self.geometry_server and self.geometry_server.is_running():
            console_logger.info("Stopping geometry generation server...")
            self.geometry_server.stop()
            console_logger.info("Geometry generation server stopped")
            self.geometry_server = None

    def _start_hssd_server(self) -> None:
        """Start HSSD retrieval server (if general_asset_source == 'hssd')."""
        # Only start if at least one agent uses HSSD strategy.
        furniture_uses_hssd = (
            self.cfg.furniture_agent.asset_manager.general_asset_source == "hssd"
        )
        manipuland_uses_hssd = (
            self.cfg.manipuland_agent.asset_manager.general_asset_source == "hssd"
        )
        wall_uses_hssd = (
            self.cfg.wall_agent.asset_manager.general_asset_source == "hssd"
        )
        ceiling_uses_hssd = (
            self.cfg.ceiling_agent.asset_manager.general_asset_source == "hssd"
        )

        if not (
            furniture_uses_hssd
            or manipuland_uses_hssd
            or wall_uses_hssd
            or ceiling_uses_hssd
        ):
            return

        # Get server configuration from experiment config.
        server_config = self.cfg.experiment.hssd_retrieval_server
        # Re-check at start time to avoid races with other concurrent runs.
        port, mode = self._resolve_server_port(
            server_label="HSSD retrieval server",
            host=str(server_config.host),
            preferred_port=int(server_config.port),
        )
        if mode == "reuse":
            self._reuse_external_servers.add("hssd")
            return
        if mode == "reassigned":
            with open_dict(self.cfg):
                self.cfg.experiment.hssd_retrieval_server.port = port
            server_config = self.cfg.experiment.hssd_retrieval_server
        # Get HSSD data configuration from asset manager config.
        hssd_config = self.cfg.furniture_agent.asset_manager.hssd

        retrieval_device = _get_retrieval_gpu_device()
        console_logger.info(
            f"Starting HSSD retrieval server on "
            f"{server_config.host}:{server_config.port} "
            f"(CLIP device: {retrieval_device or 'default'})"
        )

        self.hssd_server = HssdRetrievalServer(
            host=server_config.host,
            port=server_config.port,
            preload_retriever=True,  # Always preload CLIP for consistent performance.
            hssd_data_path=str(hssd_config.data_path),
            hssd_preprocessed_path=str(hssd_config.preprocessed_path),
            hssd_top_k=hssd_config.use_top_k,
            clip_device=retrieval_device,
        )

        self.hssd_server.start()
        # Longer timeout for CLIP loading.
        self.hssd_server.wait_until_ready(timeout_s=60.0)
        console_logger.info("HSSD retrieval server ready")

    def _stop_hssd_server(self) -> None:
        """Stop the HSSD retrieval server."""
        if "hssd" in self._reuse_external_servers:
            return
        if self.hssd_server and self.hssd_server.is_running():
            console_logger.info("Stopping HSSD retrieval server...")
            self.hssd_server.stop()
            console_logger.info("HSSD retrieval server stopped")
            self.hssd_server = None

    def _start_objaverse_server(self) -> None:
        """Start Objaverse retrieval server (if general_asset_source == 'objaverse')."""
        # Only start if at least one agent uses objaverse strategy.
        furniture_uses_objaverse = (
            self.cfg.furniture_agent.asset_manager.general_asset_source == "objaverse"
        )
        manipuland_uses_objaverse = (
            self.cfg.manipuland_agent.asset_manager.general_asset_source == "objaverse"
        )
        wall_uses_objaverse = (
            self.cfg.wall_agent.asset_manager.general_asset_source == "objaverse"
        )
        ceiling_uses_objaverse = (
            self.cfg.ceiling_agent.asset_manager.general_asset_source == "objaverse"
        )

        if not (
            furniture_uses_objaverse
            or manipuland_uses_objaverse
            or wall_uses_objaverse
            or ceiling_uses_objaverse
        ):
            return

        # Get server configuration from experiment config.
        server_config = self.cfg.experiment.objaverse_retrieval_server
        # Re-check at start time to avoid races with other concurrent runs.
        port, mode = self._resolve_server_port(
            server_label="Objaverse retrieval server",
            host=str(server_config.host),
            preferred_port=int(server_config.port),
        )
        if mode == "reuse":
            self._reuse_external_servers.add("objaverse")
            return
        if mode == "reassigned":
            with open_dict(self.cfg):
                self.cfg.experiment.objaverse_retrieval_server.port = port
            server_config = self.cfg.experiment.objaverse_retrieval_server
        # Get Objaverse data configuration from asset manager config.
        objaverse_config = self.cfg.furniture_agent.asset_manager.objaverse

        retrieval_device = _get_retrieval_gpu_device()
        console_logger.info(
            f"Starting Objaverse retrieval server on "
            f"{server_config.host}:{server_config.port} "
            f"(CLIP device: {retrieval_device or 'default'})"
        )

        self.objaverse_server = ObjaverseRetrievalServer(
            host=server_config.host,
            port=server_config.port,
            preload_retriever=True,
            objaverse_data_path=str(objaverse_config.data_path),
            objaverse_preprocessed_path=str(objaverse_config.preprocessed_path),
            objaverse_top_k=objaverse_config.use_top_k,
            clip_device=retrieval_device,
        )

        self.objaverse_server.start()
        # Longer timeout for CLIP loading.
        self.objaverse_server.wait_until_ready(timeout_s=60.0)
        console_logger.info("Objaverse retrieval server ready")

    def _stop_objaverse_server(self) -> None:
        """Stop the Objaverse retrieval server."""
        if "objaverse" in self._reuse_external_servers:
            return
        if self.objaverse_server and self.objaverse_server.is_running():
            console_logger.info("Stopping Objaverse retrieval server...")
            self.objaverse_server.stop()
            console_logger.info("Objaverse retrieval server stopped")
            self.objaverse_server = None

    def _start_articulated_server(self) -> None:
        """Start articulated retrieval server (if articulated strategy is enabled)."""
        # Check if articulated strategy is enabled for any agent.
        furniture_articulated_enabled = (
            self.cfg.furniture_agent.asset_manager.router.strategies.articulated.enabled
        )
        manipuland_articulated_enabled = (
            self.cfg.manipuland_agent.asset_manager.router.strategies.articulated.enabled
        )
        wall_articulated_enabled = (
            self.cfg.wall_agent.asset_manager.router.strategies.articulated.enabled
        )
        ceiling_articulated_enabled = (
            self.cfg.ceiling_agent.asset_manager.router.strategies.articulated.enabled
        )

        if not (
            furniture_articulated_enabled
            or manipuland_articulated_enabled
            or wall_articulated_enabled
            or ceiling_articulated_enabled
        ):
            return

        # Get server configuration from experiment config.
        server_config = self.cfg.experiment.articulated_retrieval_server
        # Re-check at start time to avoid races with other concurrent runs.
        port, mode = self._resolve_server_port(
            server_label="Articulated retrieval server",
            host=str(server_config.host),
            preferred_port=int(server_config.port),
        )
        if mode == "reuse":
            self._reuse_external_servers.add("articulated")
            return
        if mode == "reassigned":
            with open_dict(self.cfg):
                self.cfg.experiment.articulated_retrieval_server.port = port
            server_config = self.cfg.experiment.articulated_retrieval_server

        # Get articulated data configuration from furniture agent config.
        articulated_config = self.cfg.furniture_agent.asset_manager.articulated

        retrieval_device = _get_retrieval_gpu_device()
        console_logger.info(
            f"Starting articulated retrieval server on "
            f"{server_config.host}:{server_config.port} "
            f"(CLIP device: {retrieval_device or 'default'})"
        )

        self.articulated_server = ArticulatedRetrievalServer(
            host=server_config.host,
            port=server_config.port,
            preload_retriever=True,  # Always preload CLIP for consistent performance.
            articulated_config=articulated_config,
            clip_device=retrieval_device,
        )

        self.articulated_server.start()
        # Longer timeout for CLIP loading.
        self.articulated_server.wait_until_ready(timeout_s=60.0)
        console_logger.info("Articulated retrieval server ready")

    def _stop_articulated_server(self) -> None:
        """Stop the articulated retrieval server."""
        if "articulated" in self._reuse_external_servers:
            return
        if self.articulated_server and self.articulated_server.is_running():
            console_logger.info("Stopping articulated retrieval server...")
            self.articulated_server.stop()
            console_logger.info("Articulated retrieval server stopped")
            self.articulated_server = None

    def _start_materials_server(self) -> None:
        """Start materials retrieval server."""
        # Get server configuration from experiment config.
        server_config = self.cfg.experiment.materials_retrieval_server
        # Re-check at start time to avoid races with other concurrent runs.
        port, mode = self._resolve_server_port(
            server_label="Materials retrieval server",
            host=str(server_config.host),
            preferred_port=int(server_config.port),
        )
        if mode == "reuse":
            self._reuse_external_servers.add("materials")
            return
        if mode == "reassigned":
            with open_dict(self.cfg):
                self.cfg.experiment.materials_retrieval_server.port = port
            server_config = self.cfg.experiment.materials_retrieval_server

        retrieval_device = _get_retrieval_gpu_device()
        console_logger.info(
            f"Starting materials retrieval server on "
            f"{server_config.host}:{server_config.port} "
            f"(CLIP device: {retrieval_device or 'default'})"
        )

        self.materials_server = MaterialsRetrievalServer(
            host=server_config.host,
            port=server_config.port,
            preload_retriever=True,  # Always preload CLIP for consistent performance.
            materials_config=server_config,  # Pass DictConfig directly.
            clip_device=retrieval_device,
        )

        self.materials_server.start()
        # Longer timeout for CLIP loading.
        self.materials_server.wait_until_ready(timeout_s=60.0)
        console_logger.info("Materials retrieval server ready")

    def _stop_materials_server(self) -> None:
        """Stop the materials retrieval server."""
        if "materials" in self._reuse_external_servers:
            return
        if self.materials_server and self.materials_server.is_running():
            console_logger.info("Stopping materials retrieval server...")
            self.materials_server.stop()
            console_logger.info("Materials retrieval server stopped")
            self.materials_server = None

    @staticmethod
    def _generate_single_scene(
        prompt: str,
        scene_id: int,
        output_dir: Path,
        cfg_dict: dict,
        capture_logs: bool = False,
        experiment_run_id: str | None = None,
        render_gpu_id: int | None = None,
    ) -> None:
        """Generate a single scene (static method for parallel execution).

        Pipeline stages run in order:
        floor_plan → furniture → wall_mounted → ceiling_mounted → manipulands
        Use config pipeline.start_stage and pipeline.stop_stage to control execution.

        Args:
            prompt: Scene description.
            scene_id: Scene identifier.
            output_dir: Base output directory for the experiment.
            cfg_dict: Configuration as dictionary.
            capture_logs: If True, suppress stdout and only write to file.
            experiment_run_id: Unique ID for this experiment run.
            render_gpu_id: GPU device ID for Blender rendering. When set, uses
                bubblewrap to isolate the BlenderServer to this GPU.
        """
        # Reset any SDK state inherited via fork (defense in depth).
        _reset_inherited_sdk_state()

        faulthandler.enable()

        scene_generation_start_time = time.time()

        # Create scene directory.
        scene_dir = output_dir / f"scene_{scene_id:03d}"
        scene_dir.mkdir(parents=True, exist_ok=True)

        # Always create log file.
        log_path = scene_dir / "scene.log"

        # Log start message before potential suppression.
        if capture_logs:
            console_logger.info(
                f"Scene {scene_id:03d} started (logs → {log_path})\n"
                f"Prompt: {prompt}"
            )
        else:
            console_logger.info(
                f"Scene {scene_id:03d} started (debug mode)\nPrompt: {prompt}"
            )

        # Create a logger for this scene.
        logger = ConsoleLogger(output_dir=scene_dir)
        scene_tracking = TrackingContext(experiment_dir=output_dir, scene_dir=scene_dir)

        # Get pipeline stage configuration.
        pipeline_cfg = cfg_dict["experiment"]["pipeline"]
        start_stage = pipeline_cfg["start_stage"]
        stop_stage = pipeline_cfg["stop_stage"]
        skip_stages = pipeline_cfg.get("skip_stages", [])

        # Validate stages.
        if start_stage not in PIPELINE_STAGES:
            raise ValueError(
                f"Invalid start_stage '{start_stage}'. "
                f"Valid options: {PIPELINE_STAGES}"
            )
        if stop_stage not in PIPELINE_STAGES:
            raise ValueError(
                f"Invalid stop_stage '{stop_stage}'. "
                f"Valid options: {PIPELINE_STAGES}"
            )
        for s in skip_stages:
            if s not in PIPELINE_STAGES:
                raise ValueError(
                    f"Invalid skip_stage '{s}'. Valid options: {PIPELINE_STAGES}"
                )
            if s in {"floor_plan", "furniture"}:
                raise ValueError(f"Cannot skip '{s}' stage via skip_stages.")

        start_idx = PIPELINE_STAGES.index(start_stage)
        stop_idx = PIPELINE_STAGES.index(stop_stage)
        if start_idx > stop_idx:
            raise ValueError(
                f"start_stage '{start_stage}' cannot be after stop_stage '{stop_stage}'"
            )

        console_logger.info(
            f"Pipeline: start_stage='{start_stage}', stop_stage='{stop_stage}', "
            f"skip={skip_stages}"
        )

        # Handle resume from checkpoint if resume_from_path is specified.
        resume_from_path = pipeline_cfg.get("resume_from_path")
        if resume_from_path and start_stage != "floor_plan":
            source_experiment_dir = Path(resume_from_path)
            if not source_experiment_dir.exists():
                raise FileNotFoundError(
                    f"resume_from_path does not exist: {resume_from_path}"
                )
            _copy_checkpoint_for_stage(
                source_scene_dir=source_experiment_dir / f"scene_{scene_id:03d}",
                target_scene_dir=scene_dir,
                start_stage=start_stage,
                stop_stage=stop_stage,
                skip_stages=skip_stages,
            )

        with contextlib.ExitStack() as scene_stack:
            scene_stack.enter_context(
                live_wandb_tracking(
                    scene_dir=scene_dir,
                    experiment_dir=output_dir,
                    cfg_dict=cfg_dict,
                    scene_id=scene_id,
                    prompt=prompt,
                    experiment_run_id=experiment_run_id,
                )
            )
            scene_stack.enter_context(tracking_context(scene_tracking))
            with FileLoggingContext(
                log_file_path=log_path, suppress_stdout=capture_logs
            ):
                try:
                    # Create trace metadata for this scene.
                    trace_metadata = {
                        "scene_id": f"scene_{scene_id:03d}",
                        "experiment_name": cfg_dict["name"],
                        "scene_dir": str(scene_dir),
                        "prompt": prompt,
                    }
                    if experiment_run_id:
                        trace_metadata["experiment_run_id"] = experiment_run_id

                    console_logger.info(f"Generating scene for prompt: {prompt}")

                    # Task analysis step: run before floor plan or load from disk when resuming.
                    task_req_path = scene_dir / "task_requirements.json"
                    task_requirements: TaskRequirements | None = None
                    rl_task_context: str = ""

                    if start_stage == "floor_plan":
                        with _tracked_stage(
                            "task_analysis",
                            output_paths=lambda: [str(task_req_path)],
                        ):
                            console_logger.info("Running task analysis...")
                            task_agent = TaskAnalysisAgent(cfg=cfg_dict)
                            task_requirements = task_agent.analyze(task_prompt=prompt)
                            task_requirements.save(task_req_path)
                            emit_artifact(
                                artifact_type="task_requirements",
                                path=task_req_path,
                                stage_name="task_analysis",
                            )
                    elif task_req_path.exists():
                        task_requirements = TaskRequirements.load(task_req_path)
                        console_logger.info(
                            f"Loaded task requirements: {task_requirements.task_summary}"
                        )

                    if task_requirements is not None:
                        augmented_prompt = task_requirements.build_augmented_prompt()
                        rl_task_context = (
                            task_requirements.build_rl_task_context_string()
                        )
                    else:
                        augmented_prompt = prompt

                    # Single trace wraps entire scene generation (floor plan + rooms).
                    with trace(
                        workflow_name=f"scene_{scene_id:03d}_generation",
                        metadata=trace_metadata,
                    ):
                        # Stage 1: Floor plan generation (or load from saved state).
                        if start_stage == "floor_plan":
                            console_logger.info(
                                "Generating house layout (in isolated subprocess)"
                            )
                            layout_start_time = time.time()

                            results = run_parallel_isolated(
                                tasks=[
                                    (
                                        "floor_plan",
                                        _generate_floor_plan_worker,
                                        {
                                            "prompt": augmented_prompt,
                                            "scene_dir": str(scene_dir),
                                            "cfg_dict": cfg_dict,
                                            "experiment_run_id": experiment_run_id,
                                            "render_gpu_id": render_gpu_id,
                                            "rl_task_context": rl_task_context,
                                        },
                                    )
                                ],
                                max_workers=1,
                            )

                            success, error = results["floor_plan"]
                            if not success:
                                raise RuntimeError(
                                    f"Floor plan generation failed: {error}"
                                )

                            house_layout_path = scene_dir / "house_layout.json"
                            with open(house_layout_path) as f:
                                house_layout_dict = json.load(f)
                            house_layout = HouseLayout.from_dict(
                                house_layout_dict, house_dir=scene_dir
                            )

                            layout_end_time = time.time()
                            console_logger.info(
                                f"House layout generated in "
                                f"{timedelta(seconds=layout_end_time - layout_start_time)}"
                            )
                        elif start_stage == "reward_spec":
                            rl_artifacts_path = scene_dir / "rl_artifacts.json"
                            task_req_path = scene_dir / "task_requirements.json"
                            if task_requirements is None and task_req_path.exists():
                                task_requirements = TaskRequirements.load(task_req_path)
                            if not rl_artifacts_path.exists():
                                raise RuntimeError(
                                    "reward_spec stage requires rl_artifacts.json in scene_dir. "
                                    "Run the full pipeline first or use resume_from_path."
                                )
                            rl_artifacts_data = json.loads(
                                rl_artifacts_path.read_text()
                            )
                            should_run_reward_spec = (
                                start_idx <= _REWARD_SPEC_IDX <= stop_idx
                                and "reward_spec" not in skip_stages
                            )
                            should_run_morphology_task = (
                                start_idx <= _MORPHOLOGY_TASK_IDX <= stop_idx
                                and "morphology_task" not in skip_stages
                            )

                            if should_run_reward_spec:
                                if task_requirements is None:
                                    raise RuntimeError(
                                        "reward_spec stage requires task_requirements.json in "
                                        "scene_dir. Run the full pipeline first or use "
                                        "resume_from_path."
                                    )
                                with _tracked_stage(
                                    "reward_spec",
                                    output_paths=lambda: [
                                        str(scene_dir / "reward_spec.yaml"),
                                        str(scene_dir / "curriculum_spec.yaml"),
                                    ],
                                ):
                                    reward_agent = RewardSpecAgent(cfg=cfg_dict)
                                    reward_spec, curriculum_spec = (
                                        reward_agent.generate(
                                            rl_artifacts=rl_artifacts_data,
                                            task_requirements=task_requirements,
                                        )
                                    )
                                    reward_spec.save(scene_dir / "reward_spec.yaml")
                                    curriculum_spec.save(
                                        scene_dir / "curriculum_spec.yaml"
                                    )
                                    emit_artifact(
                                        artifact_type="reward_spec",
                                        path=scene_dir / "reward_spec.yaml",
                                        stage_name="reward_spec",
                                    )
                                    emit_artifact(
                                        artifact_type="curriculum_spec",
                                        path=scene_dir / "curriculum_spec.yaml",
                                        stage_name="reward_spec",
                                    )

                            if should_run_morphology_task:
                                with _tracked_stage(
                                    "morphology_task",
                                    output_paths=lambda: [
                                        str(scene_dir / "morphology_task_config.json")
                                    ],
                                ):
                                    _generate_morphology_task_config(
                                        scene_dir=scene_dir,
                                        cfg_dict=cfg_dict,
                                        rl_artifacts_data=rl_artifacts_data,
                                    )
                                    emit_artifact(
                                        artifact_type="morphology_task_config",
                                        path=scene_dir / "morphology_task_config.json",
                                        stage_name="morphology_task",
                                    )
                            finalize_scene_analysis(scene_dir, cfg_dict)
                            console_logger.info(
                                "Scene generation completed successfully in "
                                f"{timedelta(seconds=time.time() - scene_generation_start_time)}"
                            )
                            return
                        elif start_stage == "morphology_task":
                            with _tracked_stage(
                                "morphology_task",
                                output_paths=lambda: [
                                    str(scene_dir / "morphology_task_config.json")
                                ],
                            ):
                                _generate_morphology_task_config(
                                    scene_dir=scene_dir,
                                    cfg_dict=cfg_dict,
                                )
                                emit_artifact(
                                    artifact_type="morphology_task_config",
                                    path=scene_dir / "morphology_task_config.json",
                                    stage_name="morphology_task",
                                )
                            finalize_scene_analysis(scene_dir, cfg_dict)
                            console_logger.info(
                                "Scene generation completed successfully in "
                                f"{timedelta(seconds=time.time() - scene_generation_start_time)}"
                            )
                            return
                        else:
                            house_layout_path = scene_dir / "house_layout.json"
                            if not house_layout_path.exists():
                                raise FileNotFoundError(
                                    f"Cannot start from '{start_stage}' stage: "
                                    f"house_layout.json not found at {house_layout_path}. "
                                    "Run with start_stage='floor_plan' first."
                                )
                            console_logger.info(
                                f"Loading house layout from {house_layout_path}"
                            )
                            with open(house_layout_path) as f:
                                house_layout_dict = json.load(f)
                            house_layout = HouseLayout.from_dict(
                                house_layout_dict, house_dir=scene_dir
                            )

                        if stop_stage == "floor_plan":
                            finalize_scene_analysis(scene_dir, cfg_dict)
                            console_logger.info(
                                "Stopping after floor_plan stage as configured"
                            )
                            console_logger.info(
                                "Scene generation completed successfully in "
                                f"{timedelta(seconds=time.time() - scene_generation_start_time)}"
                            )
                            return

                        room_start_stage = (
                            "furniture" if start_stage == "floor_plan" else start_stage
                        )
                        room_stop_stage = (
                            stop_stage
                            if stop_stage in _ROOM_STAGES
                            else _ROOM_STAGES[-1]
                        )

                        parallel_rooms = pipeline_cfg["parallel_rooms"]
                        max_parallel_rooms = pipeline_cfg["max_parallel_rooms"]
                        num_rooms = len(house_layout.room_ids)
                        use_parallel = (
                            parallel_rooms and max_parallel_rooms > 1 and num_rooms > 1
                        )

                        if use_parallel:
                            rooms = _run_parallel_room_generation(
                                house_layout=house_layout,
                                output_dir=scene_dir,
                                cfg_dict=cfg_dict,
                                start_stage=room_start_stage,
                                stop_stage=room_stop_stage,
                                max_workers=max_parallel_rooms,
                                scene_id=scene_id,
                                experiment_run_id=experiment_run_id,
                                render_gpu_id=render_gpu_id,
                                rl_task_context=rl_task_context,
                            )
                        else:
                            rooms = _run_sequential_room_generation(
                                house_layout=house_layout,
                                logger=logger,
                                cfg_dict=cfg_dict,
                                start_stage=room_start_stage,
                                stop_stage=room_stop_stage,
                                render_gpu_id=render_gpu_id,
                                rl_task_context=rl_task_context,
                            )

                        house_scene = HouseScene(layout=house_layout, rooms=rooms)
                        snapshots = [
                            ("combined_house_after_furniture", [ObjectType.FURNITURE]),
                            (
                                "combined_house_after_wall_objects",
                                [ObjectType.FURNITURE, ObjectType.WALL_MOUNTED],
                            ),
                            (
                                "combined_house_after_ceiling",
                                [
                                    ObjectType.FURNITURE,
                                    ObjectType.WALL_MOUNTED,
                                    ObjectType.CEILING_MOUNTED,
                                ],
                            ),
                            ("combined_house", None),
                        ]
                        stage_to_count = {
                            "furniture": 1,
                            AgentType.WALL_MOUNTED.value: 2,
                            AgentType.CEILING_MOUNTED.value: 3,
                        }
                        snapshot_count = stage_to_count.get(stop_stage, len(snapshots))

                        for name, types in snapshots[:snapshot_count]:
                            house_scene.assemble(
                                cfg=cfg_dict,
                                output_name=name,
                                include_object_types=types,
                            )

                        try:
                            render_rooms_for_analysis(
                                scene_dir,
                                cfg_dict,
                                rooms=rooms,
                                render_gpu_id=render_gpu_id,
                            )
                        except Exception as exc:
                            console_logger.warning(
                                "Experiment analysis renders failed (continuing): %s",
                                exc,
                            )

                        if task_requirements is not None and stop_stage not in (
                            "floor_plan",
                            "furniture",
                            AgentType.WALL_MOUNTED.value,
                            AgentType.CEILING_MOUNTED.value,
                        ):
                            with _tracked_stage(
                                "task_artifacts",
                                output_paths=lambda: [
                                    str(scene_dir / "rl_artifacts.json")
                                ],
                            ):
                                console_logger.info("Exporting RL artifacts...")
                                rl_agent = RLArtifactAgent(cfg=cfg_dict)
                                render_paths = [
                                    Path(p)
                                    for p in first_analysis_render_paths(scene_dir)
                                ]
                                rl_artifacts = rl_agent.export(
                                    house_scene=house_scene,
                                    task_requirements=task_requirements,
                                    output_path=scene_dir / "rl_artifacts.json",
                                    render_paths=render_paths[:4],
                                )

                                verification = (rl_artifacts or {}).get(
                                    "verification"
                                ) or {}
                                if (
                                    verification.get("all_required_objects_present")
                                    is False
                                ):
                                    missing_objects = (
                                        verification.get("missing_objects") or []
                                    )
                                    console_logger.warning(
                                        "RL artifact verification flagged missing/"
                                        "substituted required objects: %s. See "
                                        "rl_artifacts.json:verification for details.",
                                        missing_objects,
                                    )
                                emit_artifact(
                                    artifact_type="rl_artifacts",
                                    path=scene_dir / "rl_artifacts.json",
                                    stage_name="task_artifacts",
                                )
                                console_logger.info(
                                    "RL artifacts saved to rl_artifacts.json"
                                )

                        should_run_reward_spec = (
                            start_idx <= _REWARD_SPEC_IDX <= stop_idx
                            and "reward_spec" not in skip_stages
                        )
                        if should_run_reward_spec:
                            rl_artifacts_path = scene_dir / "rl_artifacts.json"
                            task_req_path = scene_dir / "task_requirements.json"
                            if task_requirements is None and task_req_path.exists():
                                task_requirements = TaskRequirements.load(task_req_path)
                            missing = [
                                str(p)
                                for p in [rl_artifacts_path, task_req_path]
                                if not p.exists()
                            ]
                            if missing or task_requirements is None:
                                console_logger.warning(
                                    "Skipping reward_spec: missing files %s",
                                    missing or ["task_requirements (failed to load)"],
                                )
                            else:
                                with _tracked_stage(
                                    "reward_spec",
                                    output_paths=lambda: [
                                        str(scene_dir / "reward_spec.yaml"),
                                        str(scene_dir / "curriculum_spec.yaml"),
                                    ],
                                ):
                                    console_logger.info(
                                        "Generating reward spec and curriculum spec..."
                                    )
                                    reward_agent = RewardSpecAgent(cfg=cfg_dict)
                                    rl_artifacts_data = json.loads(
                                        rl_artifacts_path.read_text()
                                    )
                                    reward_spec, curriculum_spec = (
                                        reward_agent.generate(
                                            rl_artifacts=rl_artifacts_data,
                                            task_requirements=task_requirements,
                                        )
                                    )
                                    reward_spec.save(scene_dir / "reward_spec.yaml")
                                    curriculum_spec.save(
                                        scene_dir / "curriculum_spec.yaml"
                                    )
                                    emit_artifact(
                                        artifact_type="reward_spec",
                                        path=scene_dir / "reward_spec.yaml",
                                        stage_name="reward_spec",
                                    )
                                    emit_artifact(
                                        artifact_type="curriculum_spec",
                                        path=scene_dir / "curriculum_spec.yaml",
                                        stage_name="reward_spec",
                                    )
                                console_logger.info(
                                    "Reward spec saved to reward_spec.yaml and curriculum_spec.yaml"
                                )

                        should_run_morphology_task = (
                            start_idx <= _MORPHOLOGY_TASK_IDX <= stop_idx
                            and "morphology_task" not in skip_stages
                        )
                        if should_run_morphology_task:
                            with _tracked_stage(
                                "morphology_task",
                                output_paths=lambda: [
                                    str(scene_dir / "morphology_task_config.json")
                                ],
                            ):
                                _generate_morphology_task_config(
                                    scene_dir=scene_dir,
                                    cfg_dict=cfg_dict,
                                )
                                emit_artifact(
                                    artifact_type="morphology_task_config",
                                    path=scene_dir / "morphology_task_config.json",
                                    stage_name="morphology_task",
                                )

                        finalize_scene_analysis(
                            scene_dir,
                            cfg_dict,
                            render_gpu_id=render_gpu_id,
                        )
                        console_logger.info(
                            "Scene generation completed successfully in "
                            f"{timedelta(seconds=time.time() - scene_generation_start_time)}"
                        )

                except Exception as e:
                    console_logger.error(f"Scene generation failed: {e}")
                    finalize_scene_analysis(scene_dir, cfg_dict)
                    raise

    def _run_serial_generation(
        self,
        prompts_with_ids: list[tuple[int, str]],
        cfg_dict: dict,
        experiment_run_id: str,
    ) -> None:
        """Run scene generation in serial."""
        console_logger.info("Running scene generation serially in main thread")

        # GPU distribution is useful for parallel rooms within each scene.
        gpu_allocator = RenderGPUAllocator()

        for scene_id, prompt in prompts_with_ids:
            render_gpu_id = gpu_allocator.allocate()
            self._generate_single_scene(
                prompt=prompt,
                scene_id=scene_id,
                output_dir=self.output_dir,
                cfg_dict=cfg_dict,
                capture_logs=False,
                experiment_run_id=experiment_run_id,
                render_gpu_id=render_gpu_id,
            )
            console_logger.info(f"Completed scene {scene_id:03d}")

    def _run_parallel_generation(
        self,
        prompts_with_ids: list[tuple[int, str]],
        cfg_dict: dict,
        experiment_run_id: str,
        num_workers: int,
    ) -> None:
        """Run scene generation in parallel with fault tolerance.

        Uses isolated processes per scene instead of a shared executor pool.
        This ensures that if one scene crashes (e.g., GPU OOM), other scenes
        continue running unaffected.

        Raises:
            RuntimeError: If any scene generation fails.
        """
        console_logger.info(f"Running in parallel with {num_workers} workers")

        # Create GPU allocator for distributing Blender rendering.
        gpu_allocator = RenderGPUAllocator()

        # Build task list.
        tasks: list[tuple[str, Callable, dict]] = []
        for scene_id, prompt in prompts_with_ids:
            render_gpu_id = gpu_allocator.allocate()
            task_id = f"scene_{scene_id:03d}"
            kwargs = {
                "prompt": prompt,
                "scene_id": scene_id,
                "output_dir": self.output_dir,
                "cfg_dict": cfg_dict,
                "capture_logs": True,
                "experiment_run_id": experiment_run_id,
                "render_gpu_id": render_gpu_id,
            }
            tasks.append(
                (
                    task_id,
                    IndoorSceneGenerationExperiment._generate_single_scene,
                    kwargs,
                )
            )
            console_logger.info(f"Queued {task_id} (GPU {render_gpu_id}): {prompt}")

        # Run with fault tolerance - one crash doesn't affect others.
        results = run_parallel_isolated(tasks=tasks, max_workers=num_workers)

        # Report failures.
        failed_scenes = [
            (task_id, error)
            for task_id, (success, error) in results.items()
            if not success
        ]
        if failed_scenes:
            failure_details = "\n".join(
                f"  - {task_id}: {error}" for task_id, error in failed_scenes
            )
            raise RuntimeError(
                f"{len(failed_scenes)}/{len(tasks)} scene(s) failed:\n{failure_details}"
            )

    def generate_scenes(self) -> None:
        """Generate scenes with parallel support."""
        # Load prompts from CSV or YAML config.
        csv_path = self.cfg.experiment.csv_path
        if csv_path:
            prompts_with_ids = _load_prompts_from_csv(csv_path)
            console_logger.info(
                f"Loaded {len(prompts_with_ids)} prompts from CSV: {csv_path}"
            )
        else:
            prompts = self.cfg.experiment.prompts
            prompts_with_ids = list(enumerate(prompts))

        num_workers = min(self.cfg.experiment.num_workers, len(prompts_with_ids))

        # Get pipeline stage configuration.
        pipeline_cfg = self.cfg.experiment.pipeline
        start_stage = pipeline_cfg.start_stage
        stop_stage = pipeline_cfg.stop_stage
        skip_stages = list(pipeline_cfg.get("skip_stages", []))
        parallel_rooms = pipeline_cfg.parallel_rooms

        # Validate mutual exclusion: parallel scenes vs parallel rooms.
        if parallel_rooms and num_workers > 1:
            raise ValueError(
                "Cannot use both parallel rooms and parallel scenes. "
                "Set num_workers=1 to use parallel_rooms, or set parallel_rooms=false."
            )

        # Generate experiment run ID for trace filtering.
        experiment_run_id = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        )

        console_logger.info(f"Starting scene generation with {num_workers} workers")
        console_logger.info(f"Processing {len(prompts_with_ids)} scenes")
        console_logger.info(f"Experiment run ID: {experiment_run_id}")
        console_logger.info(
            f"Pipeline stages: start='{start_stage}', stop='{stop_stage}', "
            f"skip={skip_stages}"
        )

        try:
            # Resolve server ports before starting servers so workers inherit them.
            with open_dict(self.cfg):
                # Geometry server: only relevant when enabled by agent config.
                if (
                    self.cfg.furniture_agent.asset_manager.general_asset_source
                    == "generated"
                    or self.cfg.manipuland_agent.asset_manager.general_asset_source
                    == "generated"
                ):
                    cfg = self.cfg.experiment.geometry_generation_server
                    port, mode = self._resolve_server_port(
                        server_label="Geometry generation server",
                        host=str(cfg.host),
                        preferred_port=int(cfg.port),
                    )
                    cfg.port = port
                    if mode == "reuse":
                        self._reuse_external_servers.add("geometry")

                # HSSD server: only relevant when at least one agent uses it.
                if (
                    self.cfg.furniture_agent.asset_manager.general_asset_source
                    == "hssd"
                    or self.cfg.manipuland_agent.asset_manager.general_asset_source
                    == "hssd"
                    or self.cfg.wall_agent.asset_manager.general_asset_source == "hssd"
                    or self.cfg.ceiling_agent.asset_manager.general_asset_source
                    == "hssd"
                ):
                    cfg = self.cfg.experiment.hssd_retrieval_server
                    port, mode = self._resolve_server_port(
                        server_label="HSSD retrieval server",
                        host=str(cfg.host),
                        preferred_port=int(cfg.port),
                    )
                    cfg.port = port
                    if mode == "reuse":
                        self._reuse_external_servers.add("hssd")

                # Objaverse server: only relevant when at least one agent uses it.
                if (
                    self.cfg.furniture_agent.asset_manager.general_asset_source
                    == "objaverse"
                    or self.cfg.manipuland_agent.asset_manager.general_asset_source
                    == "objaverse"
                    or self.cfg.wall_agent.asset_manager.general_asset_source
                    == "objaverse"
                    or self.cfg.ceiling_agent.asset_manager.general_asset_source
                    == "objaverse"
                ):
                    cfg = self.cfg.experiment.objaverse_retrieval_server
                    port, mode = self._resolve_server_port(
                        server_label="Objaverse retrieval server",
                        host=str(cfg.host),
                        preferred_port=int(cfg.port),
                    )
                    cfg.port = port
                    if mode == "reuse":
                        self._reuse_external_servers.add("objaverse")

                # Articulated server: only relevant when any agent enables articulated.
                if (
                    self.cfg.furniture_agent.asset_manager.router.strategies.articulated.enabled
                    or self.cfg.manipuland_agent.asset_manager.router.strategies.articulated.enabled
                    or self.cfg.wall_agent.asset_manager.router.strategies.articulated.enabled
                    or self.cfg.ceiling_agent.asset_manager.router.strategies.articulated.enabled
                ):
                    cfg = self.cfg.experiment.articulated_retrieval_server
                    port, mode = self._resolve_server_port(
                        server_label="Articulated retrieval server",
                        host=str(cfg.host),
                        preferred_port=int(cfg.port),
                    )
                    cfg.port = port
                    if mode == "reuse":
                        self._reuse_external_servers.add("articulated")

                # Materials server: always used (agents default to 7008).
                cfg = self.cfg.experiment.materials_retrieval_server
                port, mode = self._resolve_server_port(
                    server_label="Materials retrieval server",
                    host=str(cfg.host),
                    preferred_port=int(cfg.port),
                )
                cfg.port = port
                if mode == "reuse":
                    self._reuse_external_servers.add("materials")

            # Convert config to dictionary for static method after port resolution.
            cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)

            # Start GPU servers (CUDA init happens here).
            if "geometry" not in self._reuse_external_servers:
                self._start_geometry_server()
            if "hssd" not in self._reuse_external_servers:
                self._start_hssd_server()
            if "objaverse" not in self._reuse_external_servers:
                self._start_objaverse_server()
            if "articulated" not in self._reuse_external_servers:
                self._start_articulated_server()
            if "materials" not in self._reuse_external_servers:
                self._start_materials_server()

            if num_workers == 1:
                self._run_serial_generation(
                    prompts_with_ids=prompts_with_ids,
                    cfg_dict=cfg_dict,
                    experiment_run_id=experiment_run_id,
                )
            else:
                self._run_parallel_generation(
                    prompts_with_ids=prompts_with_ids,
                    cfg_dict=cfg_dict,
                    experiment_run_id=experiment_run_id,
                    num_workers=num_workers,
                )

            console_logger.info("All scenes completed")

            # Log clear completion message.
            console_logger.info("=" * 60)
            console_logger.info(bold_green("ALL SCENES COMPLETED!"))
            console_logger.info("=" * 60)
            console_logger.info(yellow("Press Ctrl+C to exit the script."))
            console_logger.info("=" * 60)

        finally:
            # Stop GPU servers.
            self._stop_materials_server()
            self._stop_articulated_server()
            self._stop_objaverse_server()
            self._stop_hssd_server()
            self._stop_geometry_server()

    def evaluate_scenes(self) -> None:
        """
        Evaluate previously generated scenes.
        """
        raise NotImplementedError
