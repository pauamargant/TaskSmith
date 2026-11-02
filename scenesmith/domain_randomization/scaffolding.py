"""Folder scaffolding for domain randomization variations.

Thin wrapper around the checkpoint-copy logic from indoor_scene_generation.py.
"""

from __future__ import annotations

import logging
import shutil

from pathlib import Path

from scenesmith.experiments.indoor_scene_generation import (
    STAGE_CHECKPOINTS,
    _copy_checkpoint_for_stage,
    _fix_paths_in_json_file,
    _fix_paths_in_yaml_file,
    _rewrite_copied_resume_paths,
)

console_logger = logging.getLogger(__name__)

# Stages the domain randomizer can resume from (floor_plan not supported; too expensive).
SUPPORTED_RESUME_STAGES = ["furniture", "wall_mounted", "ceiling_mounted", "manipuland"]


def get_room_scene_dirs(scene_dir: Path) -> list[Path]:
    """Return actual room scene directories, excluding helper dirs like room_geometry."""
    return sorted(
        room_dir
        for room_dir in scene_dir.glob("room_*")
        if room_dir.is_dir() and (room_dir / "scene_states").is_dir()
    )


def scaffold_variation(
    base_scene_dir: Path,
    variation_scene_dir: Path,
    resume_stage: str,
) -> None:
    """Create the variation directory tree and copy the required checkpoint.

    Calls the existing _copy_checkpoint_for_stage logic from the main pipeline,
    which handles asset copying and absolute path rebasing.

    Args:
        base_scene_dir: Source scene_NNN directory from the base experiment.
        variation_scene_dir: Target scene_NNN directory inside the variation folder.
        resume_stage: Pipeline stage to resume from. Determines which checkpoint
            and asset directories are copied.

    Raises:
        ValueError: If resume_stage is not in SUPPORTED_RESUME_STAGES.
        FileNotFoundError: If base_scene_dir does not exist.
    """
    if resume_stage not in SUPPORTED_RESUME_STAGES:
        raise ValueError(
            f"Unsupported resume_stage '{resume_stage}'. "
            f"Must be one of: {SUPPORTED_RESUME_STAGES}"
        )

    console_logger.info(
        f"Scaffolding variation: {variation_scene_dir.name} "
        f"(resume_stage={resume_stage})"
    )
    _copy_checkpoint_for_stage(
        source_scene_dir=base_scene_dir,
        target_scene_dir=variation_scene_dir,
        start_stage=resume_stage,
    )


def load_checkpoint_scene_state(
    variation_scene_dir: Path,
    resume_stage: str,
) -> tuple[Path, dict]:
    """Load the scene_state.json for the checkpoint at the given resume stage.

    Args:
        variation_scene_dir: The variation's scene directory.
        resume_stage: Stage to resume from.

    Returns:
        (state_path, state_dict): path to the JSON file and its parsed content.

    Raises:
        FileNotFoundError: If the checkpoint state file is missing.
    """
    import json

    checkpoint_name = STAGE_CHECKPOINTS[resume_stage]
    if checkpoint_name is None:
        raise ValueError(
            f"Stage '{resume_stage}' has no prior checkpoint to load from."
        )

    # Domain randomization currently operates on the first real room scene only.
    room_dirs = get_room_scene_dirs(variation_scene_dir)
    if not room_dirs:
        raise FileNotFoundError(f"No room_* directories found in {variation_scene_dir}")

    state_paths = []
    for room_dir in room_dirs:
        state_path = room_dir / "scene_states" / checkpoint_name / "scene_state.json"
        if not state_path.exists():
            raise FileNotFoundError(
                f"Checkpoint state not found: {state_path}. "
                f"Ensure scaffold_variation ran successfully."
            )
        state_paths.append(state_path)

    # Return first room's state (single-room scenes).
    state_path = state_paths[0]
    with open(state_path) as f:
        state_dict = json.load(f)

    return state_path, state_dict


def _prune_v2_variation_tree(variation_dir: Path) -> None:
    """Remove run_analysis, intermediate checkpoints, and agent session DBs."""
    run_analysis = variation_dir / "run_analysis"
    if run_analysis.exists():
        shutil.rmtree(run_analysis)

    for db_path in variation_dir.rglob("*.db"):
        db_path.unlink()

    for room_dir in get_room_scene_dirs(variation_dir):
        states_dir = room_dir / "scene_states"
        if not states_dir.is_dir():
            continue
        for checkpoint_dir in list(states_dir.iterdir()):
            if checkpoint_dir.is_dir() and checkpoint_dir.name != "final_scene":
                shutil.rmtree(checkpoint_dir)


def _promote_final_scene_states(variation_dir: Path) -> None:
    """Copy final room scene states to v2 layout paths."""
    room_dirs = get_room_scene_dirs(variation_dir)
    if not room_dirs:
        raise FileNotFoundError(f"No room_* directories found in {variation_dir}")

    for room_dir in room_dirs:
        final_state = room_dir / "scene_states" / "final_scene" / "scene_state.json"
        if not final_state.exists():
            raise FileNotFoundError(f"Missing final scene state: {final_state}")
        shutil.copy2(final_state, room_dir / "scene_state.json")

    shutil.copy2(
        room_dirs[0] / "scene_states" / "final_scene" / "scene_state.json",
        variation_dir / "scene_state.json",
    )


def _rebase_v2_variation_paths(
    base_scene_dir: Path,
    variation_dir: Path,
) -> None:
    """Rebase absolute paths from base_scene_dir to variation_dir."""
    warnings = _rewrite_copied_resume_paths(base_scene_dir, variation_dir)
    if warnings:
        raise RuntimeError(
            "Path rewrite incomplete after scaffold_variation_v2: "
            + "; ".join(warnings)
        )

    room_dirs = get_room_scene_dirs(variation_dir)
    if not room_dirs:
        return

    primary_room = room_dirs[0]
    root_state = variation_dir / "scene_state.json"
    if root_state.exists():
        _fix_paths_in_json_file(
            json_path=root_state,
            new_room_dir=primary_room,
            new_scene_dir=variation_dir,
        )
        if str(base_scene_dir.resolve()) in root_state.read_text():
            raise RuntimeError(f"Unresolvable absolute path remains in {root_state}")

    for room_dir in room_dirs:
        flat_state = room_dir / "scene_state.json"
        if flat_state.exists():
            _fix_paths_in_json_file(
                json_path=flat_state,
                new_room_dir=room_dir,
                new_scene_dir=variation_dir,
            )
            if str(base_scene_dir.resolve()) in flat_state.read_text():
                raise RuntimeError(
                    f"Unresolvable absolute path remains in {flat_state}"
                )

    house_yaml = variation_dir / "combined_house" / "house.dmd.yaml"
    if house_yaml.exists():
        _fix_paths_in_yaml_file(
            yaml_path=house_yaml,
            new_room_dir=primary_room,
            new_scene_dir=variation_dir,
        )
        if str(base_scene_dir.resolve()) in house_yaml.read_text():
            raise RuntimeError(f"Unresolvable absolute path remains in {house_yaml}")


def scaffold_variation_v2(
    base_scene_dir: Path,
    variation_dir: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Flat copy of a finished scene_NNN/ tree for DR v2 (no checkpoints).

    Returns the variation_dir on success. Copies scene artifacts, combined_house,
    and room final scene states; rebases absolute paths to the variation directory.

    Does NOT copy run_analysis/, intermediate scene_states checkpoints, or agent DBs.
    """
    base_scene_dir = base_scene_dir.resolve()
    variation_dir = variation_dir.resolve()

    if not base_scene_dir.is_dir():
        raise FileNotFoundError(f"Base scene directory not found: {base_scene_dir}")

    if variation_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Variation directory already exists: {variation_dir}"
            )
        shutil.rmtree(variation_dir)

    console_logger.info(
        "Scaffolding DR v2 variation from %s -> %s",
        base_scene_dir,
        variation_dir,
    )
    shutil.copytree(base_scene_dir, variation_dir)
    _prune_v2_variation_tree(variation_dir)
    _promote_final_scene_states(variation_dir)
    _rebase_v2_variation_paths(base_scene_dir, variation_dir)
    return variation_dir
