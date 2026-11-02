from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from scenesmith.domain_randomization.cli import _load_description, _load_existing_plan
from scenesmith.domain_randomization.orchestrator import (
    DomainRandomizationOrchestrator,
    _extract_object_ids,
)
from scenesmith.domain_randomization.scaffolding import get_room_scene_dirs
from scenesmith.domain_randomization.variation_spec import (
    OrchestrationPlan,
    VariationSpec,
    VariationStatus,
)


def _make_variation_status(
    variation_id: str,
    status: str,
) -> VariationStatus:
    return VariationStatus(
        spec=VariationSpec(
            variation_id=variation_id,
            description=f"variation {variation_id}",
            strategy="programmatic",
        ),
        status=status,
    )


def test_extract_object_ids_supports_string_and_dict_entries() -> None:
    assert _extract_object_ids(["cup_a", "cup_b"]) == ["cup_a", "cup_b"]
    assert _extract_object_ids(
        [{"object_id": "mug_a"}, {"object_id": "plate_b"}, {"other": "ignored"}]
    ) == ["mug_a", "plate_b"]
    assert _extract_object_ids("not-a-list") == []


def test_load_existing_plan_uses_output_dir_plan(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    output_dir = tmp_path / "randomizations" / "run_001"
    output_dir.mkdir(parents=True)

    plan = OrchestrationPlan(
        created_at="2026-05-06T00:00:00",
        base_experiment_dir=str(base_dir),
        description="resume me",
        num_variations=2,
        orchestrator_reasoning="test",
        variations={
            "variation_000": _make_variation_status("variation_000", "completed"),
            "variation_001": _make_variation_status("variation_001", "pending"),
        },
    )
    plan.save(output_dir / "orchestration_plan.json")

    loaded = _load_existing_plan(
        base_dir,
        {"output_dir": str(output_dir)},
    )

    assert loaded is not None
    assert loaded.base_experiment_dir == str(base_dir)
    assert set(loaded.variations) == {"variation_000", "variation_001"}


def test_load_description_prefers_description_file(tmp_path: Path) -> None:
    description_file = tmp_path / "dr_instructions.txt"
    description_file.write_text(
        "Keep the mug in place, vary clutter, and allow commas, semicolons, and quotes."
    )

    description = _load_description(
        {
            "description": "fallback text",
            "description_file": str(description_file),
        }
    )

    assert (
        description
        == "Keep the mug in place, vary clutter, and allow commas, semicolons, and quotes."
    )


def test_get_room_scene_dirs_ignores_room_geometry(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene_000"
    (scene_dir / "room_geometry").mkdir(parents=True)
    (scene_dir / "room_kitchen" / "scene_states").mkdir(parents=True)
    (scene_dir / "room_office" / "scene_states").mkdir(parents=True)

    room_dirs = get_room_scene_dirs(scene_dir)

    assert room_dirs == [
        scene_dir / "room_kitchen",
        scene_dir / "room_office",
    ]


@pytest.mark.asyncio
async def test_execute_resume_skips_completed_and_retries_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_dir = tmp_path / "experiment"
    scene_dir = base_dir / "scene_000"
    scene_dir.mkdir(parents=True)

    cfg = OmegaConf.create(
        {
            "openai": {"model": "test-model"},
            "domain_randomization": {
                "output_dir": str(tmp_path / "randomizations" / "run_001")
            },
        }
    )
    monkeypatch.setattr(
        "scenesmith.domain_randomization.orchestrator.make_chat_completions_model",
        lambda _model_name: object(),
    )

    orchestrator = DomainRandomizationOrchestrator(
        base_experiment_dir=base_dir,
        num_variations=99,
        cfg=cfg,
        logger=SimpleNamespace(),
    )
    plan = OrchestrationPlan(
        created_at="2026-05-06T00:00:00",
        base_experiment_dir=str(base_dir),
        description="resume me",
        num_variations=4,
        orchestrator_reasoning="test",
        variations={
            "completed_var": _make_variation_status("completed_var", "completed"),
            "pending_var": _make_variation_status("pending_var", "pending"),
            "failed_var": _make_variation_status("failed_var", "failed"),
            "in_progress_var": _make_variation_status("in_progress_var", "in_progress"),
        },
    )

    prepared = orchestrator._prepare_plan_for_resume(plan)
    scheduled: list[str] = []

    async def fake_run_single_variation(
        var_status: VariationStatus, _plan: OrchestrationPlan
    ) -> None:
        scheduled.append(var_status.spec.variation_id)

    monkeypatch.setattr(
        orchestrator, "_run_single_variation", fake_run_single_variation
    )

    await orchestrator._execute(prepared)

    assert orchestrator.num_variations == 4
    assert prepared.variations["in_progress_var"].status == "pending"
    assert set(scheduled) == {"pending_var", "failed_var", "in_progress_var"}


def test_root_config_includes_domain_randomization_defaults() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_dir = repo_root / "configurations"
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            cfg = compose(config_name="config", overrides=["+name=test_root_config"])
    finally:
        GlobalHydra.instance().clear()

    assert "domain_randomization" in cfg
    assert cfg.domain_randomization.num_variations == 10


def test_programmatic_collision_check_ignores_structural_objects_by_default(
    tmp_path: Path,
) -> None:
    import numpy as np

    from pydrake.all import RigidTransform

    from scenesmith.agent_utils.room import ObjectType, RoomScene, SceneObject, UniqueID
    from scenesmith.domain_randomization.programmatic.collision_checker import (
        check_collisions,
    )

    scene = RoomScene(
        room_geometry=None,
        scene_dir=tmp_path,
        text_description="structural overlap test",
    )
    wall = SceneObject(
        object_id=UniqueID("wall_0"),
        object_type=ObjectType.WALL,
        name="wall",
        description="structural wall",
        transform=RigidTransform(p=[0.0, 0.0, 0.0]),
        bbox_min=np.array([-1.0, -1.0, -1.0]),
        bbox_max=np.array([1.0, 1.0, 1.0]),
    )
    chair = SceneObject(
        object_id=UniqueID("chair_0"),
        object_type=ObjectType.FURNITURE,
        name="chair",
        description="chair near wall",
        transform=RigidTransform(p=[0.0, 0.0, 0.0]),
        bbox_min=np.array([-0.2, -0.2, -0.2]),
        bbox_max=np.array([0.2, 0.2, 0.2]),
    )
    scene.add_object(wall)
    scene.add_object(chair)

    assert check_collisions(scene) == []
    assert check_collisions(scene, ignore_structural=False) == [
        "Collision: wall_0 (wall) overlaps chair_0 (chair)"
    ]
