"""Render whole-room analysis views for a DR v2 variant scene.

The shipped ``scripts/analyze_experiment_run.py --render-analysis`` path
builds a minimal cfg in-line (analyzer.py:576) that's missing
``furniture_agent.rendering.retry_count``, so it crashes immediately
when run standalone against a v2 variant. This script bypasses that by
composing the full Hydra config and calling ``render_rooms_for_analysis``
directly — same code path used by the main experiment.

Usage:
    uv run python scripts/render_dr_variant.py <scene_dir>
"""

from __future__ import annotations

import sys

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scenesmith.experiment_tracking.analysis_renders import render_rooms_for_analysis


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: render_dr_variant.py <scene_dir>")
        sys.exit(2)
    scene_dir = Path(sys.argv[1]).resolve()
    if not scene_dir.is_dir():
        print(f"scene_dir not found: {scene_dir}")
        sys.exit(2)

    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "configurations"
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=["+name=render_dr_variant"])
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    # Force-enable analysis renders regardless of experiment.tracking defaults.
    cfg_dict.setdefault("experiment", {}).setdefault("tracking", {}).setdefault(
        "analysis_renders", {}
    )["enabled"] = True

    rendered = render_rooms_for_analysis(scene_dir, cfg_dict)
    if not rendered:
        print("no rooms rendered")
        sys.exit(1)
    for room_id, pngs in sorted(rendered.items()):
        print(f"room {room_id}:")
        for p in pngs:
            print(f"  {p}")


if __name__ == "__main__":
    main()
