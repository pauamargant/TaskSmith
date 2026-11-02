#!/usr/bin/env python3
"""Analyze TaskSmith experiment outputs and optionally export them to W&B."""

from __future__ import annotations

import argparse
import sys

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scenesmith.experiment_tracking import analyze_experiment, finalize_scene_analysis
from scenesmith.experiment_tracking.wandb_export import export_experiment_to_wandb


def _load_run_config(path: Path) -> dict:
    """Find a usable cfg dict for analysis-time helpers.

    Order tried:
      1. ``<path>/.hydra/config.yaml`` — the Hydra snapshot the
         experiment originally wrote next to its outputs. Present for
         `main.py` experiments; absent for DR variant directories
         (DR scaffolds copy artifacts only).
      2. Walk parents up to 4 levels looking for a `.hydra/config.yaml`
         (handles the DR variant case where the snapshot lives at the
         base experiment dir several levels up).
      3. Hydra compose against `configurations/` — gives a complete cfg
         identical to what a fresh `main.py` run would see. Required so
         downstream callers like `render_rooms_for_analysis` find
         `furniture_agent.rendering.retry_count` and friends.
    """
    hydra_cfg = path / ".hydra" / "config.yaml"
    if hydra_cfg.exists():
        with open(hydra_cfg, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        data.setdefault("experiment", {}).setdefault("tracking", {}).setdefault(
            "analysis_renders", {}
        )["enabled"] = True
        return data

    # Walk up looking for an experiment-level .hydra snapshot.
    walker = path
    for _ in range(4):
        walker = walker.parent
        candidate = walker / ".hydra" / "config.yaml"
        if candidate.exists():
            with open(candidate, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            data.setdefault("experiment", {}).setdefault("tracking", {}).setdefault(
                "analysis_renders", {}
            )["enabled"] = True
            return data

    # Last resort: compose against configurations/ via Hydra so analysis-
    # time helpers (render_rooms_for_analysis etc.) find every key they
    # expect. Previous behaviour was a 3-key minimal stub that crashed
    # immediately with `Missing key retry_count` on the first render call.
    try:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf

        config_dir = REPO_ROOT / "configurations"
        GlobalHydra.instance().clear()
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            cfg = compose(
                config_name="config",
                overrides=["+name=analyze_experiment_run"],
            )
        data = OmegaConf.to_container(cfg, resolve=True) or {}
        data.setdefault("experiment", {}).setdefault("tracking", {}).setdefault(
            "analysis_renders", {}
        )["enabled"] = True
        return data
    except Exception:
        # Compose itself failed — fall back to the historical stub so the
        # non-render code paths still work.
        return {"experiment": {"tracking": {"analysis_renders": {"enabled": True}}}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="Scene or experiment output directory")
    parser.add_argument(
        "--render-analysis",
        action="store_true",
        help="Render whole-room analysis views before aggregating (requires Blender)",
    )
    parser.add_argument(
        "--wandb-project",
        help="Export analyzed experiment summary to this W&B project",
    )
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--wandb-mode",
        default="online",
        choices=["online", "offline", "disabled"],
    )
    args = parser.parse_args()

    path = args.path.resolve()
    if (path / "run_analysis" / "usage_records.jsonl").exists() or path.name.startswith(
        "scene_"
    ):
        cfg_dict = _load_run_config(
            path.parent if path.name.startswith("scene_") else path
        )
        if args.render_analysis:
            from scenesmith.experiment_tracking.analysis_renders import (
                render_rooms_for_analysis,
            )

            render_rooms_for_analysis(path, cfg_dict)
            from scenesmith.experiment_tracking import analyze_scene

            summary = analyze_scene(path)
        else:
            summary = finalize_scene_analysis(path, cfg_dict)
        print(f"Analyzed scene: {summary['scene_id']}")
        print(f"  total_tokens={summary['totals']['total_tokens']:,}")
        print(f"  output={path / 'run_analysis'}")
        return

    cfg_dict = _load_run_config(path)
    if args.render_analysis:
        from scenesmith.experiment_tracking import analyze_scene
        from scenesmith.experiment_tracking.analysis_renders import (
            render_rooms_for_analysis,
        )

        for scene_dir in sorted(path.glob("scene_*")):
            if scene_dir.is_dir():
                render_rooms_for_analysis(scene_dir, cfg_dict)
                analyze_scene(scene_dir)
        from scenesmith.experiment_tracking import analyze_experiment as analyze_exp

        summary = analyze_exp(path)
    else:
        summary = analyze_experiment(path)
    print(f"Analyzed experiment: {path}")
    print(f"  scenes={summary['num_scenes']}")
    print(f"  total_tokens={summary['totals']['total_tokens']:,}")
    print(f"  output={path / 'run_analysis'}")

    if args.wandb_project:
        run_id = export_experiment_to_wandb(
            experiment_dir=path,
            project=args.wandb_project,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
        )
        print(f"Exported to W&B run: {run_id}")


if __name__ == "__main__":
    main()
