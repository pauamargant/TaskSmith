"""Post-run W&B export from analyzed TaskSmith experiments.

Reads ``run_analysis/experiment_summary.yaml`` + per-scene
``aggregated.json`` and posts a single W&B run with tables and bar
charts for every by_* slice the analyzer computes.
"""

from __future__ import annotations

import importlib
import json
import logging

from pathlib import Path
from typing import Any

import yaml

console_logger = logging.getLogger(__name__)


TOKEN_COLUMNS = [
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cached_tokens",
    "total_tokens",
    "num_images",
    "num_calls",
]


def _safe_int(value: Any) -> int:
    return int(value or 0)


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _sanitize_key(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace("|", "_")
    )


def _import_wandb():
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb is required for W&B export. Install project dependencies with "
            "`uv sync` before using --wandb-project."
        ) from exc

    # If a local ``./wandb/`` directory shadows the SDK, reload from sys.path.
    if not hasattr(wandb, "init"):
        import sys

        repo_root = str(Path.cwd())
        sys.modules.pop("wandb", None)
        sys.path = [p for p in sys.path if p not in ("", repo_root)]
        wandb = importlib.import_module("wandb")
        if not hasattr(wandb, "init"):
            raise RuntimeError(
                "Imported `wandb` module does not expose `wandb.init`. "
                "Rename/move any local `./wandb/` directory or run from a "
                "different CWD."
            )
    return wandb


def _row(prefix: list[Any], totals: dict[str, Any]) -> list[Any]:
    return prefix + [_safe_int(totals.get(col)) for col in TOKEN_COLUMNS]


def _bar(
    wandb: Any, *, rows: list[list[Any]], label: str, value: str, title: str
) -> Any:
    table = wandb.Table(columns=[label, value], data=rows)
    return wandb.plot.bar(
        table, label=label, value=value, title=title, split_table=True
    )


def _load_aggregated(scene_dir: Path) -> dict[str, Any] | None:
    path = scene_dir / "run_analysis" / "aggregated.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        console_logger.warning("failed to read %s: %s", path, exc)
        return None


def export_experiment_to_wandb(
    *,
    experiment_dir: Path,
    project: str,
    entity: str | None = None,
    mode: str = "online",
    tags: list[str] | None = None,
) -> str:
    """Post per-scene tables + per-slice bar charts to one W&B run."""

    wandb = _import_wandb()

    summary_path = experiment_dir / "run_analysis" / "experiment_summary.yaml"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Experiment summary not found at {summary_path}. Run the analyzer first."
        )
    with open(summary_path, encoding="utf-8") as f:
        summary = yaml.safe_load(f) or {}

    run = wandb.init(
        project=project,
        entity=entity,
        mode=mode,
        job_type="analysis",
        name=experiment_dir.name,
        config={
            "experiment_dir": str(experiment_dir),
            "num_scenes": summary.get("num_scenes", 0),
            "run_config": summary.get("run_config") or {},
        },
        tags=tags or ["tasksmith", "experiment-tracking"],
    )

    try:
        scene_rows: list[list[Any]] = []
        stage_rows: list[list[Any]] = []
        subsystem_rows: list[list[Any]] = []
        tool_rows: list[list[Any]] = []
        agent_role_rows: list[list[Any]] = []
        agent_type_rows: list[list[Any]] = []
        timing_rows: list[list[Any]] = []

        for scene in summary.get("scenes", []):
            scene_id = scene.get("scene_id")
            scene_dir = Path(scene.get("scene_dir") or "")
            agg = _load_aggregated(scene_dir) or {}
            totals = agg.get("totals") or scene.get("totals") or {}
            scene_rows.append(_row([scene_id], totals))

            for stage, t in (agg.get("by_stage") or {}).items():
                stage_rows.append(_row([scene_id, stage], t))
            for sub, t in (agg.get("by_subsystem") or {}).items():
                subsystem_rows.append(_row([scene_id, sub], t))
            for tool, t in (agg.get("by_tool") or {}).items():
                tool_rows.append(_row([scene_id, tool], t))
            for role, t in (agg.get("by_agent_role") or {}).items():
                agent_role_rows.append(_row([scene_id, role], t))
            for type_, t in (agg.get("by_agent_type") or {}).items():
                agent_type_rows.append(_row([scene_id, type_], t))
            for stage, seconds in (agg.get("stage_wall_time_s") or {}).items():
                timing_rows.append([scene_id, stage, _safe_float(seconds)])

        # Run-level scalar summaries.
        run_totals = summary.get("totals") or {}
        for metric in TOKEN_COLUMNS:
            run.summary[f"summary/{metric}"] = _safe_int(run_totals.get(metric))
        run.summary["summary/num_scenes"] = _safe_int(summary.get("num_scenes"))

        def _log_table(name: str, columns: list[str], rows: list[list[Any]]) -> None:
            if not rows:
                return
            wandb.log({name: wandb.Table(columns=columns, data=rows)})

        _log_table("tables/scene_summary", ["scene_id", *TOKEN_COLUMNS], scene_rows)
        _log_table(
            "tables/stage_usage", ["scene_id", "stage", *TOKEN_COLUMNS], stage_rows
        )
        _log_table(
            "tables/subsystem_usage",
            ["scene_id", "subsystem", *TOKEN_COLUMNS],
            subsystem_rows,
        )
        _log_table("tables/tool_usage", ["scene_id", "tool", *TOKEN_COLUMNS], tool_rows)
        _log_table(
            "tables/agent_role_usage",
            ["scene_id", "agent_role", *TOKEN_COLUMNS],
            agent_role_rows,
        )
        _log_table(
            "tables/agent_type_usage",
            ["scene_id", "agent_type", *TOKEN_COLUMNS],
            agent_type_rows,
        )
        _log_table(
            "tables/stage_timing",
            ["scene_id", "stage", "duration_s"],
            timing_rows,
        )

        def _aggregate_for_chart(
            rows: list[list[Any]], group_col_idx: int, value_col: str
        ) -> list[list[Any]]:
            """Sum a token column across scenes by ``group_col_idx``."""
            value_idx = 1 + TOKEN_COLUMNS.index(value_col)
            bucket: dict[str, int] = {}
            for row in rows:
                key = row[group_col_idx]
                bucket[key] = bucket.get(key, 0) + _safe_int(row[value_idx])
            return [[k, v] for k, v in sorted(bucket.items())]

        if stage_rows:
            wandb.log(
                {
                    "charts/stage_input_tokens": _bar(
                        wandb,
                        rows=_aggregate_for_chart(stage_rows, 1, "input_tokens"),
                        label="stage",
                        value="input_tokens",
                        title="Input tokens per stage",
                    ),
                    "charts/stage_output_tokens": _bar(
                        wandb,
                        rows=_aggregate_for_chart(stage_rows, 1, "output_tokens"),
                        label="stage",
                        value="output_tokens",
                        title="Output tokens per stage",
                    ),
                    "charts/stage_num_images": _bar(
                        wandb,
                        rows=_aggregate_for_chart(stage_rows, 1, "num_images"),
                        label="stage",
                        value="num_images",
                        title="Images per stage",
                    ),
                }
            )
        if subsystem_rows:
            wandb.log(
                {
                    "charts/subsystem_total_tokens": _bar(
                        wandb,
                        rows=_aggregate_for_chart(subsystem_rows, 1, "total_tokens"),
                        label="subsystem",
                        value="total_tokens",
                        title="Tokens per subsystem (physics/router/vlm_direct)",
                    ),
                }
            )
        if tool_rows:
            wandb.log(
                {
                    "charts/tool_total_tokens": _bar(
                        wandb,
                        rows=_aggregate_for_chart(tool_rows, 1, "total_tokens"),
                        label="tool",
                        value="total_tokens",
                        title="Tokens attributed per tool",
                    ),
                    "charts/tool_num_images": _bar(
                        wandb,
                        rows=_aggregate_for_chart(tool_rows, 1, "num_images"),
                        label="tool",
                        value="num_images",
                        title="Images attributed per tool",
                    ),
                }
            )
        if agent_role_rows:
            wandb.log(
                {
                    "charts/agent_role_total_tokens": _bar(
                        wandb,
                        rows=_aggregate_for_chart(agent_role_rows, 1, "total_tokens"),
                        label="agent_role",
                        value="total_tokens",
                        title="Tokens per agent role",
                    ),
                }
            )
        if timing_rows:
            stage_time: dict[str, float] = {}
            for _scene, stage, seconds in timing_rows:
                stage_time[stage] = stage_time.get(stage, 0.0) + _safe_float(seconds)
            wandb.log(
                {
                    "charts/stage_wall_time_s": _bar(
                        wandb,
                        rows=[[k, v] for k, v in sorted(stage_time.items())],
                        label="stage",
                        value="duration_s",
                        title="Wall time per stage (s)",
                    ),
                }
            )

        # Figures.
        for scene in summary.get("scenes", []):
            scene_id = scene.get("scene_id")
            figures = scene.get("figures") or {}
            for key in ("stage_progress_grid", "final_scene_grid"):
                path = figures.get(key)
                if path and Path(path).exists():
                    run.summary[f"figures/{scene_id}/{key}"] = wandb.Image(
                        path, caption=f"{scene_id} / {key}"
                    )
    finally:
        run_id = run.id
        wandb.finish()

    return run_id
