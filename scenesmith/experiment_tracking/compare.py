"""Cross-run comparison for ablations and report tables.

Given multiple experiment directories, load each one's analyzed
summary + per-scene ``aggregated.json``, sum slices across scenes,
compute deltas against a baseline run, and emit markdown / CSV
output for the ``experiments.md`` ablation tables.
"""

from __future__ import annotations

import csv
import io
import json

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_NULL = "—"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return None


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _find_experiment_summary(path: Path) -> Path | None:
    """Return the experiment-level summary path if present."""
    direct = path / "run_analysis" / "experiment_summary.yaml"
    if direct.exists():
        return direct
    return None


@dataclass
class RunRollup:
    """Per-experiment rollup, summed across scenes."""

    name: str
    path: Path
    run_config: dict[str, Any] = field(default_factory=dict)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_num_images: int = 0
    total_num_calls: int = 0
    total_duration_s: float = 0.0
    wall_span_s: float = 0.0
    by_stage: dict[str, dict[str, float]] = field(default_factory=dict)
    by_subsystem: dict[str, dict[str, float]] = field(default_factory=dict)
    by_tool: dict[str, dict[str, float]] = field(default_factory=dict)
    by_agent_role: dict[str, dict[str, float]] = field(default_factory=dict)
    stage_wall_time_s: dict[str, float] = field(default_factory=dict)


def _merge_slice(
    target: dict[str, dict[str, float]],
    src: dict[str, dict[str, Any]] | None,
) -> None:
    if not src:
        return
    for key, totals in src.items():
        bucket = target.setdefault(
            key,
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "num_images": 0,
                "num_calls": 0,
                "duration_s": 0.0,
            },
        )
        bucket["input_tokens"] += int(totals.get("input_tokens") or 0)
        bucket["output_tokens"] += int(totals.get("output_tokens") or 0)
        bucket["total_tokens"] += int(totals.get("total_tokens") or 0)
        bucket["num_images"] += int(totals.get("num_images") or 0)
        bucket["num_calls"] += int(totals.get("num_calls") or 0)
        bucket["duration_s"] += float(totals.get("duration_s") or 0.0)


def load_run(path: Path) -> RunRollup:
    """Load an experiment (or scene) directory into a RunRollup."""

    path = path.resolve()
    rollup = RunRollup(name=path.name, path=path)

    summary_path = _find_experiment_summary(path)
    if summary_path is None:
        # Fall back to single-scene: path is itself a scene dir.
        if (path / "run_analysis" / "aggregated.json").exists():
            scene_dirs = [path]
            run_config: dict[str, Any] = {}
        else:
            return rollup
    else:
        summary = _load_yaml(summary_path) or {}
        run_config = summary.get("run_config") or {}
        scene_dirs = [
            Path(s.get("scene_dir") or "")
            for s in summary.get("scenes", [])
            if s.get("scene_dir")
        ]

    rollup.run_config = run_config

    for scene_dir in scene_dirs:
        agg = _load_json(scene_dir / "run_analysis" / "aggregated.json") or {}
        totals = agg.get("totals") or {}
        rollup.total_input_tokens += int(totals.get("input_tokens") or 0)
        rollup.total_output_tokens += int(totals.get("output_tokens") or 0)
        rollup.total_tokens += int(totals.get("total_tokens") or 0)
        rollup.total_num_images += int(totals.get("num_images") or 0)
        rollup.total_num_calls += int(totals.get("num_calls") or 0)
        rollup.total_duration_s += float(totals.get("duration_s") or 0.0)
        wall = agg.get("wall_span_s")
        if wall is not None:
            try:
                rollup.wall_span_s += float(wall)
            except Exception:
                pass

        _merge_slice(rollup.by_stage, agg.get("by_stage"))
        _merge_slice(rollup.by_subsystem, agg.get("by_subsystem"))
        _merge_slice(rollup.by_tool, agg.get("by_tool"))
        _merge_slice(rollup.by_agent_role, agg.get("by_agent_role"))

        stage_wall = agg.get("stage_wall_time_s") or {}
        for stage, seconds in stage_wall.items():
            try:
                rollup.stage_wall_time_s[stage] = rollup.stage_wall_time_s.get(
                    stage, 0.0
                ) + float(seconds)
            except Exception:
                continue

    return rollup


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _delta(value: float, baseline: float) -> str:
    if baseline == 0:
        return _NULL
    return f"{(value - baseline) / baseline * 100:+.1f}%"


def _fmt_int(value: int | float) -> str:
    return f"{int(value):,}"


def _fmt_float(value: float, decimals: int = 1) -> str:
    return f"{float(value):,.{decimals}f}"


@dataclass
class ComparisonResult:
    runs: list[RunRollup]
    baseline: RunRollup

    def to_markdown(self, per_stage: bool = False, per_tool: bool = False) -> str:
        baseline = self.baseline
        lines: list[str] = []

        # Main table.
        lines.append(
            "| Config | Input tokens | Δ in | Output tokens | Δ out | Total tokens | "
            "Wall span (s) | Δ wall | Calls | Images | Physics tok | Router tok |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for run in self.runs:
            phys_tok = int(
                (run.by_subsystem.get("physics_analysis") or {}).get("total_tokens")
                or 0
            )
            router_tok = int(
                (run.by_subsystem.get("router_validation") or {}).get("total_tokens")
                or 0
            )
            lines.append(
                "| {name} | {inp} | {dinp} | {out} | {dout} | {tot} | {wall} | {dwall} | "
                "{calls} | {imgs} | {phys} | {router} |".format(
                    name=run.name,
                    inp=_fmt_int(run.total_input_tokens),
                    dinp=_delta(run.total_input_tokens, baseline.total_input_tokens),
                    out=_fmt_int(run.total_output_tokens),
                    dout=_delta(run.total_output_tokens, baseline.total_output_tokens),
                    tot=_fmt_int(run.total_tokens),
                    wall=_fmt_float(run.wall_span_s),
                    dwall=_delta(run.wall_span_s, baseline.wall_span_s),
                    calls=_fmt_int(run.total_num_calls),
                    imgs=_fmt_int(run.total_num_images),
                    phys=_fmt_int(phys_tok),
                    router=_fmt_int(router_tok),
                )
            )

        if per_stage:
            lines.append("")
            lines.append("### Per-stage breakdown (tokens)")
            stages = sorted({s for run in self.runs for s in run.by_stage.keys()})
            header = "| Stage | " + " | ".join(run.name for run in self.runs) + " |"
            sep = "|---|" + "|".join(["---:" for _ in self.runs]) + "|"
            lines.append(header)
            lines.append(sep)
            for stage in stages:
                cells = [
                    _fmt_int((run.by_stage.get(stage) or {}).get("total_tokens") or 0)
                    for run in self.runs
                ]
                lines.append(f"| {stage} | " + " | ".join(cells) + " |")

            lines.append("")
            lines.append("### Per-stage wall time (s)")
            lines.append(header)
            lines.append(sep)
            for stage in stages:
                cells = [
                    _fmt_float(run.stage_wall_time_s.get(stage, 0.0))
                    for run in self.runs
                ]
                lines.append(f"| {stage} | " + " | ".join(cells) + " |")

        if per_tool:
            lines.append("")
            lines.append("### Per-tool token attribution")
            tools = sorted({t for run in self.runs for t in run.by_tool.keys()})
            header = "| Tool | " + " | ".join(run.name for run in self.runs) + " |"
            sep = "|---|" + "|".join(["---:" for _ in self.runs]) + "|"
            lines.append(header)
            lines.append(sep)
            for tool in tools:
                cells = [
                    _fmt_int((run.by_tool.get(tool) or {}).get("total_tokens") or 0)
                    for run in self.runs
                ]
                lines.append(f"| {tool} | " + " | ".join(cells) + " |")

        if any(run.run_config for run in self.runs):
            lines.append("")
            lines.append("### Run config")
            keys = sorted({k for run in self.runs for k in run.run_config.keys()})
            header = "| Key | " + " | ".join(run.name for run in self.runs) + " |"
            sep = "|---|" + "|".join(["---" for _ in self.runs]) + "|"
            lines.append(header)
            lines.append(sep)
            for key in keys:
                cells = [
                    str(run.run_config.get(key, "")) if run.run_config else ""
                    for run in self.runs
                ]
                lines.append(f"| {key} | " + " | ".join(cells) + " |")

        return "\n".join(lines)

    def to_csv(self) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            [
                "name",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "wall_span_s",
                "num_calls",
                "num_images",
                "physics_total_tokens",
                "router_total_tokens",
            ]
        )
        for run in self.runs:
            writer.writerow(
                [
                    run.name,
                    run.total_input_tokens,
                    run.total_output_tokens,
                    run.total_tokens,
                    run.wall_span_s,
                    run.total_num_calls,
                    run.total_num_images,
                    int(
                        (run.by_subsystem.get("physics_analysis") or {}).get(
                            "total_tokens"
                        )
                        or 0
                    ),
                    int(
                        (run.by_subsystem.get("router_validation") or {}).get(
                            "total_tokens"
                        )
                        or 0
                    ),
                ]
            )
        return buf.getvalue()


def compare_experiments(
    paths: list[Path],
    baseline: Path | None = None,
) -> ComparisonResult:
    runs = [load_run(p) for p in paths]
    if not runs:
        raise ValueError("compare_experiments needs at least one path")
    if baseline is None:
        baseline_run = runs[0]
    else:
        baseline_path = baseline.resolve()
        baseline_run = next((r for r in runs if r.path == baseline_path), None)
        if baseline_run is None:
            baseline_run = load_run(baseline)
            runs = [baseline_run] + runs
    return ComparisonResult(runs=runs, baseline=baseline_run)
