"""Live Weights & Biases tracking for in-progress TaskSmith scene generation.

Polls the per-LLM-call JSONL files written by the tracker and ships
aggregate tables/charts to W&B every ``poll_interval_s`` seconds. All
aggregation reuses :mod:`scenesmith.experiment_tracking.analyzer` so
the live view matches the post-run analysis exactly.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import threading

from pathlib import Path
from typing import Any

from .analyzer import aggregate, analyze_scene, load_events, load_records

console_logger = logging.getLogger(__name__)


def _get_nested(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def _safe_int(value: Any) -> int:
    return int(value or 0)


def _sanitize_key(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace("|", "_")
    )


class LiveWandbTracker:
    """Poll local tracking files and stream aggregate progress to W&B."""

    def __init__(
        self,
        *,
        scene_dir: Path,
        experiment_dir: Path,
        cfg_dict: dict[str, Any],
        scene_id: int,
        prompt: str,
        experiment_run_id: str | None,
    ) -> None:
        wandb_cfg = _get_nested(cfg_dict, "experiment", "tracking", "wandb", default={})
        self.enabled = bool(wandb_cfg.get("enabled", False))
        self.project = wandb_cfg.get("project")
        self.entity = wandb_cfg.get("entity")
        self.mode = wandb_cfg.get("mode", "online")
        self.poll_interval_s = float(wandb_cfg.get("poll_interval_s", 30.0))
        self.log_images = bool(wandb_cfg.get("log_images", True))
        self.fail_on_error = bool(wandb_cfg.get("fail_on_error", False))
        self.scene_dir = scene_dir
        self.experiment_dir = experiment_dir
        self.cfg_dict = cfg_dict
        self.scene_id = scene_id
        self.scene_name = f"scene_{scene_id:03d}"
        self.prompt = prompt
        self.experiment_run_id = experiment_run_id
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._wandb: Any = None
        self._run: Any = None
        self._last_event_count = 0
        self._last_record_count = 0
        self._logged_figure_mtimes: dict[str, float] = {}
        self._step = 0

    def start(self) -> None:
        if not self.enabled:
            return
        if not self.project:
            message = "experiment.tracking.wandb.enabled=true requires wandb.project"
            if self.fail_on_error:
                raise ValueError(message)
            console_logger.warning(message)
            return
        try:
            import wandb

            if not hasattr(wandb, "init"):
                import sys

                repo_root = str(Path.cwd())
                sys.modules.pop("wandb", None)
                sys.path = [p for p in sys.path if p not in ("", repo_root)]
                wandb = importlib.import_module("wandb")
            self._wandb = wandb
            self._run = wandb.init(
                project=self.project,
                entity=self.entity,
                mode=self.mode,
                job_type="generation",
                name=f"{self.scene_name}-{self.experiment_dir.name}",
                group=self.experiment_run_id,
                config={
                    "experiment_dir": str(self.experiment_dir),
                    "scene_dir": str(self.scene_dir),
                    "scene_id": self.scene_name,
                    "prompt": self.prompt,
                    "experiment_name": self.cfg_dict.get("name"),
                    "live_tracking": True,
                },
                tags=["tasksmith", "live-generation"],
            )
            self._thread = threading.Thread(
                target=self._run_loop,
                name=f"wandb-live-{self.scene_name}",
                daemon=True,
            )
            self._thread.start()
        except Exception as exc:
            if self.fail_on_error:
                raise
            console_logger.warning(
                "Live W&B tracking disabled after init error: %s", exc
            )
            self._run = None

    def stop(self) -> None:
        if self._run is None:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.poll_interval_s + 1.0))
        try:
            self._poll_once(final=True)
        except Exception as exc:
            console_logger.warning("Final live W&B poll failed: %s", exc)
        try:
            self._wandb.finish()
        finally:
            self._run = None

    def _run_loop(self) -> None:
        while not self._stop_event.wait(self.poll_interval_s):
            try:
                self._poll_once(final=False)
            except Exception as exc:
                console_logger.warning("Live W&B poll failed: %s", exc)

    def _poll_once(self, *, final: bool) -> None:
        if self._run is None or self._wandb is None:
            return

        records = load_records(self.scene_dir)
        events = load_events(self.scene_dir)
        agg = aggregate(records, events).as_dict()

        payload: dict[str, Any] = {
            "live/events_seen": len(events),
            "live/llm_calls_seen": len(records),
        }
        if events:
            latest = events[-1]
            self._run.summary["live/current_stage"] = latest.get("stage_name")
            self._run.summary["live/current_status"] = latest.get("status")
            self._run.summary["live/latest_event_type"] = latest.get("event_type")

        totals = agg.get("totals") or {}
        for key, value in totals.items():
            payload[f"live_totals/{key}"] = value
            self._run.summary[f"live_totals/{key}"] = value

        if len(events) != self._last_event_count:
            payload["tables/live_events"] = self._events_table(events)
            self._last_event_count = len(events)

        if len(records) != self._last_record_count:
            payload.update(self._slice_tables(agg))
            payload.update(self._slice_charts(agg))
            self._last_record_count = len(records)

        self._log_figures_if_available(payload)

        if final:
            try:
                summary = analyze_scene(self.scene_dir)
                self._run.summary["final/total_tokens"] = _safe_int(
                    summary.get("totals", {}).get("total_tokens")
                )
                self._run.summary["final/num_calls"] = _safe_int(
                    summary.get("totals", {}).get("num_calls")
                )
                self._run.summary["final/num_images"] = _safe_int(
                    summary.get("totals", {}).get("num_images")
                )
            except Exception as exc:
                console_logger.warning("final analyze_scene failed: %s", exc)
            self._log_figures_if_available(payload)

        self._step += 1
        self._wandb.log(payload, step=self._step)

    def _events_table(self, events: list[dict[str, Any]]) -> Any:
        columns = [
            "timestamp",
            "stage_name",
            "event_type",
            "status",
            "room_id",
            "duration_s",
            "notes",
            "output_paths",
        ]
        rows = [
            [
                e.get("timestamp"),
                e.get("stage_name") or "unknown",
                e.get("event_type") or "event",
                e.get("status") or "unknown",
                e.get("room_id"),
                e.get("duration_s"),
                e.get("notes"),
                e.get("output_paths") or [],
            ]
            for e in events
        ]
        return self._wandb.Table(columns=columns, data=rows)

    @staticmethod
    def _slice_to_rows(slice_dict: dict[str, dict[str, Any]]) -> list[list[Any]]:
        return [
            [
                key,
                _safe_int(totals.get("input_tokens")),
                _safe_int(totals.get("output_tokens")),
                _safe_int(totals.get("total_tokens")),
                _safe_int(totals.get("num_images")),
                _safe_int(totals.get("num_calls")),
            ]
            for key, totals in sorted(slice_dict.items())
        ]

    def _slice_tables(self, agg: dict[str, Any]) -> dict[str, Any]:
        cols = [
            "key",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "num_images",
            "num_calls",
        ]
        out: dict[str, Any] = {}
        for slice_name in ("by_stage", "by_subsystem", "by_tool", "by_agent_role"):
            rows = self._slice_to_rows(agg.get(slice_name) or {})
            if not rows:
                continue
            out[f"tables/live_{slice_name}"] = self._wandb.Table(
                columns=cols, data=rows
            )
        return out

    def _slice_charts(self, agg: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if not self._wandb:
            return out

        def _bar(rows: list[list[Any]], label: str, value: str, title: str) -> Any:
            table = self._wandb.Table(columns=[label, value], data=rows)
            return self._wandb.plot.bar(
                table, label=label, value=value, title=title, split_table=True
            )

        for slice_name, label in (
            ("by_stage", "stage"),
            ("by_subsystem", "subsystem"),
            ("by_tool", "tool"),
            ("by_agent_role", "agent_role"),
        ):
            slice_dict = agg.get(slice_name) or {}
            if not slice_dict:
                continue
            input_rows = [
                [k, _safe_int(t.get("input_tokens"))]
                for k, t in sorted(slice_dict.items())
            ]
            out[f"charts/live_{slice_name}_input_tokens"] = _bar(
                input_rows,
                label,
                "input_tokens",
                f"Live: input tokens per {label}",
            )
            image_rows = [
                [k, _safe_int(t.get("num_images"))]
                for k, t in sorted(slice_dict.items())
            ]
            if any(r[1] for r in image_rows):
                out[f"charts/live_{slice_name}_num_images"] = _bar(
                    image_rows,
                    label,
                    "num_images",
                    f"Live: images per {label}",
                )

        stage_wall = agg.get("stage_wall_time_s") or {}
        if stage_wall:
            out["charts/live_stage_wall_time_s"] = _bar(
                [[k, float(v)] for k, v in sorted(stage_wall.items())],
                "stage",
                "duration_s",
                "Live: wall time per stage (s)",
            )
        return out

    def _log_figures_if_available(self, payload: dict[str, Any]) -> None:
        if not self.log_images:
            return
        figures = {
            "stage_progress_grid": self.scene_dir
            / "run_analysis"
            / "figures"
            / "stage_progress_grid.png",
            "final_scene_grid": self.scene_dir
            / "run_analysis"
            / "figures"
            / "final_scene_grid.png",
        }
        for name, path in figures.items():
            if not path.exists():
                continue
            mtime = path.stat().st_mtime
            if self._logged_figure_mtimes.get(name) == mtime:
                continue
            payload[f"live_figures/{name}"] = self._wandb.Image(
                str(path), caption=f"{self.scene_name} / {name}"
            )
            self._logged_figure_mtimes[name] = mtime


class _NullLiveWandbTracker:
    def start(self) -> None:
        return

    def stop(self) -> None:
        return


def build_live_wandb_tracker(
    *,
    scene_dir: Path,
    experiment_dir: Path,
    cfg_dict: dict[str, Any],
    scene_id: int,
    prompt: str,
    experiment_run_id: str | None,
) -> LiveWandbTracker | _NullLiveWandbTracker:
    tracker = LiveWandbTracker(
        scene_dir=scene_dir,
        experiment_dir=experiment_dir,
        cfg_dict=cfg_dict,
        scene_id=scene_id,
        prompt=prompt,
        experiment_run_id=experiment_run_id,
    )
    return tracker if tracker.enabled else _NullLiveWandbTracker()


@contextlib.contextmanager
def live_wandb_tracking(
    *,
    scene_dir: Path,
    experiment_dir: Path,
    cfg_dict: dict[str, Any],
    scene_id: int,
    prompt: str,
    experiment_run_id: str | None,
):
    """Context manager: start a tracker on enter, stop it on exit."""

    tracker = build_live_wandb_tracker(
        scene_dir=scene_dir,
        experiment_dir=experiment_dir,
        cfg_dict=cfg_dict,
        scene_id=scene_id,
        prompt=prompt,
        experiment_run_id=experiment_run_id,
    )
    try:
        tracker.start()
        yield tracker
    finally:
        tracker.stop()
