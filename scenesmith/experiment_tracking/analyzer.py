"""Post-run analysis for TaskSmith experiment tracking.

Consumes the JSONL produced at runtime:
- ``run_analysis/usage_records.jsonl`` — one row per LLM API call.
- ``run_analysis/events.jsonl`` — stage start/end events.
- ``run_analysis/artifacts.jsonl`` — file outputs.

and produces:
- ``run_analysis/aggregated.json`` — single source of truth for all
  by_* slices (per-stage, per-agent, per-tool, per-subsystem).
- ``run_analysis/scene_summary.yaml`` — high-level summary.
- ``run_analysis/media_manifest.json`` + figure PNGs.

Per-tool token attribution rule: an LLM call's tokens are added to
every tool listed in ``tool_calls_in_response``. Calls with no tool
(text-only final messages) go to the synthetic ``"_no_tool"`` bucket.
``by_stage`` and the run totals are computed without this duplication.
"""

from __future__ import annotations

import json

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import yaml

from PIL import Image, ImageDraw

from .analysis_renders import ANALYSIS_RENDER_BATCH, ANALYSIS_RENDER_SUBDIR
from .io import load_jsonl, load_scene_and_room_jsonl
from .records import LLMCallRecord, parse_agent_label
from .writer import write_scene_json, write_scene_summary

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _coerce_record(row: dict[str, Any]) -> LLMCallRecord | None:
    """Normalize a JSONL row to ``LLMCallRecord``, tolerating old shapes."""

    if not isinstance(row, dict):
        return None
    try:
        agent_label = row.get("agent_label")
        agent_role = row.get("agent_role")
        agent_type = row.get("agent_type")
        if agent_label and (agent_role is None or agent_type is None):
            parsed_role, parsed_type = parse_agent_label(agent_label)
            agent_role = agent_role or parsed_role
            agent_type = agent_type or parsed_type

        # Old runs used source="runtime_direct" / "runtime_agentsdk".
        source = row.get("source") or "vlm_direct"
        if source == "runtime_direct":
            source = "vlm_direct"
        elif source == "runtime_agentsdk":
            source = "agents_sdk"

        return LLMCallRecord(
            timestamp=str(row.get("timestamp") or ""),
            source=source,
            provider_path=str(row.get("provider_path") or row.get("provider") or ""),
            scene_id=row.get("scene_id"),
            room_id=row.get("room_id"),
            stage_name=row.get("stage_name"),
            agent_label=agent_label,
            agent_role=agent_role,
            agent_type=agent_type,
            subsystem=row.get("subsystem"),
            tool_call_target=row.get("tool_call_target"),
            tool_calls_in_response=list(row.get("tool_calls_in_response") or []),
            model=row.get("model"),
            request_index=row.get("request_index"),
            input_tokens=int(row.get("input_tokens") or 0),
            output_tokens=int(row.get("output_tokens") or 0),
            reasoning_tokens=int(row.get("reasoning_tokens") or 0),
            cached_tokens=int(row.get("cached_tokens") or 0),
            total_tokens=int(row.get("total_tokens") or 0),
            num_images=int(row.get("num_images") or 0),
            duration_s=(
                float(row["duration_s"]) if row.get("duration_s") is not None else None
            ),
        )
    except Exception:
        return None


def _dedupe_records(records: Iterable[LLMCallRecord]) -> list[LLMCallRecord]:
    """Remove exact duplicates created by scene/room mirror writes."""

    seen: set[tuple[Any, ...]] = set()
    unique: list[LLMCallRecord] = []
    for record in records:
        key = (
            record.timestamp,
            record.source,
            record.agent_label,
            record.subsystem,
            record.stage_name,
            record.request_index,
            record.input_tokens,
            record.output_tokens,
            record.total_tokens,
            record.num_images,
            record.duration_s,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def load_records(scene_dir: Path) -> list[LLMCallRecord]:
    """Load and dedupe per-LLM-call records for one scene."""

    raw = load_scene_and_room_jsonl(scene_dir, "usage_records.jsonl")
    coerced = [r for r in (_coerce_record(row) for row in raw) if r is not None]
    return _dedupe_records(coerced)


def _dedupe_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    unique: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        key = (
            event.get("event_type"),
            event.get("stage_name"),
            event.get("status"),
            event.get("scene_id"),
            event.get("room_id"),
            event.get("started_at"),
            event.get("ended_at"),
            event.get("duration_s"),
            event.get("notes"),
            tuple(event.get("output_paths") or []),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(event)
    unique.sort(key=lambda row: str(row.get("timestamp") or ""))
    return unique


def load_events(scene_dir: Path) -> list[dict[str, Any]]:
    """Load stage events for one scene, deduped against the room mirror."""

    events = load_jsonl(scene_dir / "run_analysis" / "events.jsonl")
    events.extend(load_scene_and_room_jsonl(scene_dir, "stage_events.jsonl"))
    return _dedupe_events(events)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class TokenTotals:
    """Token + image + timing rollup for any slice."""

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    num_images: int = 0
    num_calls: int = 0  # number of LLM API calls contributing to this slice
    duration_s: float = 0.0

    def add(self, record: LLMCallRecord) -> None:
        self.input_tokens += record.input_tokens
        self.output_tokens += record.output_tokens
        self.reasoning_tokens += record.reasoning_tokens
        self.cached_tokens += record.cached_tokens
        self.total_tokens += record.total_tokens
        self.num_images += record.num_images
        self.num_calls += 1
        if record.duration_s is not None:
            self.duration_s += float(record.duration_s)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _new_totals() -> TokenTotals:
    return TokenTotals()


def _stage_time_from_events(events: list[dict[str, Any]]) -> dict[str, float]:
    """Sum stage durations from event log (stage start/end give wall time)."""

    by_stage: dict[str, float] = defaultdict(float)
    for event in events:
        duration = event.get("duration_s")
        if duration is None:
            continue
        stage = event.get("stage_name") or "unknown"
        try:
            by_stage[stage] += float(duration)
        except (TypeError, ValueError):
            continue
    return dict(by_stage)


def _wall_span_from_events(events: list[dict[str, Any]]) -> float | None:
    """Return wall-clock span (earliest to latest timestamp) across events."""

    earliest: datetime | None = None
    latest: datetime | None = None
    for event in events:
        for key in ("started_at", "timestamp", "ended_at"):
            raw = event.get(key)
            if not raw:
                continue
            try:
                if isinstance(raw, str) and raw.endswith("Z"):
                    raw = raw[:-1] + "+00:00"
                ts = datetime.fromisoformat(raw) if isinstance(raw, str) else None
            except Exception:
                continue
            if ts is None:
                continue
            if earliest is None or ts < earliest:
                earliest = ts
            if latest is None or ts > latest:
                latest = ts
    if earliest is None or latest is None:
        return None
    return max(0.0, (latest - earliest).total_seconds())


@dataclass
class AggregatedScene:
    """All by_* slices for one scene. Written as ``aggregated.json``."""

    totals: TokenTotals = field(default_factory=_new_totals)
    by_stage: dict[str, TokenTotals] = field(default_factory=dict)
    by_agent_label: dict[str, TokenTotals] = field(default_factory=dict)
    by_agent_role: dict[str, TokenTotals] = field(default_factory=dict)
    by_agent_type: dict[str, TokenTotals] = field(default_factory=dict)
    by_subsystem: dict[str, TokenTotals] = field(default_factory=dict)
    by_tool: dict[str, TokenTotals] = field(default_factory=dict)
    by_stage_agent_role: dict[str, dict[str, TokenTotals]] = field(default_factory=dict)
    by_stage_agent_type: dict[str, dict[str, TokenTotals]] = field(default_factory=dict)
    by_stage_subsystem: dict[str, dict[str, TokenTotals]] = field(default_factory=dict)
    by_stage_tool: dict[str, dict[str, TokenTotals]] = field(default_factory=dict)
    stage_wall_time_s: dict[str, float] = field(default_factory=dict)
    wall_span_s: float | None = None

    def as_dict(self) -> dict[str, Any]:
        def _flat(d: dict[str, TokenTotals]) -> dict[str, Any]:
            return {k: v.as_dict() for k, v in sorted(d.items())}

        def _matrix(d: dict[str, dict[str, TokenTotals]]) -> dict[str, Any]:
            return {k: _flat(v) for k, v in sorted(d.items())}

        return {
            "totals": self.totals.as_dict(),
            "by_stage": _flat(self.by_stage),
            "by_agent_label": _flat(self.by_agent_label),
            "by_agent_role": _flat(self.by_agent_role),
            "by_agent_type": _flat(self.by_agent_type),
            "by_subsystem": _flat(self.by_subsystem),
            "by_tool": _flat(self.by_tool),
            "by_stage_agent_role": _matrix(self.by_stage_agent_role),
            "by_stage_agent_type": _matrix(self.by_stage_agent_type),
            "by_stage_subsystem": _matrix(self.by_stage_subsystem),
            "by_stage_tool": _matrix(self.by_stage_tool),
            "stage_wall_time_s": dict(sorted(self.stage_wall_time_s.items())),
            "wall_span_s": self.wall_span_s,
        }


def _bucket(d: dict[str, TokenTotals], key: str, record: LLMCallRecord) -> None:
    if key not in d:
        d[key] = TokenTotals()
    d[key].add(record)


def _matrix_bucket(
    d: dict[str, dict[str, TokenTotals]],
    outer: str,
    inner: str,
    record: LLMCallRecord,
) -> None:
    if outer not in d:
        d[outer] = {}
    if inner not in d[outer]:
        d[outer][inner] = TokenTotals()
    d[outer][inner].add(record)


_UNKNOWN = "unknown"
_NO_TOOL = "_no_tool"


def aggregate(
    records: list[LLMCallRecord],
    events: list[dict[str, Any]] | None = None,
) -> AggregatedScene:
    """Roll up records into every by_* slice. Pure function, no I/O."""

    agg = AggregatedScene()
    for record in records:
        agg.totals.add(record)
        stage = record.stage_name or _UNKNOWN
        _bucket(agg.by_stage, stage, record)

        if record.agent_label:
            _bucket(agg.by_agent_label, record.agent_label, record)
        if record.agent_role:
            _bucket(agg.by_agent_role, record.agent_role, record)
            _matrix_bucket(agg.by_stage_agent_role, stage, record.agent_role, record)
        if record.agent_type:
            _bucket(agg.by_agent_type, record.agent_type, record)
            _matrix_bucket(agg.by_stage_agent_type, stage, record.agent_type, record)
        subsystem_key = record.subsystem or record.source or _UNKNOWN
        _bucket(agg.by_subsystem, subsystem_key, record)
        _matrix_bucket(agg.by_stage_subsystem, stage, subsystem_key, record)

        tools = record.tool_calls_in_response or []
        if not tools:
            _bucket(agg.by_tool, _NO_TOOL, record)
            _matrix_bucket(agg.by_stage_tool, stage, _NO_TOOL, record)
        else:
            for tool in tools:
                _bucket(agg.by_tool, tool, record)
                _matrix_bucket(agg.by_stage_tool, stage, tool, record)

    if events:
        agg.stage_wall_time_s = _stage_time_from_events(events)
        agg.wall_span_s = _wall_span_from_events(events)

    return agg


# ---------------------------------------------------------------------------
# Media discovery (bug-fixed)
# ---------------------------------------------------------------------------


def _discover_stage_media(scene_dir: Path) -> dict[str, list[str]]:
    """Find PNG renders per stage; tolerant of missing dirs."""

    media: dict[str, list[str]] = defaultdict(list)

    floor_plan_dir = scene_dir / "floor_plans" / "floor_plan_renders"
    if floor_plan_dir.exists():
        for render_dir in sorted(floor_plan_dir.glob("renders_*"))[:1]:
            media["floor_plan"].extend(
                sorted(str(path) for path in render_dir.glob("*.png"))
            )

    for room_dir in sorted(scene_dir.glob("room_*")):
        room_id = room_dir.name.removeprefix("room_")
        render_root = room_dir / "scene_renders"
        if not render_root.exists():
            continue
        # Whole-room analysis renders (final room state, furniture-style views).
        analysis_dir = render_root / ANALYSIS_RENDER_SUBDIR / ANALYSIS_RENDER_BATCH
        if analysis_dir.exists():
            analysis_pngs = sorted(str(p) for p in analysis_dir.glob("*.png"))
            if analysis_pngs:
                media[f"analysis:{room_id}"].extend(analysis_pngs)
                media["final_scene"].extend(analysis_pngs)
        # Furniture: a single subdir name.
        furniture_dir = render_root / "furniture"
        if furniture_dir.exists():
            first_batch = next(iter(sorted(furniture_dir.glob("renders_*"))), None)
            if first_batch is not None:
                media[f"furniture:{room_id}"].extend(
                    sorted(str(p) for p in first_batch.glob("*.png"))
                )
        # Manipuland: subdir name varies by furniture target — glob it.
        for manip_dir in sorted(render_root.glob("manipulands_*")):
            first_batch = next(iter(sorted(manip_dir.glob("renders_*"))), None)
            if first_batch is not None:
                tag = manip_dir.name.removeprefix("manipulands_")
                media[f"manipuland:{room_id}:{tag}"].extend(
                    sorted(str(p) for p in first_batch.glob("*.png"))
                )

    # Final scene fallback: combined_house/ usually has no PNGs in real runs.
    if not media["final_scene"]:
        for candidate in (
            "combined_house_after_ceiling",
            "combined_house_after_wall_objects",
            "combined_house_after_furniture",
            "combined_house",
        ):
            d = scene_dir / candidate
            if not d.exists():
                continue
            pngs = sorted(str(p) for p in d.glob("*.png"))
            if pngs:
                media["final_scene"].extend(pngs)
                break

    return dict(media)


def _make_image_grid(
    image_paths: list[str], output_path: Path, *, title: str
) -> str | None:
    if not image_paths:
        return None

    selected = [Path(p) for p in image_paths if Path(p).exists()][:4]
    if not selected:
        return None

    images = [Image.open(p).convert("RGB") for p in selected]
    thumb_size = (384, 384)
    thumbs = []
    for image in images:
        copy = image.copy()
        copy.thumbnail(thumb_size)
        canvas = Image.new("RGB", thumb_size, color=(245, 245, 245))
        offset = (
            (thumb_size[0] - copy.width) // 2,
            (thumb_size[1] - copy.height) // 2,
        )
        canvas.paste(copy, offset)
        thumbs.append(canvas)

    columns = 2
    rows = (len(thumbs) + columns - 1) // columns
    header_h = 50
    grid = Image.new(
        "RGB", (thumb_size[0] * columns, header_h + thumb_size[1] * rows), "white"
    )
    draw = ImageDraw.Draw(grid)
    draw.text((16, 16), title, fill="black")
    for idx, thumb in enumerate(thumbs):
        x = (idx % columns) * thumb_size[0]
        y = header_h + (idx // columns) * thumb_size[1]
        grid.paste(thumb, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    return str(output_path)


def _figures(scene_dir: Path, media: dict[str, list[str]]) -> dict[str, str | None]:
    figures_dir = scene_dir / "run_analysis" / "figures"
    stage_progress: list[str] = []
    final_scene: list[str] = []
    for key, paths in media.items():
        if key == "final_scene":
            final_scene.extend(paths)
        else:
            stage_progress.extend(paths[:1])

    return {
        "stage_progress_grid": _make_image_grid(
            stage_progress,
            figures_dir / "stage_progress_grid.png",
            title="Stage Progress",
        ),
        "final_scene_grid": _make_image_grid(
            final_scene, figures_dir / "final_scene_grid.png", title="Final Scene"
        ),
    }


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def analyze_scene(scene_dir: Path) -> dict[str, Any]:
    """Analyze one scene; write outputs into ``scene_dir/run_analysis/``."""

    scene_dir = scene_dir.resolve()
    tracking_dir = scene_dir / "run_analysis"
    tracking_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(scene_dir)
    events = load_events(scene_dir)
    agg = aggregate(records, events)

    media = _discover_stage_media(scene_dir)
    figures = _figures(scene_dir, media)

    aggregated_dict = agg.as_dict()
    summary = {
        "scene_dir": str(scene_dir),
        "scene_id": scene_dir.name,
        "num_calls": agg.totals.num_calls,
        "totals": aggregated_dict["totals"],
        "by_stage": aggregated_dict["by_stage"],
        "by_agent_role": aggregated_dict["by_agent_role"],
        "by_agent_type": aggregated_dict["by_agent_type"],
        "by_subsystem": aggregated_dict["by_subsystem"],
        "by_tool": aggregated_dict["by_tool"],
        "stage_wall_time_s": aggregated_dict["stage_wall_time_s"],
        "wall_span_s": aggregated_dict["wall_span_s"],
        "figures": figures,
    }

    write_scene_summary(scene_dir, summary)
    write_scene_json(scene_dir, "aggregated.json", aggregated_dict)
    write_scene_json(
        scene_dir,
        "media_manifest.json",
        {"media": media, "figures": figures},
    )
    return summary


def _extract_run_config(experiment_dir: Path) -> dict[str, Any]:
    """Best-effort snapshot of ablation-relevant Hydra config values."""

    hydra_cfg = experiment_dir / ".hydra" / "config.yaml"
    if not hydra_cfg.exists():
        return {}
    try:
        with open(hydra_cfg, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return {}

    def _select(obj: Any, dotted: str) -> Any:
        cur: Any = obj
        for part in dotted.split("."):
            if isinstance(cur, dict):
                cur = cur.get(part)
            else:
                return None
            if cur is None:
                return None
        return cur

    keys = [
        "name",
        "provider.model",
        "experiment.pipeline.start_stage",
        "experiment.pipeline.resume_from_path",
        "experiment.pipeline.skip_stages",
        "furniture_agent.max_critique_rounds",
        "manipuland_agent.max_critique_rounds",
        "furniture_agent.openai.vision_detail",
        "manipuland_agent.openai.vision_detail",
        "furniture_agent.rendering.resolution_preset",
        "manipuland_agent.rendering.resolution_preset",
        "furniture_agent.rendering.side_view_count",
        "manipuland_agent.rendering.side_view_count",
        "furniture_agent.session_memory.intra_turn_observation_stripping.keep_last_n_observations",
        "manipuland_agent.session_memory.intra_turn_observation_stripping.keep_last_n_observations",
        "furniture_agent.asset_manager.validation_rendering.resolution_preset",
        "manipuland_agent.asset_manager.validation_rendering.resolution_preset",
    ]
    out: dict[str, Any] = {}
    for k in keys:
        v = _select(cfg, k)
        if v is not None:
            out[k] = v
    return out


def analyze_experiment(experiment_dir: Path) -> dict[str, Any]:
    """Analyze every ``scene_*`` under ``experiment_dir``."""

    experiment_dir = experiment_dir.resolve()
    run_config = _extract_run_config(experiment_dir)
    # Prefer the experiment's frozen .hydra snapshot, then walk up a few
    # levels (DR variants don't carry their own snapshot), then fall back
    # to composing a fresh Hydra config — the minimal in-line stub used
    # to live here was missing `furniture_agent.rendering.retry_count`
    # and crashed render_rooms_for_analysis.
    cfg_dict: dict[str, Any] | None = None
    candidates = [experiment_dir / ".hydra" / "config.yaml"]
    walker = experiment_dir
    for _ in range(4):
        walker = walker.parent
        candidates.append(walker / ".hydra" / "config.yaml")
    for candidate in candidates:
        if candidate.exists():
            try:
                with open(candidate, encoding="utf-8") as f:
                    cfg_dict = yaml.safe_load(f) or None
                if cfg_dict:
                    break
            except Exception:
                continue
    if cfg_dict is None:
        try:
            from pathlib import Path as _Path

            from hydra import compose, initialize_config_dir
            from hydra.core.global_hydra import GlobalHydra
            from omegaconf import OmegaConf

            repo_root = _Path(__file__).resolve().parents[2]
            GlobalHydra.instance().clear()
            with initialize_config_dir(
                version_base=None, config_dir=str(repo_root / "configurations")
            ):
                cfg = compose(
                    config_name="config",
                    overrides=["+name=analyze_experiment"],
                )
            cfg_dict = OmegaConf.to_container(cfg, resolve=True) or {}
        except Exception:
            cfg_dict = {}
    cfg_dict.setdefault("experiment", {}).setdefault("tracking", {}).setdefault(
        "analysis_renders", {}
    )["enabled"] = True

    from .analysis_renders import finalize_scene_analysis

    scene_summaries: list[dict[str, Any]] = []
    for scene_dir in sorted(experiment_dir.glob("scene_*")):
        if scene_dir.is_dir():
            scene_summaries.append(finalize_scene_analysis(scene_dir, cfg_dict))

    # Sum totals over scenes.
    totals = TokenTotals()
    for summary in scene_summaries:
        s = summary.get("totals") or {}
        totals.input_tokens += int(s.get("input_tokens") or 0)
        totals.output_tokens += int(s.get("output_tokens") or 0)
        totals.reasoning_tokens += int(s.get("reasoning_tokens") or 0)
        totals.cached_tokens += int(s.get("cached_tokens") or 0)
        totals.total_tokens += int(s.get("total_tokens") or 0)
        totals.num_images += int(s.get("num_images") or 0)
        totals.num_calls += int(s.get("num_calls") or 0)
        totals.duration_s += float(s.get("duration_s") or 0.0)

    payload = {
        "experiment_dir": str(experiment_dir),
        "num_scenes": len(scene_summaries),
        "totals": totals.as_dict(),
        "run_config": run_config,
        "scenes": scene_summaries,
    }
    out_dir = experiment_dir / "run_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "experiment_summary.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    with open(out_dir / "scene_summary_table.json", "w", encoding="utf-8") as f:
        json.dump(scene_summaries, f, indent=2, sort_keys=True)
    if run_config:
        with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=2, sort_keys=True)
    return payload
