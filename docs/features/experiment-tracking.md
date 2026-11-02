# Experiment tracking

TaskSmith records every LLM API call to local JSONL during generation, can stream live progress to W&B, and ships a post-run analyzer with a cross-run comparison CLI.

The tracker is disabled with `experiment.tracking.enabled=false` (or `TASKSMITH_TRACKING_ENABLED=0`). Disabled means nothing: no files, no Model wrapping, no overhead. Every emit path is wrapped in `try/except`, so a tracking bug never aborts a generation run.

During generation, three emitters write to `scene_NNN/run_analysis/`: the tracked Model wrapper in `scenesmith.provider` (Agents SDK calls), the `run_structured_vlm_call` helpers in `scenesmith.agent_utils.sdk_vlm_runner` (single-shot VLM calls), and `_summarize_turn` in `scenesmith.agent_utils.turn_trimming_session` (context-compression calls). After generation, `scripts/analyze_experiment_run.py` builds an `aggregated.json` rollup; `scripts/compare_experiments.py` produces a markdown/CSV ablation table. The startup capability probe in `scenesmith/provider_validation.py` is not tracked.

## Local outputs

Each scene's `run_analysis/` directory contains:

```text
scene_000/
  run_analysis/
    usage_records.jsonl     # one row per LLM API call (the source of truth)
    events.jsonl            # stage lifecycle events
    artifacts.jsonl         # important intermediate outputs
    aggregated.json         # all by_* slices (single rollup file)
    scene_summary.yaml      # high-level summary for human reading
    media_manifest.json     # render PNG paths (incl. experiment_analysis views)
    figures/
      stage_progress_grid.png
      final_scene_grid.png
```

Whole-room analysis renders (same multiview layout as furniture-stage `observe_scene`) are written automatically at the end of room generation to:

```text
scene_000/room_<name>/scene_renders/experiment_analysis/renders_001/
  0_top.png
  0_side.png
  ...
```

They are picked up by `media_manifest.json` under `analysis:<room_id>` and `final_scene`, and feed `figures/final_scene_grid.png`. Toggle with `experiment.tracking.analysis_renders.enabled` (default `true`). Re-render an existing run with:

```bash
uv run python scripts/analyze_experiment_run.py outputs/YYYY-MM-DD/<run>/scene_000 --render-analysis
```

Experiment-level outputs:

```text
outputs/YYYY-MM-DD/<name>_HH-MM-SS/
  run_analysis/
    experiment_summary.yaml
    scene_summary_table.json
    run_config.json         # snapshot of ablation-relevant Hydra knobs
```

## Per-LLM-call record schema

Each row of `usage_records.jsonl` is an `LLMCallRecord`:

| Field | Meaning |
|---|---|
| `timestamp` | ISO 8601 UTC, milliseconds |
| `source` | `"agents_sdk"` (wrapped Model) or `"vlm_direct"` (backward-compat label in records from pre-refactor runs) |
| `provider_path` | `"agents_sdk"` or `"openai_direct"` (string label in records; direct openai-client adapter) |
| `scene_id`, `room_id`, `stage_name` | From the active `TrackingContext` |
| `agent_label` | Full canonical label, e.g. `"DESIGNER (FURNITURE)"` |
| `agent_role` | `"designer"`, `"critic"`, `"planner"` - parsed from label |
| `agent_type` | `"FURNITURE"`, `"MANIPULAND"`, ... - parsed from label |
| `subsystem` | For single-shot VLM calls: `"physics_analysis"`, `"router_validation"`, `"asset_analysis"`, `"task_analysis"`, `"rl_artifact_export"`, `"scene_analysis"`, `"turn_summarization"` |
| `tool_calls_in_response` | List of tool names this LLM call's response decided to invoke |
| `model`, `request_index` | Model id, 0-based index within the agent run (Agents SDK only) |
| `input_tokens`, `output_tokens`, `reasoning_tokens`, `cached_tokens`, `total_tokens` | Token counts for this single API call |
| `num_images` | **Exact** count of image content blocks in the outgoing input; same fidelity as direct VLM, including images that travel multiple turns |
| `duration_s` | Wall-clock duration of the single call |

## `aggregated.json` slices

The analyzer rolls per-call records into:

- `totals` - input/output/reasoning/cached/total tokens, plus `num_images`, `num_calls`, summed `duration_s`
- `by_stage` - totals per pipeline stage
- `by_agent_label` - totals per full agent label
- `by_agent_role` - totals per `designer` / `critic` / `planner`
- `by_agent_type` - totals per `FURNITURE` / `MANIPULAND` / ...
- `by_subsystem` - totals per subsystem: single-shot VLM calls appear under their label (`physics_analysis`, `router_validation`, `asset_analysis`, `task_analysis`, `rl_artifact_export`, `scene_analysis`, `turn_summarization`); Agents SDK calls appear under `agents_sdk`
- `by_tool` - tokens of each LLM call attributed to every tool listed in `tool_calls_in_response`. Calls with no tool go to the synthetic `"_no_tool"` bucket. A call that invokes two tools contributes to both; `by_stage` totals are the authoritative non-duplicated number.
- `by_stage_agent_role`, `by_stage_agent_type`, `by_stage_subsystem`, `by_stage_tool` - matrix slices
- `stage_wall_time_s` - from `events.jsonl` (end - start per stage)
- `wall_span_s` - earliest to latest event timestamp across the scene

## Analyze a run

Analyze a full experiment directory:

```bash
uv run python scripts/analyze_experiment_run.py outputs/YYYY-MM-DD/<name>_HH-MM-SS
```

Or one scene directory:

```bash
uv run python scripts/analyze_experiment_run.py outputs/YYYY-MM-DD/<name>_HH-MM-SS/scene_000
```

## Compare runs (E0 vs E1 ablations)

```bash
uv run python scripts/compare_experiments.py \
    outputs/2026-05-13/e0_base_22-41-07_599672 \
    outputs/2026-05-15/e1a_rounds1_02-36-36_805351 \
    outputs/2026-05-15/e1b_low_vision_22-55-33_213358 \
    --per-stage --per-tool --csv /tmp/e1.csv
```

The first positional arg is the baseline for delta columns; override with `--baseline`. The script emits a markdown table suitable for ablation comparisons.

## W&B export

Install + log in once:

```bash
uv sync
uv run wandb login
```

Export a completed run:

```bash
uv run python scripts/analyze_experiment_run.py outputs/YYYY-MM-DD/<name>_HH-MM-SS \
  --wandb-project tasksmith \
  --wandb-mode online
```

The export logs tables and bar charts for every by_* slice (stage, subsystem, tool, agent_role, agent_type) plus the discovered figure grids.

## Live W&B during generation

```bash
uv run python main.py +name=tracking_smoke \
  "experiment.prompts=['A simple kitchen with a table and countertop.']" \
  experiment.num_workers=1 \
  experiment.tracking.wandb.enabled=true \
  experiment.tracking.wandb.project=tasksmith-test \
  experiment.tracking.wandb.poll_interval_s=10
```

The live tracker polls the per-call JSONL files and ships totals + per-slice tables + bar charts every `poll_interval_s` seconds. It uses the same aggregator as the post-run analyzer, so live and final numbers match exactly. With `fail_on_error=false` (the default), W&B failures degrade to warnings and generation continues.

## Tracking flags

| Flag | Effect |
|---|---|
| `experiment.tracking.enabled: true` (default) | All emission, all aggregation, all files written |
| `experiment.tracking.enabled: true` + `experiment.tracking.agents_sdk_wrapping=false` | Direct VLM still tracked; Agents SDK calls not wrapped (raw inner Model returned). Honored by both `main.py` and the DR CLI. |
| `experiment.tracking.enabled: false` | Zero tracking. Zero files. Raw Model. |
| `TASKSMITH_TRACKING_ENABLED=0` env var | Same as `enabled: false`; wins over YAML |

Tracking bugs don't abort generation: every `emit_*` call is wrapped at the writer boundary, and the Model wrapper catches its own record-emission errors. Failures appear as WARNING-level log lines.

## Image cost

On many providers, `prompt_tokens` for multimodal turns scale much more with **image resolution** than with `vision_detail`:

- Room observation images: `*_agent.rendering.resolution_preset` (`null|low|medium|high`)
- Per-object physics/router renders: `*_agent.asset_manager.validation_rendering.resolution_preset` (`low|medium|high`)

Preset mapping: `low=256`, `medium=512`, `high=1024` (square).
