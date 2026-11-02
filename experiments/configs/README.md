# Per-experiment config bundles

Each subfolder is a **full Hydra config tree** (same layout as repo `configurations/`).

## Bundles

| Bundle | Purpose |
|--------|---------|
| `E0_baseline/` | Paper baseline: full pipeline, kitchen task prompt, Qwen3 capability flags. Use for reproducing paper experiments. |
| `E0_baseline_gemma/` | Same as E0_baseline but with Gemma 4 capability flags. Requires a vLLM Gemma 4 deployment (`--reasoning-parser gemma4 --tool-call-parser gemma4`). |

## How it works

**`--config-path <dir>`** makes `main.py` load Hydra's primary `config.yaml` from that directory instead of `configurations/`. Everything needed to resolve defaults lives under each bundle; no `hydra.searchpath` is required.

Per-stage settings are the usual Hydra groups: `floor_plan_agent`, `furniture_agent`, `wall_agent`, `ceiling_agent`, `manipuland_agent`, `task_analysis_agent`, `reward_spec_agent`, `morphology_task_agent`, `domain_randomization`, plus `experiment` for prompts, pipeline, tracking, servers, etc.

## Run

From the TaskSmith repo root:

```bash
uv run python main.py --config-path experiments/configs/E0_baseline +name=my_run
```

CLI overrides still apply on top of the bundle, for example:

```bash
uv run python main.py --config-path experiments/configs/E0_baseline +name=ablation \
  experiment.pipeline.stop_stage=furniture \
  wall_agent=stateful_wall_agent
```

## Adding a new bundle

```bash
cp -a configurations experiments/configs/my_study
```

Then edit `my_study/config.yaml` defaults and/or any group YAML under `my_study/`. Add a new `experiment/my_study.yaml` if you want a named preset.
