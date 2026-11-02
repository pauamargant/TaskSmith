# Custom config directories (`--config-path`)

Hydra loads its **primary** `config.yaml` and all **config groups** (`floor_plan_agent/`, `furniture_agent/`, `experiment/`, …) from a single directory on disk. By default that directory is **`configurations/`** at the repo root.

If you want a **self-contained config** for a study (edit YAMLs without touching the
shared `configurations/` tree), you don’t need a special “bundle” format; you just
need a directory containing a full Hydra config tree and pass it via `--config-path`.

## Default layout

```text
configurations/
  config.yaml              # provider, openai, hydra.run.dir, defaults list
  experiment/              # prompts, pipeline, tracking, servers, projection, …
  floor_plan_agent/
  furniture_agent/
  wall_agent/
  ceiling_agent/
  manipuland_agent/
  task_analysis_agent/
  reward_spec_agent/
  morphology_task_agent/
  domain_randomization/
  provider/                # vLLM/API capability presets (qwen, gemma, neutral)
```

Run as usual from the repo root:

```bash
uv run python main.py +name=my_run ...
```

## Use a custom config directory

Pass **`--config-path <dir>`** where `<dir>` contains the copied Hydra tree (absolute path, or path relative to the **current working directory**). TaskSmith passes this to Hydra as `config_path` instead of `configurations`.

The repo ships two ready-made bundles under `experiments/configs/`:

| Bundle | Purpose |
|--------|---------|
| `E0_baseline/` | Paper baseline: full pipeline, kitchen task prompt, Qwen3 provider flags |
| `E0_baseline_gemma/` | Same pipeline with Gemma 4 capability flags (requires vLLM Gemma deployment) |

```bash
uv run python main.py --config-path experiments/configs/E0_baseline +name=my_run
```

Work from the **TaskSmith repo root** so relative paths and tool scripts behave consistently.

## Write outputs to a custom log directory

By default, TaskSmith writes runs under `outputs/...` via `hydra.run.dir`.
To write a run under a custom directory (for example `logs/experiments/...`), override
`hydra.run.dir` and keep `hydra.job.chdir=false` so relative paths still resolve from
the repo root:

```bash
mkdir -p logs/experiments/e0_baseline

uv run python main.py --config-path experiments/configs/E0_baseline +name=e0_baseline \
  hydra.job.chdir=false \
  "hydra.run.dir=logs/experiments/e0_baseline/${now:%Y-%m-%d}/${name}_${now:%H-%M-%S}_${now:%f}"
```

## What to edit

| Location | Role |
| --- | --- |
| **`config.yaml`** | Top-level **`defaults:`** select which **preset file** runs for each Hydra group (`experiment: …`, `wall_agent: …`, …). This is where you switch per-**stage** agent configs. |
| **`experiment/<preset>.yaml`** | **`experiment.`** subtree: `prompts`, `pipeline` (`start_stage`, `stop_stage`, `skip_stages`, `resume_from_path`, …), `tracking.wandb`, retrieval servers, `projection`, `sceneeval_export`, … (see **`configurations/experiment/base_experiment.yaml`** for the full shape). |
| **`floor_plan_agent/*.yaml`**, **`furniture_agent/*.yaml`**, … | Stage-specific parameters (models, `max_critique_rounds`, vision, asset routing, …). |

**Note:** Files under **`experiment/`** only define the **`experiment`** config group. They **do not** replace top-level **`floor_plan_agent`** / **`wall_agent`** choices; those come from **`config.yaml`** `defaults` or from CLI overrides like `wall_agent=stateful_wall_agent`.

CLI overrides still apply when using a bundle, for example:

```bash
uv run python main.py --config-path experiments/configs/E0_baseline +name=ablation \
  experiment.pipeline.stop_stage=furniture \
  wall_agent=stateful_wall_agent
```

Legacy environment variable support (still works, but prefer `--config-path`):

```bash
TASKSMITH_CONFIG_PATH=experiments/configs/E0_baseline \
  uv run python main.py +name=my_run
```

## Create your own config directory

Copy the shared tree and adjust:

```bash
cp -a configurations experiments/configs/my_study
```

Then:

1. Edit `experiments/configs/my_study/config.yaml` `defaults:`.
2. Add `experiments/configs/my_study/experiment/my_study.yaml` (usually `defaults: [indoor_scene_generation, _self_]` plus your overrides).
3. Point **`defaults`** → **`experiment: my_study`** in that bundle’s **`config.yaml`**.

To start your own bundle, copy the E0_baseline (or the shared `configurations/`) and adjust from there. Use `E0_baseline_gemma/` as a reference if running on a Gemma 4 deployment.

## Notes

- Work from the repo root so relative paths (e.g. `data/...`) resolve consistently.
- Legacy environment variable support still works, but prefer `--config-path`.

See also: [Configuration system](configuration.md), [Pipeline overview](../pipeline/overview.md), [Stage control](../pipeline/stage-control.md), [Experiment tracking](../features/experiment-tracking.md), [Installation](installation.md).
