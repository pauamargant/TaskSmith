# Quickstart

Prerequisites: follow [Installation](installation.md).

## Concurrent runs

Multiple `main.py` processes can run simultaneously with different configs or run names. If the default local retrieval-server ports are in use, TaskSmith reuses a healthy existing server or starts a new one on a free port. The `outputs/<date>/latest-run` symlink is disabled by default to avoid races; set `experiment.write_latest_run_symlink=true` to re-enable it.

## Plain scene generation

```bash
uv run python main.py +name=tasksmith_kitchen \
  "experiment.prompts=['A well-lit residential kitchen featuring a central wooden dining table and a fixed stone countertop along the wall.']" \
  experiment.num_workers=1
```

## Task-aware generation

Task-focused prompts trigger `TaskAnalysisAgent` before layout and produce task artifacts:

```bash
uv run python main.py +name=tasksmith_task \
  "experiment.prompts=['An agent must pick up items scattered on a kitchen table and place them neatly on the countertop.']" \
  experiment.num_workers=1
```

Outputs land under your Hydra output directory, e.g. `outputs/<date>/<run_name>_<time>/scene_000/`:

- `task_requirements.json` - structured task spec from analysis
- `rl_artifacts.json` - grounded object IDs and success conditions
- `reward_spec.yaml`, `curriculum_spec.yaml` - after the `reward_spec` stage
- `morphology_task_config.json` - after the `morphology_task` stage (when enabled)

See [Task-aware generation](../features/task-aware-generation.md) and [Stage control](../pipeline/stage-control.md).

## Run with a dedicated config directory

To use a **full copy** of the Hydra tree for one study (in any directory you choose) instead of shared **`configurations/`**, pass `--config-path`. The repo ships `experiments/configs/E0_baseline/` (Qwen3) and `experiments/configs/E0_baseline_gemma/` (Gemma 4) as ready-made examples. See [Custom config directories](experiment-bundles.md).

```bash
uv run python main.py --config-path experiments/configs/E0_baseline +name=my_bundle_run
```

## Analyze outputs

After a run completes, summarize local tracking files and token usage:

```bash
uv run python scripts/analyze_experiment_run.py outputs/YYYY-MM-DD/<name>_HH-MM-SS
```

This writes `run_analysis/experiment_summary.yaml` at run level and `scene_000/run_analysis/scene_summary.yaml` plus token/media summaries at scene level. To also export the completed run to W&B, add `--wandb-project <project>`. See [Experiment tracking](../features/experiment-tracking.md).

## Write outputs and console logs under `logs/`

To write the Hydra run directory under `logs/` (instead of `outputs/`) and tee the
console output to a file:

```bash
mkdir -p logs/experiments/e0_baseline

uv run python main.py +name=e0_baseline \
  hydra.job.chdir=false \
  "hydra.run.dir=logs/experiments/e0_baseline/${now:%Y-%m-%d}/${name}_${now:%H-%M-%S}_${now:%f}" \
  2>&1 | tee logs/experiments/e0_baseline/console.log
```

## Resume from a checkpoint

Example: resume at `wall_mounted` using a previous run directory:

```bash
uv run python main.py +name=resume_wall \
  "experiment.prompts=['same prompt as source run']" \
  experiment.pipeline.start_stage=wall_mounted \
  experiment.pipeline.resume_from_path=outputs/YYYY-MM-DD/<name>_HH-MM-SS
```

This creates a **new complete output directory** seeded from the source run. The source
scene stays unchanged, copied paths are rebased into the new run, and artifacts from
`wall_mounted` onward are cleared before rerun. See [Stage control](../pipeline/stage-control.md)
for the stage-by-stage invalidation rules and `resume_manifest.json`.

See also: [Custom config directories](experiment-bundles.md), [Pipeline overview](../pipeline/overview.md), [Domain randomization](../features/domain-randomization.md), [Experiment tracking](../features/experiment-tracking.md).
