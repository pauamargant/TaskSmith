# TaskSmith documentation

**TaskSmith** extends [SceneSmith](https://arxiv.org/abs/2602.09153): agentic generation of simulation-ready indoor scenes with **task-aware** artifacts (`task_requirements.json`, `rl_artifacts.json`, `reward_spec.yaml`, `curriculum_spec.yaml`, `morphology_task_config.json`) and exports (Drake DMD, MuJoCo).

This site covers running and configuring the fork. The **root `README.md`** is the long-form SceneSmith install and data guide; use both together.

## SceneSmith vs TaskSmith

SceneSmith is the paper codebase with a dataset-focused workflow (branding and install instructions live in the root `README.md`). TaskSmith builds on the same core and adds task analysis, reward specs, morphology-task selection, domain randomization, and the provider-centric LLM setup documented here.

## Where to start

1. [Installation](getting-started/installation.md) — `uv`, virtualenv, optional Docker.
2. [Quickstart](getting-started/quickstart.md) — first scene generation and task-aware runs.
3. [Custom configs](getting-started/experiment-bundles.md) — `--config-path`, per-study config folders (any directory with a full copied Hydra tree).
4. [Pipeline overview](pipeline/overview.md) — stages and outputs.
5. [Stage control](pipeline/stage-control.md) — `start_stage`, `stop_stage`, `skip_stages`, self-contained resume.
6. [Providers & models](features/providers.md) — `PROVIDER_*`, the direct openai-client adapter, and reasoning-model compatibility rules.

## Build these docs locally

```bash
uv sync --group docs
uv run mkdocs serve
```

Strict build (same as CI):

```bash
uv run mkdocs build --strict
```
