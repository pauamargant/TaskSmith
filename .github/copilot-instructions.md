# Copilot Instructions for TaskSmith

Welcome to TaskSmith! This guide provides context to help you be immediately productive in this project.

## Architecture & Big Picture
- **TaskSmith** is a pipeline for generating simulation-ready indoor manipulation scenes and task artifacts (e.g., for RL).
- **Main Pipeline**: The primary entrypoint is `scenesmith/experiments/indoor_scene_generation.py`.
- **Stages Pipeline**: Generating floor plans -> placing objects via specialized agents (furniture, wall, ceiling, manipuland) -> RL task analysis -> reward spec generation.
- **State Checkpoints**: The pipeline creates real checkpoints under `scene_NNN/room_<name>/scene_states/` (e.g., `scene_after_furniture`), enabling robust resuming.

## Critical Developer Workflows
- **Running a Scene Generation**:
  ```bash
  uv run python main.py +name=tasksmith_kitchen "experiment.prompts=['Prompt text here.']" experiment.num_workers=1
  ```
- **Resuming a Pipeline Run**: To resume from a previous run at a specific stage, pass the `start_stage` and `resume_from_path`:
  ```bash
  uv run python main.py +name=resume_wall "experiment.prompts=['same prompt']" experiment.pipeline.start_stage=wall_mounted experiment.pipeline.resume_from_path=outputs/YYYY-MM-DD/<path>
  ```
- **Analysis & Tracking**: Run post-run summaries & W&B exports using `python scripts/analyze_experiment_run.py outputs/<path>`.

## Specific Conventions & Patterns
- **Provider & Agent SDKs**: All Agent SDK calls MUST route through `scenesmith/provider.py` via `make_chat_completions_model()`. Direct single-turn/structured requests route via `scenesmith/openai_adapter.py`.
- **Reasoning Fields Compatibility**: Never consume `reasoning`, `reasoning_content`, or similar trace fields as a fallback for final text; this breaks provider compatibility.
- **Handoff History**: When using the Agents SDK, prefer `handoff(..., nest_handoff_history=False)` to prevent `No user query found...` transcript errors on EPFL/vLLM providers.
- **Environment Variables**: `PROVIDER_MODEL`, `PROVIDER_BASE_URL`, and `PROVIDER_API_KEY` are canonical. Read them via Hydra resolvers (e.g., `${oc.env:PROVIDER_MODEL,...}`) in configurations.
- **Prompts**: Prompts live strictly in `scenesmith/prompts/data/` and are registered via `PromptRegistry` (`scenesmith/prompts/registry.py`). Template variables require exact matches.

## Integration & Assets
- **HSSD Assets**: HSSD retrieval is the default data core for furniture, wall, ceiling, and manipuland agents. See configurations in `configurations/*_agent/base_*_agent.yaml`.
- **Exporting**: The MuJoCo export is intentionally standalone and is run via `scripts/export_scene_to_mujoco.py`.
