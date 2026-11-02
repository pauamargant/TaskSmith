# Pipeline overview

The main entrypoint is `scenesmith/experiments/indoor_scene_generation.py`, driven by Hydra (`main.py`).

## Stages

Room and layout stages follow `AgentType` order; then two post-processing stages run at scene level:

```text
floor_plan → furniture → wall_mounted → ceiling_mounted → manipuland → reward_spec → morphology_task
```

| Stage | Role |
| --- | --- |
| `floor_plan` | House layout / room structure |
| `furniture` | Furniture placement per room |
| `wall_mounted` | Wall-mounted objects |
| `ceiling_mounted` | Ceiling-mounted objects |
| `manipuland` | Small objects / manipulands |
| `reward_spec` | Reward + curriculum YAML from scene + task artifacts |
| `morphology_task` | Morphology task config for downstream RL |

## Physical post-processing

During physical-feasibility forward simulation, the pipeline may optionally remove manipulands that fall off surfaces (via `remove_fallen_manipulands`).

Task objects (manipulands with `metadata.is_task_object=true`) are treated specially: if simulation would classify them as fallen, the pipeline restores them to their **pre-simulation pose** instead of deleting them. If that pre-simulation pose is unavailable, TaskSmith raises immediately rather than leaving a required task object in a fallen pose. This avoids downstream task-artifact export failing due to missing or invalid required objects.

Valid values for `experiment.pipeline.start_stage`, `stop_stage`, and `skip_stages` are exactly the seven stages in the table above (with restrictions noted in [Stage control](stage-control.md)).

## Outputs

Hydra saves each run under `outputs/<date>/<name>_<time>_<uuid>/`, where `<name>` is the required `+name=` label from the command line (see `configurations/config.yaml`).

Per run, expect directories such as (layout varies slightly by config):

- `scene_000/` (or more scenes) - room folders, `scene_states/`, assets
- Scene-level JSON/YAML - `task_requirements.json`, `rl_artifacts.json`, `reward_spec.yaml`, `curriculum_spec.yaml`, `morphology_task_config.json` when those stages apply
- `combined_house/` - assembled house DMD and related exports
- `resume_manifest.json` - written only for resumed scenes; records source run, invalidated stages, and deleted stale paths
- `run_analysis/` - local tracking records, token summaries, media manifests, and post-run experiment summaries

## Configuration

Hydra configs live under **`configurations/`** by default. Experiment defaults are composed from the primary **`config.yaml`** **`defaults:`** list ( **`experiment`** preset and each **`_*_agent`** group) and the YAML files in the matching folders. For the four room-object agent groups (`furniture_agent/`, `wall_agent/`, `ceiling_agent/`, `manipuland_agent/`), `_base_agent.yaml` holds shared defaults and `base_*_agent.yaml` holds agent-specific overrides. The remaining groups (`floor_plan_agent/`, `reward_spec_agent/`, `morphology_task_agent/`, `task_analysis_agent/`) contain only their own `base_*_agent.yaml`.

To use a **self-contained copy** of the full tree for one study, pass `--config-path` pointing at a directory containing a copied Hydra tree (for example `experiments/configs/E0_baseline/`). See [Custom config directories](../getting-started/experiment-bundles.md).

## Image resolution presets

TaskSmith cost scales strongly with **image resolution** for multimodal prompts. There are two independent resolution knobs:

- **Room observation renders** (agent `observe_scene` images):
  - `*_agent.rendering.resolution_preset: null | low | medium | high`
  - Default `null` uses explicit `top_view_width/height` and `side_view_width/height`.
  - When set: top view becomes `preset×preset`, side views become `(preset/2)×(preset/2)`.
  - **Gotcha**: when `image_quality_preset` is non-null it overrides explicit `side_view_width/height` and forces sides to `top/2`. Leave both presets `null` to honour the explicit dims in `_base_agent.yaml` (sides default to 1024², matched to top, for legible critic input).
- **Default views per `observe_scene` call (6 images):** `0_top.png`, `0_side.png` … `(N-1)_side.png`, plus `{N}_iso.png` (the isometric / corner-above view at elevation 45° azimuth 45°). Controlled by `*_agent.rendering.include_isometric_view` (default `true`) and `isometric_elevation_degrees` / `isometric_azimuth_degrees`. The isometric view helps the critic read chair/table relationships at a glance.
- **Framing margins**: `*_agent.rendering.side_view_margin_scale` (default `1.2`) and `top_view_margin_scale` (default `1.4`) control how much empty border each view leaves around the scene. Historical defaults were 1.40 / 1.8 (≈55% scene area); the tighter defaults give ≈75% scene area at the cost of more clipping risk in very large rooms (lower bound ~1.1).
- **Per-object validation / physics renders** (mesh physics analysis + router validation):
  - `*_agent.asset_manager.validation_rendering.resolution_preset: low | medium | high`
  - Default `medium`. Not affected by the room-rendering knobs above.

Preset mapping (square): `low=256`, `medium=512`, `high=1024`.

With the current defaults, a single `observe_scene` call produces ~3× the pixel count of the pre-2026-05-19 setup (sides bumped from 512² to 1024², plus the new isometric view). If your provider charges per image token, dial `side_view_width/height` back to 512 in your experiment override.

## Parallel VLM physics analysis

Per-asset physics analysis (material, mass, orientation) makes one VLM call per object. In a batch of 8 furniture items this adds up to ~16 min of serial VLM wait. Enable parallel execution across the batch with:

```yaml
asset_manager:
  parallel_vlm_physics: true
```

or as a CLI override: `asset_manager.parallel_vlm_physics=true`.

Disabled by default. The Blender rendering steps (GLB→GLTF conversion, multi-view render, mesh canonicalization) remain sequential because `BlenderServer` is single-process. Only the VLM HTTP calls are parallelized. Safe to enable when the inference server supports concurrent requests; check server load limits before using with large batches.

See also: [Stage control](stage-control.md), [Task-aware generation](../features/task-aware-generation.md), [Experiment tracking](../features/experiment-tracking.md), [Custom config directories](../getting-started/experiment-bundles.md), [Providers & models](../features/providers.md).
