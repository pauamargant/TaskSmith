# Pipeline stage control

The generation pipeline runs **seven** stages in order:

```text
floor_plan → furniture → wall_mounted → ceiling_mounted → manipuland → reward_spec → morphology_task
```

Three config knobs control which stages actually run:

| Option | Type | Default | Purpose |
| --- | --- | --- | --- |
| `start_stage` | string | `"floor_plan"` | Resume from this stage (loads previous checkpoint) |
| `stop_stage` | string | `"morphology_task"` | Stop after this stage (saves checkpoint for later) |
| `skip_stages` | list | `[]` | Skip specific stages mid-pipeline |

---

## `skip_stages` - skip specific stages

Use this to skip individual stages while still running everything else. Skipped stages
still write a checkpoint (current scene state) under their checkpoint name so that
downstream stages and future resume runs work correctly. New tracked runs also record
skipped stages in `scene_000/run_analysis/events.jsonl` with `status: skipped`.

**Example: skip wall and ceiling objects**

```bash
uv run python main.py +name=my_run \
  'experiment.pipeline.skip_stages=[wall_mounted,ceiling_mounted]'
```

**In config:**

```yaml
pipeline:
  skip_stages: []
```

**Valid values:** `wall_mounted`, `ceiling_mounted`, `manipuland`, `reward_spec`, `morphology_task`

(`floor_plan` and `furniture` cannot be skipped via this option. When `manipuland` is skipped, TaskSmith still writes `final_scene` from the current ceiling-stage scene so downstream artifacts and resume runs have the expected checkpoint.)

---

## `start_stage` / `stop_stage` - contiguous ranges

### Stop early

```bash
uv run python main.py +name=my_run experiment.pipeline.stop_stage=furniture
```

Stops after furniture, saves `scene_after_furniture` checkpoint.

### Resume from a later stage

Use `start_stage` alone when checkpoints already exist in the current output directory — for example, after a crashed or interrupted run. Hydra creates a new timestamped directory for each invocation, so this form is only useful when rerunning into the same output directory.

```bash
uv run python main.py +name=my_run experiment.pipeline.start_stage=manipuland
```

Loads the `scene_after_ceiling_objects` checkpoint and runs the remaining stages from manipuland onward. If the checkpoint is missing, the run aborts with an error.

To branch from a **different** completed run (the common case), use `resume_from_path` as described in the section below.

### Saved checkpoints

| After stage | Checkpoint location |
| --- | --- |
| `floor_plan` | `scene_000/house_layout.json` |
| `furniture` | `scene_000/room_*/scene_states/scene_after_furniture/` |
| `wall_mounted` | `scene_000/room_*/scene_states/scene_after_wall_objects/` |
| `ceiling_mounted` | `scene_000/room_*/scene_states/scene_after_ceiling_objects/` |
| `manipuland` | `scene_000/room_*/scene_states/final_scene/` |
| `reward_spec` | `scene_000/reward_spec.yaml` and `scene_000/curriculum_spec.yaml` |
| `morphology_task` | `scene_000/morphology_task_config.json` |

---

## Resuming from a previous experiment

Use `resume_from_path` to create a **new self-contained run directory** from an existing
scene output, then rerun from `start_stage` onward. This is useful for A/B testing a
later stage without mutating the source run in place.

```bash
# First run: generate up to furniture
uv run python main.py +name=base_run experiment.pipeline.stop_stage=furniture

# Resume A: add manipulands with default config
uv run python main.py +name=resume_a \
  experiment.pipeline.start_stage=manipuland \
  experiment.pipeline.resume_from_path=outputs/2026-03-20/base_run_10-30-45

# Resume B: same base, different manipuland config
uv run python main.py +name=resume_b \
  experiment.pipeline.start_stage=manipuland \
  experiment.pipeline.resume_from_path=outputs/2026-03-20/base_run_10-30-45 \
  manipuland_agent.some_param=other_value
```

Resume behavior:

- the source run stays unchanged
- the new run starts as a full copy of the source scene directory
- absolute paths inside copied checkpoint files are rewritten to the new output directory; if any path cannot be rewritten, the run **aborts with an error** rather than continuing with stale paths
- outputs from `start_stage` onward are deleted before rerun so stale files do not survive
- resumed scenes write `scene_000/resume_manifest.json` with provenance and cleanup details
- new tracking summaries in `scene_000/run_analysis/` describe the rerun, while `resume_manifest.json` remains the provenance record for copied source artifacts

For post-processing-only resumes:

- `start_stage=reward_spec` keeps the finished room outputs and final scene, deletes reward/morphology outputs, then reruns reward-spec generation; **`rl_artifacts.json` must be present** in the source scene. If missing the run aborts with a clear error.
- `start_stage=morphology_task` keeps earlier outputs and reruns only morphology-task generation; **`rl_artifacts.json` must be present** in the source scene. If missing the run aborts with a clear error. Run the full pipeline through at least `reward_spec` first.

### What gets invalidated

Resume keeps the last trusted upstream checkpoint and removes everything that depends on later stages:

| `start_stage` | Retained room outputs | Deleted before rerun |
| --- | --- | --- |
| `furniture` | no room-stage outputs | all room-stage outputs, `combined_house*`, `rl_artifacts.json`, `reward_spec.yaml`, `curriculum_spec.yaml`, `morphology_task_config.json` |
| `wall_mounted` | `generated_assets/furniture/`, `scene_after_furniture/` | wall/ceiling/manipuland room outputs and all scene-level downstream outputs |
| `ceiling_mounted` | furniture + wall assets, `scene_after_wall_objects/` | ceiling/manipuland room outputs and all scene-level downstream outputs |
| `manipuland` | furniture + wall + ceiling assets, `scene_after_ceiling_objects/` | manipuland/final room outputs and all scene-level downstream outputs |
| `reward_spec` | all room outputs and final scene | `reward_spec.yaml`, `curriculum_spec.yaml`, `morphology_task_config.json` |
| `morphology_task` | everything earlier | `morphology_task_config.json` |

Inherited runtime artifacts such as `*.db`, copied logs, `scene_renders/`, `simulation/`,
and room `action_log.json` files are always cleared on resume.

---

## Combining options

`skip_stages` composes with `start_stage`/`stop_stage`:

```bash
# Run floor_plan + furniture + manipuland only (skip wall and ceiling)
uv run python main.py +name=my_run \
  'experiment.pipeline.skip_stages=[wall_mounted,ceiling_mounted]'

# Resume from furniture, skip ceiling, stop after manipuland
uv run python main.py +name=my_run \
  experiment.pipeline.start_stage=furniture \
  'experiment.pipeline.skip_stages=[ceiling_mounted]'
```
