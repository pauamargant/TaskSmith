# Domain Randomization

Generates N varied copies of a completed base scene for RL training diversity.
Takes a finished scenesmith output directory and produces a batch of scene
variations, each with its own `scene_state.json`, reward spec, and exports.

---

## Quick Start

```sh
uv run python -m scenesmith.domain_randomization.cli \
    +name=dr_batch_01 \
    +domain_randomization.base_experiment_dir=outputs/2024-01-01/kitchen_baseline_12-00-00 \
    +domain_randomization.num_variations=15 \
    '+domain_randomization.description=vary clutter and object placement'
```

Output lands in `<base_experiment_dir>/randomizations/<timestamp>/` unless
`domain_randomization.output_dir` is set explicitly.

---

## Architecture

The orchestrator runs five phases:

```
ANALYZE → PLAN → CRITIQUE → EXECUTE → REWARD/EXPORT
```

### Analyze
Pure Python, no LLM. Reads from the base scene directory:
- `scene_state.json` - object inventory by type
- `task_requirements.json` → task-critical object IDs, task type
- `rl_artifacts.json` → `clutter_ids`, `furniture_movable`, position bounds per object
- `reward_spec.yaml` → reward invariants (spatial relationships that must be preserved)
- `curriculum_spec.yaml` → difficulty axis

Builds a `SceneContext` that constrains all downstream decisions.

### Plan
A multi-turn planning agent proposes N `VariationSpec` objects using these tools:
- `propose_variation(id, description, strategy, ops, ...)` - adds a variation to the plan
- `get_task_critical_objects()` - locked IDs that must not be modified
- `get_clutter_objects()` - safe dropout targets from `rl_artifacts`
- `finalize_plan()` - signals completion

### Critique
A single-turn structured-output pass checks whether variations are actually distinct, whether all specs respect locked objects and reward invariants, and whether the plan addresses the user description. If not approved, issues are sent back to the planner (max 2 revision rounds).

### Execute
All variations run concurrently. Per variation: copy the checkpoint and assets from the base scene, restore `RoomScene` from checkpoint, apply programmatic ops (or LLM-directed edits in `llm_directed` mode), write `scene_state.json`, copy/patch/regenerate the reward spec depending on what changed, then assemble `combined_house/` and run the Drake export.

### Current Limitation

The randomizer is single-room: it restores and modifies the first `room_*`
directory only. Multi-room scenes are not supported.

---

## Variation Strategies

Each variation uses one of two strategies:

| Strategy | How | Cost |
|---|---|---|
| `programmatic` | Direct scene_state.json manipulation, no LLM | Fast |
| `llm_directed` | LLM edits individual object properties via structured output | LLM cost per object |

Always set `strategy` explicitly in `VariationSpec`. If an unrecognised value is given and `ops` are present, the strategy is auto-corrected to `"programmatic"` and a `WARNING` is logged; this indicates a misconfigured spec.

### Programmatic Operations

All ops skip `task_critical_object_ids` automatically.

| Op | What it does | Key params |
|---|---|---|
| `pose_perturbation` | Gaussian noise on XY position + yaw rotation | `translation_std_m`, `rotation_std_deg` |
| `scale_jitter` | Random scale factor within a range | `scale_range: [0.85, 1.15]` |
| `object_dropout` | Remove random N% of objects (prefer `clutter_ids`) | `rate`, `min_count`, `preferred_ids` |
| `asset_substitution` | Swap geometry/SDF with catalog alternative (same semantic class) | `catalog_dir` |

---

## Edit Mode

When `edit_instruction` is set and the scene already has objects of the relevant
type, the designer receives a prompt that shows the existing layout and asks for
targeted edits rather than regenerating from scratch. Applies to:

- `StatefulFurnitureAgent.add_furniture(scene, edit_instruction=...)`
- `StatefulWallAgent.add_wall_objects(scene, edit_instruction=...)`
- `StatefulCeilingAgent.add_ceiling_objects(scene, edit_instruction=...)`
- `StatefulManipulandAgent.add_manipulands(scene, edit_instruction=...)`

---

## Reward Spec Updates

After each variation the orchestrator classifies the impact on the reward spec:

| Impact | Condition | Action |
|---|---|---|
| `unchanged` | No task-critical objects touched | Copy base `reward_spec.yaml` as-is |
| `patch` | Task-critical objects pose/scale changed | Update position references in YAML |
| `regenerate` | Structural change (asset_substitution on task-critical objects) | Re-run `RewardSpecAgent` |

Each variation gets its own `reward_spec.yaml` and `curriculum_spec.yaml`.

---

## Output Structure

```
<base_experiment_dir>/
└── randomizations/
    └── 2026-04-27_143000/
        ├── orchestration_plan.json     ← plan + per-variation status
        ├── orchestrator_planner.db     ← planning agent session
        ├── variation_000/
        │   └── scene_000/
        │       ├── room_000/
        │       │   ├── scene_states/
        │       │   │   ├── scene_after_<stage>/  ← copied checkpoint
        │       │   │   └── final_scene/
        │       │   │       └── scene_state.json  ← modified scene
        │       │   └── generated_assets/
        │       ├── combined_house/
        │       │   └── house.dmd.yaml            ← Drake directives
        │       ├── reward_spec.yaml              ← updated if needed
        │       ├── curriculum_spec.yaml
        │       └── exports/
        │           └── scene.dmd.yaml            ← Drake (copy)
        ├── variation_001/
        └── ...
```

### `orchestration_plan.json`

Tracks the full plan and live execution status.

If `domain_randomization.output_dir` points at an existing randomization run,
the CLI loads `orchestration_plan.json` and resumes from it. Variations already
marked `completed` are skipped; `pending` and `failed` variations are rerun.

> **Note:** resume via `output_dir` is only supported for `mode="deterministic"`. Setting `output_dir` with `mode="agentic"` raises an error; the agentic Director always starts a fresh plan.

```json
{
  "created_at": "2026-04-27T14:30:00",
  "base_experiment_dir": "/path/to/experiment",
  "description": "vary clutter level",
  "num_variations": 15,
  "orchestrator_reasoning": "...",
  "variations": {
    "variation_000": {
      "spec": {
        "strategy": "programmatic",
        "ops": [{"op_type": "object_dropout", "rate": 0.3}],
        "task_critical_object_ids": ["cup_a8f3b2"],
        "difficulty": 0.2
      },
      "status": "completed",
      "started_at": "...",
      "completed_at": "..."
    }
  }
}
```

---

## Configuration

```yaml
# configurations/domain_randomization/base_domain_randomization.yaml
base_experiment_dir: null     # required
num_variations: 10
mode: "deterministic"         # "deterministic" = v1 programmatic path; "agentic" = v2 Director path
description: null             # optional natural language guidance
description_file: null        # optional text file for punctuation-heavy guidance
output_dir: null              # defaults to <base_dir>/randomizations/<timestamp>

exports:
  drake: true
  blender: false

# Additional blocks available for mode="agentic":
# director.self_critique: true/false
# diversity.two_wave: true/false
# diversity.min_distance: float
# subflow.collision_check.*: per-step collision feedback settings
# openai.reasoning_effort: "low"/"medium"/"high"
```

CLI overrides:

```sh
uv run python -m scenesmith.domain_randomization.cli \
    +name=dr_batch_02 \
    +domain_randomization.base_experiment_dir=/path/to/experiment \
    +domain_randomization.num_variations=20 \
    '+domain_randomization.description=focus on lighting variation' \
    +domain_randomization.exports.blender=true
```

For punctuation-heavy or multiline instructions, prefer a text file:

```sh
uv run python -m scenesmith.domain_randomization.cli \
    +name=dr_batch_02 \
    +domain_randomization.base_experiment_dir=/path/to/experiment \
    +domain_randomization.description_file=/path/to/dr_instructions.txt
```

Resume a prior run by reusing its output directory (supported only for `mode="deterministic"`):

```sh
uv run python -m scenesmith.domain_randomization.cli \
    +name=dr_resume_01 \
    +domain_randomization.base_experiment_dir=/path/to/experiment \
    domain_randomization.output_dir=/path/to/experiment/randomizations/2026-04-27_143000
```

---

## Module Layout

```
scenesmith/domain_randomization/
├── orchestrator.py          # DomainRandomizationOrchestrator
├── variation_spec.py        # VariationSpec, OrchestrationPlan dataclasses
├── scaffolding.py           # folder creation + checkpoint copying
├── director.py              # v2 agentic Director (VariationGoal → DRSubflowAgent)
├── dr_scene_loader.py       # scene state loading helpers
├── llm_editor.py            # LLM-directed object edits (v2 subflow)
├── rl_artifacts_updater.py  # RL artifact patching after variation
├── reward_spec_updater.py   # reward spec impact classification + update
├── export.py                # combined house assembly + Drake/Blender
├── cli.py                   # Hydra CLI entry point
├── programmatic/
│   ├── base.py              # BaseRandomizationOp ABC
│   ├── collision_checker.py # AABB + room bounds validation; skips structural walls/floor
│   ├── pose_perturbation.py
│   ├── scale_jitter.py
│   ├── object_dropout.py
│   └── asset_substitution.py
└── subflow/                 # v2 agentic subflow (used when mode="agentic")
    ├── stateful_subflow.py  # DRSubflowAgent (planner/designer/critic loop)
    ├── tools.py             # subflow tool definitions
    └── dr_subflow_cfg.py    # subflow config dataclass
```

---

## Reused components

The DR orchestrator calls stage agents and scene helpers directly, bypassing `indoor_scene_generation.py`:

- `_copy_checkpoint_for_stage()` — checkpoint scaffolding
- `StatefulFurnitureAgent`, `StatefulWallAgent`, `StatefulCeilingAgent`, `StatefulManipulandAgent`
- `RewardSpecAgent.generate()` — for the `regenerate` reward spec path
- `HouseScene.assemble()` — produces `combined_house/house.dmd.yaml`
- `ConsoleLogger` — per-variation logging
