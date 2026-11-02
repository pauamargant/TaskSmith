# Task-aware generation

When an RL task prompt is supplied, the pipeline runs `TaskAnalysisAgent` before floor plan generation, saves `task_requirements.json`, and passes the extracted task context to every downstream agent. After generation, `RLArtifactAgent` reviews the finished scene and writes `rl_artifacts.json` with grounded object IDs, success conditions, and domain-randomization specs. Reward and morphology-task stages are documented in the [Pipeline overview](../pipeline/overview.md).

Plain scene-description prompts continue to work unchanged — see [Backward compatibility](#backward-compatibility).

---

## Overview

Plain scene description:
```
"A cozy studio apartment with a kitchen area and a reading nook."
```

Task-focused prompt:
```
"An agent must pick up a ceramic mug from the kitchen table and place it in the sink."
```

When a task prompt is detected the pipeline:
1. Runs `TaskAnalysisAgent` → saves `task_requirements.json`
2. Builds an augmented scene prompt carrying RL context
3. Passes `rl_task_context` (compact version) to every critic agent
4. Each critic scores a new **RL Task Suitability** dimension
5. Furniture and manipuland designers get a `tag_as_task_object` tool
6. After full generation runs `RLArtifactAgent` → saves `rl_artifacts.json`

---

## Writing task prompts

Specify the action, the objects by specific name ("ceramic mug" not "cup" — asset retrieval uses these directly), and both the initial and goal positions. Room type is inferred from context ("kitchen table" implies a kitchen).

Examples:
```
"An agent must pick up a ceramic mug from the kitchen table and place it in the sink."
"A character needs to open the refrigerator door and retrieve a bottle from inside."
"A robot navigates from the living room to the bedroom while avoiding obstacles."
"An agent sorts red and blue blocks on the dining table into separate containers."
```

---

## Pipeline flow

```
Task prompt
    │
    ▼
TaskAnalysisAgent                ← single VLM call, saves task_requirements.json
    │
    ▼  augmented_prompt + rl_task_context
    │
    ▼
Floor plan agent                 ← rl_task_context injected, critic scores RL suitability
    │
    ▼
Furniture agent (per room)       ← designer has tag_as_task_object tool, critic scores RL
    │
    ▼
Wall / Ceiling agents            ← critic scores RL suitability
    │
    ▼
Manipuland agent                 ← designer has tag_as_task_object tool, critic scores RL
    │
    ▼
RLArtifactAgent                  ← reviews final scene, saves rl_artifacts.json
```

---

## `task_requirements.json` - schema

Produced by `TaskAnalysisAgent` before floor plan generation.

```json
{
  "raw_prompt": "An agent must pick up a ceramic mug...",
  "task_summary": "Pick and place: mug from table to sink",
  "task_type": "pick_and_place",
  "scene_description": "A kitchen with a dining table, sink, and counter...",
  "required_objects": [
    {
      "name": "ceramic coffee mug",
      "is_task_object": true,
      "placement_stage": "manipuland",
      "properties": {"approx_size_cm": [10, 8, 12], "must_be_upright": true},
      "placement_hint": "on dining table surface"
    },
    {
      "name": "kitchen sink",
      "is_task_object": true,
      "placement_stage": "furniture",
      "properties": {},
      "placement_hint": "mounted in kitchen counter"
    }
  ],
  "task_objects": ["ceramic coffee mug", "kitchen sink"],
  "spatial_constraints": [
    "sink must be accessible from all sides",
    "table must be in open area with clear approach"
  ],
  "scene_requirements": {
    "room_types": ["kitchen"],
    "needs_open_floor": true
  },
  "rl_metadata": {
    "initial_state": "Mug upright on dining table surface",
    "goal_state": "Mug placed inside kitchen sink"
  },
  "domain_randomization": {
    "task_object_variants": [
      {
        "name": "ceramic coffee mug",
        "can_swap_with": ["glass mug", "ceramic cup"],
        "randomizable_properties": ["color", "material", "position_on_table"],
        "position_range": "within 0.3m on table surface"
      }
    ],
    "furniture_movable": [
      {
        "name": "dining table",
        "can_reposition": true,
        "can_swap_with": [],
        "constraint": "must remain in kitchen area with 1m clearance on all sides"
      }
    ],
    "clutter_objects": [
      {
        "surface": "dining table",
        "can_add": true,
        "can_remove": true,
        "density_range": ["sparse", "moderate"]
      }
    ],
    "scene_level": {
      "lighting_intensity": [0.6, 1.4],
      "floor_texture": true,
      "wall_color": true
    },
    "notes": "Sink position must remain fixed as it is the goal location."
  }
}
```

---

## `rl_artifacts.json` - schema

Produced by `RLArtifactAgent` after full scene generation. Contains grounded object IDs and positions from the actual scene.

```json
{
  "task_objects": [
    {
      "object_id": "mug_a8f3b2",
      "name": "ceramic coffee mug",
      "position_xyz": [1.2, 0.8, 0.85],
      "bbox_dims": [0.10, 0.08, 0.12],
      "sdf_path": "assets/generated/mug_a8f3b2/model.sdf",
      "parent_surface": "dining_table_c4d5e6",
      "is_initial_object": true,
      "is_goal_location": false
    },
    {
      "object_id": "sink_f1e2d3",
      "name": "kitchen sink",
      "position_xyz": [3.1, 0.6, 0.90],
      "bbox_dims": [0.60, 0.50, 0.25],
      "sdf_path": "assets/hssd/sink_f1e2d3/model.sdf",
      "parent_surface": null,
      "is_initial_object": false,
      "is_goal_location": true
    }
  ],
  "success_conditions": {
    "description": "Mug is placed inside the kitchen sink basin",
    "goal_objects": ["mug_a8f3b2"],
    "goal_location": "sink_f1e2d3",
    "tolerance_m": 0.15
  },
  "initial_state": {
    "mug_a8f3b2": {"position_xyz": [1.2, 0.8, 0.85]}
  },
  "domain_randomization": {
    "task_object_variants": [
      {
        "object_id": "mug_a8f3b2",
        "can_swap_with": ["ceramic cup", "glass mug"],
        "randomizable_properties": ["color", "material"],
        "position_range_xy_on_surface": [0.3, 0.3]
      }
    ],
    "furniture_movable": [
      {
        "object_id": "dining_table_c4d5e6",
        "can_reposition": true,
        "reposition_range_xy": [0.5, 0.5],
        "can_swap_with": [],
        "constraint": "must remain in kitchen area with 1m clearance"
      }
    ],
    "clutter_ids": ["plate_b2c3d4", "napkin_e5f6a7"],
    "scene_level": {
      "lighting_intensity_range": [0.6, 1.4],
      "floor_texture_ok": true,
      "wall_color_ok": true
    }
  },
  "verification": {
    "all_required_objects_present": true,
    "missing_objects": [],
    "notes": "All required objects found. Mug tagged as task object by furniture agent."
  }
}
```

---

## Domain randomization spec

The domain randomization fields in `rl_artifacts.json` tell training scripts what can vary across episodes without breaking the task.

`task_object_variants` describes what aspects of the task objects can change: alternative asset names (`can_swap_with`), randomizable appearance properties (`randomizable_properties`), and how far from the nominal position the object can start (`position_range_xy_on_surface`).

`furniture_movable` lists which furniture pieces can be repositioned, how far (`reposition_range_xy`), what can substitute (`can_swap_with`), and any constraint that must remain true.

`clutter_ids` are non-task objects on shared surfaces that can be added, removed, or swapped freely without affecting task success.

`scene_level` covers global randomization: `lighting_intensity_range` is a min/max multiplier for scene lighting; `floor_texture_ok` and `wall_color_ok` control whether surface appearance can vary.

---

## RL Task Suitability scoring

Each critic agent evaluates an **RL Task Suitability** dimension (0–10) when `rl_task_context` is non-empty:

| Stage | What is evaluated |
|---|---|
| Floor plan | Required room types present? Adequate open floor? Room adjacencies support task workflow? |
| Furniture | Required furniture objects present (explicit checklist)? Accessible? Clear approach paths? |
| Wall objects | Wall-mounted items don't interfere with task execution or block access to task areas? |
| Ceiling objects | Fixtures don't obstruct agent movement? Adequate lighting for task environment? |
| Manipulands | Task objects in correct initial state (upright, exposed, on correct surface)? Goal locations unobstructed? |

The `rl_task_context` passed to the furniture critic contains a **"Required Furniture"** section listing every `required_object` with `placement_stage: "furniture"`. The critic's STEP 2b explicitly checks each item against the `get_current_scene_state()` output and scores `rl_task_suitability ≤ 3` if any are missing, driving the designer to add them in subsequent rounds.

### `placement_stage` field

Each entry in `required_objects` carries a `placement_stage` that determines which pipeline stage is responsible:

| Value | Placed by | Examples |
|---|---|---|
| `"furniture"` | Furniture agent | dining table, stone countertop, shelving unit |
| `"manipuland"` | Manipuland agent | ceramic mug, plate, apple, sponge |

Objects with names containing words like "fixed" or "built-in" are **not** treated as pre-installed structural elements; `placement_stage` is authoritative. The augmented prompt seen by furniture/floor-plan designers explicitly says "NOT pre-installed structural or architectural elements, regardless of descriptive language like 'fixed' or 'built-in'."

Scores appear in each stage's `scores.yaml` alongside the standard dimensions (realism, functionality, layout, prompt_following).

---

## Configuration

The `task_analysis_agent` block is inherited from `configurations/task_analysis_agent/base_task_analysis_agent.yaml` via Hydra defaults. Override in your experiment config:

```yaml
provider:
  model_traits:
    reasoning_native: false

task_analysis_agent:
  openai:
    model: "openai/gpt-oss-120b"
    service_tier: null   # or "flex", "priority"
```

The task analysis agent uses the same direct openai-client path as the other single-turn VLM helpers. Endpoint capabilities are declared in the root `provider` block; model/service-tier settings still come from the agent/global `openai` config; and TaskSmith still consumes only the model's final answer text, never reasoning traces.

---

## Resuming runs

When resuming from a later stage (`start_stage: "furniture"` or later):

1. The pipeline checks for `task_requirements.json` in the scene directory
2. If found, loads `TaskRequirements` and reconstructs `rl_task_context`
3. Passes `rl_task_context` to all room-level agents as normal

This means RL context is preserved across resume points without re-running the task analysis VLM call.

When `resume_from_path` is used, the preserved `task_requirements.json` lives inside a new
self-contained output directory. The source run is not edited in place.

---

## Backward compatibility

Old-style scene description prompts continue to work unchanged:
- No `task_requirements.json` is produced
- `rl_task_context` is an empty string throughout
- All `{% if rl_task_context %}` blocks in critic prompts are skipped
- Critic output uses base dataclasses (no `rl_task_suitability` field)
- No `rl_artifacts.json` is produced

No configuration changes are needed to use the old behavior.
