# Tabletop Mode

Generates a minimal scene: flat open floor, one LLM-placed table, and objects
on that table. No walls, no ceiling, no room enclosure. Useful for manipulation
datasets, fast iteration, or any task where a full room would be overhead.

---

## Quick Start

```bash
uv run python main.py +name=tabletop_kitchen \
  experiment=tabletop \
  manipuland_agent=tabletop_manipuland_agent \
  furniture_agent=tabletop_furniture_agent \
  "experiment.prompts=['A wooden table with a coffee mug, open notebook, and a pen.']" \
  experiment.num_workers=1
```

Output lands in `outputs/<date>/tabletop_kitchen_<time>_<id>/scene_000/room_tabletop/`.

---

## How It Works

The standard pipeline is:

```
floor_plan → furniture → wall_mounted → ceiling_mounted → manipuland → reward_spec → morphology_task
```

Tabletop mode changes three things.

**Floor plan.** `_generate_floor_plan_worker` detects `experiment.tabletop_mode=true`
and calls `scenesmith.tabletop.floor_plan.build_tabletop_house_layout()` instead of
the LLM floor plan agent. This produces a single flat-floor `HouseLayout`: one room
named `tabletop` at the configured dimensions (default 4×4 m), a box SDF floor with no
wall geometry, and `wall_height=10.0` so physics-validation height checks always pass.
No Blender render, no LLM call, no geometry server — the stage completes in under a second.

**Furniture.** The room prompt is the user's original prompt plus a hard constraint
appended by the floor plan builder:

```
TABLETOP SCENE CONSTRAINT: This is a minimalist, wall-free open surface
(no room enclosure). Place exactly ONE table in the centre of the floor
area — no chairs, shelves, or any other furniture. The table should be
large enough to hold multiple small objects on its top surface.
```

`furniture_agent=tabletop_furniture_agent` reduces the critique round count to 1 so
the critic cannot ask the agent to add more items.

**Skipped stages.** `skip_stages: [wall_mounted, ceiling_mounted, reward_spec, morphology_task]`
is set in the tabletop experiment config. The manipuland stage runs normally on the
single table's support surfaces.

---

## Configuration

### Experiment preset

```yaml
# configurations/experiment/tabletop.yaml
tabletop_mode: true
tabletop_room_size: [4.0, 4.0]   # [width_m, length_m]
pipeline:
  skip_stages: [wall_mounted, ceiling_mounted, reward_spec, morphology_task]
```

Override floor size on the CLI:

```bash
experiment.tabletop_room_size=[6.0,3.0]
```

Use a larger floor when the table itself is unusually big (e.g. a workbench).

### Manipuland preset (`tabletop_manipuland_agent`)

| Key | Tabletop value | Default |
|---|---|---|
| `max_critique_rounds` | 4 | 2 |
| `asset_manager.hssd.use_top_k` | 12 | 5 |

More critique rounds give the agent additional passes to fill the tabletop
naturally and resolve placement conflicts. The wider HSSD search (12 candidates
per query instead of 5) increases object variety.

### Furniture preset (`tabletop_furniture_agent`)

| Key | Tabletop value | Default |
|---|---|---|
| `max_critique_rounds` | 1 | 2 |

One critique round is enough because the prompt already constrains the agent to a
single table; extra rounds risk the critic requesting additional furniture.

---

## Adjusting Object Variety

To use Objaverse instead of HSSD (much larger library, slower retrieval):

```bash
manipuland_agent.asset_manager.general_asset_source=objaverse
```

To increase the candidate pool further without switching sources:

```bash
manipuland_agent.asset_manager.hssd.use_top_k=20
```

---

## Output Structure

```
scene_000/
├── house_layout.json                    ← minimal tabletop HouseLayout
├── floor_plans/
│   └── tabletop/
│       └── room_geometry.sdf            ← flat floor box, no walls
└── room_tabletop/
    ├── generated_assets/
    │   └── furniture/                   ← one table asset
    │   └── manipuland/                  ← tabletop objects
    ├── scene_states/
    │   ├── scene_after_furniture/
    │   ├── scene_after_wall_objects/    ← copy of furniture state (skipped stage)
    │   ├── scene_after_ceiling_objects/ ← copy of furniture state (skipped stage)
    │   └── final_scene/
    │       └── scene_state.json
    └── scene_renders/
        ├── furniture/
        └── experiment_analysis/
```

---

## Limitations

- `wall_mounted` and `ceiling_mounted` stages are skipped; the tabletop scene has no
  surfaces for those objects.
- `reward_spec` and `morphology_task` are also skipped by default. To generate RL
  artifacts, run a task-aware prompt (see [Task-aware generation](task-aware-generation.md))
  and remove those stages from `skip_stages`.
- The floor has no visual mesh, only a collision box. For rendering, Blender will show
  a plain grey box. Add a floor material override if visual fidelity matters.
- Multi-table scenes are not supported by this mode. For two or more tables, use the
  standard pipeline with a standard room prompt.

---

## Implementation

| File | Role |
|---|---|
| `scenesmith/tabletop/floor_plan.py` | `build_tabletop_house_layout()` - programmatic HouseLayout builder |
| `scenesmith/experiments/indoor_scene_generation.py` | Bypass injected in `_generate_floor_plan_worker` |
| `configurations/experiment/tabletop.yaml` | Experiment preset |
| `configurations/manipuland_agent/tabletop_manipuland_agent.yaml` | Manipuland overrides |
| `configurations/furniture_agent/tabletop_furniture_agent.yaml` | Furniture overrides |
| `tests/unit/test_tabletop_floor_plan.py` | Unit tests |
