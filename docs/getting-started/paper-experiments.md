# Paper experiment reproduction

All commands run from the TaskSmith repo root with the venv active and `.env` present.
See [Installation](installation.md) and [Providers & models](../features/providers.md) for setup.

Outputs land under `outputs/YYYY-MM-DD/<name>_HH-MM-SS/scene_000/`.

---

## A. Kitchen scene generation (E0 / E1)

The baseline experiment generates a kitchen-dining area with full task artifacts (task spec, RL artifacts, reward spec).

```bash
uv run python main.py \
  --config-path experiments/configs/E0_baseline \
  +name=kitchen_c3 \
  "experiment.prompts=['A compact kitchen-dining area with a fixed stone countertop along one wall, a wooden sideboard, a tall potted plant in one corner, and a rectangular dining table with four chairs. On the table: a dinner plate, a bowl, a glass, a fork, and a knife. On the countertop: a cutting board, a sponge, and a small pot. Task: arrange the plate, bowl, glass, fork, and knife in a proper place setting on the table.']"
```

Key outputs:

| File | Contents |
|------|----------|
| `task_requirements.json` | Structured task spec: required objects and placement stages |
| `rl_artifacts.json` | Resolved object IDs and support surface positions |
| `reward_spec.yaml` | Dense reward function specification |
| `combined_house/house.dmd.yaml` | Full scene descriptor |
| `combined_house/house.blend` | Blender scene for rendering |

---

## B. Multi-scene generalization (E2)

Run the same pipeline on a different room type to evaluate layout generalization. No code changes; only the prompt differs.

Office scene:

```bash
uv run python main.py \
  --config-path experiments/configs/E0_baseline \
  +name=office \
  "experiment.prompts=['A compact home office with a wooden writing desk along one wall, an office chair in front of it, a two-shelf bookcase against the opposite wall, and a small filing cabinet in one corner. On the desk: a laptop computer, a ceramic coffee mug, a pen holder with several pens, and a spiral-bound notebook. Task: clear the desk surface by placing the coffee mug on the bookcase shelf and the notebook next to the filing cabinet.']"
```

Living room scene:

```bash
uv run python main.py \
  --config-path experiments/configs/E0_baseline \
  +name=living_room \
  "experiment.prompts=['A cozy living room with a three-seat sofa along one wall, a rectangular coffee table centered in front of it, and a tall bookshelf against the opposite wall. Scattered on the sofa: three hardcover books and a paperback novel. Task: pick up each book from the sofa and place it upright on the bookshelf.']"
```

---

## C. Tabletop mode (E3)

Tabletop mode generates a flat-floor scene with one LLM-placed table and manipulands, skipping walls, ceiling, and full room enclosure. See [Tabletop mode](../features/tabletop-mode.md).

Research desk:

```bash
uv run python main.py \
  +name=tabletop_desk \
  experiment=tabletop \
  manipuland_agent=tabletop_manipuland_agent \
  furniture_agent=tabletop_furniture_agent \
  "experiment.prompts=['A researcher'\''s desk with a laptop, a ceramic mug, a spiral notebook, and a pen. Task: place the mug to the right of the laptop and the notebook in front of it.']"
```

Kitchen prep table:

```bash
uv run python main.py \
  +name=tabletop_kitchen \
  experiment=tabletop \
  manipuland_agent=tabletop_manipuland_agent \
  furniture_agent=tabletop_furniture_agent \
  "experiment.prompts=['A kitchen prep table with a cutting board, a knife, two tomatoes, and a bowl. Task: place the tomatoes on the cutting board and the knife beside it.']"
```

---

## D. Domain randomization (E4 / E5)

Domain randomization generates N task-valid variants from a completed base scene. Run experiment A or B first, then pass the output directory. See [Domain randomization](../features/domain-randomization.md).

**v1 - programmatic (`mode=deterministic`):**

```bash
uv run python -m scenesmith.domain_randomization.cli \
  +name=dr_batch_v1 \
  +domain_randomization.base_experiment_dir=outputs/YYYY-MM-DD/<run> \
  +domain_randomization.num_variations=10 \
  domain_randomization.mode=deterministic
```

**v2 - agentic (`mode=agentic`):**

```bash
uv run python -m scenesmith.domain_randomization.cli \
  +name=dr_batch_v2 \
  +domain_randomization.base_experiment_dir=outputs/YYYY-MM-DD/<run> \
  +domain_randomization.num_variations=10 \
  domain_randomization.mode=agentic \
  '+domain_randomization.description=vary object placement and clutter level'
```

DR outputs land alongside the base scene under `outputs/YYYY-MM-DD/<dr_name>_<time>/`.

---

## Parallel runs

`experiment.num_workers=N` runs up to N prompts simultaneously, each in its own subprocess. Workers are capped to the number of prompts, so pass at least N prompts:

```bash
uv run python main.py \
  --config-path experiments/configs/E0_baseline \
  +name=kitchen_batch \
  experiment.num_workers=4 \
  "experiment.prompts=['Prompt A.', 'Prompt B.', 'Prompt C.', 'Prompt D.']"
```

Note: `num_workers > 1` is mutually exclusive with `experiment.pipeline.parallel_rooms`. See [Quickstart](quickstart.md) for resume, logging, and output analysis commands.
