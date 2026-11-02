# Configuration system

TaskSmith uses [Hydra](https://hydra.cc) for configuration. Every run's final config is assembled from a directory of YAML files at startup. No config is hardcoded in Python; changing behaviour means changing a YAML or passing a CLI override.

---

## The shared config tree: `configurations/`

By default, Hydra reads from `configurations/` at the repo root:

```text
configurations/
  config.yaml              # entry point: provider block, defaults list, hydra.run.dir
  provider/                # capability presets: qwen.yaml, gemma.yaml, neutral.yaml
  experiment/              # prompts, pipeline stages, tracking, retrieval servers
  floor_plan_agent/        # floor plan LLM agent settings
  furniture_agent/         # furniture placement agent (planner/designer/critic loop)
  wall_agent/              # wall-mounted object agent
  ceiling_agent/           # ceiling-mounted object agent
  manipuland_agent/        # graspable object agent
  task_analysis_agent/     # task requirement extraction
  reward_spec_agent/       # reward function generation
  morphology_task_agent/   # robot morphology and task config
  domain_randomization/    # DR orchestrator settings
```

Each subdirectory is a **config group**. Hydra loads one YAML file per group (selected by the `defaults:` list in `config.yaml`) and merges everything into a single config object for the run.

---

## How the defaults list works

`config.yaml` contains a `defaults:` list that names which YAML to load from each group:

```yaml
# configurations/config.yaml (simplified)
defaults:
  - provider: qwen                          # → loads configurations/provider/qwen.yaml
  - experiment: indoor_scene_generation     # → loads configurations/experiment/indoor_scene_generation.yaml
  - floor_plan_agent: stateful_floor_plan_agent
  - furniture_agent: stateful_furniture_agent
  - wall_agent: stateful_wall_agent
  - ceiling_agent: stateful_ceiling_agent
  - manipuland_agent: stateful_manipuland_agent
  - task_analysis_agent: base_task_analysis_agent
  - reward_spec_agent: base_reward_spec_agent
  - morphology_task_agent: base_morphology_task_agent
  - domain_randomization: base_domain_randomization
  - _self_                                  # merge config.yaml's own top-level keys last
```

Hydra loads each file, merges them in order, and the result is the config object your run sees. The `_self_` entry controls where `config.yaml`'s own keys (the `provider:` block, `openai:`, `name:`, `hydra:`) land in the merge order.

---

## Provider credentials: env vars, not Hydra

Provider credentials live in `.env` at the repo root, **not** in the Hydra tree:

```bash
PROVIDER_BASE_URL="https://inference.rcp.epfl.ch/v1"
PROVIDER_API_KEY="<your-key>"
PROVIDER_MODEL="Qwen/Qwen3.6-27B-FP8"
```

`scenesmith/provider.py` loads `.env` at import time. The `provider.model` key in `config.yaml` reads this value via an OmegaConf resolver (`${oc.env:PROVIDER_MODEL,...}`), so the model name flows into all agent configs automatically without you touching any YAML.

Switching providers is done by setting env vars and choosing a preset (see below).

---

## CLI overrides

Any key in the merged config can be overridden from the CLI. Hydra uses dotted paths:

```bash
# Change the experiment preset
uv run python main.py +name=my_run experiment=tabletop

# Change a provider preset (swaps the whole provider/qwen.yaml block)
uv run python main.py +name=my_run provider=gemma

# Override a leaf value
uv run python main.py +name=my_run experiment.num_workers=4

# Stop the pipeline after the furniture stage
uv run python main.py +name=my_run experiment.pipeline.stop_stage=furniture

# Override a key in the furniture agent config
uv run python main.py +name=my_run furniture_agent.max_critique_rounds=1

# Pass a list value (quote the whole override)
uv run python main.py +name=my_run \
  "experiment.pipeline.skip_stages=[wall_mounted,ceiling_mounted]"
```

Use `+key=value` for **new** keys not declared in any YAML (like `+name=`, which is required). Use `key=value` (no `+`) for keys that already exist in the merged config.

---

## Provider presets

The `provider/` group contains three presets. Switch from the CLI:

| CLI flag | File loaded | Use when |
|---|---|---|
| `provider=qwen` *(default)* | `provider/qwen.yaml` | Qwen3 on vLLM (EPFL RCP) |
| `provider=gemma` | `provider/gemma.yaml` | Gemma 4 on vLLM |
| `provider=neutral` | `provider/neutral.yaml` | OpenAI or Azure (native reasoning_effort) |

Note: the bundled configs under `experiments/configs/E0_baseline/` embed the Qwen3 capability block directly in `config.yaml` rather than using a separate `provider/` group, so `provider=qwen` is not a valid CLI override for that bundle. Use `experiments/configs/E0_baseline_gemma/` for Gemma runs.

---

## LLM settings

### Model and service tier

Each agent's `openai:` block controls which model it uses and how it calls the API:

```yaml
# From configurations/furniture_agent/_base_agent.yaml (shared by all agents)
openai:
  model: "${provider.model}"          # resolves to PROVIDER_MODEL env var
  service_tier: ${openai.service_tier} # resolves to top-level openai.service_tier
  vision_detail: "high"
```

`provider.model` is an OmegaConf resolver that reads `PROVIDER_MODEL` at startup. All agents reference the same `${provider.model}` interpolation, so changing the env var (or overriding `provider.model` on the CLI) changes the model for every agent at once.

`service_tier` in `config.yaml` defaults to `null` (use the provider's project default). Valid values:

| Value | Effect |
|---|---|
| `null` | Provider project default |
| `"default"` | Standard pricing and throughput |
| `"flex"` | Lower cost, slower (Batch API rates) |
| `"priority"` | Lower latency, premium pricing |

Override from the CLI: `openai.service_tier=flex`.

### Reasoning effort and verbosity per role

Each pipeline agent runs multiple LLM roles (planner, designer, critic, and several analysis calls). The `openai.reasoning_effort` and `openai.verbosity` keys in `_base_agent.yaml` tune each role independently:

```yaml
openai:
  reasoning_effort:
    planner: "low"
    designer: "high"
    critic: "high"
    mesh_analysis: "minimal"
    asset_analysis: "medium"
    asset_validation: "minimal"
  verbosity:
    planner: "medium"
    designer: "low"
    critic: "medium"
    mesh_analysis: "medium"
    asset_analysis: "medium"
    asset_validation: "medium"
```

How `reasoning_effort` reaches the model depends on `thinking_control_mode` in the active provider preset:

| Mode | Set by | Behaviour |
|---|---|---|
| `"openai"` | `provider=neutral` | `reasoning_effort` forwarded as a native kwarg (OpenAI / Azure) |
| `"binary"` | `provider=qwen`, `provider=gemma` | `minimal` → `enable_thinking: false`; any other value → `enable_thinking: true` via `extra_body.chat_template_kwargs` |

Override a single role from the CLI: `furniture_agent.openai.reasoning_effort.designer=medium`.

### Sampling parameters

Sampling parameters (`temperature`, `top_p`, `top_k`, …) live in the active **provider preset**, not in the per-agent `openai:` block. They are forwarded with every request.

Qwen preset defaults (from `configurations/provider/qwen.yaml`):

```yaml
sampling:
  temperature: 1.0
  top_p: 0.95
  top_k: 20
  min_p: 0.0
  presence_penalty: 1.5
  repetition_penalty: 1.0
```

Gemma preset defaults (from `configurations/provider/gemma.yaml`):

```yaml
sampling:
  temperature: 1.0
  top_p: 0.95
  top_k: 64
```

The neutral preset ships with `sampling: {}` — all keys left to the provider default, since OpenAI ignores vLLM-only params like `top_k` and `min_p`.

Override a sampling param from the CLI: `provider.sampling.temperature=0.8`.

### Retry and timeout knobs

The retry layer has two independent levels. Both read their values from environment variables at startup; the YAML values under `provider.retry` document the defaults only.

**Transport wrapper** (`scenesmith/llm_retry.py`) — wraps each `Runner.run()` call:

| Env var | Default | Meaning |
|---|---|---|
| `TASKSMITH_PROVIDER_MAX_ATTEMPTS` | `3` | Total attempts (not retries — 3 means try up to 3 times) |
| `TASKSMITH_PROVIDER_RETRY_BACKOFF_S` | `2.0` | Base backoff between attempts (seconds) |

**OpenAI SDK client** (`scenesmith/provider.py`) — SDK-level retries inside each transport attempt:

| Env var | Default | Meaning |
|---|---|---|
| `PROVIDER_TIMEOUT_S` | `360` | Per-request timeout for Agents SDK calls |
| `PROVIDER_MAX_RETRIES` | `5` | SDK retries per attempt (5 retries = 6 total tries) |

**Direct OpenAI client** (`scenesmith/openai_adapter.py`) — used for turn-trimming calls:

| Env var | Default | Meaning |
|---|---|---|
| `TASKSMITH_LLM_TIMEOUT_S` | `360` | Per-request timeout |
| `TASKSMITH_LLM_MAX_RETRIES` | `5` | Retries per call (5 retries = 6 total tries) |

Worst-case total attempts per `Runner.run()` call: `TASKSMITH_PROVIDER_MAX_ATTEMPTS` × (`PROVIDER_MAX_RETRIES` + 1) = 3 × 6 = 18.

Note the naming asymmetry: `TASKSMITH_PROVIDER_MAX_ATTEMPTS` counts **total attempts**, while `PROVIDER_MAX_RETRIES` and `TASKSMITH_LLM_MAX_RETRIES` count **retries** (so the total is retries + 1).

---

## Self-contained bundles: `--config-path`

`--config-path <dir>` makes Hydra load its primary `config.yaml` from `<dir>` instead of `configurations/`. The directory must contain a full Hydra config tree (same layout as `configurations/`).

Use this when:
- you want to freeze a study's settings without touching the shared `configurations/`
- you want to share a single directory that fully reproduces a result

The repo ships two bundles under `experiments/configs/`:

| Bundle | Purpose |
|--------|---------|
| `E0_baseline/` | Paper baseline: full pipeline, Qwen3 flags, kitchen task prompt |
| `E0_baseline_gemma/` | Same pipeline with Gemma 4 capability flags |

```bash
uv run python main.py --config-path experiments/configs/E0_baseline +name=my_run
```

CLI overrides still work on top of the bundle config.

---

## Worked example: E0_baseline kitchen run

```bash
uv run python main.py \
  --config-path experiments/configs/E0_baseline \
  +name=kitchen_c3 \
  "experiment.prompts=['A compact kitchen-dining area...']"
```

`--config-path experiments/configs/E0_baseline` tells Hydra to read `experiments/configs/E0_baseline/config.yaml` as the primary config. That file's `defaults:` list loads one YAML per group from inside the bundle directory (not the shared `configurations/`), so `experiment: example_bundle` resolves to `experiments/configs/E0_baseline/experiment/example_bundle.yaml`.

```yaml
defaults:
  - experiment: example_bundle
  - floor_plan_agent: stateful_floor_plan_agent
  - furniture_agent: stateful_furniture_agent
  - wall_agent: stateful_wall_agent
  - ceiling_agent: stateful_ceiling_agent
  - manipuland_agent: stateful_manipuland_agent
  - task_analysis_agent: base_task_analysis_agent
  - reward_spec_agent: base_reward_spec_agent
  - morphology_task_agent: base_morphology_task_agent
  - domain_randomization: base_domain_randomization
  - _self_
```

CLI overrides apply on top: `+name=kitchen_c3` sets the required `name` key; `experiment.prompts=[...]` overrides the prompt list from `example_bundle.yaml`. After merge, OmegaConf resolvers run — `provider.model: "${oc.env:PROVIDER_MODEL,...}"` reads `PROVIDER_MODEL` from `.env`. Every agent config that references `${provider.model}` gets the same value.

`initialize_provider_runtime(cfg)` in `scenesmith/provider_validation.py` validates declared capabilities against the provider and, when `provider.validation.probe=true`, sends a test request to confirm the endpoint responds. Then `build_experiment(cfg)` runs the 7-stage pipeline (`floor_plan → … → morphology_task`); outputs land in `outputs/YYYY-MM-DD/kitchen_c3_HH-MM-SS/scene_000/`.

---

## Creating your own bundle

```bash
cp -a configurations experiments/configs/my_study
```

Then:

1. Edit `experiments/configs/my_study/config.yaml` `defaults:` to select different agent presets.
2. Add `experiments/configs/my_study/experiment/my_study.yaml` with your prompts and pipeline settings.
3. Point `defaults: - experiment: my_study` in that bundle's `config.yaml`.

Run it:

```bash
uv run python main.py --config-path experiments/configs/my_study +name=my_run
```
