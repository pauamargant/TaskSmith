# Installation

Data setup and upstream SceneSmith installation is documented in the root `README.md`. The notes below cover Python version, dependencies, GPU paths, and provider credentials.

## Python and uv

- **Python**: `3.11` (see repo `.python-version`).
- **uv**: install from [astral.sh](https://docs.astral.sh/uv/).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Project dependencies

From the repository root:

```bash
uv sync
```

This installs the full runtime, including the direct openai-client adapter for non-Agents-SDK provider calls.

Omit dev/test dependencies:

```bash
uv sync --no-dev
```

Activate the virtualenv:

```bash
source .venv/bin/activate
```

## Documentation tooling

To build or preview this site:

```bash
uv sync --group docs
uv run mkdocs serve
```

## GPU setup

### PyTorch (CLIP retrieval)

CLIP similarity search (`scenesmith/agent_utils/clip_embeddings.py`) runs on CUDA if available.
The project requires **torch ≥ 2.7** because sm_120 (Blackwell / RTX 50-series) support was not backported to earlier releases.
`pyproject.toml` already points `torch` and `torchvision` at the `pytorch-cu128` index, so `uv sync` installs the right wheel automatically.

If you see a warning like:
```
GPU architecture sm_120 is not supported by the installed PyTorch build. Falling back to CPU for CLIP inference.
```
your environment has an older PyTorch. Re-run `uv sync` to pick up the cu128 wheel.

### Blender (scene rendering)

Blender rendering has two separate GPU paths:

**EEVEE Next** (used for all scene-observation and asset-validation renders):
EEVEE Next uses Vulkan. In headless sessions (no `DISPLAY`) the Vulkan loader silently falls
back to Mesa LavaPipe (CPU-based software renderer) unless told otherwise.
`BlenderServer.start()` automatically sets `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`
when that file exists and the variable is not already set, forcing GPU rendering.
No manual action required; the log line `Set VK_ICD_FILENAMES=...` confirms it.

To override (e.g. for a different GPU or a multi-ICD setup), set `VK_ICD_FILENAMES` in your shell
before launching and the automatic assignment is skipped.

**Cycles** (used for the 8-view CLIP asset pre-renders):
`render_multiview_for_clip_embedding()` calls `setup_cycles_gpu_rendering()` which enables
OPTIX (preferred) or CUDA on NVIDIA hardware, with graceful CPU fallback when no compatible GPU is found.

### Datasets and assets

Asset data is expected under `data/` at the repo root:

| Path | Contents |
|------|----------|
| `data/hssd-models/` | HSSD furniture and object meshes |
| `data/objathor-assets/` | Objaverse-derived 3D assets |
| `data/materials/` | PBR material library + CLIP embeddings |
| `data/preprocessed/` | Precomputed CLIP embeddings for HSSD/Objaverse |

Follow the upstream [SceneSmith](https://github.com/nepfaff/scenesmith) data download instructions to populate these directories.

## Provider credentials

LLM access is configured via environment variables, typically a `.env` file in the repo root. See [Providers & models](../features/providers.md) for model configuration and startup validation.
