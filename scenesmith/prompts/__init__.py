"""Prompt management system for scene agent."""

import os

from pathlib import Path

from .manager import PromptManager
from .registry import (
    AssetRouterPrompts,
    DRSubflowPrompts,
    FloorPlanAgentPrompts,
    FurnitureAgentPrompts,
    ImageGenerationPrompts,
    ManipulandAgentPrompts,
    MeshPhysicsPrompts,
    MorphologyTaskAgentPrompts,
    PromptRegistry,
    RewardSpecAgentPrompts,
    TaskAnalysisAgentPrompts,
)

# Initialize the prompt manager with the data directory.
# Override with SCENESMITH_PROMPTS_DIR env var to use a different prompt variant.
_default_data_dir = Path(__file__).parent / "data"
PROMPTS_DATA_DIR = Path(
    os.environ.get("SCENESMITH_PROMPTS_DIR", str(_default_data_dir))
)

prompt_manager = PromptManager(prompts_dir=PROMPTS_DATA_DIR)

# Initialize the prompt registry.
prompt_registry = PromptRegistry(prompt_manager)

__all__ = [
    "prompt_manager",
    "prompt_registry",
    "AssetRouterPrompts",
    "DRSubflowPrompts",
    "FloorPlanAgentPrompts",
    "FurnitureAgentPrompts",
    "ImageGenerationPrompts",
    "ManipulandAgentPrompts",
    "MeshPhysicsPrompts",
    "MorphologyTaskAgentPrompts",
    "RewardSpecAgentPrompts",
    "TaskAnalysisAgentPrompts",
]
