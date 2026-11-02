from omegaconf import DictConfig

from .base_experiment import BaseExperiment
from .indoor_scene_generation import IndoorSceneGenerationExperiment

# Each key corresponds to an `experiment/<name>.yaml` config-group option. Hydra
# sets `cfg.experiment._name` to that option name, so any preset file we ship
# must be registered here (even if it reuses the same experiment class).
exp_registry = {
    "indoor_scene_generation": IndoorSceneGenerationExperiment,
    # Presets that layer defaults/overrides on top of the same experiment class.
    "base_experiment": IndoorSceneGenerationExperiment,
    "tasksmith_defaults_wandb": IndoorSceneGenerationExperiment,
    "example_bundle": IndoorSceneGenerationExperiment,
    "locallm": IndoorSceneGenerationExperiment,
    "tabletop": IndoorSceneGenerationExperiment,
}


def build_experiment(cfg: DictConfig) -> BaseExperiment:
    """
    Build an experiment instance based on registry

    Args:
        cfg (DictConfig): The experiment configuration.

    Returns:
        BaseExperiment: The experiment instance.
    """
    if cfg.experiment._name not in exp_registry:
        raise ValueError(
            f"Experiment {cfg.experiment._name} not found in registry "
            f"{list(exp_registry.keys())}. Make sure you register it correctly in "
            "'experiment/__init__.py' under the same name as yaml file."
        )

    return exp_registry[cfg.experiment._name](cfg)
