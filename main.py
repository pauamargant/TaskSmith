"""Hydra CLI entry point for TaskSmith scene generation.

Basic invocation (requires +name):

    uv run python main.py +name=my_run "experiment.prompts=['Your scene prompt here.']"

Key overrides:

    +name=<label>                    Required. Labels the output directory.
    experiment.prompts=[...]          One or more scene generation prompts.
    experiment.num_workers=N          Run N scenes in parallel (default 1).
    --config-path experiments/configs/E0_baseline
                                     Use a self-contained experiment bundle.

Pipeline control:
    experiment.pipeline.start_stage=furniture   Resume from a checkpoint stage.
    experiment.pipeline.stop_stage=manipuland   Stop after a stage.
    "experiment.pipeline.skip_stages=[wall_mounted,ceiling_mounted]"

Tabletop mode (flat surface, no walls or ceiling):
    experiment=tabletop
    manipuland_agent=tabletop_manipuland_agent
    furniture_agent=tabletop_furniture_agent

Provider (set via env vars or .env file):
    PROVIDER_BASE_URL, PROVIDER_MODEL, PROVIDER_API_KEY

Provider preset:
    provider=qwen     Qwen3 on vLLM (EPFL RCP, default)
    provider=gemma    Gemma 4 on vLLM
    provider=neutral  OpenAI or Azure

Outputs land in outputs/YYYY-MM-DD/<name>_HH-MM-SS/.
"""

import logging
import os
import sys
import time

from datetime import timedelta
from pathlib import Path

import hydra

from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict

# isort: off
# Need to import bpy first to avoid potential symbol loading issues.
import bpy  # noqa: F401

# isort: on

from scenesmith.provider import load_project_dotenv
from scenesmith.utils.logging import FileLoggingContext
from scenesmith.utils.omegaconf import register_resolvers
from scenesmith.utils.print_utils import cyan

console_logger = logging.getLogger(__name__)

# Ensure .env values are present before Hydra resolves `${oc.env:...}` fields.
load_project_dotenv()


def run_local(cfg: DictConfig):
    # Delay some imports in case they are not needed in non-local envs for submission.
    from scenesmith.experiment_tracking import init_tracking_from_cfg
    from scenesmith.experiments import build_experiment

    # Prime the tracking flag cache before any agent / VLM call. Reads
    # experiment.tracking.enabled / .agents_sdk_wrapping from cfg.
    init_tracking_from_cfg(cfg)

    start_time = time.time()

    # Get yaml names.
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    cfg_choice = OmegaConf.to_container(hydra_cfg.runtime.choices)

    # Inject _name into each agent config group so build_*_agent() can dispatch by name.
    agent_groups = [
        "floor_plan_agent",
        "furniture_agent",
        "manipuland_agent",
        "wall_agent",
        "ceiling_agent",
        "task_analysis_agent",
        "reward_spec_agent",
        "morphology_task_agent",
    ]
    with open_dict(cfg):
        if cfg_choice["experiment"] is not None:
            cfg.experiment._name = cfg_choice["experiment"]
        for group in agent_groups:
            if cfg_choice.get(group) is not None and hasattr(cfg, group):
                getattr(cfg, group)._name = cfg_choice[group]

    # Set up the output directory.
    output_dir = Path(hydra_cfg.runtime.output_dir)
    with open_dict(cfg):
        cfg.experiment.output_dir = output_dir

    # Set up experiment-level logging to file while preserving stdout.
    experiment_log_path = output_dir / "experiment.log"
    experiment_log_path.parent.mkdir(parents=True, exist_ok=True)

    with FileLoggingContext(log_file_path=experiment_log_path, suppress_stdout=False):
        console_logger.info(f"Outputs will be saved to: {output_dir}")
        print(cyan(f"Outputs will be saved to:"), output_dir)

        if cfg.experiment.get("write_latest_run_symlink", False):
            (output_dir.parents[1] / "latest-run").unlink(missing_ok=True)
            (output_dir.parents[1] / "latest-run").symlink_to(
                output_dir, target_is_directory=True
            )
        else:
            console_logger.info("Skipping latest-run symlink (disabled by config)")

        # Log and save resolved configuration.
        resolved_config_yaml = OmegaConf.to_yaml(cfg)
        console_logger.info("Resolved configuration:\n" + resolved_config_yaml)
        print(cyan("Resolved configuration:"))
        print(resolved_config_yaml)

        # Save config to output directory for reproducibility.
        config_file = output_dir / "resolved_config.yaml"
        with open(config_file, "w") as f:
            f.write(resolved_config_yaml)
        console_logger.info(f"Saved resolved config to: {config_file}")
        print(cyan(f"Saved resolved config to: {config_file}"))

        # Launch experiment.
        console_logger.info("Starting experiment execution")
        experiment = build_experiment(cfg=cfg)
        for task in cfg.experiment.tasks:
            console_logger.info(f"Executing task: {task}")
            experiment.exec_task(task)
            console_logger.info(f"Completed task: {task}")

        console_logger.info(
            "Experiment execution completed in "
            f"{timedelta(seconds=time.time() - start_time)}"
        )


def _hydra_config_path() -> str:
    """Directory passed to Hydra as ``config_path``.

    Default: ``configurations`` (relative to the process working directory).

    Preferred: pass ``--config-path <dir>`` on the command line (path may be absolute
    or cwd-relative). This is a thin wrapper that strips that flag from ``sys.argv``
    before Hydra parses overrides.

    Back-compat: ``TASKSMITH_CONFIG_PATH`` env var is still supported, but should be
    considered deprecated in favor of the explicit CLI flag.
    """
    raw: str | None = None

    # CLI flag: `python main.py --config-path <dir> ...`
    # Must be removed from sys.argv before Hydra processes arguments.
    flag_names = ("--config-path", "--config_path")
    for flag in flag_names:
        if flag in sys.argv:
            idx = sys.argv.index(flag)
            if idx + 1 >= len(sys.argv):
                raise ValueError(f"{flag} requires a path argument")
            raw = sys.argv[idx + 1]
            del sys.argv[idx : idx + 2]
            break

    # Back-compat env var.
    if raw is None:
        raw = os.environ.get("TASKSMITH_CONFIG_PATH")

    if not raw:
        return "configurations"

    path = Path(raw)
    return str(path if path.is_absolute() else (Path.cwd() / path).resolve())


@hydra.main(version_base=None, config_path=_hydra_config_path(), config_name="config")
def run(cfg: DictConfig):
    if OmegaConf.is_missing(cfg, "name"):
        raise ValueError(
            "Must specify a name for the run with command line argument '+name=[name]'"
        )

    # Configure logging level from LOGLEVEL environment variable.
    log_level = os.environ.get("LOGLEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    from scenesmith.provider_validation import initialize_provider_runtime

    register_resolvers()
    OmegaConf.resolve(cfg)
    initialize_provider_runtime(cfg)

    # Configure the agents SDK to use the provider-configured async client.
    from agents import set_default_openai_api, set_default_openai_client
    from agents.tracing import set_tracing_disabled, set_tracing_export_api_key

    from scenesmith.provider import make_async_client

    set_default_openai_client(make_async_client())
    set_default_openai_api("chat_completions")

    # Disable tracing by default; enable only if a dedicated key is provided.
    tracing_api_key = os.environ.get("OPENAI_TRACING_KEY")
    if tracing_api_key:
        set_tracing_export_api_key(tracing_api_key)
        console_logger.info("Using separate API key for tracing exports")
    else:
        set_tracing_disabled(True)

    run_local(cfg)


if __name__ == "__main__":
    run()
