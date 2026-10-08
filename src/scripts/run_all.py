"""Orchestrates the full toolbox pipeline: runs s1 through s6 in
order, skipping any step whose output has already been produced."""

# Copyright (c) 2026 Denis Reiter
# Licensed under the EUPL-1.2. See the LICENSE file in the project root for details.

import argparse
import importlib
import geopandas as gpd
import yaml
from datetime import datetime
from pathlib import Path
from types import ModuleType


def load_steps(run_all_config_path: str) -> list[dict]:
    """Read the run_all config and return its ordered list of steps.

    Each step's "config" path is resolved relative to this run_all
    config file's own location, not the current working directory —
    so run_all.py behaves the same regardless of where it's launched
    from.
    """
    config_file = Path(run_all_config_path)
    with open(config_file) as f:
        config = yaml.safe_load(f)

    base_dir = config_file.parent
    for step in config["steps"]:
        step["config"] = str(base_dir / step["config"])

    return config["steps"]


def load_cli_module(step_name: str) -> ModuleType:
    """Import a pipeline script's cli module by its subpackage name.

    Lets run_all.py stay in sync with whatever steps are listed in its
    YAML config, without needing a hardcoded import statement for every
    script.
    """
    module_path = f"scripts.{step_name}.cli"
    return importlib.import_module(module_path)


def output_exists(args) -> bool:
    """Check whether a script's output layer already exists.

    All scripts share one workspace file across the pipeline
    (s1 creates it, s2-s6 add layers to it).
    """
    if not args.workspace.exists():
        return False

    existing_layers = gpd.list_layers(args.workspace)["name"].tolist()
    return args.output_layer in existing_layers


def run_step(step: dict) -> None:
    """Run one script's full pipeline, or skip it if already done.

    Parameters
    ----------
    step : dict
        A single entry from run_all's step list, with keys "name"
        (subpackage folder name, e.g. "s1_preprocessing") and "config"
        (path to that script's own YAML config file).
    """
    name = step["name"]
    print(f"\n=== {name} ===")

    module = load_cli_module(name)
    args = module.parse_args(["--config", step["config"]])

    if output_exists(args):
        print(f"Output already exists for {name} — skipping.")
        return

    step_start = datetime.now()
    result = module.run_pipeline(args, step_start)

    print(f"# Writing output to layer '{args.output_layer}'...")
    result.to_file(args.workspace, driver="GPKG", layer=args.output_layer)

    step_end = datetime.now()
    print(f"{name} runtime: {step_end - step_start}")


def main(argv: list[str] | None = None) -> None:
    """Entry point: parse the run_all config, run every step in order."""
    parser = argparse.ArgumentParser(
        description="Run the full toolbox pipeline (s1 through s6)."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the run_all YAML file listing all pipeline steps.",
    )
    args = parser.parse_args(argv)

    print("### run_all: Start ###")
    start_time = datetime.now()
    print(f"Start time: {start_time:%Y-%m-%d %H:%M:%S}")

    steps = load_steps(args.config)
    for step in steps:
        run_step(step)

    end_time = datetime.now()
    print(f"\nEnd time: {end_time:%Y-%m-%d %H:%M:%S}")
    print(f"Total pipeline runtime: {end_time - start_time}")
    print("### run_all: End ###")


if __name__ == "__main__":
    main()