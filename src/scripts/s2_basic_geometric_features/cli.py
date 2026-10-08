"""Command-line interface for the building processing toolset."""

# Copyright (c) 2026 Denis Reiter
# Licensed under the EUPL-1.2. See the LICENSE file in the project root for details.

import argparse
import geopandas as gpd

from pathlib import Path
from datetime import datetime

from .geometry import (
    add_shape_index,
    add_perimeter_area_ratio,
    add_minimum_rotated_rectangle_metrics,
    add_node_count,
    add_hole_count
)
from .attributes import (
    copy_basic_attributes,
    add_hull_metrics
)

from scripts.utils import log_checkpoint, parse_args_with_config

REQUIRED_FIELDS = [
    "workspace",
    "input_layer", 
    "output_layer"
]

DEFAULTS = {
    "output_layer": "s2_basic_geometric_features"
}


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the toolbox.

    Returns
    -------
    argparse.ArgumentParser
        The configured parser, ready to have `.parse_args()` called on it.
    """
    parser = argparse.ArgumentParser(
        description="Calculate basic geometric features using preprocessed building footprints."
    )
    parser.add_argument(
        "--config", 
        help="Optional YAML file providing default values for any flag below."
    )
    parser.add_argument(
        "--workspace",
        type=Path, 
        help="Path to input & output .gpkg."
    )
    parser.add_argument(
        "--input-layer", 
        help="Feature class name for preprocessed building footprints."
    )
    parser.add_argument(
        "--height-field", 
        help="Building height field."
    )
    parser.add_argument(
        "--volume-field", 
        help="Building volume field."
    )
    parser.add_argument(
        "--wall-area-field", 
        help="Wall area field, without ground area and roof."
    )
    parser.add_argument(
        "--roof-area-field", 
        help="Roof area field."
    )
    parser.add_argument(
        "--output-layer", 
        help="Layer name to use inside the output .gpkg. Default: 's2_basic_geometric_features'."
    )

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    See scripts.utils.parse_args_with_config for the shared merge
    behavior (config file precedence, type handling, validation).
    """
    parser = build_parser()
    return parse_args_with_config(
        parser,
        argv,
        required_fields=REQUIRED_FIELDS,
        defaults=DEFAULTS,
    )


def run_pipeline(args: argparse.Namespace, start_time: datetime) -> gpd.GeoDataFrame:
    """Run the full processing pipeline and return the result."""
    checkpoint = start_time

    print("# Reading input data...")
    footprints = gpd.read_file(args.workspace, layer=args.input_layer)
    checkpoint = log_checkpoint("Reading input data", start_time, checkpoint)

    print("# Calculating Shape Index...")
    footprints = add_shape_index(footprints)
    checkpoint = log_checkpoint("Calculating Shape Index", start_time, checkpoint)

    print("# Calculating Perimeter-to-Area-Ratio...")
    footprints = add_perimeter_area_ratio(footprints)
    checkpoint = log_checkpoint("Calculating Perimeter-to-Area-Ratio", start_time, checkpoint)

    print("# Calculating Minimum Rotated Rectangle...")
    footprints = add_minimum_rotated_rectangle_metrics(footprints)
    checkpoint = log_checkpoint("Calculating Minimum Rotated Rectangle", start_time, checkpoint)

    print("# Calculating number of nodes...")
    footprints = add_node_count(footprints)
    checkpoint = log_checkpoint("Calculating number of nodes", start_time, checkpoint)

    print("# Calculating number of holes...")
    footprints = add_hole_count(footprints)
    checkpoint = log_checkpoint("Calculating number of holes", start_time, checkpoint)

    print("# Copying basic attribute fields...")
    footprints = copy_basic_attributes(footprints, args.height_field, args.volume_field)
    checkpoint = log_checkpoint("Copying basic attribute fields", start_time, checkpoint)

    print("# Calculating Hull-to-Volume-Ratio...")
    footprints = add_hull_metrics(footprints, args.wall_area_field, args.roof_area_field, args.volume_field)
    checkpoint = log_checkpoint("Calculating Hull-to-Volume-Ratio", start_time, checkpoint)

    return footprints


def main(argv: list[str] | None = None) -> None:
    """Entry point: parse arguments, run the pipeline, write the output."""
    print("### Start ###")
    start_time = datetime.now()
    print(f"Start time: {start_time:%Y-%m-%d %H:%M:%S}")

    args = parse_args(argv)

    if args.output_layer == args.input_layer:
        raise ValueError(
            "WARNING: --output-layer must differ from --input-layer!"
        )

    result = run_pipeline(args, start_time)

    print(f"# Writing output to {args.workspace} (layer: {args.output_layer})...")
    result.to_file(args.workspace, driver="GPKG", layer=args.output_layer)
    end_time = datetime.now()

    print(f"End time: {end_time:%Y-%m-%d %H:%M:%S}")
    print(f"Total script runtime: {end_time - start_time}")
    print("### End ###")


if __name__ == "__main__":
    main()