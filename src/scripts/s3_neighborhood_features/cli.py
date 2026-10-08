"""Command-line interface for the building processing toolset."""

# Copyright (c) 2026 Denis Reiter
# Licensed under the EUPL-1.2. See the LICENSE file in the project root for details.

import argparse
import geopandas as gpd

from datetime import datetime
from pathlib import Path

from scripts.s3_neighborhood_features.direct_neighbors import (
    calc_nearest_neighbor_distance,
    calc_shared_neighbors,
    calc_shared_wall_ratio,
)
from scripts.s3_neighborhood_features.radius_analysis import (
    calc_coverage_within_radius,
    calc_neighbors_within_radius,
)

from scripts.utils import log_checkpoint, parse_args_with_config

REQUIRED_FIELDS = [
    "workspace",
    "input_layer", 
    "output_layer"
]

DEFAULTS = {
    "radius": 50.0, 
    "precision": 0.001, 
    "skip_coverage": True,
    "output_layer": "s3_neighborhood_features"
}

def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the toolbox.

    Returns
    -------
    argparse.ArgumentParser
        The configured parser, ready to have `.parse_args()` called on it.
    """
    parser = argparse.ArgumentParser(
        description=("Calculate direct-neighbor and radius-based neighborhood features for building footprints.")
    )
    parser.add_argument(
        "--config", 
        help="Optional YAML file providing default values for any flag below."
    )
    parser.add_argument(
        "--workspace",
        type=Path, 
        help="Path to input & output .gpkg"
    )
    parser.add_argument(
        "--input-layer", 
        help="Feature class name for preprocessed building footprints."
    )
    parser.add_argument(
        "--radius",
        type=float,
        help="Search radius in meters for radius-based neighborhood analysis. Default: 50.",
    )
    parser.add_argument(
        "--precision",
        type=float,
        help="Coordinate snapping precision in meters for direct-neighbor adjacency. Default: 0.001.",
    )
    parser.add_argument(
        "--skip-coverage",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip the built-up coverage calculation. Warning: enabling this does significantly increase "
        "calculation time and resource demand. Use only for datasets < 1,000,000 rows. Default: True.",
    )
    parser.add_argument(
        "--output-layer", 
        help="Layer name to use inside the output .gpkg. Default: 's3_neighborhood_features'."
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

    print("# Calculating nearest-neighbor distance...")
    footprints = calc_nearest_neighbor_distance(footprints)
    checkpoint = log_checkpoint("Calculating nearest-neighbor distance", start_time, checkpoint)

    print("# Calculating shared neighbors...")
    footprints = calc_shared_neighbors(footprints, precision=args.precision)
    checkpoint = log_checkpoint("Calculating shared neighbors", start_time, checkpoint)

    print("# Calculating shared-wall ratio...")
    footprints = calc_shared_wall_ratio(footprints)
    checkpoint = log_checkpoint("Calculating shared-wall ratio", start_time, checkpoint)

    print("# Calculating neighbors within radius...")
    footprints = calc_neighbors_within_radius(footprints, radius=args.radius)
    checkpoint = log_checkpoint("Calculating neighbors within radius", start_time, checkpoint)

    if args.skip_coverage:
        print("# Skipping coverage calculation.")
    else:
        print("# Calculating coverage within radius...")
        footprints = calc_coverage_within_radius(footprints, radius=args.radius)
        checkpoint = log_checkpoint("Calculating coverage within radius", start_time, checkpoint)

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