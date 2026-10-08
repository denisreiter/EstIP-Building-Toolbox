"""Command-line interface for the building processing toolset."""

# Copyright (c) 2026 Denis Reiter
# Licensed under the EUPL-1.2. See the LICENSE file in the project root for details.

import argparse
import geopandas as gpd

from datetime import datetime
from pathlib import Path

from scripts.s6_population_mapping.population_mapping import map_population

from scripts.utils import log_checkpoint, parse_args_with_config

REQUIRED_FIELDS = [
    "workspace",
    "input_layer",
    "output_layer",
    "grid",
    "bldg_id_field",
    "grid_id_field",
    "population_field",
]

DEFAULTS = {
    "output_layer": "s6_population_mapping",
    "min_area": 40.0,
    "min_volume": 250.0,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the toolbox.

    Returns
    -------
    argparse.ArgumentParser
        The configured parser, ready to have `.parse_args()` called on it.
    """
    parser = argparse.ArgumentParser(
        description=("Map census grid population to buildings, weighted by building volume.")
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
        help="Feature class name for building footprints, expected to carry Script 5's MAPPED_CELLCODE field."
    )
    parser.add_argument(
        "--bldg-id-field",
        help="Unique building identifier field, used to join mapped population back onto the full building set."
    )
    parser.add_argument(
        "--grid",
        help="Census grid layer."
    )
    parser.add_argument(
        "--grid-id-field",
        help="Cell identifier field in the census grid layer, matched against buildings' MAPPED_CELLCODE."
    )
    parser.add_argument(
        "--population-field",
        help="Population field in the census grid layer."
    )
    parser.add_argument(
        "--min-area",
        type=float,
        help="Minimum footprint area (m^2) for a building to be considered residential. Default: 40.0."
    )
    parser.add_argument(
        "--min-volume",
        type=float,
        help="Minimum volume (m^3) for a building to be considered residential. Default: 250.0."
    )
    parser.add_argument(
        "--output-layer",
        help="Layer name to use inside the output .gpkg. Default: 's6_population_mapping'."
    )

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for s6_population_mapping.

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

    print("# Reading census grid...")
    grid = gpd.read_file(args.workspace, layer=args.grid)
    checkpoint = log_checkpoint("Reading census grid", start_time, checkpoint)

    print("# Mapping population to buildings...")
    footprints = map_population(
        footprints,
        grid,
        field_id=args.bldg_id_field,
        grid_id_field=args.grid_id_field,
        population_field=args.population_field,
        min_area=args.min_area,
        min_volume=args.min_volume,
    )
    checkpoint = log_checkpoint("Mapping population to buildings", start_time, checkpoint)

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