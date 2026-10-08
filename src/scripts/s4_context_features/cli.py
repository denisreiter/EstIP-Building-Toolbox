"""Command-line interface for the building processing toolset."""

# Copyright (c) 2026 Denis Reiter
# Licensed under the EUPL-1.2. See the LICENSE file in the project root for details.

import argparse
import geopandas as gpd

from datetime import datetime
from pathlib import Path

from scripts.s4_context_features.streets import (
    calc_distance_to_street
)
from scripts.s4_context_features.address_points import (
    calc_address_count
)
from scripts.s4_context_features.blocks import (
    assign_blocks,
    calc_block_statistics,
    calc_block_ratios,
    classify_residential_extended
)

from scripts.utils import (
    log_checkpoint, 
    parse_args_with_config,
    clean_geometries
)


REQUIRED_FIELDS = [
    "workspace",
    "input_layer", 
    "output_layer"
]

DEFAULTS = {
    "clean_geometries": False, 
    "output_layer": "s4_context_features"
}


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the toolbox.

    Returns
    -------
    argparse.ArgumentParser
        The configured parser, ready to have `.parse_args()` called on it.
    """
    parser = argparse.ArgumentParser(
        description=("Calculate context-based features for building footprints using ancillary datasets.")
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
        "--bldg-id-field", 
        help="Building unique identifier."
    )
    parser.add_argument(
        "--volume-field", 
        help="Building volume field."
    )
    parser.add_argument(
        "--streets",
        help="Optional streets layer, should be Polyline features."
    )
    parser.add_argument(
        "--address-points",
        help="Optional address points layer, should be Point features."
    )
    parser.add_argument(
        "--address-id-field", 
        help="Address points unique identifier."
    )
    parser.add_argument(
        "--blocks",
        help="Optional blocks layer, should be Polygon features."
    )
    parser.add_argument(
        "--block-id-field",
        help="Blocks unique identifier."
    )
    parser.add_argument(
        "--block-type-field",
        help="Block type field."
    )
    parser.add_argument(
        "--clean-geometries",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Repair and delete invalid geometries before all other processing steps. Default: 'False'."
    )
    parser.add_argument(
        "--output-layer", 
        help="Layer name to use inside the output .gpkg. Default: 's4_context_features'."
    )

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for s4_context_features.

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
    if args.streets:
         streets = gpd.read_file(args.workspace, layer=args.streets)
    if args.address_points:
         address_points = gpd.read_file(args.workspace, layer=args.address_points)
    if args.blocks:
         blocks = gpd.read_file(args.workspace, layer=args.blocks)
    checkpoint = log_checkpoint("Reading input data", start_time, checkpoint)

    if args.clean_geometries:
        print("# Cleaning geometries...")
        if args.streets:
            streets = clean_geometries(streets, label="Streets")
        if args.address_points:
            address_points = clean_geometries(address_points, label="Address points")
        if args.blocks:
            blocks = clean_geometries(blocks, label="Blocks")
        checkpoint = log_checkpoint("Cleaning geometries", start_time, checkpoint)
    else:
        print("# Cleaning geometries skipped.")

    if args.streets:
        print("# Calculating distance to nearest street...")
        footprints = calc_distance_to_street(footprints, streets)
        checkpoint = log_checkpoint("Calculating distance to nearest street", start_time, checkpoint)
    else:
        print("# No streets provided, skipping these features.")

    if args.address_points:
        print("# Counting address points...")
        footprints = calc_address_count(footprints, address_points, id_field=args.bldg_id_field, address_id_field=args.address_id_field)
        checkpoint = log_checkpoint("Counting address points", start_time, checkpoint)
    else:
        print("# No address points provided, skipping these features.")

    if args.blocks:
        print("# Assigning footprints to blocks...")
        footprints = assign_blocks(footprints, blocks, id_field=args.bldg_id_field, block_id_field=args.block_id_field, block_type_field=args.block_type_field)
        checkpoint = log_checkpoint("Assigning footprints to blocks", start_time, checkpoint)

        print("# Calculating block statistics and ratios...")
        footprints = calc_block_statistics(footprints, volume_field=args.volume_field)
        footprints = calc_block_ratios(footprints, volume_field=args.volume_field)
        checkpoint = log_checkpoint("Calculating block statistics and ratios", start_time, checkpoint)

        print("# Updating residential buildings...")
        footprints = classify_residential_extended(footprints)
        checkpoint = log_checkpoint("Updating residential buildings", start_time, checkpoint)
    else:
        print("# No blocks provided, skipping these features.")
    
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