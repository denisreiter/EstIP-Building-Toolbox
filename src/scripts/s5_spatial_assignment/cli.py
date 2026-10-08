"""Command-line interface for the building processing toolset."""

# Copyright (c) 2026 Denis Reiter
# Licensed under the EUPL-1.2. See the LICENSE file in the project root for details.

import argparse
import geopandas as gpd

from datetime import datetime
from pathlib import Path

from scripts.s5_spatial_assignment.attribute_joins import (
    assign_municipality_typologies
)
from scripts.s5_spatial_assignment.grid_assignment import (
    assign_cellcode_to_addresses,
    assign_cellcode_by_overlap,
    assign_cellcodes
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
    "add_id_field": "add_id",
    "cnt_field": "CNT_ADD",
    "clean_geometries": False,
    "output_layer": "s5_spatial_assignment"
}


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the toolbox.

    Returns
    -------
    argparse.ArgumentParser
        The configured parser, ready to have `.parse_args()` called on it.
    """
    parser = argparse.ArgumentParser(
        description=("Assign municipality typologies and 100m grid cell codes to building footprints.")
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
        help="Feature class name for building footprints, expected to carry Script 4's CNT_ADD and add_id fields."
    )
    parser.add_argument(
        "--municipalities",
        help="Optional municipality layer, should be Polygon features.",
    )
    parser.add_argument(
        "--state-field",
        help="State code field in the municipality layer."
    )
    parser.add_argument(
        "--region-field",
        help="Region code field in the municipality layer."
    )
    parser.add_argument(
        "--municipality-field",
        help="Municipality code field in the municipality layer."
    )
    parser.add_argument(
        "--typology-field",
        help="Special-type code field in the municipality layer."
    )
    parser.add_argument(
        "--grid",
        help="Optional 100m census grid layer, should be Polygon features."
    )
    parser.add_argument(
        "--grid-id-field",
        help="Cell code field in the grid layer.",
    )
    parser.add_argument(
        "--address-points",
        help="Optional address points layer, should be Point features. Required together with --grid."
    )
    parser.add_argument(
        "--address-id-field",
        help="Unique ID field in the address points layer itself. Required together with --grid and --address-points."
    )
    parser.add_argument(
        "--add-id-field",
        help="Building field holding the linked address ID, set by Script 4. Default: 'add_id'."
    )
    parser.add_argument(
        "--cnt-field",
        help="Building field holding address-point count from Script 4. Default: 'CNT_ADD'."
    )
    parser.add_argument(
        "--clean-geometries",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Repair and delete invalid geometries before all other processing steps. Default: 'False'."
    )
    parser.add_argument(
        "--output-layer",
        help="Layer name to use inside the output .gpkg. Default: 's5_spatial_assignment'."
    )

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for s5_spatial_assignment.

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
    if args.municipalities:
        municipalities = gpd.read_file(args.workspace, layer=args.municipalities)
    if args.grid:
        grid = gpd.read_file(args.workspace, layer=args.grid)
    if args.address_points:
        address_points = gpd.read_file(args.workspace, layer=args.address_points)
    checkpoint = log_checkpoint("Reading input data", start_time, checkpoint)

    if args.clean_geometries:
        print("# Cleaning geometries...")
        if args.municipalities:
            municipalities = clean_geometries(municipalities, label="Municipalities")
        if args.grid:
            grid = clean_geometries(grid, label="Grid")
        if args.address_points:
            address_points = clean_geometries(address_points, label="Address points")
        checkpoint = log_checkpoint("Cleaning geometries", start_time, checkpoint)
    else:
        print("# Cleaning geometries skipped.")

    if args.municipalities:
        print("# Assigning municipality typologies...")
        footprints = assign_municipality_typologies(footprints, municipalities, 
                                                    args.state_field,args.region_field, args.municipality_field, args.typology_field)
        checkpoint = log_checkpoint("Assigning municipality typologies", start_time, checkpoint)
    else:
        print("# No municipality layer provided, skipping these features.")

    if args.grid and args.address_points:
        print("# Assigning grid cells to address points...")
        address_points = assign_cellcode_to_addresses(address_points, grid, args.grid_id_field)
        checkpoint = log_checkpoint("Assigning grid cells to address points", start_time, checkpoint)

        print("# Assigning grid cells to buildings...")
        footprints = assign_cellcodes(
            footprints, address_points, grid, args.grid_id_field, args.address_id_field,
            cnt_field=args.cnt_field, add_id_field=args.add_id_field
        )
        checkpoint = log_checkpoint("Assigning grid cells to buildings", start_time, checkpoint)
    elif args.grid:
        print("# No address points provided, assigning all grid cells by largest overlap...")
        footprints = assign_cellcode_by_overlap(footprints, grid, args.grid_id_field)
        checkpoint = log_checkpoint("Assigning grid cells to buildings (overlap only)", start_time, checkpoint)
    else:
        print("# No grid provided, skipping grid cell assignment.")

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

    if args.municipalities and not (args.state_field or args.region_field or args.municipality_field or args.typology_field):
        raise ValueError(
            "WARNING: at least one of --state-field, --region-field, --municipality-field or "
            "--typology-field must be provided if --municipalities is set!"
        )
    
    if args.grid and not args.grid_id_field:
        raise ValueError(
            "WARNING: --grid-id-field must be provided if --grid is set!"
    )

    if args.address_points and not (args.grid and args.address_id_field):
        raise ValueError(
            "WARNING: --grid and --address-id-field must be provided if --address-points is set!"
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