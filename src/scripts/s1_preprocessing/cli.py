"""Command-line interface for the building processing toolset."""

# Copyright (c) 2026 Denis Reiter
# Licensed under the EUPL-1.2. See the LICENSE file in the project root for details.

import argparse
import geopandas as gpd
import pandas as pd

from pathlib import Path
from datetime import datetime

from .cleaning import remove_non_buildings
from .geometry import reproject, flag_overlapping_geometries
from .attributes import (
    find_main_building_part,
    join_part_count,
    assign_roof_type,
    assign_residential_function,
)

from scripts.utils import (
    log_checkpoint, 
    parse_args_with_config,
    clean_geometries
)

REQUIRED_FIELDS = [
    "workspace", "lod2_footprints", "lod2_functions_table",
    "bldg_id_field", "bldg_function_field", "output_layer"
]

DEFAULTS = {
    "clean_geometries": True, 
    "overlap_area_threshold": 1.0,
    "output_layer": "s1_preprocessed_buildings"
}

def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the toolbox.

    Returns
    -------
    argparse.ArgumentParser
        The configured parser, ready to have `.parse_args()` called on it.
    """
    parser = argparse.ArgumentParser(
        description="Clean, reproject, and enrich building data."
    )
    parser.add_argument(
        "--config",
        help="Optional YAML file providing default values for any flag below."
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        help="Path to the shared workspace .gpkg. Must already contain the "
             "footprints layer (and parts if used) as input;"
             " this script's output layer is written here too."
    )
    parser.add_argument(
        "--lod2-footprints",
        help="Layer name for LoD2 footprints within the workspace .gpkg."
    )
    parser.add_argument(
        "--lod2-parts",
        help="Optional layer name for LoD2 parts within the workspace .gpkg. "
             "If omitted, all part-dependent processing (main part selection, "
             "part count, part-based roof type) is skipped."
    )
    parser.add_argument(
        "--lod2-functions-table",
        type=Path,
        help="Path to a semicolon-delimited CSV file containing building "
             "function assignments (see docs/package default for expected columns)."
    )
    parser.add_argument(
        "--bldg-id-field",
        help="Building unique identifier."
    )
    parser.add_argument(
        "--bldg-function-field",
        help="Building function field."
    )
    parser.add_argument(
        "--roof-type-field",
        help="Column on the footprints already holding roof type, if "
             "available. Used as the fallback when parts are not supplied."
    )
    parser.add_argument(
        "--part-selector-field",
        help="Required if --lod2-parts is given. Max. value of this field "
             "determines which LoD2 part is the main building part. "
             "Recommended is building volume."
    )
    parser.add_argument(
        "--roof-type-field-parts",
        help="Column on the parts holding roof type, used only if "
             "--roof-type-field is not given and parts are supplied."
    )
    parser.add_argument(
        "--clean-geometries",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Repair and delete invalid geometries before all other processing steps. Default: 'True'."
    )
    parser.add_argument(
        "--overlap-area-threshold",
        type=float,
        help="Minimum overlap area in square metres to flag a pair as a real overlap. Defaults to 1.0."
    )
    parser.add_argument(
        "--output-layer",
        help="Layer name to use inside the output .gpkg. Default: 's1_preprocessed_buildings'."
    )

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    See scripts.utils.parse_args_with_config for the shared merge
    behavior (config file precedence, type handling, validation).
    """
    parser = build_parser()
    args = parse_args_with_config(
        parser,
        argv,
        required_fields=REQUIRED_FIELDS,
        defaults=DEFAULTS,
    )

    has_parts = bool(args.lod2_parts)
    has_selector = bool(args.part_selector_field)
    if has_parts != has_selector:
        parser.error(
            "--lod2-parts and --part-selector-field must be given together, or both omitted."
        )
    if args.roof_type_field_parts and not has_parts:
        parser.error(
            "--roof-type-field-parts requires --lod2-parts to also be given."
        )
    return args


def run_pipeline(args: argparse.Namespace, start_time: datetime) -> gpd.GeoDataFrame:
    """Run the full processing pipeline and return the result."""
    checkpoint = start_time
    has_parts = bool(args.lod2_parts)

    print("# Reading input data...")
    footprints = gpd.read_file(args.workspace, layer=args.lod2_footprints)
    try:
        residential_lookup = pd.read_csv(args.lod2_functions_table, sep=";")
    except UnicodeDecodeError:
        raise ValueError(
            f"Could not read {args.lod2_functions_table} as UTF-8 — please ensure the functions table CSV is UTF-8 encoded."
        )
    parts = gpd.read_file(args.workspace, layer=args.lod2_parts) if has_parts else None
    checkpoint = log_checkpoint("Reading input data", start_time, checkpoint)

    print("# Removing non-buildings...")
    footprints = remove_non_buildings(footprints, args.bldg_function_field)
    if has_parts:
        parts = remove_non_buildings(parts, args.bldg_function_field)
    checkpoint = log_checkpoint("Removing non-buildings", start_time, checkpoint)

    if args.clean_geometries:
        print("# Cleaning geometries...")
        footprints = clean_geometries(footprints, label="LoD2 footprints")
        if has_parts:
            parts = clean_geometries(parts, label="LoD2 parts")
        checkpoint = log_checkpoint("Cleaning geometries", start_time, checkpoint)
    else:
        print("# Cleaning geometries skipped.")

    print("# Reprojecting...")
    footprints = reproject(footprints)
    if has_parts:
        parts = reproject(parts)
    checkpoint = log_checkpoint("Reprojecting", start_time, checkpoint)

    print("Flagging overlapping building footprints...")
    footprints = flag_overlapping_geometries(footprints, overlap_area_threshold=args.overlap_area_threshold)
    n_overlaps = footprints["has_overlap"].sum()
    pct_overlaps = 100 * n_overlaps / len(footprints)
    print(f"  --> {n_overlaps} of {len(footprints)} footprints overlap another building ({pct_overlaps:.5f}%)")
    checkpoint = log_checkpoint("Flagging overlapping footprints", start_time, checkpoint)

    if has_parts:
        print("# Finding main building part...")
        parts = find_main_building_part(parts, args.bldg_id_field, args.part_selector_field)
        checkpoint = log_checkpoint("Finding main building part", start_time, checkpoint)

        print("# Joining part count...")
        footprints = join_part_count(footprints, parts, args.bldg_id_field)
        checkpoint = log_checkpoint("Joining part count", start_time, checkpoint)
    else:
        if "CNT_PRTS" in footprints.columns:
            print("# No parts supplied — CNT_PRTS already present, leaving as-is.")
        else:
            print("# No parts supplied — skipping part count, CNT_PRTS set to null.")
            footprints["CNT_PRTS"] = pd.NA

    print("# Assigning roof type...")
    footprints = assign_roof_type(
        footprints, parts, args.bldg_id_field,
        roof_type_field=args.roof_type_field,
        roof_type_field_parts=args.roof_type_field_parts,
    )
    checkpoint = log_checkpoint("Assigning roof type", start_time, checkpoint)

    print("# Assigning residential function...")
    footprints = assign_residential_function(footprints, residential_lookup, args.bldg_function_field)
    checkpoint = log_checkpoint("Assigning residential function", start_time, checkpoint)

    return footprints


def main(argv: list[str] | None = None) -> None:
    """Entry point: parse arguments, run the pipeline, write the output."""
    print("### Start ###")
    start_time = datetime.now()
    print(f"Start time: {start_time:%Y-%m-%d %H:%M:%S}")

    args = parse_args(argv)

    result = run_pipeline(args, start_time)

    print(f"# Writing output to {args.workspace} (layer: {args.output_layer})...")
    result.to_file(args.workspace, driver="GPKG", layer=args.output_layer)
    end_time = datetime.now()

    print(f"End time: {end_time:%Y-%m-%d %H:%M:%S}")
    print(f"Total script runtime: {end_time - start_time}")
    print("### End ###")


if __name__ == "__main__":
    main()