"""Shared utility functions used across the toolbox scripts."""

# Copyright (c) 2026 Denis Reiter
# Licensed under the EUPL-1.2. See the LICENSE file in the project root for details.

import argparse
import geopandas as gpd
import pandas as pd
import warnings
import yaml
from pathlib import Path
from datetime import datetime

# Supress geopandas warning of old .notna() behaviour in _repair_geometries
warnings.filterwarnings(
    "ignore",
    message="GeoSeries.notna\\(\\) previously returned False",
    category=UserWarning,
)


def log_checkpoint(step_name: str, start_time: datetime, last_checkpoint: datetime) -> datetime:
    """Print progress for a completed pipeline step and return the new checkpoint time.

    Intended to be called once after each major step in a script's
    `run_pipeline` function, to track both that step's own duration
    and the cumulative runtime since the script started.

    Parameters
    ----------
    step_name : str
        Human-readable name of the step that just finished.
    start_time : datetime
        When the overall script started (for cumulative runtime).
    last_checkpoint : datetime
        When the previous step finished (for this step's own duration).

    Returns
    -------
    datetime
        The current time, to be passed as `last_checkpoint` for the next call.
    """
    now = datetime.now()
    print(f"  --> Executed in {now - last_checkpoint} (total time elapsed: {now - start_time})")
    return now


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    argv: list[str] | None = None,
    required_fields: list[str] | None = None,
    defaults: dict | None = None,
) -> argparse.Namespace:
    """Parse CLI arguments, applying a config file's values as defaults.

    Shared across all scripts. Any flag also given directly on
    the command line takes priority over the config file's value for that
    same flag.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        A parser built by the calling script's own build_parser(). Must
        include a `--config` flag of type str. Any argument declared with
        type=Path is automatically detected, so config-file values for
        those same fields are cast to pathlib.Path for consistency with
        their CLI-parsed counterparts.
    argv : list of str or None, default None
        Argument list to parse. If None, argparse reads the real command
        line (sys.argv).
    required_fields : list of str, optional
        Argument names that must be resolved (via CLI flag or config file)
        before the pipeline can run. If any remain None after merging,
        parser.error() is called.
    defaults : dict, optional
        Fallback values applied after the CLI/config merge, for arguments
        still None at that point.

    Returns
    -------
    argparse.Namespace
        The fully resolved, validated arguments.

    Raises
    ------
    SystemExit
        Raised by argparse (via parser.error) if the config file contains
        an unrecognized key, or if any required field is still missing
        after merging flags, config file, and defaults.
    """
    required_fields = required_fields or []
    defaults = defaults or {}

    # Discover which argument names were declared with type=Path in the
    # calling script's build_parser(), so config-file values for those same
    # fields can be cast consistently with their CLI-parsed counterparts --
    # without needing a separately maintained path_fields set that could
    # drift out of sync with build_parser() itself.
    path_fields = {action.dest for action in parser._actions if action.type is Path}

    args = parser.parse_args(argv)

    if args.config is not None:
        config_values = yaml.safe_load(Path(args.config).read_text())
        for key, value in config_values.items():
            # Fail loudly on an unrecognized config key
            if not hasattr(args, key):
                parser.error(f"Unrecognized key in config file: '{key}'")
            # Fail loudly if wrong None keyword is used
            if isinstance(value, str) and value.strip().lower() in ("none", "null", "~"):
                            parser.error(
                                f"Config key '{key}' has value {value!r}, which YAML "
                                f"treats as a literal string, not a null. Leave the "
                                f"key blank or use YAML's 'null' instead."
                            )

            current_value = getattr(args, key)
            # `is None`, not an equality/membership check against
            # (None, False): a genuine 0, 0.0, or explicit False is a real
            # value, not "unset", and must never be silently overwritten by
            # the config file.
            if current_value is None:
                if key in path_fields:
                    value = Path(value)
                setattr(args, key, value)

    # Apply hardcoded fallback defaults for anything still unresolved.
    for key, default_value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, default_value)

    missing = [f for f in required_fields if getattr(args, f) is None]
    if missing:
        parser.error(f"Missing required arguments (via flag or --config): {', '.join(missing)}")

    return args


def _repair_geometries(
        gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
    """Fix invalid geometries and drop rows with no geometry at all.

    Rows where repair collapses the geometry into a GeometryCollection
    (a known make_valid() artifact for sufficiently degenerate shapes,
    e.g. a bowtie polygon splitting into a sliver polygon plus a line)
    are dropped and logged, rather than passed downstream where a single
    leftover non-polygon row could cause remove_null_or_empty_polygons to
    treat the entire dataset as non-polygonal and skip cleaning it
    altogether.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame, potentially containing invalid or missing
        geometries.

    Returns
    -------
    gpd.GeoDataFrame
        A new GeoDataFrame with invalid geometries repaired, rows with
        missing (None) geometry removed, and any row that collapsed into
        a GeometryCollection during repair dropped.
    """
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf["geometry"] = gdf.geometry.make_valid()

    collection_mask = gdf.geometry.geom_type == "GeometryCollection"
    dropped = int(collection_mask.sum())
    if dropped:
        print(f"  --> Dropped {dropped} geometries that collapsed into a GeometryCollection during repair.")
    gdf = gdf[~collection_mask].copy()

    return gdf


def _remove_null_or_empty_polygons(
        gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
    """Remove rows with missing, empty, or zero-area polygon geometry.

    Skipped entirely for non-polygon geometry types (points, lines), since
    area is not a meaningful filter for those - the input is returned
    unchanged in that case.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame.

    Returns
    -------
    gpd.GeoDataFrame
        A new GeoDataFrame excluding null, empty, and zero-area rows, if
        the input is polygonal. Otherwise, the input unchanged.
    """
    geom_types = gdf.geom_type.dropna().unique()
    allowed = {"Polygon", "MultiPolygon"}
    if not set(geom_types).issubset(allowed):
        print(f"  --> Skipping null/empty polygon removal: geometry type(s) {sorted(set(geom_types) - allowed)} are not polygonal.")
        return gdf

    has_area = gdf.geometry.notna() & ~gdf.geometry.is_empty & (gdf.geometry.area > 0)
    return gdf[has_area].copy()


def clean_geometries(gdf, label="dataset"):
    """
    Repair invalid geometries and drop null/empty ones from a polygon
    GeoDataFrame. Intended to be called explicitly wherever a script's
    author has decided cleaning is warranted for a given input.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Polygon dataset to clean.
    label : str, default "dataset"
        Human-readable name used in the log message, e.g. "buildings".

    Returns
    -------
    geopandas.GeoDataFrame
        Cleaned copy of `gdf`.
    """
    before = len(gdf)
    gdf = _repair_geometries(gdf)
    gdf = _remove_null_or_empty_polygons(gdf)
    dropped = before - len(gdf)
    if dropped:
        print(f"  --> Cleaned {label}: dropped {dropped} null/empty/unrepairable geometries.")
    return gdf


def assign_by_largest_overlap(left, right, value_fields, predicate="intersects", log_label=None):
    """
    Join attributes from `right` onto `left` by spatial predicate, resolving
    ties (a `left` feature matching multiple `right` features) by keeping
    only the match with the largest area of overlap.

    Uses a left join, so every row of `left` is preserved even where no
    match exists (matching fields are filled with NaN in that case).

    Parameters
    ----------
    left : geopandas.GeoDataFrame
        Features to receive attributes.
    right : geopandas.GeoDataFrame
        Features supplying attributes.
    value_fields : list of str
        Columns in `right` to carry over to `left`.
    predicate : str, default "intersects"
        Spatial predicate passed to geopandas.sjoin.
    log_label : str, optional
        If given, a short plural noun phrase (e.g. "buildings with multiple
        municipality matches") used to print how many rows needed tie-
        breaking. If omitted, no message is printed - use this for cases
        where multi-matches are the expected norm rather than a rare event.

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of `left` with `value_fields` appended.
    """
    right_subset = right[value_fields + ["geometry"]]

    joined = gpd.sjoin(left, right_subset, how="left", predicate=predicate)

    match_counts = joined.index.value_counts()
    duplicated_idx = match_counts[match_counts > 1].index

    single_matches = joined[~joined.index.isin(duplicated_idx)].copy()
    multi_matches = joined[joined.index.isin(duplicated_idx)].copy()

    if not multi_matches.empty:
        if log_label:
            print(f"  --> Resolved {len(duplicated_idx)} {log_label} via largest-overlap-area.")

        right_geometry = right.geometry.rename("_right_geometry")
        multi_matches = multi_matches.join(right_geometry, on="index_right")
        multi_matches["_overlap_area"] = multi_matches.geometry.intersection(multi_matches["_right_geometry"]).area

        # multi_matches has a non-unique index (each duplicated left-hand
        # row keeps its original label across all its candidate matches),
        # so groupby(index).idxmax() -> .loc[label] cannot select a single
        # winning row: .loc[label] returns every row sharing that label,
        # not just the one with the largest area. Sorting by overlap area
        # and dropping duplicates on the index as a column (not as a
        # label) avoids that trap - this mirrors the safe pattern already
        # used in s4_context_features.blocks.assign_blocks.
        multi_matches = multi_matches.rename_axis("_left_index").reset_index()
        multi_matches = multi_matches.sort_values("_overlap_area", ascending=False)
        multi_matches = multi_matches.drop_duplicates(subset="_left_index", keep="first")
        multi_matches = multi_matches.set_index("_left_index")
        multi_matches.index.name = None
        multi_matches = multi_matches.drop(columns=["_right_geometry", "_overlap_area"])

    resolved = pd.concat([single_matches, multi_matches]).sort_index()
    resolved = resolved.drop(columns=["index_right"])

    return resolved