"""Functions for calculating block statistics for buildings."""

# Copyright (c) 2026 Denis Reiter
# Licensed under the EUPL-1.2. See the LICENSE file in the project root for details.

import geopandas as gpd
import numpy as np


def assign_blocks(buildings, blocks, id_field, block_id_field, block_type_field, min_overlap_ratio=2/3):
    """
    Assign each building to the block it overlaps most, and attach
    block attributes.

    A building is only assigned to a block if at least
    `min_overlap_ratio` of its own area overlaps that block; buildings
    that merely touch or barely clip a block's boundary are left
    unassigned (blck_id is null).

    Geometry lookups use each GeoDataFrame's positional index rather
    than `id_field`/`block_id_field` values.

    Parameters
    ----------
    buildings : geopandas.GeoDataFrame
        Building footprints, identified by `id_field`.
    blocks : geopandas.GeoDataFrame
        Block polygons. Reprojected to match `buildings` CRS if needed.
    id_field : str
        Name of the building identifier column.
    block_id_field : str
        Name of the block identifier column in `blocks`. Not assumed
        to be unique.
    block_type_field : str
        Name of the block type/category column in `blocks`.
    min_overlap_ratio : float, default 2/3
        Minimum fraction of a building's area that must overlap a
        block for that block to be considered a valid assignment.

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of `buildings` with new "blck_id", "BLCK_TYPE", and
        "blck_a" columns (block ID, block type, and block area of the
        assigned block). Null for buildings with no qualifying overlap.
    """
    if blocks.crs != buildings.crs:
        blocks = blocks.to_crs(buildings.crs)

    blocks = blocks.copy()
    blocks["blck_a"] = blocks.geometry.area

    # find every block each building's footprint intersects at all
    candidates = gpd.sjoin(
        buildings[[id_field, "geometry"]],
        blocks[[block_id_field, block_type_field, "geometry", "blck_a"]],
        how="left", predicate="intersects",
    )

    # buildings with no intersecting block at all have index_right = NaN
    # -- these have no candidate to evaluate, so set them aside before
    # doing any geometry lookups
    has_candidate = candidates["index_right"].notna()
    candidates = candidates[has_candidate].copy()
    candidates["index_right"] = candidates["index_right"].astype(int)

    # look up geometries by row position (index_right / candidates'
    # own index), not by id_field/block_id_field values -- those
    # identifier fields are not guaranteed unique, but positional
    # indices always are
    candidates["block_geometry"] = blocks.geometry.loc[candidates["index_right"]].values
    candidates["building_geometry"] = buildings.geometry.loc[candidates.index].values
    candidates["building_area"] = buildings["A"].loc[candidates.index].values

    candidates["overlap_area"] = candidates["building_geometry"].intersection(
        candidates["block_geometry"]
    ).area
    candidates["overlap_ratio"] = candidates["overlap_area"] / candidates["building_area"]

    candidates = candidates[candidates["overlap_ratio"] >= min_overlap_ratio]

    assignments = (
        candidates.sort_values("overlap_ratio", ascending=False)
        .drop_duplicates(subset=id_field, keep="first")
        .rename(columns={block_id_field: "blck_id", block_type_field: "BLCK_TYPE"})
    )
    assignments = assignments[[id_field, "blck_id", "BLCK_TYPE", "blck_a"]]

    result = buildings.merge(assignments, on=id_field, how="left")
    return result


def calc_block_statistics(buildings, volume_field):
    """
    Calculate per-block summary statistics and attach them to each
    building in that block.

    Parameters
    ----------
    buildings : geopandas.GeoDataFrame
        Building footprints with a "blck_id" column already assigned
        (see `assign_blocks`).
    volume_field : str
        Name of the building volume column.

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of `buildings` with new columns: "CNT_BLCK", "A_SUM_BLCK",
        "A_MEAN_BLCK", "A_STD_BLCK", "V_SUM_BLCK", "V_MEAN_BLCK",
        "V_STD_BLCK", "MEAN_DIST_NGHB_BLCK". Buildings with no assigned
        block ("blck_id" is null) receive null values throughout.
    """
    buildings = buildings.copy()
    buildings["_area"] = buildings["A"]

    stats = buildings.groupby("blck_id").agg(
        CNT_BLCK=("blck_id", "size"),
        A_SUM_BLCK=("_area", "sum"),
        A_MEAN_BLCK=("_area", "mean"),
        A_STD_BLCK=("_area", "std"),
        V_SUM_BLCK=(volume_field, "sum"),
        V_MEAN_BLCK=(volume_field, "mean"),
        V_STD_BLCK=(volume_field, "std"),
        MEAN_DIST_NGHB_BLCK=("DIST_NGHB", "mean"),
    ).reset_index()

    result = buildings.drop(columns=["_area"]).merge(stats, on="blck_id", how="left")
    return result


def calc_block_ratios(buildings, volume_field):
    """
    Calculate ratios of each building's area and volume relative to its
    block's aggregate statistics.

    Parameters
    ----------
    buildings : geopandas.GeoDataFrame
        Building footprints with block statistics already attached
        (see `assign_blocks`, `calc_block_statistics`).
    volume_field : str
        Name of the building volume column.

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of `buildings` with new columns: "RATIO_A_BLCK",
        "RATIO_A_SUM", "RATIO_V_SUM", "RATIO_A_MEAN", "RATIO_V_MEAN".
    """
    buildings = buildings.copy()
    area = buildings["A"]
    volume = buildings[volume_field]

    # treat a zero value (in either the numerator or the denominator)
    # as missing, matching the original's truthy-based guard
    def _safe_ratio(numerator, denominator):
        # only the denominator is guarded: division by zero is genuinely
        # undefined and should become NaN. A zero numerator is a valid
        # result (0), not an error -- if it reflects a zero-area or
        # zero-volume building, that's a data quality issue that should
        # surface upstream in s1/s2, not be masked here as missing.
        denominator = denominator.replace(0, np.nan)
        return numerator / denominator

    buildings["RATIO_A_BLCK"] = _safe_ratio(area, buildings["blck_a"])
    buildings["RATIO_A_SUM"] = _safe_ratio(area, buildings["A_SUM_BLCK"])
    buildings["RATIO_V_SUM"] = _safe_ratio(volume, buildings["V_SUM_BLCK"])
    buildings["RATIO_A_MEAN"] = _safe_ratio(area, buildings["A_MEAN_BLCK"])
    buildings["RATIO_V_MEAN"] = _safe_ratio(volume, buildings["V_MEAN_BLCK"])

    return buildings


def classify_residential_extended(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Extend residential classification using block-level context.

    The initial `is_res` classification (0, 1, 2) is assigned in Script 1
    from building-level attributes alone. This function adds two further
    categories for buildings located in blocks that are themselves
    predominantly residential (FNC_CODE == "310019998"), distinguishing
    them by block type:

    - BLCK_TYPE == "41001" -> is_res = 11
    - BLCK_TYPE == "41006" -> is_res = 22

    All other buildings retain their existing `is_res` value from Script 1.
    `FNC_CODE` and `BLCK_TYPE` are treated as categorical codes and compared
    as strings, since they identify a category rather than a numeric
    quantity, and their stored dtype is not guaranteed to be numeric.
    `BLCK_TYPE` is an optional field (only present if Script 4's block
    statistics were run). If it is missing, this function is skipped and
    `is_res` is returned unchanged.

    Parameters
    ----------
    buildings : gpd.GeoDataFrame
        Buildings with existing `is_res` and `FNC_CODE` fields. `BLCK_TYPE`
        (from Script 4's block statistics) is used if present.

    Returns
    -------
    gpd.GeoDataFrame
        Copy of `buildings` with `is_res` overwritten to include categories
        11 and 22 where applicable, or unchanged if `BLCK_TYPE` is absent.
    """
    if "BLCK_TYPE" not in buildings.columns:
        print("  --> 'BLCK_TYPE' field not found, skipping extended residential classification (is_res left unchanged).")
        return buildings.copy()

    buildings = buildings.copy()

    fnc_code = buildings["FNC_CODE"].astype("string")
    blck_type = buildings["BLCK_TYPE"].astype("string")

    is_residential_block = (fnc_code == "310019998").fillna(False).to_numpy(dtype=bool)
    is_block_type_41001 = (blck_type == "41001").fillna(False).to_numpy(dtype=bool)
    is_block_type_41006 = (blck_type == "41006").fillna(False).to_numpy(dtype=bool)

    conditions = [
        is_residential_block & is_block_type_41001,
        is_residential_block & is_block_type_41006,
    ]
    choices = [11, 22]

    buildings["is_res"] = np.select(conditions, choices, default=buildings["is_res"])

    return buildings