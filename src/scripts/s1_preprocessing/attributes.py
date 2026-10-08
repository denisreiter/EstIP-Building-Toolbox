"""Functions for computing building attributes from part geometries."""

# Copyright (c) 2026 Denis Reiter
# Licensed under the EUPL-1.2. See the LICENSE file in the project root for details.

import geopandas as gpd
import pandas as pd


def find_main_building_part(
        gdf: gpd.GeoDataFrame, 
        id_field: str, 
        selector_field: str
    ) -> gpd.GeoDataFrame:
    """Flag the main part of each building and count parts per building.

    For each building (grouped by `id_field`), the "main" part is the
    one with the highest `selector_field` value. If multiple parts tie for 
    the maximum, the first one in the input row order is selected.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Building parts, with one row per part.
    id_field : str
        Column identifying which building each part belongs to.
    selector_field : str
        Column whose maximum value determines the main part.

    Returns
    -------
    gpd.GeoDataFrame
        A copy of `gdf` with two new columns:
        - ``is_selected_part`` : 1 for the main part, 0 otherwise.
        - ``CNT_PRTS`` : total number of parts for that building.
    """
    gdf = gdf.copy()

    gdf["CNT_PRTS"] = gdf.groupby(id_field)[id_field].transform("count")

    max_per_building = gdf.groupby(id_field)[selector_field].transform("max")
    is_max = gdf[selector_field] == max_per_building

    gdf["is_selected_part"] = 0
    first_max_index = gdf[is_max].groupby(gdf[id_field]).head(1).index
    gdf.loc[first_max_index, "is_selected_part"] = 1

    return gdf


def join_part_count(
    footprints: gpd.GeoDataFrame, 
    parts: gpd.GeoDataFrame, 
    id_field: str
) -> gpd.GeoDataFrame:
    """Attach each building's part count to its footprint.

    Parameters
    ----------
    footprints : gpd.GeoDataFrame
        Building footprints, one row per building.
    parts : gpd.GeoDataFrame
        Building parts, already processed by `find_main_building_part`
        (must contain ``is_selected_part`` and ``CNT_PRTS``).
    id_field : str
        Column identifying which building each footprint/part belongs to.

    Returns
    -------
    gpd.GeoDataFrame
        A copy of `footprints` with a new ``CNT_PRTS`` column.
    """
    main_parts = parts[parts["is_selected_part"] == 1][[id_field, "CNT_PRTS"]]
    return footprints.merge(main_parts, on=id_field, how="left")


def assign_roof_type(
    footprints: gpd.GeoDataFrame,
    parts: gpd.GeoDataFrame | None,
    id_field: str,
    roof_type_field: str | None = None,
    roof_type_field_parts: str | None = None,
) -> gpd.GeoDataFrame:
    """Resolve a single ``RF_TYPE`` column from footprint or part data.

    Priority order:
    1. `roof_type_field` on footprints, if given — used as-is.
    2. `roof_type_field_parts` on parts, if given — taken from each
       building's main part (see `find_main_building_part`).
    3. Neither given — ``RF_TYPE`` is added as an empty (null) column,
       so downstream scripts can rely on the column always existing.

    Parameters
    ----------
    footprints : gpd.GeoDataFrame
        Building footprints, one row per building.
    parts : gpd.GeoDataFrame or None
        Building parts, already processed by `find_main_building_part`.
        Required if `roof_type_field_parts` is given; otherwise unused.
    id_field : str
        Column identifying which building each footprint/part belongs to.
    roof_type_field : str or None, default None
        Column on `footprints` already holding roof type, if available.
    roof_type_field_parts : str or None, default None
        Column on `parts` holding roof type, used only if
        `roof_type_field` is not given.

    Returns
    -------
    gpd.GeoDataFrame
        A copy of `footprints` with a resolved ``RF_TYPE`` column.
        Null-filled if neither source field was usable.

    Raises
    ------
    ValueError
        If `roof_type_field` is given but not present in `footprints`;
        if `roof_type_field_parts` is given but `parts` is None or the
        field is not present in `parts`; or if `footprints` already has
        an ``RF_TYPE`` column when merging from parts (would otherwise
        collide silently during the merge).
    """
    if roof_type_field is not None:
        if roof_type_field not in footprints.columns:
            raise ValueError(
                f"roof_type_field={roof_type_field!r} not found in "
                f"footprints columns: {footprints.columns.tolist()}"
            )
        return footprints.rename(columns={roof_type_field: "RF_TYPE"})

    if roof_type_field_parts is not None:
        if parts is None:
            raise ValueError(
                "roof_type_field_parts was given, but no parts data was "
                "supplied — cannot derive roof type from parts."
            )
        if roof_type_field_parts not in parts.columns:
            raise ValueError(
                f"roof_type_field_parts={roof_type_field_parts!r} not "
                f"found in parts columns: {parts.columns.tolist()}"
            )
        if "RF_TYPE" in footprints.columns:
            raise ValueError(
                "footprints already has an 'RF_TYPE' column; refusing to "
                "merge ambiguously from parts."
            )
        main_parts = (
            parts[parts["is_selected_part"] == 1][[id_field, roof_type_field_parts]]
            .rename(columns={roof_type_field_parts: "RF_TYPE"})
        )
        return footprints.merge(main_parts, on=id_field, how="left")

    footprints = footprints.copy()
    footprints["RF_TYPE"] = pd.NA
    return footprints

def assign_residential_function(
    footprints: gpd.GeoDataFrame,
    residential_lookup: gpd.GeoDataFrame,
    function_field: str,
) -> gpd.GeoDataFrame:
    """Join residential-use info onto footprints from a lookup table.

    Buildings whose function code has no match in the lookup table
    are assumed non-residential and unclassified: both ``is_res`` and
    ``FNC_CODE`` are set to ``0``.

    Parameters
    ----------
    footprints : gpd.GeoDataFrame
        Building footprints.
    residential_lookup : gpd.GeoDataFrame
        Lookup table with columns ``orig_fkt`` (function code),
        ``FNC_CODE``, and ``is_res``.
    function_field : str
        Column on `footprints` holding the function code to match
        against ``orig_fkt``.

    Returns
    -------
    gpd.GeoDataFrame
        A copy of `footprints` with ``FNC_CODE`` and ``is_res`` columns
        joined in as integers; unmatched rows get both set to ``0``.
    """
    merged = footprints.merge(
        residential_lookup[["orig_fkt", "FNC_CODE", "is_res"]],
        left_on=function_field,
        right_on="orig_fkt",
        how="left",
    )
    merged["is_res"] = merged["is_res"].fillna(0).astype(int)
    merged["FNC_CODE"] = merged["FNC_CODE"].fillna(0).astype(int)
    return merged