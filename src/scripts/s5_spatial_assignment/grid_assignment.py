"""Assign buildings to grid cells."""

# Copyright (c) 2026 Denis Reiter
# Licensed under the EUPL-1.2. See the LICENSE file in the project root for details.

import geopandas as gpd
import pandas as pd

from scripts.utils import assign_by_largest_overlap


def assign_cellcode_to_addresses(addresses, grid, cellcode_field):
    """
    Assign each address point the code of the grid cell it falls within.

    Address points lying outside every grid cell are dropped. 
    Such points contribute no cell, so any single-address building 
    relying on them is later removed as an outlier.

    Parameters
    ----------
    addresses : geopandas.GeoDataFrame
        Address points. Must carry the address ID field used elsewhere in the
        toolbox to link addresses back to buildings.
    grid : geopandas.GeoDataFrame
        100m census grid cells.
    cellcode_field : str
        Column in `grid` holding the cell code.

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of `addresses` with an added ``cellcode`` column.
    """
    # Only the cell code is needed from the grid; geometry drives the join.
    grid_subset = grid[[cellcode_field, "geometry"]]

    # "within" places each point inside exactly one cell (grid cells tile
    # without overlap, so no duplicate matches are expected). The inner join
    # mirrors KEEP_COMMON: points outside the grid extent are discarded.
    joined = gpd.sjoin(addresses, grid_subset, how="inner", predicate="within")

    joined = joined.rename(columns={cellcode_field: "cellcode"}).drop(
        columns=["index_right"]
    )

    return joined


def assign_cellcode_by_overlap(buildings, grid, cellcode_field):
    """
    Assign each building the code of the grid cell it overlaps most.

    A general-purpose overlap-based assignment: makes no assumption about
    which buildings are passed in. Used both for the zero/multi-address
    subset (as part of assign_cellcodes) and, when no address dataset is
    available, for all buildings directly. Buildings that fall entirely
    outside the grid receive no cell and are removed later as outliers.

    Parameters
    ----------
    buildings : geopandas.GeoDataFrame
        Building footprints to assign.
    grid : geopandas.GeoDataFrame
        100m census grid cells.
    cellcode_field : str
        Column in `grid` holding the cell code.

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of `buildings` with an added ``cellcode`` column.
    """
    resolved = assign_by_largest_overlap(buildings, grid, [cellcode_field], predicate="intersects")
    resolved = resolved.rename(columns={cellcode_field: "cellcode"})

    return resolved


def assign_cellcodes(
    buildings,
    addresses,
    grid,
    cellcode_field,
    id_field,
    cnt_field="CNT_ADD",
    add_id_field="add_id",
):
    """
    Assign each building a grid cell code, routed by its address-point count.

    Routing rule:
      * exactly 1 address -> inherit the cell of that address point
      * 0 or 2+ addresses -> assign by largest footprint overlap

    Buildings that end up with no cell (outside the grid, or whose sole
    address point fell outside the grid) are dropped as outliers.

    Parameters
    ----------
    buildings : geopandas.GeoDataFrame
        Building footprints. Must carry the address-count and address-ID
        fields produced in Script 4.
    addresses : geopandas.GeoDataFrame
        Address points already carrying a ``cellcode`` column
        (see ``assign_cellcode_to_addresses``) and the address ID field.
    grid : geopandas.GeoDataFrame
        100m census grid cells.
    cellcode_field : str
        Column in `grid` holding the cell code.
    id_field : str
        Address ID column in `addresses`, whose values are stored in each
        single-address building's `add_id_field`.
    cnt_field : str, default "CNT_ADD"
        Building column holding the address-point count (from Script 4).
    add_id_field : str, default "add_id"
        Building column holding the linked address ID (from Script 4).

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of `buildings` (outliers removed) with an added
        ``MAPPED_CELLCODE`` column.

    Raises
    ------
    ValueError
        If no single-address building's `add_id_field` matches any address
        ID. This usually means Script 4 and Script 5 were run against
        different address datasets.
    """
    # Route buildings by address count: exactly one address takes the cheap
    # attribute lookup; zero or several fall back to the spatial overlap path.
    single_mask = buildings[cnt_field] == 1
    single_buildings = buildings[single_mask].copy()
    other_buildings = buildings[~single_mask].copy()

    # --- Single-address path -------------------------------------------------
    # The building-to-address link already exists from Script 4 (add_id), so no
    # spatial rejoin is needed - just look the address's cell code up directly.
    # Cast both sides to str so an int-vs-text ID mismatch between datasets
    # can't silently prevent every match.
    cellcode_by_address = addresses.set_index(id_field)["cellcode"]
    cellcode_by_address.index = cellcode_by_address.index.astype(str)
    linked_ids = single_buildings[add_id_field].astype(str)

    # A total miss here almost always means s4 and s5 were given different
    # address datasets, not a genuine spatial outcome - fail loudly rather
    # than silently dropping every single-address building later.
    if len(single_buildings) and not linked_ids.isin(cellcode_by_address.index).any():
        raise ValueError("  --> No single-address building matched an address ID - check that Script 4 and Script 5 used the same address dataset.")

    single_buildings["cellcode"] = linked_ids.map(cellcode_by_address)

    # --- Zero / multi-address path ------------------------------------------
    other_buildings = assign_cellcode_by_overlap(other_buildings, grid, cellcode_field)

    # Recombine the two routes; concat aligns on column name, sort_index
    # restores the original building order.
    combined = pd.concat([single_buildings, other_buildings]).sort_index()

    # Standardize to the toolbox's output field name.
    combined = combined.rename(columns={"cellcode": "MAPPED_CELLCODE"})

    # Drop outliers: buildings that received no cell from either path.
    outlier_mask = combined["MAPPED_CELLCODE"].isna()
    dropped = int(outlier_mask.sum())
    if dropped:
        print(f"  --> Dropped {dropped} buildings with no grid cell (outliers).")

    combined = combined[~outlier_mask].copy()

    return combined