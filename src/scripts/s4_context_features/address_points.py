"""Functions for calculating address point attributes."""

# Copyright (c) 2026 Denis Reiter
# Licensed under the EUPL-1.2. See the LICENSE file in the project root for details.

import geopandas as gpd
import pandas as pd


def calc_address_count(buildings, addresses, id_field, address_id_field):
    """
    Count address points associated with each building, and record
    their identifiers.

    Address points that fall directly within
    a building count toward it. Points that fall outside every building
    are attributed to their single nearest building, but only if within
    1.2 meters.

    In addition to the count, the values of `address_id_field` for all
    matched address points are written to a new "add_id" text field:
    None if a building has no matches, the value itself (unaltered) if
    it has exactly one match, and a comma-separated string of values if
    it has multiple matches.

    Buildings with no associated address points receive a count of 0
    and an "add_id" of None.

    Parameters
    ----------
    buildings : geopandas.GeoDataFrame
        Building footprints, identified by `id_field`.
    addresses : geopandas.GeoDataFrame
        Address points. Reprojected to match `buildings` CRS if needed.
    id_field : str
        Name of the unique building identifier column.
    address_id_field : str
        Name of the column in `addresses` whose values should be
        recorded per building in the new "add_id" field.

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of `buildings` with new "CNT_ADD" and "add_id" columns.
    """
    if addresses.crs != buildings.crs:
        addresses = addresses.to_crs(buildings.crs)

    join_cols = [id_field, "geometry"]

    # stage 1: points that fall directly inside (or on the boundary of)
    # a building
    direct = gpd.sjoin(
        addresses, buildings[join_cols],
        how="inner", predicate="intersects",
    )

    # stage 2: leftover points, matched to their nearest building only
    # if within the 1.2m tolerance
    leftover = addresses.loc[~addresses.index.isin(direct.index)]
    nearby = gpd.sjoin_nearest(
        leftover, buildings[join_cols],
        how="inner", max_distance=1.2,
    )

    # combine both sets of matches
    matched = pd.concat([
        direct[[id_field, address_id_field]],
        nearby[[id_field, address_id_field]],
    ])

    counts = matched.groupby(id_field).size().rename("CNT_ADD")
    id_lists = matched.groupby(id_field)[address_id_field].apply(list)

    result = buildings.merge(counts, on=id_field, how="left")
    result = result.merge(id_lists.rename("add_id"), on=id_field, how="left")

    result["CNT_ADD"] = result["CNT_ADD"].fillna(0).astype(int)
    result["add_id"] = result["add_id"].apply(_format_address_ids)

    return result


def _format_address_ids(id_list):
    """
    Collapse a list of matched address IDs into a single text value.

    Private helper for `calc_address_count`.

    Parameters
    ----------
    id_list : list or float
        List of matched address ID values for one building, or NaN
         if no address points matched.

    Returns
    -------
    str, object, or None
        None if there were no matches, the single value unaltered if
        there was exactly one match, or a comma-separated string of
        values if there were multiple.
    """
    if not isinstance(id_list, list):
        # merge produced NaN for buildings with no matches at all
        return None
    if len(id_list) == 1:
        return id_list[0]
    return ", ".join(str(v) for v in id_list)