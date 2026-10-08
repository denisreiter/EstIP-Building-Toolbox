"""Functions for calculating street proximity attributes."""

# Copyright (c) 2026 Denis Reiter
# Licensed under the EUPL-1.2. See the LICENSE file in the project root for details.

import geopandas as gpd


def calc_distance_to_street(buildings, streets):
    """
    Calculate the planar distance from each building to its nearest street.

    Parameters
    ----------
    buildings : geopandas.GeoDataFrame
        Building footprints, already in EPSG:3035 (enforced upstream in
        s1_preprocessing).
    streets : geopandas.GeoDataFrame
        Street centerlines used as the reference layer. Reprojected to
        EPSG:3035 if not already in that CRS.

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of `buildings` with a new "DIST_STRT" column giving the
        planar distance in meters.
    """
    # streets is a new input not covered by s1_preprocessing's CRS
    # handling, so check and reproject it here if needed
    if streets.crs != buildings.crs:
        streets = streets.to_crs(buildings.crs)

    joined = gpd.sjoin_nearest(
        buildings,
        streets[["geometry"]],
        how="left",
        distance_col="DIST_STRT",
    )

    # sjoin_nearest can return more than one row per building if two or
    # more streets are exactly tied for closest. Keep just the first match
    # so the output has the same row count as the input.
    joined = joined[~joined.index.duplicated(keep="first")]

    # drop the join artifact column left behind by sjoin_nearest
    joined = joined.drop(columns=["index_right"])

    return joined