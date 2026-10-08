"""Functions for calculating neighbor polygon attributes."""

# Copyright (c) 2026 Denis Reiter
# Licensed under the EUPL-1.2. See the LICENSE file in the project root for details.

import geopandas as gpd
import numpy as np
import shapely


def calc_nearest_neighbor_distance(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Calculate the planar distance from each polygon to its closest neighbor.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Building polygons. Must have a unique index, since sjoin_nearest
        uses the index to match results back to input rows.

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of `gdf` with a new column "DIST_NGHB" containing the
        distance (in the CRS's linear unit, e.g. meters for EPSG:3035)
        to the nearest other polygon.
    """
    # Reset index for the same reason as the radius_analysis functions:
    # sindex operates positionally, so a clean 0..n-1 index is required to
    # correctly attribute results back to the right rows.
    result = gdf.reset_index(drop=True)

    if len(result) < 2:
        # No possible neighbor to measure against.
        result["DIST_NGHB"] = np.nan
        return result

    # exclusive=True excludes a geometry matching itself, equivalent to the
    # self-match filtering done manually in the earlier sjoin_nearest
    # version. return_all=False returns exactly one nearest match per input
    # geometry, breaking ties arbitrarily -- matching the earlier
    # keep="first" behavior in spirit (a single, deterministic-enough pick
    # among equally-close neighbors).
    positions, distances = result.sindex.nearest(
        result.geometry, return_distance=True, exclusive=True, return_all=False
    )
    src_pos = positions[0]

    # Attach distances at their correct positions. np.full pre-fills with
    # NaN so that, in principle, any input row sindex.nearest did not
    # return a match for keeps a well-defined "no neighbor found" value
    # rather than an error -- though in practice every row should get a
    # match whenever len(result) >= 2.
    dist_arr = np.full(len(result), np.nan)
    dist_arr[src_pos] = distances

    result["DIST_NGHB"] = dist_arr

    return result


def calc_shared_neighbors(gdf: gpd.GeoDataFrame, precision: float = 0.001) -> gpd.GeoDataFrame:
    """Count direct neighbors and measure shared border length per polygon.

    Two polygons are considered neighbors if they touch along a boundary or at a point; 
    polygons that overlap by area are excluded entirely.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Building polygons, projected (e.g. EPSG:3035) so that length units
        are meaningful. Must have a unique index.
    precision : float, default 0.001
        Grid size (in the CRS's linear unit) used to snap geometries before
        testing adjacency.

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of `gdf` with two new columns:
        "CNT_NGHB_D" -- number of directly touching neighbors.
        "sum_length" -- total length of shared border with those neighbors.
    """


def calc_shared_neighbors(gdf: gpd.GeoDataFrame, precision: float = 0.001) -> gpd.GeoDataFrame:
    """Count direct neighbors and measure shared border length per polygon.

    Two polygons are considered neighbors if they touch along a 
    boundary or at a point; polygons that overlap by area are excluded entirely.

    Uses the spatial index directly rather than ``gpd.sjoin``, since sjoin
    duplicates every attribute column from gdf into the intermediate
    result for every candidate pair. ``sindex.query()`` returns bare positional index
    pairs instead, so only geometry data is duplicated across pairs, never
    the full attribute table.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Building polygons, projected (e.g. EPSG:3035) so that length units
        are meaningful.
    precision : float, default 0.001
        Grid size (in the CRS's linear unit) used to snap geometries before
        testing adjacency.

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of `gdf`, with its index reset to a default 0..n-1 range, with
        two new columns:
        "CNT_NGHB_D" -- number of directly touching neighbors.
        "sum_length" -- total length of shared border with those neighbors.
    """
    # Reset index for the same reason as the radius_analysis functions:
    # sindex and np.bincount operate positionally, so a clean 0..n-1 index
    # is required to correctly attribute results back to the right rows.
    result = gdf.reset_index(drop=True)

    # Snap geometries to a coordinate grid to collapse near-coincident
    # vertices, same purpose as ArcGIS's XY tolerance. Only a bare geometry
    # array is snapped here, not a full copy of every attribute column.
    snapped_geom = shapely.set_precision(result.geometry.values, grid_size=precision)

    # Candidate pairs: which polygons' snapped geometries touch or overlap.
    # Returns plain positional index arrays, not a full duplicated table.
    src_pos, nbr_pos = gpd.GeoSeries(snapped_geom).sindex.query(
        snapped_geom, predicate="intersects"
    )

    # Drop self-matches (a polygon always intersects itself).
    keep = src_pos != nbr_pos
    src_pos, nbr_pos = src_pos[keep], nbr_pos[keep]

    # For each remaining pair, compute the actual shared geometry between
    # the two snapped polygons. This can be a Point (corner touch), a
    # LineString/MultiLineString (shared edge), or a Polygon.
    shared_geom = shapely.intersection(snapped_geom[src_pos], snapped_geom[nbr_pos])

    # Exclude pairs that overlap by area -- NO_AREA_OVERLAP means such
    # pairs are not neighbors at all.
    is_touch = shapely.area(shared_geom) == 0
    src_pos = src_pos[is_touch]
    shared_geom = shared_geom[is_touch]

    # Length of a Point is 0, so corner-touching pairs correctly contribute
    # 0 to shared length while still counting as a neighbor.
    shared_length = shapely.length(shared_geom)

    # Aggregate per source polygon using np.bincount, matching the pattern
    # used in radius_analysis.py for the same reason: src_pos is already a
    # small, dense, non-negative integer array, making bincount a fast fit.
    n = len(result)
    counts = np.bincount(src_pos, minlength=n)
    sums = np.bincount(src_pos, weights=shared_length, minlength=n)

    result["CNT_NGHB_D"] = counts
    result["sum_length"] = sums

    return result


def calc_shared_wall_ratio(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Calculate the ratio of shared border length to a polygon's circumference.

    RATIO_SW expresses what fraction of a building's outline is shared
    with direct neighbors -- e.g. 0.3 means 30% of the perimeter borders
    another building.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Building polygons with columns "sum_length" (total shared border
        length, from calc_shared_neighbors) and "C" (circumference, from
        script 2).

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of `gdf` with a new column "RATIO_SW".
    """
    result = gdf.copy()
    result["RATIO_SW"] = (result["sum_length"] / result["C"]).where(result["C"] != 0)

    return result