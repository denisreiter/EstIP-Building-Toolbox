"""Functions for calculating neighbor polygon attributes in circular radius."""

# Copyright (c) 2026 Denis Reiter
# Licensed under the EUPL-1.2. See the LICENSE file in the project root for details.

import geopandas as gpd
import numpy as np
import shapely


def calc_neighbors_within_radius(gdf: gpd.GeoDataFrame, radius: float = 50.0) -> gpd.GeoDataFrame:
    """Count and average-distance neighbors within a given search radius.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Building polygons, projected (e.g. EPSG:3035) so that distance
        units are meaningful.
    radius : float
        Search radius in the CRS's linear unit (meters for EPSG:3035).
        Converted to int for use in output column names, e.g. a radius
        of 50 (or 50.0, or "50") produces "CNT_NGHB_R50" and
        "DIST_NGHB_R50". Default = 50.

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of `gdf`, with its index reset to a default 0..n-1 range,
        with two new columns:
        "CNT_NGHB_R{radius}" -- number of neighbors within the radius.
        "DIST_NGHB_R{radius}" -- mean distance to those neighbors.
    """
    radius_int = int(radius)
    result = gdf.reset_index(drop=True)

    # Low-resolution buffer to find candidate neighbors
    buffered_geom = result.geometry.buffer(radius, resolution=4)

    # Query the spatial index directly instead of gpd.sjoin()
    src_pos, nbr_pos = result.sindex.query(buffered_geom, predicate="intersects")

    # Exclude self-matches
    keep = src_pos != nbr_pos
    src_pos, nbr_pos = src_pos[keep], nbr_pos[keep]

    # Exact boundary-to-boundary distance for each candidate pair
    src_geom = result.geometry.values[src_pos]
    nbr_geom = result.geometry.values[nbr_pos]
    near_dist = gpd.GeoSeries(src_geom).distance(gpd.GeoSeries(nbr_geom), align=False).values

    within_radius = near_dist <= radius
    src_pos = src_pos[within_radius]
    near_dist = near_dist[within_radius]

    # Aggregate per source polygon using NumPy
    n = len(result)
    counts = np.bincount(src_pos, minlength=n)
    sums = np.bincount(src_pos, weights=near_dist, minlength=n)

    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / counts, np.nan)

    result[f"CNT_NGHB_R{radius_int}"] = counts
    result[f"DIST_NGHB_R{radius_int}"] = means

    return result


def calc_coverage_within_radius(gdf: gpd.GeoDataFrame, radius: float = 50.0) -> gpd.GeoDataFrame:
    """Calculate the built-up fraction of a buffer around each building.

    For each building, a buffer of the given radius is drawn and the total
    building footprint area falling inside that buffer is measured, then
    divided by the buffer's area. The result is a density measure in the
    range [0, 1]: higher values indicate more built-up neighborhoods. The
    source building's own footprint is included in the built-up area, since
    it is part of its own neighborhood.

    WARNING: This operation is recommended only for a maximum number of 
    1 million building footprints. Otherwise, memory issues may occur 
    and computation time increases significantly.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Building polygons, projected (e.g. EPSG:3035) so that area units
        are meaningful.
    radius : float
        Buffer radius in the CRS's linear unit (meters for EPSG:3035).
        Converted to int for use in the output column name, e.g. a radius
        of 50 produces "CVR_NGHB_R50". Default = 50.

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of `gdf`, with its index reset to a default 0..n-1 range, and
        a new column "CVR_NGHB_R{radius}" holding the built-up fraction.

    Notes
    -----
    Edge effects are not corrected: a building near the boundary of the
    dataset has a buffer extending into un-surveyed space, which lowers its
    coverage value because the denominator still uses the full buffer area.

    If building footprints overlap one another, overlapping regions are 
    counted more than once and coverage can exceed 1.
    """
    radius_int = int(radius)
    result = gdf.reset_index(drop=True)

    buffers = result.geometry.buffer(radius)
    buffer_area = buffers.area.values

    # Candidate pairs: which buildings intersect which buffer
    src_pos, nbr_pos = result.sindex.query(buffers, predicate="intersects")

    # Clip each candidate building to the source buffer and take the area of the overlap
    buf_geom = buffers.values[src_pos]
    bld_geom = result.geometry.values[nbr_pos]
    clipped_area = shapely.area(shapely.intersection(buf_geom, bld_geom))

    # Sum the clipped built-up area per source building.
    n = len(result)
    built_up = np.bincount(src_pos, weights=clipped_area, minlength=n)

    # Relative coverage = built-up area within buffer / buffer area
    result[f"CVR_NGHB_R{radius_int}"] = built_up / buffer_area

    return result