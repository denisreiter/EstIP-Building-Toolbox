"""Functions for reprojecting geometries."""

# Copyright (c) 2026 Denis Reiter
# Licensed under the EUPL-1.2. See the LICENSE file in the project root for details.

import geopandas as gpd


def reproject(
        gdf: gpd.GeoDataFrame, 
        target_epsg: int = 3035
    ) -> gpd.GeoDataFrame:
    """Reproject a GeoDataFrame to the target coordinate reference system.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame. Must have a defined CRS.
    target_epsg : int, default 3035
        EPSG code to reproject into. Defaults to ETRS89-extended /
        LAEA Europe (3035).

    Returns
    -------
    gpd.GeoDataFrame
        A new GeoDataFrame in the target CRS.

    Raises
    ------
    ValueError
        If the input GeoDataFrame has no CRS defined, since there is
        nothing to reproject from.
    """
    if gdf.crs is None:
        raise ValueError(
            "Input GeoDataFrame has no CRS defined; cannot reproject."
        )
    return gdf.to_crs(epsg=target_epsg)


def flag_overlapping_geometries(
    gdf: gpd.GeoDataFrame,
    overlap_area_threshold: float = 1.0,
    ) -> gpd.GeoDataFrame:
    """Flag polygons that meaningfully overlap another polygon in the layer.

    Uses a spatial self-join to find every pair of intersecting
    polygons, then measures the actual overlap area of each pair.
    Pairs that only touch along a shared edge (zero-area intersection)
    or overlap by a negligible sliver are not flagged.

    Requires `gdf` to be in a projected (metric) CRS, since the
    threshold is an absolute area in square metres.

    This function only detects and flags overlaps; it does not attempt
    to resolve them, since which polygon is "correct" generally cannot
    be determined from geometry alone.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input polygons, with a unique index, in a projected CRS.
    overlap_area_threshold : float, default 1.0
        Minimum overlap area, in square metres, for a pair to be
        considered a real overlap rather than a shared-edge touch or
        a negligible sliver.

    Returns
    -------
    gpd.GeoDataFrame
        A copy of `gdf` with a new ``has_overlap`` boolean column.

    Raises
    ------
    ValueError
        If `gdf`'s index is not unique, since the self-join relies on
        index-based lookups that would otherwise silently return
        wrong-length results.
    """
    if not gdf.index.is_unique:
        raise ValueError(
            "flag_overlapping_geometries requires a unique index on gdf."
        )

    gdf = gdf.copy()
    geometry_only = gdf[["geometry"]]

    pairs = gpd.sjoin(geometry_only, geometry_only, predicate="intersects")
    pairs = pairs[pairs.index != pairs["index_right"]]

    if pairs.empty:
        gdf["has_overlap"] = False
        return gdf

    geom_a = gdf.geometry.loc[pairs.index]
    geom_b = gdf.geometry.loc[pairs["index_right"]]
    overlap_area = geom_a.intersection(geom_b, align=False).area.to_numpy()

    real_overlap_index = pairs.index[overlap_area > overlap_area_threshold]
    gdf["has_overlap"] = gdf.index.isin(real_overlap_index)
    return gdf