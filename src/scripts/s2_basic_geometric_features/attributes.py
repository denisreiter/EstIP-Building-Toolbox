"""Attribute calculations for building footprints."""

# Copyright (c) 2026 Denis Reiter
# Licensed under the EUPL-1.2. See the LICENSE file in the project root for details.

import geopandas as gpd


def copy_basic_attributes(
    gdf: gpd.GeoDataFrame,
    field_height: str | None = None,
    field_volume: str | None = None,
) -> gpd.GeoDataFrame:
    """Copy source attribute fields into standardized short column names.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input footprints.
    field_area : str
        Name of the source column holding building footprint area.
        Copied into column ``A``.
    field_cir : str
        Name of the source column holding building circumference.
        Copied into column ``C``.
    field_height : str or None, default None
        Name of the source column holding building height. Copied into
        column ``H``. Pass ``None`` if this field was not supplied.
    field_volume : str or None, default None
        Name of the source column holding building volume. Copied into
        column ``V``. Pass ``None`` if this field was not supplied.

    Returns
    -------
    geopandas.GeoDataFrame
        The input GeoDataFrame with columns ``A``, ``C``, and (if
        supplied) ``H`` and ``V`` added.
    """

    gdf["A"] = gdf.geometry.area
    gdf["C"] = gdf.geometry.length

    if field_height is not None:
        gdf["H"] = gdf[field_height]
    if field_volume is not None:
        gdf["V"] = gdf[field_volume]

    return gdf


def add_hull_metrics(
    gdf: gpd.GeoDataFrame,
    field_wallarea: str | None,
    field_roofarea: str | None,
    field_volume: str | None,
) -> gpd.GeoDataFrame:
    """Add total hull area and the hull-to-volume ratio.

    Hull area is the sum of footprint area, wall area, and roof area 
    (the total exterior surface area of the building envelope). The
    hull-to-volume ratio relates that surface area to the building's
    volume.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input footprints.
    field_area : str
        Name of the source column holding building footprint area.
    field_wallarea : str or None
        Name of the source column holding total wall area. Pass ``None``
        if this field was not supplied.
    field_roofarea : str or None
        Name of the source column holding total roof area. Pass ``None``
        if this field was not supplied.
    field_volume : str or None
        Name of the source column holding building volume. Pass ``None``
        if this field was not supplied.

    Returns
    -------
    geopandas.GeoDataFrame
        The input GeoDataFrame with columns ``hull_area`` and
        ``RATIO_HLL_V`` added, if and only if `field_wallarea`,
        `field_roofarea`, and `field_volume` were all supplied. Otherwise
        the GeoDataFrame is returned unchanged.
    """
    if field_wallarea is None or field_roofarea is None or field_volume is None:
            print("  Skipping hull-to-volume ratio: wall area, roof area, and volume fields are all required.")
            return gdf

    gdf["hull_area"] = gdf.geometry.area + gdf[field_wallarea] + gdf[field_roofarea]
    gdf["RATIO_HLL_V"] = gdf["hull_area"] / gdf[field_volume]

    return gdf