"""Geometric feature calculations for building footprints."""

# Copyright (c) 2026 Denis Reiter
# Licensed under the EUPL-1.2. See the LICENSE file in the project root for details.

import math

import geopandas as gpd
import numpy as np
from shapely.geometry.base import BaseGeometry


def add_shape_index(
        gdf: gpd.GeoDataFrame, 
        field_name: str = "SHPX_2D"
    ) -> gpd.GeoDataFrame:
    """Add a 2D shape index (circularity) column.

    The shape index is ``4 * pi * area / perimeter**2``. A value of 1.0
    corresponds to a perfect circle; values approach 0 for elongated or
    highly irregular shapes.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input footprints. Must have a valid geometry column.
    field_name : str, default "SHPX_2D"
        Name of the column to create.

    Returns
    -------
    geopandas.GeoDataFrame
        The input GeoDataFrame with `field_name` added.
    """
    gdf[field_name] = (4 * math.pi * gdf.geometry.area) / (gdf.geometry.length ** 2)
    return gdf


def add_perimeter_area_ratio(
        gdf: gpd.GeoDataFrame, 
        field_name: str = "RATIO_C_A"
    ) -> gpd.GeoDataFrame:
    """Add a perimeter-to-area ratio column.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input footprints.
    field_name : str, default "RATIO_C_A"
        Name of the column to create.

    Returns
    -------
    geopandas.GeoDataFrame
        The input GeoDataFrame with `field_name` added.
    """
    gdf[field_name] = gdf.geometry.length / gdf.geometry.area
    return gdf


def _mrr_length_and_width(
        geometry: BaseGeometry
    ) -> tuple[float, float]:
    """Return the (length, width) of a geometry's minimum rotated rectangle.

    Parameters
    ----------
    geometry : shapely.geometry.base.BaseGeometry
        A single polygon or multipolygon geometry.

    Returns
    -------
    tuple of float
        ``(length, width)`` where length is the longer side and width is
        the shorter side of the rectangle. Returns ``(nan, nan)`` if the
        geometry is empty or invalid.
    """
    rectangle = geometry.minimum_rotated_rectangle
    if rectangle.geom_type != "Polygon" or rectangle.is_empty:
        return (np.nan, np.nan)
 
    coords = list(rectangle.exterior.coords)
    if len(coords) < 4:
        return (np.nan, np.nan)
 
    edge_1 = math.dist(coords[0], coords[1])
    edge_2 = math.dist(coords[1], coords[2])
    return (max(edge_1, edge_2), min(edge_1, edge_2))


def add_minimum_rotated_rectangle_metrics(
        gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
    """Add minimum rotated rectangle (MRR) metrics.

    Adds three columns:

    - ``mrr_area``: area of each geometry's minimum rotated rectangle
    - ``RATIO_LW_MRR``: MRR length divided by MRR width
    - ``RATIO_MRR_A``: MRR area divided by the original geometry's area

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input footprints.

    Returns
    -------
    geopandas.GeoDataFrame
        The input GeoDataFrame with the three columns above added.
    """
    mrr_geometry = gdf.geometry.minimum_rotated_rectangle()
    gdf["mrr_area"] = mrr_geometry.area

    length_width = gdf.geometry.apply(_mrr_length_and_width)
    gdf["RATIO_LW_MRR"] = [lw[0] / lw[1] for lw in length_width]
    gdf["RATIO_MRR_A"] = gdf["mrr_area"] / gdf.geometry.area

    return gdf


def _count_vertices(
        geometry: BaseGeometry
    ) -> int:
    """Count the number of unique vertices in a polygon or multipolygon.

    Parameters
    ----------
    geometry : shapely.geometry.base.BaseGeometry
        A single polygon or multipolygon geometry.

    Returns
    -------
    int
        Total vertex count across all parts, not counting the repeated
        closing coordinate of each ring.
    """
    if geometry.geom_type == "MultiPolygon":
        parts = geometry.geoms
    else:
        parts = [geometry]

    total = 0
    for part in parts:
        # -1 because shapely repeats the first coordinate at the end of a ring to close it.
        total += len(part.exterior.coords) - 1
    return total


def add_node_count(
        gdf: gpd.GeoDataFrame, 
        field_name: str = "CNT_NDS"
    ) -> gpd.GeoDataFrame:
    """Add a column counting the number of vertices per geometry.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input footprints.
    field_name : str, default "CNT_NDS"
        Name of the column to create.

    Returns
    -------
    geopandas.GeoDataFrame
        The input GeoDataFrame with `field_name` added.
    """
    gdf[field_name] = gdf.geometry.apply(_count_vertices)
    return gdf


def _count_holes(
        geometry: BaseGeometry
    ) -> int:
    """Count the number of interior rings (holes) in a geometry.

    Parameters
    ----------
    geometry : shapely.geometry.base.BaseGeometry
        A single polygon or multipolygon geometry.

    Returns
    -------
    int
        Total number of interior rings across all parts.
    """
    if geometry.geom_type == "MultiPolygon":
        parts = geometry.geoms
    else:
        parts = [geometry]

    return sum(len(part.interiors) for part in parts)


def add_hole_count(
        gdf: gpd.GeoDataFrame, 
        field_name: str = "CNT_HLS"
    ) -> gpd.GeoDataFrame:
    """Add a column counting the number of holes per geometry.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input footprints.
    field_name : str, default "CNT_HLS"
        Name of the column to create.

    Returns
    -------
    geopandas.GeoDataFrame
        The input GeoDataFrame with `field_name` added.
    """
    gdf[field_name] = gdf.geometry.apply(_count_holes)
    return gdf