"""Tests for lod2_toolbox.cleaning"""

import geopandas as gpd
from shapely.geometry import Point, Polygon

from scripts.s1_preprocessing.cleaning import remove_non_buildings

from scripts.utils import (
    _repair_geometries, 
    _remove_null_or_empty_polygons
)


def test_remove_non_buildings_keeps_only_function_code_3():
    # Arrange: a tiny synthetic GeoDataFrame with mixed function codes.
    # Geometries are irrelevant here, so we use simple points as placeholders.
    gdf = gpd.GeoDataFrame(
        {
            "function": ["31001", "31001", "42000", "51000"],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3)],
        }
    )

    # Act
    result = remove_non_buildings(gdf, function_field="function")

    # Assert: only the two rows starting with '3' should remain
    assert len(result) == 2
    assert set(result["function"]) == {"31001"}


def test_repair_geometries_drops_none_geometry():
    gdf = gpd.GeoDataFrame(
        {"id": [1, 2], "geometry": [Polygon([(0, 0), (1, 0), (1, 1)]), None]}
    )
    result = _repair_geometries(gdf)
    assert len(result) == 1


def test_remove_null_or_empty_polygons_drops_zero_area():
    valid_poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    zero_area_poly = Polygon([(0, 0), (1, 0), (2, 0)])  # collinear points, no area
    gdf = gpd.GeoDataFrame(
        {"id": [1, 2, 3], "geometry": [valid_poly, zero_area_poly, None]}
    )
    result = _remove_null_or_empty_polygons(gdf)
    assert len(result) == 1
    assert result.iloc[0]["id"] == 1


def test_remove_null_or_empty_polygons_skips_non_polygon_input():
    """Point/line geometries aren't polygonal, so the function should
    return them unchanged rather than raising or attempting to filter
    by area."""
    # Arrange
    from shapely.geometry import Point
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)], crs="EPSG:3035")

    # Act
    result = _remove_null_or_empty_polygons(gdf)

    # Assert
    assert len(result) == 1
    assert result.iloc[0]["id"] == 1