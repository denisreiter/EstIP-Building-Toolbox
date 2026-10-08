"""Tests for lod2_toolbox.s2_geometric_features.geometry."""

import math

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from scripts.s2_basic_geometric_features.geometry import (
    add_shape_index,
    add_perimeter_area_ratio,
    add_minimum_rotated_rectangle_metrics,
    add_node_count,
    add_hole_count,
    _mrr_length_and_width,
    _count_vertices,
    _count_holes,
)


SQUARE = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
RECTANGLE = Polygon([(0, 0), (20, 0), (20, 5), (0, 5)])
SQUARE_WITH_HOLE = Polygon(
    [(0, 0), (10, 0), (10, 10), (0, 10)],
    holes=[[(3, 3), (6, 3), (6, 6), (3, 6)]],
)
DEGENERATE = Polygon([(0, 0), (5, 0), (10, 0)])  # collinear points, zero area
EMPTY = Polygon()


def make_gdf(geometries):
    """Build a minimal GeoDataFrame from a list of geometries for testing."""
    return gpd.GeoDataFrame(
        {"id": range(len(geometries))}, geometry=geometries, crs="EPSG:3035"
    )


def test_add_shape_index_perfect_square():
    # Arrange
    gdf = make_gdf([SQUARE])

    # Act
    result = add_shape_index(gdf)

    # Assert
    expected = (4 * math.pi * 100) / (40 ** 2)
    assert result["SHPX_2D"].iloc[0] == pytest.approx(expected)


def test_add_shape_index_custom_field_name():
    # Arrange
    gdf = make_gdf([SQUARE])

    # Act
    result = add_shape_index(gdf, field_name="my_shape_index")

    # Assert
    assert "my_shape_index" in result.columns
    assert "SHPX_2D" not in result.columns


def test_add_perimeter_area_ratio():
    # Arrange
    gdf = make_gdf([SQUARE])

    # Act
    result = add_perimeter_area_ratio(gdf)

    # Assert
    assert result["RATIO_C_A"].iloc[0] == pytest.approx(40 / 100)


def test_mrr_length_and_width_normal_rectangle():
    # Arrange / Act
    length, width = _mrr_length_and_width(RECTANGLE)

    # Assert
    assert length == pytest.approx(20)
    assert width == pytest.approx(5)


def test_mrr_length_and_width_degenerate_geometry_returns_nan():
    # Arrange / Act
    length, width = _mrr_length_and_width(DEGENERATE)

    # Assert
    assert math.isnan(length)
    assert math.isnan(width)


def test_mrr_length_and_width_empty_geometry_returns_nan():
    # Arrange / Act
    length, width = _mrr_length_and_width(EMPTY)

    # Assert
    assert math.isnan(length)
    assert math.isnan(width)


def test_add_minimum_rotated_rectangle_metrics_square_with_hole():
    # Arrange
    gdf = make_gdf([SQUARE_WITH_HOLE])

    # Act
    result = add_minimum_rotated_rectangle_metrics(gdf)

    # Assert
    # The hole is 3x3=9, so the polygon's actual area is 100 - 9 = 91.
    # The mrr ignores holes and covers the full outer square (area 100).
    assert result["mrr_area"].iloc[0] == pytest.approx(100)
    assert result["RATIO_LW_MRR"].iloc[0] == pytest.approx(1.0)  # square mrr
    assert result["RATIO_MRR_A"].iloc[0] == pytest.approx(100 / 91)


def test_add_minimum_rotated_rectangle_metrics_plain_rectangle():
    # Arrange
    gdf = make_gdf([RECTANGLE])

    # Act
    result = add_minimum_rotated_rectangle_metrics(gdf)

    # Assert
    assert result["mrr_area"].iloc[0] == pytest.approx(100)
    assert result["RATIO_LW_MRR"].iloc[0] == pytest.approx(20 / 5)
    assert result["RATIO_MRR_A"].iloc[0] == pytest.approx(1.0)


def test_count_vertices_simple_polygon():
    # Arrange / Act
    count = _count_vertices(SQUARE)

    # Assert
    assert count == 4


def test_count_vertices_polygon_with_hole_counts_exterior_only():
    # Arrange / Act
    count = _count_vertices(SQUARE_WITH_HOLE)

    # Assert
    # Only the exterior ring's vertices are counted, not the hole's.
    assert count == 4


def test_add_node_count_custom_field_name():
    # Arrange
    gdf = make_gdf([SQUARE])

    # Act
    result = add_node_count(gdf, field_name="my_node_count")

    # Assert
    assert result["my_node_count"].iloc[0] == 4


def test_count_holes_no_holes():
    # Arrange / Act
    count = _count_holes(SQUARE)

    # Assert
    assert count == 0


def test_count_holes_one_hole():
    # Arrange / Act
    count = _count_holes(SQUARE_WITH_HOLE)

    # Assert
    assert count == 1


def test_add_hole_count():
    # Arrange
    gdf = make_gdf([SQUARE, SQUARE_WITH_HOLE])

    # Act
    result = add_hole_count(gdf)

    # Assert
    assert result["CNT_HLS"].tolist() == [0, 1]