"""Tests for street proximity feature calculations."""

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point

from scripts.s4_context_features.streets import calc_distance_to_street

CRS = "EPSG:3035"


def test_calc_distance_to_street_typical_case():
    # Arrange: one building at the origin, one street 10m to the east
    buildings = gpd.GeoDataFrame(
        {"geometry": [Point(0, 0)]}, crs=CRS
    )
    streets = gpd.GeoDataFrame(
        {"geometry": [LineString([(10, -5), (10, 5)])]}, crs=CRS
    )

    # Act
    result = calc_distance_to_street(buildings, streets)

    # Assert
    assert result.loc[0, "DIST_STRT"] == pytest.approx(10.0)
    assert len(result) == 1


def test_calc_distance_to_street_multiple_buildings():
    # Arrange: two buildings at different distances from the same street
    buildings = gpd.GeoDataFrame(
        {"geometry": [Point(0, 0), Point(20, 0)]}, crs=CRS
    )
    streets = gpd.GeoDataFrame(
        {"geometry": [LineString([(10, -5), (10, 5)])]}, crs=CRS
    )

    # Act
    result = calc_distance_to_street(buildings, streets)

    # Assert
    assert len(result) == 2
    assert result.loc[0, "DIST_STRT"] == pytest.approx(10.0)
    assert result.loc[1, "DIST_STRT"] == pytest.approx(10.0)


def test_calc_distance_to_street_reprojects_mismatched_crs():
    # Arrange: streets provided in a geographic CRS instead of EPSG:3035
    buildings = gpd.GeoDataFrame(
        {"geometry": [Point(0, 0)]}, crs=CRS
    )
    streets_3035 = gpd.GeoDataFrame(
        {"geometry": [LineString([(10, -5), (10, 5)])]}, crs=CRS
    )
    streets_4326 = streets_3035.to_crs("EPSG:4326")

    # Act
    result = calc_distance_to_street(buildings, streets_4326)

    # Assert: reprojection happened internally, distance still correct
    # in the buildings' CRS units (meters)
    assert result.loc[0, "DIST_STRT"] == pytest.approx(10.0)


def test_calc_distance_to_street_tied_streets_returns_one_row_per_building():
    # Arrange: two streets equidistant from a single building -- this
    # is the scenario sjoin_nearest can return duplicate rows for
    buildings = gpd.GeoDataFrame(
        {"geometry": [Point(0, 0)]}, crs=CRS
    )
    streets = gpd.GeoDataFrame(
        {
            "geometry": [
                LineString([(10, -5), (10, 5)]),
                LineString([(-10, -5), (-10, 5)]),
            ]
        },
        crs=CRS,
    )

    # Act
    result = calc_distance_to_street(buildings, streets)

    # Assert: row count matches input, not doubled by the tie
    assert len(result) == 1
    assert result.loc[0, "DIST_STRT"] == pytest.approx(10.0)