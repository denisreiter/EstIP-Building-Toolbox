"""Tests for address point feature calculations."""

import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon

from scripts.s4_context_features.address_points import (
    calc_address_count,
    _format_address_ids,
)

CRS = "EPSG:3035"
ID_FIELD = "bldg_id"
ADDRESS_ID_FIELD = "PNT_ID"


def _make_square_building(bldg_id, x_min=0, y_min=0, size=10):
    square = Polygon([
        (x_min, y_min),
        (x_min + size, y_min),
        (x_min + size, y_min + size),
        (x_min, y_min + size),
    ])
    return {ID_FIELD: bldg_id, "geometry": square}


def test_calc_address_count_single_point_inside():
    # Arrange: one building, one address point inside it
    buildings = gpd.GeoDataFrame([_make_square_building(1)], crs=CRS)
    addresses = gpd.GeoDataFrame(
        {ADDRESS_ID_FIELD: ["ADDR1"], "geometry": [Point(5, 5)]}, crs=CRS
    )

    # Act
    result = calc_address_count(buildings, addresses, ID_FIELD, ADDRESS_ID_FIELD)

    # Assert
    assert result.loc[0, "CNT_ADD"] == 1
    assert result.loc[0, "add_id"] == "ADDR1"


def test_calc_address_count_no_points():
    # Arrange: building with no address points anywhere nearby
    buildings = gpd.GeoDataFrame([_make_square_building(1)], crs=CRS)
    addresses = gpd.GeoDataFrame(
        {ADDRESS_ID_FIELD: [], "geometry": []}, crs=CRS
    )

    # Act
    result = calc_address_count(buildings, addresses, ID_FIELD, ADDRESS_ID_FIELD)

    # Assert
    assert result.loc[0, "CNT_ADD"] == 0
    assert result.loc[0, "add_id"] is None


def test_calc_address_count_multiple_points_inside():
    # Arrange: one building, two address points inside it
    buildings = gpd.GeoDataFrame([_make_square_building(1)], crs=CRS)
    addresses = gpd.GeoDataFrame(
        {
            ADDRESS_ID_FIELD: ["ADDR1", "ADDR2"],
            "geometry": [Point(3, 3), Point(7, 7)],
        },
        crs=CRS,
    )

    # Act
    result = calc_address_count(buildings, addresses, ID_FIELD, ADDRESS_ID_FIELD)

    # Assert: count is correct and both IDs appear, order aside
    assert result.loc[0, "CNT_ADD"] == 2
    assert set(result.loc[0, "add_id"].split(", ")) == {"ADDR1", "ADDR2"}


def test_calc_address_count_point_on_boundary_counts_directly():
    # Arrange: address point sitting exactly on the building's edge --
    # should be caught by the "intersects" stage, not the tolerance stage
    buildings = gpd.GeoDataFrame([_make_square_building(1)], crs=CRS)
    addresses = gpd.GeoDataFrame(
        {ADDRESS_ID_FIELD: ["ADDR1"], "geometry": [Point(10, 5)]}, crs=CRS
    )

    # Act
    result = calc_address_count(buildings, addresses, ID_FIELD, ADDRESS_ID_FIELD)

    # Assert
    assert result.loc[0, "CNT_ADD"] == 1
    assert result.loc[0, "add_id"] == "ADDR1"


def test_calc_address_count_point_within_tolerance_counts():
    # Arrange: address point 1.0m outside the building -- within the
    # 1.2m tolerance, should be picked up by the nearest-match stage
    buildings = gpd.GeoDataFrame([_make_square_building(1)], crs=CRS)
    addresses = gpd.GeoDataFrame(
        {ADDRESS_ID_FIELD: ["ADDR1"], "geometry": [Point(11, 5)]}, crs=CRS
    )

    # Act
    result = calc_address_count(buildings, addresses, ID_FIELD, ADDRESS_ID_FIELD)

    # Assert
    assert result.loc[0, "CNT_ADD"] == 1
    assert result.loc[0, "add_id"] == "ADDR1"


def test_calc_address_count_point_beyond_tolerance_not_counted():
    # Arrange: address point 2.0m outside the building -- beyond the
    # 1.2m tolerance, should not be matched to anything
    buildings = gpd.GeoDataFrame([_make_square_building(1)], crs=CRS)
    addresses = gpd.GeoDataFrame(
        {ADDRESS_ID_FIELD: ["ADDR1"], "geometry": [Point(12, 5)]}, crs=CRS
    )

    # Act
    result = calc_address_count(buildings, addresses, ID_FIELD, ADDRESS_ID_FIELD)

    # Assert
    assert result.loc[0, "CNT_ADD"] == 0
    assert result.loc[0, "add_id"] is None


@pytest.mark.parametrize(
    "id_list, expected",
    [
        (float("nan"), None),
        (["ADDR1"], "ADDR1"),
        (["ADDR1", "ADDR2"], "ADDR1, ADDR2"),
        (["ADDR1", "ADDR2", "ADDR3"], "ADDR1, ADDR2, ADDR3"),
    ],
)
def test_format_address_ids(id_list, expected):
    # Arrange/Act/Assert combined -- pure function, trivial to invoke
    assert _format_address_ids(id_list) == expected