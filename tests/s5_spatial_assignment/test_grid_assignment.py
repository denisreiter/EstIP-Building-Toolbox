"""Tests for s5_spatial_assignment.grid_assignment."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

from scripts.s5_spatial_assignment.grid_assignment import (
    assign_cellcode_by_overlap,
    assign_cellcode_to_addresses,
    assign_cellcodes,
)

CRS = "EPSG:3035"


@pytest.fixture
def grid():
    """Two adjacent 10x10 grid cells: CELL_A spans x[0,10], CELL_B spans x[10,20]."""
    return gpd.GeoDataFrame(
        {"CELLCODE": ["CELL_A", "CELL_B"]},
        geometry=[
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            Polygon([(10, 0), (20, 0), (20, 10), (10, 10)]),
        ],
        crs=CRS,
    )


def test_assign_cellcode_to_addresses_within_grid_gets_cellcode(grid):
    """An address point inside a grid cell is assigned that cell's code."""
    # Arrange
    addresses = gpd.GeoDataFrame(
        {"ADDR_ID": ["a1"]}, geometry=[Point(2, 2)], crs=CRS
    )

    # Act
    result = assign_cellcode_to_addresses(addresses, grid, "CELLCODE")

    # Assert
    assert result.loc[result["ADDR_ID"] == "a1", "cellcode"].iloc[0] == "CELL_A"


def test_assign_cellcode_to_addresses_outside_grid_is_dropped(grid):
    """An address point outside every cell is dropped."""
    # Arrange
    addresses = gpd.GeoDataFrame(
        {"ADDR_ID": ["a1", "a2"]},
        geometry=[Point(2, 2), Point(200, 200)],
        crs=CRS,
    )

    # Act
    result = assign_cellcode_to_addresses(addresses, grid, "CELLCODE")

    # Assert
    assert list(result["ADDR_ID"]) == ["a1"]


def test_assign_cellcode_by_overlap_single_match(grid):
    """A building fully inside one cell is assigned that cell's code."""
    # Arrange
    buildings = gpd.GeoDataFrame(
        {"BLDG_ID": ["b1"]},
        geometry=[Polygon([(1, 1), (4, 1), (4, 4), (1, 4)])],
        crs=CRS,
    )

    # Act
    result = assign_cellcode_by_overlap(buildings, grid, "CELLCODE")

    # Assert
    assert result.loc[result["BLDG_ID"] == "b1", "cellcode"].iloc[0] == "CELL_A"


def test_assign_cellcode_by_overlap_straddling_boundary_picks_largest(grid):
    """A building straddling the cell boundary is assigned the cell it
    overlaps most, not merely the first one geopandas happens to match."""
    # Arrange: this footprint overlaps CELL_A on x[8,10] (2 wide) and
    # CELL_B on x[10,14] (4 wide) - CELL_B has the larger overlap.
    buildings = gpd.GeoDataFrame(
        {"BLDG_ID": ["b1"]},
        geometry=[Polygon([(8, 1), (14, 1), (14, 4), (8, 4)])],
        crs=CRS,
    )

    # Act
    result = assign_cellcode_by_overlap(buildings, grid, "CELLCODE")

    # Assert
    assert result.loc[result["BLDG_ID"] == "b1", "cellcode"].iloc[0] == "CELL_B"


def test_assign_cellcode_by_overlap_outside_grid_gets_no_cellcode(grid):
    """A building entirely outside the grid gets a null cellcode rather than
    being dropped - filtering outliers is the combiner's job, not this
    function's."""
    # Arrange
    buildings = gpd.GeoDataFrame(
        {"BLDG_ID": ["b1"]},
        geometry=[Polygon([(100, 1), (104, 1), (104, 4), (100, 4)])],
        crs=CRS,
    )

    # Act
    result = assign_cellcode_by_overlap(buildings, grid, "CELLCODE")

    # Assert
    assert len(result) == 1
    assert pd.isna(result["cellcode"].iloc[0])


@pytest.fixture
def addresses_with_cellcode():
    """A single address point, already carrying a cellcode, as produced by
    assign_cellcode_to_addresses."""
    return gpd.GeoDataFrame(
        {"ADDR_ID": ["a1"], "cellcode": ["CELL_A"]},
        geometry=[Point(2, 2)],
        crs=CRS,
    )


def test_assign_cellcodes_single_address_building_uses_address_lookup(
    grid, addresses_with_cellcode
):
    """A building with exactly one address inherits that address's cell,
    even though the building's own geometry sits in a different cell -
    proving the lookup path is used, not the overlap path."""
    # Arrange: building geometry placed in CELL_B, but its linked address
    # (via add_id) is in CELL_A. If the lookup path is used, CELL_A wins.
    buildings = gpd.GeoDataFrame(
        {
            "BLDG_ID": ["b1"],
            "CNT_ADD": [1],
            "add_id": ["a1"],
        },
        geometry=[Polygon([(11, 1), (14, 1), (14, 4), (11, 4)])],
        crs=CRS,
    )

    # Act
    result = assign_cellcodes(
        buildings, addresses_with_cellcode, grid, "CELLCODE", "ADDR_ID"
    )

    # Assert
    assert result.loc[result["BLDG_ID"] == "b1", "MAPPED_CELLCODE"].iloc[0] == "CELL_A"


@pytest.mark.parametrize("address_count", [0, 2])
def test_assign_cellcodes_non_single_address_building_uses_overlap(
    grid, addresses_with_cellcode, address_count
):
    """Buildings with zero or multiple addresses are routed to the overlap
    path, ignoring any address lookup."""
    # Arrange: building geometry sits fully in CELL_B. If the overlap path
    # is used (as it should be for cnt != 1), CELL_B wins regardless of
    # what the (irrelevant, possibly absent) address link says.
    buildings = gpd.GeoDataFrame(
        {
            "BLDG_ID": ["b1"],
            "CNT_ADD": [address_count],
            "add_id": ["a1"],
        },
        geometry=[Polygon([(11, 1), (14, 1), (14, 4), (11, 4)])],
        crs=CRS,
    )

    # Act
    result = assign_cellcodes(
        buildings, addresses_with_cellcode, grid, "CELLCODE", "ADDR_ID"
    )

    # Assert
    assert result.loc[result["BLDG_ID"] == "b1", "MAPPED_CELLCODE"].iloc[0] == "CELL_B"


def test_assign_cellcodes_drops_outlier_with_no_cellcode(
    grid, addresses_with_cellcode
):
    """A building that gets no cellcode from either path (outside the grid,
    zero addresses) is dropped from the output."""
    # Arrange
    buildings = gpd.GeoDataFrame(
        {
            "BLDG_ID": ["b1", "b2"],
            "CNT_ADD": [0, 1],
            "add_id": [None, "a1"],
        },
        geometry=[
            Polygon([(100, 1), (104, 1), (104, 4), (100, 4)]),  # outside grid
            Polygon([(1, 1), (4, 1), (4, 4), (1, 4)]),
        ],
        crs=CRS,
    )

    # Act
    result = assign_cellcodes(
        buildings, addresses_with_cellcode, grid, "CELLCODE", "ADDR_ID"
    )

    # Assert
    assert list(result["BLDG_ID"]) == ["b2"]


def test_assign_cellcodes_handles_dtype_mismatch_between_id_columns(
    grid,
):
    """A single-address building whose add_id is stored as an int still
    matches an address whose ID column is text, because both sides are
    cast to str before the lookup."""
    # Arrange
    addresses = gpd.GeoDataFrame(
        {"ADDR_ID": ["1"], "cellcode": ["CELL_A"]},
        geometry=[Point(2, 2)],
        crs=CRS,
    )
    buildings = gpd.GeoDataFrame(
        {
            "BLDG_ID": ["b1"],
            "CNT_ADD": [1],
            "add_id": [1],  # int, not str
        },
        geometry=[Polygon([(11, 1), (14, 1), (14, 4), (11, 4)])],
        crs=CRS,
    )

    # Act
    result = assign_cellcodes(buildings, addresses, grid, "CELLCODE", "ADDR_ID")

    # Assert
    assert result.loc[result["BLDG_ID"] == "b1", "MAPPED_CELLCODE"].iloc[0] == "CELL_A"


def test_assign_cellcodes_raises_on_total_id_mismatch(grid):
    """If no single-address building's add_id matches any address ID at
    all, this signals a dataset mismatch (e.g. Script 4 and Script 5 given
    different address layers) and should fail loudly rather than silently
    dropping every single-address building as an outlier."""
    # Arrange
    addresses = gpd.GeoDataFrame(
        {"ADDR_ID": ["a1"], "cellcode": ["CELL_A"]},
        geometry=[Point(2, 2)],
        crs=CRS,
    )
    buildings = gpd.GeoDataFrame(
        {
            "BLDG_ID": ["b1"],
            "CNT_ADD": [1],
            "add_id": ["does_not_exist"],
        },
        geometry=[Polygon([(1, 1), (4, 1), (4, 4), (1, 4)])],
        crs=CRS,
    )

    # Act / Assert
    with pytest.raises(ValueError, match="Script 4 and Script 5"):
        assign_cellcodes(buildings, addresses, grid, "CELLCODE", "ADDR_ID")