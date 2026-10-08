"""Tests for s6_population_poi_features.population_mapping."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from scripts.s6_population_mapping.population_mapping import (
    calc_volume_weighted_population,
    finalize_population_field,
    join_cell_population,
    map_population,
    select_residential_buildings,
)


@pytest.mark.parametrize(
    "is_res_value, expected_selected",
    [
        (1, True),
        (2, True),
        (11, True),
        (22, True),
        (0, False),
        (99, False),
    ],
)
def test_select_residential_buildings_filters_by_is_res_category(is_res_value, expected_selected):
    # Arrange
    buildings = gpd.GeoDataFrame(
        {
            "A": [100.0],
            "V": [500.0],
            "is_res": [is_res_value],
            "geometry": [Point(0, 0)],
        }
    )

    # Act
    result = select_residential_buildings(buildings, min_area=40.0, min_volume=250.0)

    # Assert
    assert (len(result) == 1) == expected_selected


@pytest.mark.parametrize(
    "area, volume, expected_selected",
    [
        (40.0, 250.0, True),   # exactly at both thresholds
        (39.9, 250.0, False),  # below area threshold
        (40.0, 249.9, False),  # below volume threshold
        (100.0, 500.0, True),  # comfortably above both
    ],
)
def test_select_residential_buildings_applies_area_and_volume_thresholds(
    area, volume, expected_selected
):
    # Arrange
    buildings = gpd.GeoDataFrame(
        {
            "A": [area],
            "V": [volume],
            "is_res": [1],
            "geometry": [Point(0, 0)],
        }
    )

    # Act
    result = select_residential_buildings(buildings, min_area=40.0, min_volume=250.0)

    # Assert
    assert (len(result) == 1) == expected_selected


def test_select_residential_buildings_returns_copy_not_view():
    # Arrange
    buildings = gpd.GeoDataFrame(
        {
            "A": [100.0],
            "V": [500.0],
            "is_res": [1],
            "geometry": [Point(0, 0)],
        }
    )

    # Act
    result = select_residential_buildings(buildings)
    result["A"] = 0.0

    # Assert
    assert buildings["A"].iloc[0] == 100.0


def test_join_cell_population_matches_cellcode_and_attaches_einwohner():
    # Arrange
    buildings = gpd.GeoDataFrame(
        {
            "MAPPED_CELLCODE": ["A1", "A2"],
            "geometry": [Point(0, 0), Point(1, 1)],
        }
    )
    census_grid = gpd.GeoDataFrame(
        {
            "CellCode": ["A1", "A2"],
            "Einwohner": [50.0, 120.0],
            "geometry": [Point(0, 0), Point(1, 1)],
        }
    )

    # Act
    result = join_cell_population(
        buildings, census_grid, grid_id_field="CellCode", population_field="Einwohner"
    )

    # Assert
    assert result.loc[result["MAPPED_CELLCODE"] == "A1", "Einwohner"].iloc[0] == 50.0
    assert result.loc[result["MAPPED_CELLCODE"] == "A2", "Einwohner"].iloc[0] == 120.0


def test_join_cell_population_assigns_nan_for_unmatched_cell():
    # Arrange
    buildings = gpd.GeoDataFrame(
        {
            "MAPPED_CELLCODE": ["NO_GRID_DATA"],
            "geometry": [Point(0, 0)],
        }
    )
    census_grid = gpd.GeoDataFrame(
        {
            "CellCode": ["A1"],
            "Einwohner": [50.0],
            "geometry": [Point(0, 0)],
        }
    )

    # Act
    result = join_cell_population(
        buildings, census_grid, grid_id_field="CellCode", population_field="Einwohner"
    )

    # Assert
    assert pd.isna(result["Einwohner"].iloc[0])


def test_join_cell_population_preserves_all_building_rows():
    # Arrange
    buildings = gpd.GeoDataFrame(
        {
            "MAPPED_CELLCODE": ["A1", "A2", "A3"],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
        }
    )
    census_grid = gpd.GeoDataFrame(
        {
            "CellCode": ["A1"],
            "Einwohner": [50.0],
            "geometry": [Point(0, 0)],
        }
    )

    # Act
    result = join_cell_population(
        buildings, census_grid, grid_id_field="CellCode", population_field="Einwohner"
    )

    # Assert
    assert len(result) == len(buildings)


def test_join_cell_population_drops_cellcode_column():
    # Arrange
    buildings = gpd.GeoDataFrame(
        {
            "MAPPED_CELLCODE": ["A1"],
            "geometry": [Point(0, 0)],
        }
    )
    census_grid = gpd.GeoDataFrame(
        {
            "CellCode": ["A1"],
            "Einwohner": [50.0],
            "geometry": [Point(0, 0)],
        }
    )

    # Act
    result = join_cell_population(
        buildings, census_grid, grid_id_field="CellCode", population_field="Einwohner"
    )

    # Assert
    assert "CellCode" not in result.columns


def test_calc_volume_weighted_population_splits_proportionally_to_volume():
    # Arrange: two buildings in the same cell, volumes 300 and 700 (30%/70% split)
    residential_buildings = gpd.GeoDataFrame(
        {
            "MAPPED_CELLCODE": ["A1", "A1"],
            "V": [300.0, 700.0],
            "Einwohner": [100.0, 100.0],
            "geometry": [Point(0, 0), Point(1, 1)],
        }
    )

    # Act
    result = calc_volume_weighted_population(residential_buildings, population_field="Einwohner")

    # Assert
    assert result["MAPPED_POPULATION"].iloc[0] == pytest.approx(30.0)
    assert result["MAPPED_POPULATION"].iloc[1] == pytest.approx(70.0)


def test_calc_volume_weighted_population_gives_full_population_to_sole_building_in_cell():
    # Arrange
    residential_buildings = gpd.GeoDataFrame(
        {
            "MAPPED_CELLCODE": ["A1"],
            "V": [500.0],
            "Einwohner": [42.0],
            "geometry": [Point(0, 0)],
        }
    )

    # Act
    result = calc_volume_weighted_population(residential_buildings, population_field="Einwohner")

    # Assert
    assert result["MAPPED_POPULATION"].iloc[0] == pytest.approx(42.0)


def test_calc_volume_weighted_population_propagates_nan_for_no_data_cell():
    # Arrange
    residential_buildings = gpd.GeoDataFrame(
        {
            "MAPPED_CELLCODE": ["A1"],
            "V": [500.0],
            "Einwohner": [np.nan],
            "geometry": [Point(0, 0)],
        }
    )

    # Act
    result = calc_volume_weighted_population(residential_buildings, population_field="Einwohner")

    # Assert
    assert pd.isna(result["MAPPED_POPULATION"].iloc[0])


def test_calc_volume_weighted_population_does_not_mix_cells():
    # Arrange: cell A1 has one building, cell A2 has one building with a
    # different population value; each should be self-contained.
    residential_buildings = gpd.GeoDataFrame(
        {
            "MAPPED_CELLCODE": ["A1", "A2"],
            "V": [500.0, 500.0],
            "Einwohner": [10.0, 90.0],
            "geometry": [Point(0, 0), Point(1, 1)],
        }
    )

    # Act
    result = calc_volume_weighted_population(residential_buildings, population_field="Einwohner")

    # Assert
    assert result.loc[result["MAPPED_CELLCODE"] == "A1", "MAPPED_POPULATION"].iloc[0] == pytest.approx(10.0)
    assert result.loc[result["MAPPED_CELLCODE"] == "A2", "MAPPED_POPULATION"].iloc[0] == pytest.approx(90.0)


def test_finalize_population_field_assigns_computed_value_to_residential_building():
    # Arrange
    buildings = gpd.GeoDataFrame(
        {
            "field_id": [1],
            "geometry": [Point(0, 0)],
        }
    )
    residential_buildings_with_population = gpd.GeoDataFrame(
        {
            "field_id": [1],
            "MAPPED_POPULATION": [42.0],
            "geometry": [Point(0, 0)],
        }
    )

    # Act
    result = finalize_population_field(
        buildings,
        residential_buildings_with_population,
        field_id="field_id",
        population_field="Einwohner",
    )

    # Assert
    assert result["MAPPED_POPULATION"].iloc[0] == pytest.approx(42.0)


def test_finalize_population_field_fills_zero_for_non_residential_building():
    # Arrange: building 2 never appears in the residential subset
    buildings = gpd.GeoDataFrame(
        {
            "field_id": [1, 2],
            "geometry": [Point(0, 0), Point(1, 1)],
        }
    )
    residential_buildings_with_population = gpd.GeoDataFrame(
        {
            "field_id": [1],
            "MAPPED_POPULATION": [42.0],
            "geometry": [Point(0, 0)],
        }
    )

    # Act
    result = finalize_population_field(
        buildings,
        residential_buildings_with_population,
        field_id="field_id",
        population_field="Einwohner",
    )

    # Assert
    assert result.loc[result["field_id"] == 2, "MAPPED_POPULATION"].iloc[0] == 0.0


def test_finalize_population_field_fills_zero_for_nan_population_result():
    # Arrange: residential building sits in a no-data cell, MAPPED_POPULATION is NaN
    buildings = gpd.GeoDataFrame(
        {
            "field_id": [1],
            "geometry": [Point(0, 0)],
        }
    )
    residential_buildings_with_population = gpd.GeoDataFrame(
        {
            "field_id": [1],
            "MAPPED_POPULATION": [np.nan],
            "geometry": [Point(0, 0)],
        }
    )

    # Act
    result = finalize_population_field(
        buildings,
        residential_buildings_with_population,
        field_id="field_id",
        population_field="Einwohner",
    )

    # Assert
    assert result["MAPPED_POPULATION"].iloc[0] == 0.0


def test_finalize_population_field_never_leaves_nan():
    # Arrange
    buildings = gpd.GeoDataFrame(
        {
            "field_id": [1, 2, 3],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
        }
    )
    residential_buildings_with_population = gpd.GeoDataFrame(
        {
            "field_id": [1],
            "MAPPED_POPULATION": [10.0],
            "geometry": [Point(0, 0)],
        }
    )

    # Act
    result = finalize_population_field(
        buildings,
        residential_buildings_with_population,
        field_id="field_id",
        population_field="Einwohner",
    )

    # Assert
    assert result["MAPPED_POPULATION"].isna().sum() == 0


def test_finalize_population_field_drops_einwohner_when_present():
    # Arrange
    buildings = gpd.GeoDataFrame(
        {
            "field_id": [1],
            "Einwohner": [50.0],
            "geometry": [Point(0, 0)],
        }
    )
    residential_buildings_with_population = gpd.GeoDataFrame(
        {
            "field_id": [1],
            "MAPPED_POPULATION": [42.0],
            "geometry": [Point(0, 0)],
        }
    )

    # Act
    result = finalize_population_field(
        buildings,
        residential_buildings_with_population,
        field_id="field_id",
        population_field="Einwohner",
    )

    # Assert
    assert "Einwohner" not in result.columns


def test_finalize_population_field_does_not_error_when_einwohner_absent():
    # Arrange: buildings never had Einwohner joined onto it
    buildings = gpd.GeoDataFrame(
        {
            "field_id": [1],
            "geometry": [Point(0, 0)],
        }
    )
    residential_buildings_with_population = gpd.GeoDataFrame(
        {
            "field_id": [1],
            "MAPPED_POPULATION": [42.0],
            "geometry": [Point(0, 0)],
        }
    )

    # Act
    result = finalize_population_field(
        buildings,
        residential_buildings_with_population,
        field_id="field_id",
        population_field="Einwohner",
    )

    # Assert
    assert "Einwohner" not in result.columns


def test_map_population_end_to_end_produces_expected_split_and_zero_fill():
    # Arrange: two residential buildings in cell A1 (30/70 volume split,
    # population 100), one non-residential building in cell A1, one
    # residential building in cell A2 which has no census data.
    buildings = gpd.GeoDataFrame(
        {
            "field_id": [1, 2, 3, 4],
            "A": [100.0, 100.0, 100.0, 100.0],
            "V": [300.0, 700.0, 500.0, 500.0],
            "is_res": [1, 1, 0, 1],
            "FNC_CODE": [1, 1, 1, 1],
            "MAPPED_CELLCODE": ["A1", "A1", "A1", "A2"],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3)],
        }
    )
    census_grid = gpd.GeoDataFrame(
        {
            "CellCode": ["A1", "A2"],
            "Einwohner": [100.0, np.nan],
            "geometry": [Point(0, 0), Point(3, 3)],
        }
    )

    # Act
    result = map_population(
        buildings,
        census_grid,
        field_id="field_id",
        grid_id_field="CellCode",
        population_field="Einwohner",
        min_area=40.0,
        min_volume=250.0,
    )

    # Assert
    result = result.set_index("field_id")
    assert result.loc[1, "MAPPED_POPULATION"] == pytest.approx(30.0)
    assert result.loc[2, "MAPPED_POPULATION"] == pytest.approx(70.0)
    assert result.loc[3, "MAPPED_POPULATION"] == 0.0  # non-residential
    assert result.loc[4, "MAPPED_POPULATION"] == 0.0  # residential, no-data cell
    assert result["MAPPED_POPULATION"].isna().sum() == 0
    assert "Einwohner" not in result.columns


def test_select_residential_buildings_excludes_null_is_res():
    # Arrange: a building with unknown/null is_res (e.g. a future LoD2 type)
    buildings = gpd.GeoDataFrame(
        {
            "A": [100.0],
            "V": [500.0],
            "is_res": [np.nan],
            "geometry": [Point(0, 0)],
        }
    )

    # Act
    result = select_residential_buildings(buildings, min_area=40.0, min_volume=250.0)

    # Assert
    assert len(result) == 0


def test_map_population_assigns_zero_for_null_is_res():
    # Arrange: unknown is_res should end up ignored, population 0.0
    buildings = gpd.GeoDataFrame(
        {
            "field_id": [1],
            "A": [100.0],
            "V": [500.0],
            "is_res": [np.nan],
            "FNC_CODE": ["1"],
            "MAPPED_CELLCODE": ["A1"],
            "geometry": [Point(0, 0)],
        }
    )
    census_grid = gpd.GeoDataFrame(
        {
            "CellCode": ["A1"],
            "Einwohner": [100.0],
            "geometry": [Point(0, 0)],
        }
    )

    # Act
    result = map_population(
        buildings, census_grid, field_id="field_id",
        grid_id_field="CellCode", population_field="Einwohner",
    )

    # Assert
    assert result["MAPPED_POPULATION"].iloc[0] == 0.0