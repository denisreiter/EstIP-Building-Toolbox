"""Tests for block feature calculations."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

from scripts.s4_context_features.blocks import (
    assign_blocks,
    calc_block_statistics,
    calc_block_ratios,
    classify_residential_extended,
)

CRS = "EPSG:3035"
ID_FIELD = "bldg_id"
BLOCK_ID_FIELD = "BLOCK_ID"
BLOCK_TYPE_FIELD = "BLOCK_TYPE"


def test_assign_blocks_single_block_containment():
    # Arrange: one building fully inside one block
    buildings = gpd.GeoDataFrame(
        {ID_FIELD: [1], "A": [4.0], "geometry": [box(2, 2, 4, 4)]}, crs=CRS
    )
    blocks = gpd.GeoDataFrame(
        {
            BLOCK_ID_FIELD: ["A"],
            BLOCK_TYPE_FIELD: ["residential"],
            "geometry": [box(0, 0, 10, 10)],
        },
        crs=CRS,
    )

    # Act
    result = assign_blocks(buildings, blocks, ID_FIELD, BLOCK_ID_FIELD, BLOCK_TYPE_FIELD)

    # Assert
    assert result.loc[0, "blck_id"] == "A"
    assert result.loc[0, "BLCK_TYPE"] == "residential"
    assert result.loc[0, "blck_a"] == pytest.approx(100.0)


def test_assign_blocks_straddling_building_uses_largest_overlap():
    # Arrange: building straddles two adjacent blocks unevenly --
    # 20 sq m overlap with block A, 40 sq m overlap with block B
    buildings = gpd.GeoDataFrame(
        {ID_FIELD: [1], "A": [60.0], "geometry": [box(8, 0, 14, 10)]}, crs=CRS
    )
    blocks = gpd.GeoDataFrame(
        {
            BLOCK_ID_FIELD: ["A", "B"],
            BLOCK_TYPE_FIELD: ["residential", "commercial"],
            "geometry": [box(0, 0, 10, 10), box(10, 0, 20, 10)],
        },
        crs=CRS,
    )

    # Act
    result = assign_blocks(buildings, blocks, ID_FIELD, BLOCK_ID_FIELD, BLOCK_TYPE_FIELD)

    # Assert: assigned to block B, the larger overlap
    assert result.loc[0, "blck_id"] == "B"
    assert result.loc[0, "BLCK_TYPE"] == "commercial"


def test_assign_blocks_building_outside_all_blocks():
    # Arrange: building nowhere near any block
    buildings = gpd.GeoDataFrame(
        {ID_FIELD: [1], "A": [16.0], "geometry": [box(100, 100, 104, 104)]}, crs=CRS
    )
    blocks = gpd.GeoDataFrame(
        {
            BLOCK_ID_FIELD: ["A"],
            BLOCK_TYPE_FIELD: ["residential"],
            "geometry": [box(0, 0, 10, 10)],
        },
        crs=CRS,
    )

    # Act
    result = assign_blocks(buildings, blocks, ID_FIELD, BLOCK_ID_FIELD, BLOCK_TYPE_FIELD)

    # Assert: no match, matching the original's KEEP_ALL behavior
    assert pd.isna(result.loc[0, "blck_id"])
    assert pd.isna(result.loc[0, "BLCK_TYPE"])


def test_assign_blocks_mixed_matched_and_unmatched_buildings():
    # Arrange: one building inside a block, one building far outside
    # every block, processed in the same call
    buildings = gpd.GeoDataFrame(
        {
            ID_FIELD: [1, 2],
            "A": [4.0, 16.0],
            "geometry": [box(2, 2, 4, 4), box(100, 100, 104, 104)],
        },
        crs=CRS,
    )
    blocks = gpd.GeoDataFrame(
        {
            BLOCK_ID_FIELD: ["A"],
            BLOCK_TYPE_FIELD: ["residential"],
            "geometry": [box(0, 0, 10, 10)],
        },
        crs=CRS,
    )

    # Act
    result = assign_blocks(buildings, blocks, ID_FIELD, BLOCK_ID_FIELD, BLOCK_TYPE_FIELD)

    # Assert: matched building gets assigned, unmatched building
    # remains null without raising a KeyError
    assert result.loc[result[ID_FIELD] == 1, "blck_id"].iloc[0] == "A"
    assert pd.isna(result.loc[result[ID_FIELD] == 2, "blck_id"].iloc[0])


def test_calc_block_statistics_typical_case():
    # Arrange: two buildings in the same block
    buildings = gpd.GeoDataFrame(
        {
            "blck_id": ["A", "A"],
            "A": [4.0, 6.0],
            "V": [100.0, 200.0],
            "DIST_NGHB": [5.0, 7.0],
            "geometry": [box(0, 0, 2, 2), box(3, 0, 5, 3)],  # areas 4, 6
        },
        crs=CRS,
    )

    # Act
    result = calc_block_statistics(buildings, volume_field="V")

    # Assert
    assert result.loc[0, "CNT_BLCK"] == 2
    assert result.loc[0, "A_SUM_BLCK"] == pytest.approx(10.0)
    assert result.loc[0, "A_MEAN_BLCK"] == pytest.approx(5.0)
    assert result.loc[0, "V_SUM_BLCK"] == pytest.approx(300.0)
    assert result.loc[0, "V_MEAN_BLCK"] == pytest.approx(150.0)
    assert result.loc[0, "MEAN_DIST_NGHB_BLCK"] == pytest.approx(6.0)


def test_calc_block_statistics_single_building_block_has_nan_std():
    # Arrange: a block with exactly one building -- std is undefined
    buildings = gpd.GeoDataFrame(
        {
            "blck_id": ["A"],
            "A": [4.0],
            "V": [100.0],
            "DIST_NGHB": [5.0],
            "geometry": [box(0, 0, 2, 2)],
        },
        crs=CRS,
    )

    # Act
    result = calc_block_statistics(buildings, volume_field="V")

    # Assert
    assert pd.isna(result.loc[0, "A_STD_BLCK"])
    assert pd.isna(result.loc[0, "V_STD_BLCK"])


def test_calc_block_statistics_unassigned_building_gets_null_stats():
    # Arrange: a building with no block assignment (blck_id is NaN)
    buildings = gpd.GeoDataFrame(
        {
            "blck_id": [np.nan],
            "A": [4.0],
            "V": [100.0],
            "DIST_NGHB": [5.0],
            "geometry": [box(0, 0, 2, 2)],
        },
        crs=CRS,
    )

    # Act
    result = calc_block_statistics(buildings, volume_field="V")

    # Assert
    assert pd.isna(result.loc[0, "CNT_BLCK"])


def test_calc_block_ratios_typical_case():
    # Arrange: a building with straightforward, nonzero block stats
    buildings = gpd.GeoDataFrame(
        {
            "V": [50.0],
            "A": [10.0],
            "blck_a": [20.0],
            "A_SUM_BLCK": [40.0],
            "A_MEAN_BLCK": [10.0],
            "V_SUM_BLCK": [200.0],
            "V_MEAN_BLCK": [25.0],
            "geometry": [box(0, 0, 2, 5)],  # area 10
        },
        crs=CRS,
    )

    # Act
    result = calc_block_ratios(buildings, volume_field="V")

    # Assert
    assert result.loc[0, "RATIO_A_BLCK"] == pytest.approx(10 / 20)
    assert result.loc[0, "RATIO_A_SUM"] == pytest.approx(10 / 40)
    assert result.loc[0, "RATIO_V_SUM"] == pytest.approx(50 / 200)
    assert result.loc[0, "RATIO_A_MEAN"] == pytest.approx(10 / 10)
    assert result.loc[0, "RATIO_V_MEAN"] == pytest.approx(50 / 25)


def test_calc_block_ratios_zero_denominator_is_nan():
    # Arrange: block-level stat is zero, which is undefined as a denominator
    buildings = gpd.GeoDataFrame(
        {
            "V": [50.0],
            "A": [10.0],
            "blck_a": [0.0],
            "A_SUM_BLCK": [40.0],
            "A_MEAN_BLCK": [10.0],
            "V_SUM_BLCK": [200.0],
            "V_MEAN_BLCK": [25.0],
            "geometry": [box(0, 0, 2, 5)],
        },
        crs=CRS,
    )

    # Act
    result = calc_block_ratios(buildings, volume_field="V")

    # Assert
    assert pd.isna(result.loc[0, "RATIO_A_BLCK"])


def test_calc_block_ratios_zero_numerator_is_zero_not_nan():
    # Arrange: building itself has zero area -- a valid ratio of 0,
    # not a missing value (this is the deliberate deviation from the
    # original ArcPy truthy guard, discussed and agreed on separately)
    buildings = gpd.GeoDataFrame(
        {
            "V": [0.0],
            "A": [0.0],
            "blck_a": [20.0],
            "A_SUM_BLCK": [40.0],
            "A_MEAN_BLCK": [10.0],
            "V_SUM_BLCK": [200.0],
            "V_MEAN_BLCK": [25.0],
            "geometry": [box(0, 0, 0, 0)],  # degenerate, zero area
        },
        crs=CRS,
    )

    # Act
    result = calc_block_ratios(buildings, volume_field="V")

    # Assert
    assert result.loc[0, "RATIO_A_BLCK"] == 0.0
    assert result.loc[0, "RATIO_V_SUM"] == 0.0


def test_classify_residential_extended_assigns_11_for_block_type_41001():
    # Arrange
    buildings = gpd.GeoDataFrame(
        {
            "is_res": [0],
            "FNC_CODE": [310019998],
            "BLCK_TYPE": [41001],
            "geometry": [Point(0, 0)],
        }
    )

    # Act
    result = classify_residential_extended(buildings)

    # Assert
    assert result["is_res"].iloc[0] == 11


def test_classify_residential_extended_assigns_22_for_block_type_41006():
    # Arrange
    buildings = gpd.GeoDataFrame(
        {
            "is_res": [0],
            "FNC_CODE": [310019998],
            "BLCK_TYPE": [41006],
            "geometry": [Point(0, 0)],
        }
    )

    # Act
    result = classify_residential_extended(buildings)

    # Assert
    assert result["is_res"].iloc[0] == 22


def test_classify_residential_extended_preserves_is_res_when_fnc_code_not_residential():
    # Arrange
    buildings = gpd.GeoDataFrame(
        {
            "is_res": [1],
            "FNC_CODE": [999999999],
            "BLCK_TYPE": [41001],
            "geometry": [Point(0, 0)],
        }
    )

    # Act
    result = classify_residential_extended(buildings)

    # Assert
    assert result["is_res"].iloc[0] == 1


def test_classify_residential_extended_preserves_is_res_when_block_type_unmatched():
    # Arrange
    buildings = gpd.GeoDataFrame(
        {
            "is_res": [2],
            "FNC_CODE": [310019998],
            "BLCK_TYPE": [99999],
            "geometry": [Point(0, 0)],
        }
    )

    # Act
    result = classify_residential_extended(buildings)

    # Assert
    assert result["is_res"].iloc[0] == 2


def test_classify_residential_extended_skips_and_returns_unchanged_when_blck_type_missing():
    # Arrange
    buildings = gpd.GeoDataFrame(
        {
            "is_res": [0, 1, 2],
            "FNC_CODE": [310019998, 310019998, 1],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
        }
    )

    # Act
    result = classify_residential_extended(buildings)

    # Assert
    pd.testing.assert_series_equal(result["is_res"], buildings["is_res"])