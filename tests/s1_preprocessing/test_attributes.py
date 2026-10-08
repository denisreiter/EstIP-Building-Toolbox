"""Tests for lod2_toolbox.attributes"""

import geopandas as gpd
from shapely.geometry import Point
import pytest

from scripts.s1_preprocessing.attributes import (
    find_main_building_part,
    join_part_count,
    assign_roof_type,
    assign_residential_function
)


def test_find_main_building_part_flags_highest_selector():
    gdf = gpd.GeoDataFrame(
        {
            "bldg_id": [1, 1, 1, 2, 2],
            "selector": [5, 9, 3, 1, 1],
            "geometry": [Point(i, i) for i in range(5)],
        }
    )
    result = find_main_building_part(gdf, id_field="bldg_id", selector_field="selector")

    # Building 1: row with selector=9 (index 1) should be selected
    selected_bldg1 = result[(result["bldg_id"] == 1) & (result["is_selected_part"] == 1)]
    assert len(selected_bldg1) == 1
    assert selected_bldg1.iloc[0]["selector"] == 9

    # CNT_PRTS should reflect part counts per building
    assert result[result["bldg_id"] == 1]["CNT_PRTS"].iloc[0] == 3
    assert result[result["bldg_id"] == 2]["CNT_PRTS"].iloc[0] == 2


def test_find_main_building_part_breaks_ties_by_first_occurrence():
    gdf = gpd.GeoDataFrame(
        {
            "bldg_id": [1, 1, 1],
            "selector": [7, 7, 2],  # first two rows tie for max
            "geometry": [Point(i, i) for i in range(3)],
        }
    )
    result = find_main_building_part(gdf, id_field="bldg_id", selector_field="selector")

    # Only the first tied row (original index 0) should be selected
    assert result.loc[0, "is_selected_part"] == 1
    assert result.loc[1, "is_selected_part"] == 0
    assert result.loc[2, "is_selected_part"] == 0
    assert result["is_selected_part"].sum() == 1


def test_join_part_count_attaches_counts():
    footprints = gpd.GeoDataFrame(
        {"bldg_id": [1, 2], "geometry": [Point(0, 0), Point(1, 1)]}
    )
    parts = gpd.GeoDataFrame(
        {
            "bldg_id": [1, 1, 2],
            "is_selected_part": [1, 0, 1],
            "CNT_PRTS": [2, 2, 1],
            "geometry": [Point(0, 0), Point(0.1, 0), Point(1, 1)],
        }
    )
    result = join_part_count(footprints, parts, id_field="bldg_id")
    assert result[result["bldg_id"] == 1]["CNT_PRTS"].iloc[0] == 2
    assert result[result["bldg_id"] == 2]["CNT_PRTS"].iloc[0] == 1


def test_assign_roof_type_uses_footprint_field_when_given():
    footprints = gpd.GeoDataFrame(
        {"bldg_id": [1], "roof": ["flat"], "geometry": [Point(0, 0)]}
    )
    parts = gpd.GeoDataFrame(
        {"bldg_id": [1], "is_selected_part": [1], "geometry": [Point(0, 0)]}
    )
    result = assign_roof_type(footprints, parts, id_field="bldg_id", roof_type_field="roof")
    assert result.loc[0, "RF_TYPE"] == "flat"
    assert "roof" not in result.columns


def test_assign_roof_type_falls_back_to_parts_field():
    footprints = gpd.GeoDataFrame({"bldg_id": [1], "geometry": [Point(0, 0)]})
    parts = gpd.GeoDataFrame(
        {
            "bldg_id": [1, 1],
            "is_selected_part": [1, 0],
            "part_roof": ["gabled", "flat"],
            "geometry": [Point(0, 0), Point(0.1, 0)],
        }
    )
    result = assign_roof_type(
        footprints, parts, id_field="bldg_id", roof_type_field_parts="part_roof"
    )
    assert result.loc[0, "RF_TYPE"] == "gabled"


def test_assign_roof_type_nulls_when_no_source_given():
    footprints = gpd.GeoDataFrame({"bldg_id": [1], "geometry": [Point(0, 0)]})
    parts = gpd.GeoDataFrame(
        {"bldg_id": [1], "is_selected_part": [1], "geometry": [Point(0, 0)]}
    )
    result = assign_roof_type(footprints, parts, id_field="bldg_id")
    assert "RF_TYPE" in result.columns
    assert result["RF_TYPE"].isna().all()


def test_assign_roof_type_raises_if_parts_field_given_without_parts():
    footprints = gpd.GeoDataFrame({"bldg_id": [1], "geometry": [Point(0, 0)]})
    with pytest.raises(ValueError):
        assign_roof_type(
            footprints, None, id_field="bldg_id", roof_type_field_parts="part_roof"
        )


def test_assign_residential_function_joins_matching_rows():
    footprints = gpd.GeoDataFrame(
        {"bldg_id": [1, 2], "function": ["31001", "31002"], "geometry": [Point(0, 0), Point(1, 1)]}
    )
    lookup = gpd.GeoDataFrame(
        {
            "orig_fkt": ["31001", "31002"],
            "FNC_CODE": [3, 4],
            "is_res": [1, 0],
            "geometry": [None, None],
        }
    )
    result = assign_residential_function(footprints, lookup, function_field="function")
    assert result.loc[result["bldg_id"] == 1, "is_res"].iloc[0] == 1
    assert result.loc[result["bldg_id"] == 1, "FNC_CODE"].iloc[0] == 3
    assert result.loc[result["bldg_id"] == 2, "is_res"].iloc[0] == 0


def test_assign_residential_function_defaults_unmatched_to_zero():
    footprints = gpd.GeoDataFrame(
        {"bldg_id": [1], "function": ["99999"], "geometry": [Point(0, 0)]}
    )
    lookup = gpd.GeoDataFrame(
        {"orig_fkt": ["31001"], "FNC_CODE": [3], "is_res": [1], "geometry": [None]}
    )
    result = assign_residential_function(footprints, lookup, function_field="function")
    assert result.loc[0, "is_res"] == 0
    assert result.loc[0, "FNC_CODE"] == 0
    assert result["is_res"].dtype.kind == "i"
    assert result["FNC_CODE"].dtype.kind == "i"


def test_assign_roof_type_raises_on_missing_footprint_field():
    footprints = gpd.GeoDataFrame({"bldg_id": [1], "geometry": [Point(0, 0)]})
    parts = gpd.GeoDataFrame(
        {"bldg_id": [1], "is_selected_part": [1], "geometry": [Point(0, 0)]}
    )
    with pytest.raises(ValueError, match="roof_type_field"):
        assign_roof_type(footprints, parts, id_field="bldg_id", roof_type_field="does_not_exist")


def test_assign_roof_type_raises_on_missing_parts_field():
    footprints = gpd.GeoDataFrame({"bldg_id": [1], "geometry": [Point(0, 0)]})
    parts = gpd.GeoDataFrame(
        {"bldg_id": [1], "is_selected_part": [1], "geometry": [Point(0, 0)]}
    )
    with pytest.raises(ValueError, match="roof_type_field_parts"):
        assign_roof_type(footprints, parts, id_field="bldg_id", roof_type_field_parts="does_not_exist")


def test_assign_roof_type_rejects_string_none_as_field_name():
    # Guards against a YAML config gotcha: "field: None" parses as the
    # literal string "None", not Python's None — this must fail loudly,
    # not silently return footprints unchanged.
    footprints = gpd.GeoDataFrame({"bldg_id": [1], "geometry": [Point(0, 0)]})
    parts = gpd.GeoDataFrame(
        {"bldg_id": [1], "is_selected_part": [1], "geometry": [Point(0, 0)]}
    )
    with pytest.raises(ValueError):
        assign_roof_type(footprints, parts, id_field="bldg_id", roof_type_field="None")