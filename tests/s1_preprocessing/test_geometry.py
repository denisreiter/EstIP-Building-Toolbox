"""Tests for lod2_toolbox.geometry"""

import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon

from scripts.s1_preprocessing.geometry import reproject, flag_overlapping_geometries


def test_reproject_changes_crs():
    gdf = gpd.GeoDataFrame(
        {"id": [1], "geometry": [Point(10, 50)]}, crs="EPSG:4326"
    )
    result = reproject(gdf, target_epsg=3035)
    assert result.crs.to_epsg() == 3035


def test_reproject_is_noop_when_already_in_target_crs():
    gdf = gpd.GeoDataFrame(
        {"id": [1], "geometry": [Point(4000000, 3000000)]}, crs="EPSG:3035"
    )
    result = reproject(gdf, target_epsg=3035)
    assert result.crs.to_epsg() == 3035
    assert result.iloc[0].geometry.x == gdf.iloc[0].geometry.x


def test_reproject_raises_on_missing_crs():
    gdf = gpd.GeoDataFrame({"id": [1], "geometry": [Point(0, 0)]})  # no crs set
    with pytest.raises(ValueError):
        reproject(gdf)


def test_flag_overlapping_geometries_flags_real_overlap():
    poly_a = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    poly_b = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])  # overlaps poly_a by 1 m²
    poly_c = Polygon([(10, 10), (11, 10), (11, 11), (10, 11)])  # isolated
    gdf = gpd.GeoDataFrame({"id": [1, 2, 3], "geometry": [poly_a, poly_b, poly_c]}, crs="EPSG:3035")

    result = flag_overlapping_geometries(gdf, overlap_area_threshold=0.5)

    assert result.loc[result["id"] == 1, "has_overlap"].iloc[0] == True
    assert result.loc[result["id"] == 2, "has_overlap"].iloc[0] == True
    assert result.loc[result["id"] == 3, "has_overlap"].iloc[0] == False


def test_flag_overlapping_geometries_ignores_mere_touching():
    poly_a = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly_b = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])  # shares only an edge
    gdf = gpd.GeoDataFrame({"id": [1, 2], "geometry": [poly_a, poly_b]}, crs="EPSG:3035")

    result = flag_overlapping_geometries(gdf)

    assert not result["has_overlap"].any()


def test_flag_overlapping_geometries_ignores_slivers_below_threshold():
    poly_a = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    # Overlaps poly_a by a 0.5 m x 0.5 m sliver = 0.25 m², below default 1.0 m² threshold
    poly_b = Polygon([(9.5, 9.5), (10.5, 9.5), (10.5, 10.5), (9.5, 10.5)])
    gdf = gpd.GeoDataFrame({"id": [1, 2], "geometry": [poly_a, poly_b]}, crs="EPSG:3035")

    result = flag_overlapping_geometries(gdf)

    assert not result["has_overlap"].any()


def test_flag_overlapping_geometries_raises_on_non_unique_index():
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    gdf = gpd.GeoDataFrame({"geometry": [poly, poly]}, index=[0, 0], crs="EPSG:3035")
    with pytest.raises(ValueError):
        flag_overlapping_geometries(gdf)