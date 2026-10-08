"""Tests for s3_neighborhood_features.radius_analysis."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box, Polygon

from scripts.s3_neighborhood_features.radius_analysis import (
    calc_coverage_within_radius,
    calc_neighbors_within_radius,
)

CRS = "EPSG:3035"


def test_calc_neighbors_within_radius_basic():
    """One neighbor well within radius: CNT=1, DIST=exact gap distance."""
    # Arrange: A and B are 10m apart (edge to edge), radius 50m
    gdf = gpd.GeoDataFrame(
        {"uid": [1, 2]},
        geometry=[box(0, 0, 10, 10), box(20, 0, 30, 10)],
        crs=CRS,
    )

    # Act
    result = calc_neighbors_within_radius(gdf, radius=50)

    # Assert
    assert result["CNT_NGHB_R50"].tolist() == [1, 1]
    assert result["DIST_NGHB_R50"].tolist() == pytest.approx([10.0, 10.0])


def test_calc_neighbors_within_radius_excludes_self():
    """Regression test: a polygon must never count itself as a neighbor,
    even though its own buffer trivially intersects itself."""
    # Arrange: single polygon, no other buildings at all
    gdf = gpd.GeoDataFrame({"uid": [1]}, geometry=[box(0, 0, 10, 10)], crs=CRS)

    # Act
    result = calc_neighbors_within_radius(gdf, radius=50)

    # Assert
    assert result["CNT_NGHB_R50"].iloc[0] == 0
    assert np.isnan(result["DIST_NGHB_R50"].iloc[0])


def test_calc_neighbors_within_radius_excludes_out_of_range():
    """A building beyond the radius must not be counted, even though it
    may be the nearest existing neighbor."""
    # Arrange: B is 60m from A, radius is only 50m
    gdf = gpd.GeoDataFrame(
        {"uid": [1, 2]},
        geometry=[box(0, 0, 10, 10), box(70, 0, 80, 10)],
        crs=CRS,
    )

    # Act
    result = calc_neighbors_within_radius(gdf, radius=50)

    # Assert
    assert result["CNT_NGHB_R50"].tolist() == [0, 0]
    assert result["DIST_NGHB_R50"].isna().all()


def test_calc_neighbors_within_radius_boundary_case_exact_radius():
    """Edge case: a neighbor at exactly the radius distance must be
    included (<=, not <)."""
    # Arrange: B is exactly 50m from A
    gdf = gpd.GeoDataFrame(
        {"uid": [1, 2]},
        geometry=[box(0, 0, 10, 10), box(60, 0, 70, 10)],
        crs=CRS,
    )

    # Act
    result = calc_neighbors_within_radius(gdf, radius=50)

    # Assert
    assert result["CNT_NGHB_R50"].tolist() == [1, 1]


def test_calc_neighbors_within_radius_counts_multiple_and_averages_distance():
    """With 3 neighbors at known distances, CNT and mean DIST must reflect
    all of them, not just the closest one -- this is the key behavioral
    difference from calc_nearest_neighbor_distance (ALL vs CLOSEST)."""
    # Arrange: B, C, D are 10m, 20m, 30m from A respectively, all within 50m
    gdf = gpd.GeoDataFrame(
        {"uid": [1, 2, 3, 4]},
        geometry=[
            box(0, 0, 10, 10),     # A
            box(20, 0, 30, 10),    # B: 10m from A
            box(30, 0, 40, 10),    # C: 20m from A
            box(40, 0, 50, 10),    # D: 30m from A
        ],
        crs=CRS,
    )

    # Act
    result = calc_neighbors_within_radius(gdf, radius=50)
    a_row = result[result["uid"] == 1].iloc[0]

    # Assert
    assert a_row["CNT_NGHB_R50"] == 3
    assert a_row["DIST_NGHB_R50"] == pytest.approx((10.0 + 20.0 + 30.0) / 3)


def test_calc_neighbors_within_radius_field_names_use_int_radius():
    """Regression test: a float or string-like radius must still produce
    clean integer field names (e.g. 50, not 50.0)."""
    # Arrange
    gdf = gpd.GeoDataFrame(
        {"uid": [1, 2]},
        geometry=[box(0, 0, 10, 10), box(20, 0, 30, 10)],
        crs=CRS,
    )

    # Act
    result = calc_neighbors_within_radius(gdf, radius=50.0)

    # Assert
    assert "CNT_NGHB_R50" in result.columns
    assert "DIST_NGHB_R50" in result.columns
    assert "CNT_NGHB_R50.0" not in result.columns


def test_calc_neighbors_within_radius_resets_gappy_index():
    """Edge case: if the input GeoDataFrame has a non-contiguous index
    (e.g. after upstream filtering), results must still be attributed to
    the correct rows -- this guards the reset_index safety net."""
    # Arrange: drop the middle row to create a gappy index [0, 2]
    gdf = gpd.GeoDataFrame(
        {"uid": [1, 2, 3]},
        geometry=[box(0, 0, 10, 10), box(200, 0, 210, 10), box(20, 0, 30, 10)],
        crs=CRS,
    ).drop(index=1)

    # Act
    result = calc_neighbors_within_radius(gdf, radius=50)

    # Assert: uid 1 and uid 3 are 10m apart and should find each other,
    # despite the gap in the original index
    assert result.set_index("uid")["CNT_NGHB_R50"].loc[1] == 1
    assert result.set_index("uid")["CNT_NGHB_R50"].loc[3] == 1


def test_calc_neighbors_within_radius_does_not_mutate_input():
    """The function must return a copy, leaving the caller's original
    GeoDataFrame untouched."""
    # Arrange
    gdf = gpd.GeoDataFrame(
        {"uid": [1, 2]},
        geometry=[box(0, 0, 10, 10), box(20, 0, 30, 10)],
        crs=CRS,
    )

    # Act
    calc_neighbors_within_radius(gdf, radius=50)

    # Assert
    assert "CNT_NGHB_R50" not in gdf.columns


def test_calc_coverage_within_radius_fully_covered_buffer_approaches_one():
    """If a building's entire buffer is covered by non-overlapping building
    footprints, coverage should approach 1.0. B is shaped as a large square
    with a hole matching A's footprint, so A and B do not overlap -- this
    isolates the 'fully covered' behavior from the documented double-count
    caveat for overlapping footprints (see calc_coverage_within_radius
    docstring)."""
    # Arrange: A is a 10x10 building; B is a huge square with a hole
    # exactly where A sits, so the two footprints are disjoint but together
    # cover the entire area around A
    building_a = box(0, 0, 10, 10)
    building_b = Polygon(
        box(-1000, -1000, 1000, 1000).exterior.coords,
        holes=[building_a.exterior.coords],
    )
    gdf = gpd.GeoDataFrame({"uid": [1, 2]}, geometry=[building_a, building_b], crs=CRS)

    # Act
    result = calc_coverage_within_radius(gdf, radius=50)
    a_coverage = result[result["uid"] == 1]["CVR_NGHB_R50"].iloc[0]

    # Assert
    assert a_coverage == pytest.approx(1.0, abs=1e-6)


def test_calc_coverage_within_radius_partial_overlap_clips_correctly():
    """A neighbor building straddling the buffer boundary should only
    contribute the portion of its area that falls inside the buffer, not
    its full footprint."""
    # Arrange: A is a point-like small building; B is a large building
    # positioned so only half of it falls inside A's 50m buffer
    gdf = gpd.GeoDataFrame(
        {"uid": [1, 2]},
        geometry=[box(0, 0, 1, 1), box(-10, -100, 10, 100)],  # B: 20 wide, 200 tall, straddling
        crs=CRS,
    )

    # Act
    result = calc_coverage_within_radius(gdf, radius=50)
    a_coverage = result[result["uid"] == 1]["CVR_NGHB_R50"].iloc[0]

    # Assert: coverage must be strictly between 0 and 1 -- if clipping were
    # broken and counted B's full area, this could exceed 1; if clipping
    # were broken and excluded B entirely, this would just be A's own
    # tiny footprint fraction
    buffer_area = gdf.geometry.iloc[0].buffer(50).area
    a_only_fraction = 1.0 / buffer_area
    assert a_coverage > a_only_fraction  # B contributes something
    assert a_coverage <= 1.0  # never exceeds the buffer itself


def test_calc_coverage_within_radius_no_neighbors_equals_self_only():
    """When no other buildings are in range, coverage should equal exactly
    the source building's own footprint fraction -- confirms include-self
    behavior precisely, not just 'greater than zero'."""
    # Arrange: B is far outside any reasonable radius
    gdf = gpd.GeoDataFrame(
        {"uid": [1, 2]},
        geometry=[box(0, 0, 10, 10), box(10000, 10000, 10010, 10010)],
        crs=CRS,
    )

    # Act
    result = calc_coverage_within_radius(gdf, radius=50)
    a_coverage = result[result["uid"] == 1]["CVR_NGHB_R50"].iloc[0]

    # Assert
    expected = 100.0 / gdf.geometry.iloc[0].buffer(50).area
    assert a_coverage == pytest.approx(expected)


def test_calc_coverage_within_radius_resets_gappy_index():
    """Same positional-indexing safety net as calc_neighbors_within_radius
    must apply here too."""
    # Arrange
    gdf = gpd.GeoDataFrame(
        {"uid": [1, 2, 3]},
        geometry=[box(0, 0, 10, 10), box(10000, 0, 10010, 10), box(15, 0, 25, 10)],
        crs=CRS,
    ).drop(index=1)

    # Act
    result = calc_coverage_within_radius(gdf, radius=50)

    # Assert: uid 1 and uid 3 are close and should cover each other
    a_coverage = result.set_index("uid")["CVR_NGHB_R50"].loc[1]
    self_only = 100.0 / gdf.geometry.iloc[0].buffer(50).area
    assert a_coverage > self_only


def test_calc_coverage_within_radius_does_not_mutate_input():
    """The function must return a copy, leaving the caller's original
    GeoDataFrame untouched."""
    # Arrange
    gdf = gpd.GeoDataFrame({"uid": [1]}, geometry=[box(0, 0, 10, 10)], crs=CRS)

    # Act
    calc_coverage_within_radius(gdf, radius=50)

    # Assert
    assert "CVR_NGHB_R50" not in gdf.columns


def test_calc_coverage_within_radius_overlapping_footprints_double_count():
    """Documented caveat: overlapping building footprints are not
    deduplicated, so coverage can exceed 1.0. This test pins that known
    behavior explicitly."""
    # Arrange: B fully contains A, so their footprints overlap
    gdf = gpd.GeoDataFrame(
        {"uid": [1, 2]},
        geometry=[box(0, 0, 10, 10), box(-1000, -1000, 1000, 1000)],
        crs=CRS,
    )

    # Act
    result = calc_coverage_within_radius(gdf, radius=50)
    a_coverage = result[result["uid"] == 1]["CVR_NGHB_R50"].iloc[0]

    # Assert: coverage exceeds 1.0 due to the double-counted overlap
    assert a_coverage > 1.0