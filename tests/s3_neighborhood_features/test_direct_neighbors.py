"""Tests for s3_neighborhood_features.direct_neighbors."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, box

from scripts.s3_neighborhood_features.direct_neighbors import (
    calc_nearest_neighbor_distance,
    calc_shared_neighbors,
    calc_shared_wall_ratio,
)

CRS = "EPSG:3035"


def test_calc_nearest_neighbor_distance_basic():
    """Two polygons 5m apart should report that exact distance both ways."""
    # Arrange: square A at x=[0,10], square B starting 5m to the right of A
    gdf = gpd.GeoDataFrame(
        {"uid": [1, 2]},
        geometry=[box(0, 0, 10, 10), box(15, 0, 25, 10)],
        crs=CRS,
    )

    # Act
    result = calc_nearest_neighbor_distance(gdf)

    # Assert
    assert result["DIST_NGHB"].tolist() == [5.0, 5.0]


def test_calc_nearest_neighbor_distance_picks_closest_of_several():
    """With three polygons, each should report distance to its true closest."""
    # Arrange: A---5m---B---20m---C  (A's closest is B, not C)
    gdf = gpd.GeoDataFrame(
        {"uid": [1, 2, 3]},
        geometry=[box(0, 0, 10, 10), box(15, 0, 25, 10), box(45, 0, 55, 10)],
        crs=CRS,
    )

    # Act
    result = calc_nearest_neighbor_distance(gdf)

    # Assert
    assert result.set_index("uid")["DIST_NGHB"].loc[1] == 5.0
    assert result.set_index("uid")["DIST_NGHB"].loc[3] == 20.0


def test_calc_nearest_neighbor_distance_single_polygon_returns_nan():
    """Edge case: a GeoDataFrame with only one polygon has no possible
    neighbor, so the distance must be NaN, not 0 or an error."""
    # Arrange
    gdf = gpd.GeoDataFrame({"uid": [1]}, geometry=[box(0, 0, 10, 10)], crs=CRS)

    # Act
    result = calc_nearest_neighbor_distance(gdf)

    # Assert
    assert np.isnan(result["DIST_NGHB"].iloc[0])


def test_calc_nearest_neighbor_distance_handles_ties():
    """Edge case: if two neighbors are exactly equidistant, the function
    must still return exactly one row per input polygon (no duplication),
    picking one of the tied distances."""
    # Arrange: B and C are both exactly 5m from A
    gdf = gpd.GeoDataFrame(
        {"uid": [1, 2, 3]},
        geometry=[box(0, 0, 10, 10), box(15, 0, 25, 10), box(-15, 0, -5, 10)],
        crs=CRS,
    )

    # Act
    result = calc_nearest_neighbor_distance(gdf)

    # Assert: no row duplication despite the tie
    assert len(result) == 3
    assert result.set_index("uid")["DIST_NGHB"].loc[1] == 5.0


def test_calc_nearest_neighbor_distance_does_not_mutate_input():
    """The function must return a copy, leaving the caller's original
    GeoDataFrame untouched."""
    # Arrange
    gdf = gpd.GeoDataFrame(
        {"uid": [1, 2]},
        geometry=[box(0, 0, 10, 10), box(15, 0, 25, 10)],
        crs=CRS,
    )

    # Act
    calc_nearest_neighbor_distance(gdf)

    # Assert
    assert "DIST_NGHB" not in gdf.columns


def test_calc_shared_neighbors_shared_edge():
    """Two polygons sharing a full 10m edge should report CNT=1 and
    sum_length=10 for each."""
    # Arrange
    gdf = gpd.GeoDataFrame(
        {"uid": [1, 2]},
        geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)],
        crs=CRS,
    )

    # Act
    result = calc_shared_neighbors(gdf)

    # Assert
    assert result["CNT_NGHB_D"].tolist() == [1, 1]
    assert result["sum_length"].tolist() == pytest.approx([10.0, 10.0])


def test_calc_shared_neighbors_corner_touch_only():
    """Edge case: two polygons touching only at a single point (e.g.
    diagonal buildings meeting at a corner) should count as a neighbor
    (CNT=1) but contribute zero shared length."""
    # Arrange: A occupies (0,0)-(10,10), B occupies (10,10)-(20,20) -- they
    # meet only at the single point (10, 10)
    gdf = gpd.GeoDataFrame(
        {"uid": [1, 2]},
        geometry=[box(0, 0, 10, 10), box(10, 10, 20, 20)],
        crs=CRS,
    )

    # Act
    result = calc_shared_neighbors(gdf)

    # Assert
    assert result["CNT_NGHB_D"].tolist() == [1, 1]
    assert result["sum_length"].tolist() == pytest.approx([0.0, 0.0])


def test_calc_shared_neighbors_isolated_polygon():
    """A polygon with no touching neighbors should report CNT=0 and
    sum_length=0, not NaN -- these are known zeros, not missing data."""
    # Arrange: B is 100m away from A, far outside any tolerance
    gdf = gpd.GeoDataFrame(
        {"uid": [1, 2]},
        geometry=[box(0, 0, 10, 10), box(110, 0, 120, 10)],
        crs=CRS,
    )

    # Act
    result = calc_shared_neighbors(gdf)

    # Assert
    assert result["CNT_NGHB_D"].tolist() == [0, 0]
    assert result["sum_length"].tolist() == [0.0, 0.0]


def test_calc_shared_neighbors_area_overlap_excluded():
    """Polygons that overlap by area (not just touch) must be excluded
    entirely -- matching ArcGIS's NO_AREA_OVERLAP setting."""
    # Arrange: B overlaps A by a 2x10 strip, rather than merely touching
    gdf = gpd.GeoDataFrame(
        {"uid": [1, 2]},
        geometry=[box(0, 0, 10, 10), box(8, 0, 18, 10)],
        crs=CRS,
    )

    # Act
    result = calc_shared_neighbors(gdf)

    # Assert
    assert result["CNT_NGHB_D"].tolist() == [0, 0]
    assert result["sum_length"].tolist() == [0.0, 0.0]


def test_calc_shared_neighbors_counts_three_plus_neighbors():
    """Regression test: a polygon touching 3+ neighbors must have its
    length and count summed across all of them, not just one."""
    # Arrange: A (0,0)-(10,10) is touched on three sides by B, C, D
    gdf = gpd.GeoDataFrame(
        {"uid": [1, 2, 3, 4]},
        geometry=[
            box(0, 0, 10, 10),    # A
            box(10, 0, 20, 10),   # B: touches A's right edge (len 10)
            box(-10, 0, 0, 10),   # C: touches A's left edge (len 10)
            box(0, 10, 10, 20),   # D: touches A's top edge (len 10)
        ],
        crs=CRS,
    )

    # Act
    result = calc_shared_neighbors(gdf)
    a_row = result[result["uid"] == 1].iloc[0]

    # Assert
    assert a_row["CNT_NGHB_D"] == 3
    assert a_row["sum_length"] == pytest.approx(30.0)


@pytest.mark.parametrize(
    "gap, expect_touching",
    [
        (0.0, True),      # exactly coincident edges
        (0.0001, True),   # sub-millimeter gap: should snap together
        (0.5, False),     # 0.5m gap: a real gap, must NOT be bridged
    ],
)
def test_calc_shared_neighbors_precision_snapping(gap, expect_touching):
    """The 0.001m precision grid should absorb floating-point-scale gaps 
    but must NOT bridge genuine gaps. This guards against accidentally 
    reverting to a coarse buffer approach, which would incorrectly 
    treat real gaps as touching."""
    # Arrange
    gdf = gpd.GeoDataFrame(
        {"uid": [1, 2]},
        geometry=[box(0, 0, 10, 10), box(10 + gap, 0, 20 + gap, 10)],
        crs=CRS,
    )

    # Act
    result = calc_shared_neighbors(gdf)

    # Assert
    expected_cnt = 1 if expect_touching else 0
    assert result["CNT_NGHB_D"].tolist() == [expected_cnt, expected_cnt]


def test_calc_shared_neighbors_does_not_mutate_input():
    """The function must return a copy, leaving the caller's original
    GeoDataFrame untouched."""
    # Arrange
    gdf = gpd.GeoDataFrame(
        {"uid": [1, 2]},
        geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)],
        crs=CRS,
    )

    # Act
    calc_shared_neighbors(gdf)

    # Assert
    assert "CNT_NGHB_D" not in gdf.columns
    assert "sum_length" not in gdf.columns


def test_calc_shared_wall_ratio_normal_case():
    """Straightforward ratio calculation."""
    # Arrange
    gdf = gpd.GeoDataFrame(
        {"sum_length": [10.0, 5.0], "C": [40.0, 20.0]},
        geometry=[Point(0, 0), Point(1, 1)],
        crs=CRS,
    )

    # Act
    result = calc_shared_wall_ratio(gdf)

    # Assert
    assert result["RATIO_SW"].tolist() == pytest.approx([0.25, 0.25])


def test_calc_shared_wall_ratio_zero_shared_length_gives_zero_not_nan():
    """Edge case: a polygon with no shared wall (sum_length=0) but a valid
    circumference should have RATIO_SW=0.0, a real and meaningful value, 
    not NaN."""
    # Arrange
    gdf = gpd.GeoDataFrame(
        {"sum_length": [0.0], "C": [40.0]},
        geometry=[Point(0, 0)],
        crs=CRS,
    )

    # Act
    result = calc_shared_wall_ratio(gdf)

    # Assert
    assert result["RATIO_SW"].iloc[0] == 0.0


def test_calc_shared_wall_ratio_zero_circumference_gives_nan():
    """Edge case: a degenerate polygon with C=0 must not raise a
    division-by-zero error, and the ratio is genuinely undefined (NaN),
    not zero."""
    # Arrange
    gdf = gpd.GeoDataFrame(
        {"sum_length": [5.0], "C": [0.0]},
        geometry=[Point(0, 0)],
        crs=CRS,
    )

    # Act
    result = calc_shared_wall_ratio(gdf)

    # Assert
    assert np.isnan(result["RATIO_SW"].iloc[0])


def test_calc_shared_wall_ratio_does_not_mutate_input():
    """The function must return a copy, leaving the caller's original
    GeoDataFrame untouched."""
    # Arrange
    gdf = gpd.GeoDataFrame(
        {"sum_length": [10.0], "C": [40.0]},
        geometry=[Point(0, 0)],
        crs=CRS,
    )

    # Act
    calc_shared_wall_ratio(gdf)

    # Assert
    assert "RATIO_SW" not in gdf.columns