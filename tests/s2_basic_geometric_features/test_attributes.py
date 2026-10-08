"""Tests for lod2_toolbox.s2_geometric_features.attributes."""

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from scripts.s2_basic_geometric_features.attributes import (
    copy_basic_attributes,
    add_hull_metrics,
)


SQUARE = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])  # area 100, length 40


def make_gdf(**extra_columns):
    """Build a single-row GeoDataFrame with a known square geometry plus
    whatever extra attribute columns a test needs."""
    return gpd.GeoDataFrame(
        {**extra_columns}, geometry=[SQUARE], crs="EPSG:3035"
    )



def test_copy_basic_attributes_area_and_circumference_from_geometry():
    # Arrange
    gdf = make_gdf()

    # Act
    result = copy_basic_attributes(gdf)

    # Assert
    assert result["A"].iloc[0] == pytest.approx(100)
    assert result["C"].iloc[0] == pytest.approx(40)


def test_copy_basic_attributes_height_and_volume_included_when_given():
    # Arrange
    gdf = make_gdf(HEIGHT=[5.0], VOLUME=[500.0])

    # Act
    result = copy_basic_attributes(gdf, field_height="HEIGHT", field_volume="VOLUME")

    # Assert
    assert result["H"].iloc[0] == pytest.approx(5.0)
    assert result["V"].iloc[0] == pytest.approx(500.0)


def test_copy_basic_attributes_height_and_volume_omitted_when_not_given():
    # Arrange
    gdf = make_gdf()

    # Act
    result = copy_basic_attributes(gdf)

    # Assert
    assert "H" not in result.columns
    assert "V" not in result.columns


def test_copy_basic_attributes_height_only_leaves_volume_out():
    # Arrange
    gdf = make_gdf(HEIGHT=[5.0])

    # Act
    result = copy_basic_attributes(gdf, field_height="HEIGHT")

    # Assert
    assert "H" in result.columns
    assert "V" not in result.columns


def test_add_hull_metrics_computes_when_all_fields_given():
    # Arrange
    gdf = make_gdf(WALL=[200.0], ROOF=[110.0], VOLUME=[500.0])

    # Act
    result = add_hull_metrics(
        gdf, field_wallarea="WALL", field_roofarea="ROOF", field_volume="VOLUME"
    )

    # Assert
    # hull_area = footprint area (100) + wall area (200) + roof area (110)
    assert result["hull_area"].iloc[0] == pytest.approx(410.0)
    assert result["RATIO_HLL_V"].iloc[0] == pytest.approx(410.0 / 500.0)


@pytest.mark.parametrize(
    "field_wallarea, field_roofarea, field_volume",
    [
        (None, "ROOF", "VOLUME"),   # wall area missing
        ("WALL", None, "VOLUME"),   # roof area missing
        ("WALL", "ROOF", None),     # volume missing
        (None, None, None),         # nothing supplied
    ],
)
def test_add_hull_metrics_skips_when_any_required_field_missing(
    field_wallarea, field_roofarea, field_volume, capsys
):
    # Arrange
    gdf = make_gdf(WALL=[200.0], ROOF=[110.0], VOLUME=[500.0])

    # Act
    result = add_hull_metrics(
        gdf,
        field_wallarea=field_wallarea,
        field_roofarea=field_roofarea,
        field_volume=field_volume,
    )

    # Assert
    assert "hull_area" not in result.columns
    assert "RATIO_HLL_V" not in result.columns
    captured = capsys.readouterr()
    assert "Skipping hull-to-volume ratio" in captured.out


def test_add_hull_metrics_does_not_mutate_input_columns():
    # Arrange
    gdf = make_gdf(WALL=[200.0], ROOF=[110.0], VOLUME=[500.0])

    # Act
    add_hull_metrics(gdf, field_wallarea=None, field_roofarea=None, field_volume=None)

    # Assert
    # Original source columns should be untouched when the calculation is skipped.
    assert gdf["WALL"].iloc[0] == pytest.approx(200.0)
    assert gdf["ROOF"].iloc[0] == pytest.approx(110.0)
    assert gdf["VOLUME"].iloc[0] == pytest.approx(500.0)