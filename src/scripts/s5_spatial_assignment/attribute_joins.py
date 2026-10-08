"""Assign spatial typology attributes to building footprints."""

# Copyright (c) 2026 Denis Reiter
# Licensed under the EUPL-1.2. See the LICENSE file in the project root for details.

import geopandas as gpd


def assign_municipality_typologies(
    buildings,
    municipalities,
    state_field=None,
    region_field=None,
    mun_field=None,
    typology_field=None,
):
    """
    Assign state, region, municipality, and special-type codes to building
    footprints, based on which municipality polygon contains each
    building's representative point.

    All four source fields are optional; only the ones provided are
    fetched, cast, and included in the output. At least one must be given.
    Provided fields are cast to pandas' nullable text dtype.

    Parameters
    ----------
    buildings : geopandas.GeoDataFrame
        Building footprints to receive the typology attributes.
    municipalities : geopandas.GeoDataFrame
        Municipality polygons carrying the source attribute fields.
        Must not contain overlapping polygons - see Raises.
    state_field, region_field, mun_field, typology_field : str, optional
        Column names in `municipalities` holding the state, region,
        municipality, and special-type codes respectively. Any not
        provided is simply omitted from the output.

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of `buildings` with one new text column per field provided
        (STATE, REGION, MUNICIP, REG_TYPE). Buildings whose representative
        point falls outside every municipality polygon receive nulls.

    Raises
    ------
    ValueError
        If none of the four fields are provided, or if any building's
        representative point falls inside more than one municipality
        polygon (indicating overlapping/duplicate geometries in
        `municipalities` rather than a legitimate multi-match).
    """
    field_map = {
        "STATE": state_field,
        "REGION": region_field,
        "MUNICIP": mun_field,
        "REG_TYPE": typology_field,
    }
    # Keep only the fields actually provided.
    field_map = {output: source for output, source in field_map.items() if source is not None}

    if not field_map:
        raise ValueError(
            "At least one of state_field, region_field, mun_field, or "
            "typology_field must be provided."
        )

    value_fields = list(field_map.values())

    municipalities = municipalities.astype({field: "string" for field in value_fields})
    mun_subset = municipalities[value_fields + ["geometry"]]

    points = buildings.copy()
    points["geometry"] = buildings.geometry.representative_point()

    joined = gpd.sjoin(points, mun_subset, how="left", predicate="within")

    duplicate_count = joined.index.duplicated().sum()
    if duplicate_count:
        raise ValueError(
            f"  --> {duplicate_count} buildings' representative points fell inside "
            "      more than one municipality polygon - check the municipality "
            "      layer for overlapping or duplicate geometries."
        )

    joined = joined.drop(columns=["index_right"])
    joined["geometry"] = buildings.geometry.values

    reverse_map = {source: output for output, source in field_map.items()}
    resolved = joined.rename(columns=reverse_map)

    return resolved