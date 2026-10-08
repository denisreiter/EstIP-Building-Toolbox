"""Dasymetric mapping of population to buildings."""

# Copyright (c) 2026 Denis Reiter
# Licensed under the EUPL-1.2. See the LICENSE file in the project root for details.

import geopandas as gpd
import pandas as pd


def select_residential_buildings(
    buildings: gpd.GeoDataFrame,
    min_area: float = 40.0,
    min_volume: float = 250.0,
) -> gpd.GeoDataFrame:
    """Select residential buildings eligible for population mapping.

    A building qualifies as residential for population-mapping purposes if
    it meets minimum area and volume thresholds and carries one of the
    residential `is_res` categories. The `is_res` values themselves are
    fixed outputs of earlier pipeline steps (Script 1 base classification,
    extended in this script's `classify_residential_extended`) and are not
    user-configurable.

    Parameters
    ----------
    buildings : gpd.GeoDataFrame
        Buildings with `A` (area), `V` (volume), and `is_res` fields. `A` is
        read as a stored field rather than recomputed from geometry, since
        geometries are not modified anywhere in the pipeline after Script 2
        derives it.
    min_area : float, default 40.0
        Minimum footprint area (m^2) for a building to be considered
        residential.
    min_volume : float, default 250.0
        Minimum volume (m^3) for a building to be considered residential.

    Returns
    -------
    gpd.GeoDataFrame
        Subset of `buildings` meeting the residential criteria.
    """
    residential_categories = [1, 2, 11, 22]

    is_large_enough = buildings["A"] >= min_area
    is_voluminous_enough = buildings["V"] >= min_volume
    is_residential_category = buildings["is_res"].isin(residential_categories)

    residential_buildings = buildings[
        is_large_enough & is_voluminous_enough & is_residential_category
    ].copy()

    return residential_buildings


def join_cell_population(
    buildings: gpd.GeoDataFrame,
    census_grid: gpd.GeoDataFrame,
    grid_id_field: str,
    population_field: str,
) -> gpd.GeoDataFrame:
    """Join census grid population values onto buildings by cell code.

    Matches each building's `MAPPED_CELLCODE` (assigned in Script 5) to the
    census grid's cell identifier field, attaching the grid's population
    field. Cells with no census data (data-protection gaps, or areas
    outside grid coverage) result in `NaN`, which is resolved later in
    `finalize_population_field`.

    Parameters
    ----------
    buildings : gpd.GeoDataFrame
        Buildings with a `MAPPED_CELLCODE` field.
    census_grid : gpd.GeoDataFrame
        Census grid with a cell identifier field and a population field.
    grid_id_field : str
        Name of the cell identifier field in `census_grid` (matched against
        `MAPPED_CELLCODE` on the buildings).
    population_field : str
        Name of the population field in `census_grid`.

    Returns
    -------
    gpd.GeoDataFrame
        Copy of `buildings` with an added population field (named
        `population_field`, `NaN` where no census data exists).
    """
    population_lookup = pd.DataFrame(census_grid[[grid_id_field, population_field]])

    buildings = buildings.merge(
        population_lookup,
        left_on="MAPPED_CELLCODE",
        right_on=grid_id_field,
        how="left",
    )

    if grid_id_field != "MAPPED_CELLCODE":
        buildings = buildings.drop(columns=grid_id_field)

    return buildings


def calc_volume_weighted_population(
    residential_buildings: gpd.GeoDataFrame,
    population_field: str,
) -> gpd.GeoDataFrame:
    """Apportion cell population to residential buildings by volume share.

    Within each cell, a building's population share is proportional to its
    volume relative to the summed volume of all residential buildings in
    that cell (larger buildings are assumed to house more people).

    Parameters
    ----------
    residential_buildings : gpd.GeoDataFrame
        Residential buildings (output of `select_residential_buildings`)
        with `V` (volume), `MAPPED_CELLCODE`, and a population field (from
        `join_cell_population`).
    population_field : str
        Name of the cell population field, as joined by
        `join_cell_population`.

    Returns
    -------
    gpd.GeoDataFrame
        Copy of `residential_buildings` with an added `MAPPED_POPULATION`
        field. `NaN` where the cell has no census population data.
    """
    residential_buildings = residential_buildings.copy()

    cell_volume_totals = residential_buildings.groupby("MAPPED_CELLCODE")["V"].transform("sum")
    volume_share = residential_buildings["V"] / cell_volume_totals

    residential_buildings["MAPPED_POPULATION"] = (
        residential_buildings[population_field] * volume_share
    )

    return residential_buildings


def finalize_population_field(
    buildings: gpd.GeoDataFrame,
    residential_buildings_with_population: gpd.GeoDataFrame,
    field_id: str,
    population_field: str,
) -> gpd.GeoDataFrame:
    """Attach final population values to the full building set.

    Merges `MAPPED_POPULATION` (computed in `calc_volume_weighted_population`
    for the residential subset) back onto all buildings. Buildings that were
    non-residential, excluded by the area/volume thresholds, or located in a
    cell with no census data all receive `0.0`.

    Parameters
    ----------
    buildings : gpd.GeoDataFrame
        Full building set (all buildings, not just residential).
    residential_buildings_with_population : gpd.GeoDataFrame
        Output of `calc_volume_weighted_population`, containing `field_id`
        and `MAPPED_POPULATION` for residential buildings only.
    field_id : str
        Name of the unique building identifier field used to join the two
        building sets.
    population_field : str
        Name of the intermediate cell population field (from
        `join_cell_population`), dropped from the output if present.

    Returns
    -------
    gpd.GeoDataFrame
        Copy of `buildings` with a final `MAPPED_POPULATION` field
        (float, no NaN). The intermediate `population_field`, if present,
        is dropped.
    """
    population_lookup = residential_buildings_with_population[[field_id, "MAPPED_POPULATION"]]

    buildings = buildings.merge(population_lookup, on=field_id, how="left")
    buildings["MAPPED_POPULATION"] = buildings["MAPPED_POPULATION"].fillna(0.0)

    if population_field in buildings.columns:
        buildings = buildings.drop(columns=population_field)

    return buildings


def map_population(
    buildings: gpd.GeoDataFrame,
    census_grid: gpd.GeoDataFrame,
    field_id: str,
    grid_id_field: str,
    population_field: str,
    min_area: float = 40.0,
    min_volume: float = 250.0,
) -> gpd.GeoDataFrame:
    """Map census grid population to buildings, weighted by building volume.

    Orchestrates the full population-mapping sequence:

    1. Select residential buildings eligible for population mapping
       (`select_residential_buildings`).
    2. Join census grid population onto the residential subset
       (`join_cell_population`).
    3. Apportion each cell's population to its residential buildings by
       volume share (`calc_volume_weighted_population`).
    4. Merge the result back onto the full building set, filling all
       non-residential, excluded, or no-data buildings with `0.0`
       (`finalize_population_field`).

    Parameters
    ----------
    buildings : gpd.GeoDataFrame
        Full building set with `is_res`, `FNC_CODE`, `V`, `A`, and
        `MAPPED_CELLCODE` fields. `BLCK_TYPE` is used if present.
    census_grid : gpd.GeoDataFrame
        Census grid with a cell identifier field and a population field.
    field_id : str
        Name of the unique building identifier field.
    grid_id_field : str
        Name of the cell identifier field in `census_grid`.
    population_field : str
        Name of the population field in `census_grid`.
    min_area : float, default 40.0
        Minimum footprint area (m^2) for residential eligibility.
    min_volume : float, default 250.0
        Minimum volume (m^3) for residential eligibility.

    Returns
    -------
    gpd.GeoDataFrame
        Copy of `buildings` with a final `MAPPED_POPULATION` field
        (float, no NaN).
    """
    residential_buildings = select_residential_buildings(
        buildings, min_area=min_area, min_volume=min_volume
    )
    residential_buildings = join_cell_population(
        residential_buildings, census_grid, grid_id_field, population_field
    )
    residential_buildings = calc_volume_weighted_population(
        residential_buildings, population_field
    )

    buildings = finalize_population_field(
        buildings, residential_buildings, field_id=field_id, population_field=population_field
    )

    return buildings