"""Functions for cleaning building geometries and attributes."""

# Copyright (c) 2026 Denis Reiter
# Licensed under the EUPL-1.2. See the LICENSE file in the project root for details.

import geopandas as gpd


def remove_non_buildings(
        gdf: gpd.GeoDataFrame, 
        function_field: str
    ) -> gpd.GeoDataFrame:
    """Keep only rows classified as buildings.

    CityGML function codes starting with '3' denote buildings.
    Rows with any other code (e.g. bridges, vegetation) are removed.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame (either building footprints or parts).
    function_field : str
        Name of the column holding the function code.

    Returns
    -------
    gpd.GeoDataFrame
        A new GeoDataFrame containing only building rows.
    """
    is_building = gdf[function_field].astype(str).str.startswith("3")
    return gdf[is_building].copy()