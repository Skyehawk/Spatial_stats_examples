import os

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import box


def create_state_intersection_mask(
    state_ids,
    dataset,
    threshold,
    shapefile_path="./Shapefiles/cb_2023_us_state_20m/cb_2023_us_state_20m.shp",
    projected_crs="EPSG:5070",  # Project to CONUS Albers Equal Area
):
    # Check if the shapefile exists
    if not os.path.exists(shapefile_path):
        raise FileNotFoundError(f"Shapefile not found at {shapefile_path}")

    # Load US states data
    us_states = gpd.read_file(shapefile_path)

    # Reproject to a projected CRS for accurate area calculations
    us_states = us_states.to_crs(projected_crs)

    # Convert state IDs to uppercase
    state_ids = [state_id.upper() for state_id in state_ids]

    # Filter states based on input state_ids
    selected_states = us_states[us_states["STUSPS"].isin(state_ids)]

    # Convert the dataset's lat/lon to a projected CRS
    lat, lon = np.meshgrid(dataset.lat.values, dataset.lon.values, indexing="ij")
    lon_lat = np.column_stack([lon.ravel(), lat.ravel()])
    points_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(lon_lat[:, 0], lon_lat[:, 1]), crs="EPSG:4326"
    )
    points_gdf = points_gdf.to_crs(projected_crs)

    # Create a binary mask with the same shape as the input dataset lat/lon grid
    mask = xr.DataArray(
        np.zeros((len(dataset.lat), len(dataset.lon)), dtype=bool),
        dims=["lat", "lon"],
        coords={"lat": dataset.lat, "lon": dataset.lon},
    )

    # Iterate through each grid cell
    for i in range(len(dataset.lat)):
        for j in range(len(dataset.lon)):
            # Create a box for the current grid cell
            cell_box = box(
                dataset.lon[j].item() - dataset.lon.diff("lon")[0].item() / 2,
                dataset.lat[i].item() - dataset.lat.diff("lat")[0].item() / 2,
                dataset.lon[j].item() + dataset.lon.diff("lon")[0].item() / 2,
                dataset.lat[i].item() + dataset.lat.diff("lat")[0].item() / 2,
            )

            # Project the cell box to the same CRS as the shapefile
            cell_box = gpd.GeoSeries([cell_box], crs="EPSG:4326").to_crs(projected_crs)

            # Calculate intersection area with selected states
            intersection_area = selected_states.geometry.intersection(
                cell_box[0]
            ).area.sum()
            cell_area = cell_box[0].area

            # Set mask value based on threshold
            if intersection_area / cell_area >= threshold:
                mask[i, j] = True

    # Return the dataset containing only the mask variable
    return xr.Dataset({"mask": mask})
