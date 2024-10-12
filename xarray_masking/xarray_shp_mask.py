#!/usr/bin/env python3

"""
A command-line tool for masking datasets based on geographical boundaries defined by state shapefiles.

Usage:
    python mask_tool.py [OPTIONS]

Options:
    -h, --help                  Show this help message and exit.
    --shapefile SHAPEFILE       Specify the path to the shapefile (default: ./Shapefiles/cb_2023_us_state_20m.shp).
    --dataset DATASET           Specify the path to the NetCDF dataset, must contain lat/lon dims only (no Time) (default: ./measurements_short.nc).
    --state-ids STATE_IDS       Comma-separated list of state codes (default: TX, OK, KS).
    --threshold THRESHOLD       Intersection threshold (0.0-1), portion of the cell that must be intersected to be included (default: 0.5).
    --save-path SAVE_PATH       Path to save the masked dataset (defaults to current working directory).

Examples:
    python mask_tool.py --shapefile ./my_shapefile.shp --dataset ./my_data.nc --state-ids TX,CA --threshold 0.75
"""
# Skye Leake Oct, 2024

import os
import numpy as np
import geopandas as gpd
import xarray as xr
import dask.array as da
import warnings
import matplotlib.pyplot as plt
import configparser
from shapely.geometry import box
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from distributed import Client
from matplotlib.colors import ListedColormap
import time
import argparse

# Set up a Dask environment
client = Client(n_workers=2, threads_per_worker=4, memory_limit="32GB")
print(client)

# Load default configurations from the config file
config = configparser.ConfigParser()
config.read('mask.config')

default_shapefile_path = config['DEFAULTS']['shapefile_path']
default_dataset_path = config['DEFAULTS']['dataset_path']
default_state_ids = config['DEFAULTS']['state_ids']
default_threshold = float(config['DEFAULTS']['threshold'])

def validate_path(prompt: str, default: str) -> str:
    """Prompt the user for a valid file path, using a default if input is empty."""
    while True:
        path = input(prompt).strip() or default
        if os.path.exists(path):
            return path
        print(f"Error: The path '{path}' does not exist. Please enter a valid path.")

def create_state_intersection_mask_dask(
    state_ids: list[str],
    dataset: xr.Dataset,
    threshold: float,
    shapefile_path: str = "./Shapefiles/cb_2023_us_state_20m.shp",
    projected_crs: str = "EPSG:5070",
) -> xr.Dataset:
    print("Starting masking...")

    # Load US states shapefile and reproject to the desired CRS
    print("Loading and reprojecting shapefile to the projected CRS...")
    us_states = gpd.read_file(shapefile_path).to_crs(projected_crs)
    print("Shapefile loaded and reprojected successfully.")

    # Convert state IDs to uppercase and filter selected states
    state_ids = [state_id.upper() for state_id in state_ids]
    print(f"Selected state IDs: {state_ids}")

    selected_states = us_states[us_states["STUSPS"].isin(state_ids)]
    print(f"Filtered selected states. Number of states found: {len(selected_states)}")

    # Confirm selection with user
    confirm = input("Do you want to proceed with these state codes? (yes/no) [default: yes]: ").strip().lower() or 'yes'
    if confirm != 'yes':
        print("Aborting operation.")
        return None  # Return None to indicate operation was aborted

    # Convert the lat/lon grid to Numpy arrays
    print(f"Lat/Lon grid loaded. Grid size: {dataset['lat'].shape}")

    # Calculate the differences for constructing grid boxes
    lat = dataset["lat"].values
    lon = dataset["lon"].values
    lat_diff = np.mean(np.diff(lat, axis=0))
    lon_diff = np.mean(np.diff(lon, axis=1))
    print(f"Calculated lat/lon differences. lat_diff: {lat_diff}, lon_diff: {lon_diff}")

    # Create grid cells (bounding boxes)
    print("Creating grid cells (bounding boxes)...")
    lon_min = lon - lon_diff / 2
    lon_max = lon + lon_diff / 2
    lat_min = lat - lat_diff / 2
    lat_max = lat + lat_diff / 2

    grid_cells = [box(lon_min[i, j], lat_min[i, j], lon_max[i, j], lat_max[i, j])
                  for i in range(lat.shape[0])
                  for j in range(lon.shape[1])]
    print(f"Grid cells created. Total number of cells: {len(grid_cells)}")

    # Convert the grid cells to a GeoSeries and project them
    print("Projecting grid cells to the projected CRS...")
    grid_cells_gdf = gpd.GeoSeries(grid_cells, crs="EPSG:4326").to_crs(projected_crs)
    print(f"Grid cells projected to {projected_crs}.")

    # Suppress RuntimeWarnings during the intersection calculation
    print("Starting intersection calculation between grid cells and states...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        intersections = grid_cells_gdf.intersection(selected_states.unary_union)
        intersection_areas = intersections.area
        cell_areas = grid_cells_gdf.area
    print("Intersection calculation completed.")

    # Calculate the mask based on the threshold
    print("Calculating mask based on the threshold...")
    mask = (intersection_areas / cell_areas).values.reshape(lat.shape) >= threshold
    print("Mask calculation completed.")

    # Convert the result back to an xarray DataArray
    print("Converting mask to xarray DataArray...")
    mask_da = xr.DataArray(
        mask,
        dims=["south_north", "west_east"],
        coords={"lat": dataset["lat"], "lon": dataset["lon"]}
    )
    print("Mask DataArray created successfully.")

    print("Function completed successfully.")
    return xr.Dataset({"mask": mask_da})

def plot_mask(masked_dataset, mask_data):
    """Plot the mask using Cartopy."""
    # Set up the map projection
    projection = ccrs.PlateCarree()

    # Create a figure with Cartopy's projection
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": projection})

    # Add coastlines, state borders, and ocean features
    ax.coastlines(resolution="10m")
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.STATES, linestyle="-", edgecolor="black")

    # Convert the boolean mask to integers (0 for False, 1 for True)
    masked_values = np.where(mask_data, 1, np.nan)  # Set False values to NaN for better visibility

    # Define a binary colormap (blue for False and red for True)
    cmap = ListedColormap(['blue', 'red'])

    # Plot using pcolormesh directly on the 2D mask data
    pcm = ax.pcolormesh(masked_dataset["lon"], masked_dataset["lat"], masked_values,
                        transform=ccrs.PlateCarree(), cmap=cmap, shading='auto')

    # Add titles and labels
    plt.title("Masked States")
    plt.show()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Mask datasets based on geographical boundaries defined by state shapefiles.')
    
    # Define command-line arguments
    parser.add_argument('--shapefile', type=str, default=default_shapefile_path,
                        help='Specify the path to the shapefile.')
    parser.add_argument('--dataset', type=str, default=default_dataset_path,
                        help='Specify the path to the NetCDF dataset.')
    parser.add_argument('--state-ids', type=str, default=default_state_ids,
                        help='Comma-separated list of state codes.')
    parser.add_argument('--threshold', type=float, default=default_threshold,
                        help='Intersection threshold (0-1).')
    parser.add_argument('--save-path', type=str,
                        help='Path to save the masked dataset (defaults to current working directory).')

    # Parse arguments
    args = parser.parse_args()

    # Validate paths
    shapefile_path = validate_path(f"Using shapefile path: {args.shapefile}. Do you want to proceed? (yes/no) [default: yes]: ", args.shapefile)
    dataset_path = validate_path(f"Using dataset path: {args.dataset}. Do you want to proceed? (yes/no) [default: yes]: ", args.dataset)

    # Load dataset
    ds = xr.open_dataset(dataset_path)

    # Get state codes
    state_ids = [state_id.strip() for state_id in args.state_ids.split(',')]
    
    # Get and validate threshold input
    threshold = args.threshold
    if not (0 <= threshold <= 1):
        print("Error: Threshold must be a float between 0 and 1. Exiting.")
        return

    # Create the mask using the state boundaries
    masked_dataset = create_state_intersection_mask_dask(state_ids, ds, threshold, shapefile_path)

    if masked_dataset is None:
        return  # Exit if the operation was aborted

    # Prepare save path and filename
    save_path_input = args.save_path
    if not save_path_input:
        save_path_input = os.getcwd()  # Default to current working directory
    else:
        save_path_input = os.path.abspath(save_path_input)  # Make it absolute

    # Create the save file name based on provided state IDs and current time
    if os.path.isdir(save_path_input):
        state_str = "_".join(state_ids)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_path_input, f"masked_{state_str}_{timestamp}.nc")
    else:
        save_path = save_path_input

    # Check if the file already exists
    if os.path.exists(save_path):
        confirm_save = input(f"The file '{save_path}' already exists. Do you want to overwrite it? (yes/no) [default: no]: ").strip().lower() or 'no'
        if confirm_save != 'yes':
            print("Aborting save operation.")
            return  # Exit if the operation was aborted

    # Ask user if they want to preview or fully plot the mask before saving
    plot_option = input("Do you want to preview (p) or fully plot (f) the mask before saving? (default: p): ").strip().lower() or 'p'
    
    if plot_option in ['f', 'full']:
        plot_mask(ds, masked_dataset["mask"].values)
    else:
        print("Previewing mask (not full plotting)...")
        plt.imshow(masked_dataset["mask"].values, cmap='gray', interpolation='nearest', origin="lower")
        plt.colorbar()
        plt.title("Preview Mask")
        plt.show()

    # Save the masked dataset
    masked_dataset.to_netcdf(save_path)
    print(f"Masked dataset saved to {save_path}.")

if __name__ == "__main__":
    main()
