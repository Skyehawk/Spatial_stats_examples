"""

"""

from __future__ import annotations

import glob

# Standard library imports
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# Cartopy stuff
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import dask.array as da

# Plotting packages
import matplotlib.pyplot as plt

# Data packages
import numpy as np
import pandas as pd
import xarray as xr

# Dask for parallel processing
from distributed import Client

# Signal Processing Stuff
from scipy.signal import correlate

# Output folder
output_folder = Path(tempfile.mkdtemp())

# Set up a Dask client
client = Client(n_workers=2, threads_per_worker=4, memory_limit="64GB")


def load_datasets(
    path0: str, path1: str, path_grid: str
) -> tuple[xr.Dataset, xr.Dataset]:
    files0 = glob.glob(path0, recursive=True)
    files1 = glob.glob(path1, recursive=True)

    grid = xr.open_dataset(os.path.join(path_grid, "geo_em.d01.nc"))

    da0 = xr.open_mfdataset(
        files0, chunks="auto", combine="by_coords", engine="netcdf4"
    )
    da0 = da0.assign_coords(lon=grid.CLONG[0], lat=grid.CLAT[0]).rename(
        {"Time": "time"}
    )

    da1 = xr.open_mfdataset(
        files1, chunks="auto", combine="by_coords", engine="netcdf4"
    )
    da1 = da1.assign_coords(lon=grid.CLONG[0], lat=grid.CLAT[0]).rename(
        {"Time": "time"}
    )

    return da0, da1


def coarsen_data(da: xr.Dataset, factor: int) -> xr.Dataset:
    return da.coarsen(south_north=factor, west_east=factor, boundary="trim").mean()


def consecutive_wet_days(wet_days_array: xr.DataArray, min_days: int) -> xr.DataArray:
    wet_days_np = wet_days_array.values
    consecutive_mask = np.zeros_like(wet_days_np, dtype=bool)

    for i in range(wet_days_np.shape[1]):
        for j in range(wet_days_np.shape[2]):
            time_series = wet_days_np[:, i, j]
            temp_mask = np.zeros_like(time_series, dtype=bool)
            count = 0
            for t in range(len(time_series)):
                if time_series[t]:
                    count += 1
                    if count >= min_days:
                        temp_mask[t - min_days + 1 : t + 1] = True
                else:
                    count = 0
            consecutive_mask[:, i, j] = temp_mask

    return xr.DataArray(
        consecutive_mask, dims=wet_days_array.dims, coords=wet_days_array.coords
    )


def extract_and_aggregate_mam(
    ds: xr.Dataset, time_var: str = "time", output_file: str = None
) -> xr.DataArray:
    if not pd.api.types.is_datetime64_any_dtype(ds[time_var].values):
        ds[time_var] = pd.to_datetime(ds[time_var].values)

    selected_months = ds.sel({time_var: ds[time_var].dt.month.isin([3, 4, 5])})
    daily_data = selected_months.groupby(f"{time_var}.dayofyear").mean()

    if output_file:
        daily_data.to_netcdf(output_file)

    return daily_data


def compute_cross_correlation(
    baseline_da: xr.DataArray, future_da: xr.DataArray
) -> xr.DataArray:
    baseline_array = baseline_da.values
    future_array = future_da.values
    correlation = np.full((baseline_array.shape[1], baseline_array.shape[2]), np.nan)

    for i in range(baseline_array.shape[1]):
        for j in range(baseline_array.shape[2]):
            baseline_series = baseline_array[:, i, j]
            future_series = future_array[:, i, j]
            corr = correlate(baseline_series, future_series, mode="full")
            correlation[i, j] = np.max(corr)

    return xr.DataArray(
        correlation,
        coords=[baseline_da.south_north.values, baseline_da.west_east.values],
        dims=["south_north", "west_east"],
    )


def compute_lag(baseline_da: xr.DataArray, future_da: xr.DataArray) -> xr.DataArray:
    baseline_array = baseline_da.values
    future_array = future_da.values
    lags = np.full((baseline_array.shape[1], baseline_array.shape[2]), np.nan)

    for i in range(baseline_array.shape[1]):
        for j in range(baseline_array.shape[2]):
            baseline_series = baseline_array[:, i, j]
            future_series = future_array[:, i, j]
            corr = correlate(baseline_series, future_series, mode="full")
            lag = np.arange(-len(baseline_series) + 1, len(baseline_series))
            lags[i, j] = lag[np.argmax(corr)]

    return xr.DataArray(
        lags,
        coords=[baseline_da.south_north.values, baseline_da.west_east.values],
        dims=["south_north", "west_east"],
    )


def plot_correlation_and_lags(correlation: xr.DataArray, lags: xr.DataArray, lon, lat):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 12), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    corr_plot = ax1.pcolormesh(
        lon, lat, correlation, transform=ccrs.PlateCarree(), cmap="bwr", vmin=-1, vmax=1
    )
    ax1.coastlines()
    ax1.add_feature(cfeature.BORDERS)
    ax1.set_title("Absolute Correlations (CWD)")
    fig.colorbar(corr_plot, ax=ax1, orientation="vertical", label="Correlation")

    lag_plot = ax2.pcolormesh(
        lon, lat, lags, transform=ccrs.PlateCarree(), cmap="BrBG", vmin=-10, vmax=10
    )
    ax2.coastlines()
    ax2.add_feature(cfeature.BORDERS)
    ax2.set_title("Lag Times (Days)")
    fig.colorbar(lag_plot, ax=ax2, orientation="vertical", label="Lag (Days)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # --- Data Loading and preprocessing ---
    # File paths for datasets
    path0 = "/home/scratch/sleake/AFWA_TOTPRECIP_PrecipDays_hist/*1p000.nc"
    path1 = "/home/scratch/sleake/AFWA_TOTPRECIP_PrecipDays_eoc_4p5/*1p000.nc"
    path_grid = "/home/scratch/sleake/WRF-BCC_Geogrid"

    # --- Load datasets (just put in a function for cleanliness) ---
    #     We assume data are the same lat/lon dimentions and time dimention is "Time",
    #     which we rename to "time", lat/lon coords can be added (if not already in data)
    #     via a WRF geogrid geo_em*.nc file
    da0, da1 = load_datasets(path0, path1, path_grid)

    # Coarsen data, assumes you want to coarsen by the same amount on lat and lon
    #   We do this first to make the computation far less intensive, however this may
    #   not be the correct order for your use case.
    factor = 21
    da0 = coarsen_data(da0, factor)
    da1 = coarsen_data(da1, factor)

    # --- Data masking and cleaning for our purposes (in this case CWD frequency) ---
    # Wet day threshold and masking
    wet_day_threshold = 0.0  # mm/day to be labeled a wet day
    wet_days0 = da0.AFWA_TOTPRECIP > wet_day_threshold
    wet_days1 = da1.AFWA_TOTPRECIP > wet_day_threshold

    # Consecutive wet days
    min_consecutive_days = (
        2  # minimum length of run for all days within run to be labled as consecutive
    )
    consecutive_wet_mask0 = consecutive_wet_days(wet_days0, min_consecutive_days)
    consecutive_wet_mask1 = consecutive_wet_days(wet_days1, min_consecutive_days)

    # Extract and aggregate data
    da0_cleaned = extract_and_aggregate_mam(
        consecutive_wet_mask0, time_var="time", output_file="daily_data_mam0.nc"
    )
    da1_cleaned = extract_and_aggregate_mam(
        consecutive_wet_mask1, time_var="time", output_file="daily_data_mam1.nc"
    )

    # --- Perform our analysis ---
    # Cross-correlation computation on cumulative distribution function
    #    We need to be careful with how data are aggregated. Yearly should be fine,
    #    multi-yearly is *likely* not and should be averaged. Similarly if we have large
    #    temporal gaps (i.e. eveluating several years of MAM data) we need to get an average
    correlation = compute_cross_correlation(da0_cleaned, da1_cleaned)
    lags = compute_lag(da0_cleaned, da1_cleaned)

    # Plot results
    plot_correlation_and_lags(correlation, lags, da0_cleaned.lon, da0_cleaned.lat)
