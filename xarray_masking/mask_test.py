import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from mask_by_state import create_state_intersection_mask


def create_conus_test_dataset(start_date="2023-01-01", num_days=10, resolution=0.125):
    # Define CONUS bounding box
    lon_min, lon_max = -124.7844079, -66.9513812
    lat_min, lat_max = 24.7433195, 49.3457868

    # Create coordinates
    lons = np.arange(lon_min, lon_max + resolution, resolution)
    lats = np.arange(lat_min, lat_max + resolution, resolution)
    times = pd.date_range(start=start_date, periods=num_days)

    # Create random data
    data = np.random.rand(len(times), len(lats), len(lons))

    # Create the dataset
    ds = xr.Dataset(
        data_vars={
            "temperature": (
                ["time", "lat", "lon"],
                data,
                {"units": "celsius", "long_name": "Surface Temperature"},
            ),
        },
        coords={
            "lon": ("lon", lons, {"units": "degrees_east", "long_name": "Longitude"}),
            "lat": ("lat", lats, {"units": "degrees_north", "long_name": "Latitude"}),
            "time": ("time", times),
        },
        attrs={
            "description": "Test dataset for CONUS with random temperature values",
            "creation_date": pd.Timestamp.now().isoformat(),
        },
    )

    return ds


# Create the test dataset
conus_ds = create_conus_test_dataset()

# Set the state IDs and threshold for masking
state_ids = ["CA", "OR", "WA"]
threshold = 0.5

# Create the mask for the states
masked_dataset = create_state_intersection_mask(state_ids, conus_ds, threshold)

# Set up the map projection (e.g., PlateCarree for lat/lon data)
projection = ccrs.PlateCarree()

# Create a figure with Cartopy's projection
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": projection})

# Add coastlines, state borders, and ocean features
ax.coastlines(resolution="10m")
ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.add_feature(cfeature.STATES, linestyle="-", edgecolor="black")

# Plot the mask data
masked_dataset.mask.plot.pcolormesh(
    ax=ax,
    transform=ccrs.PlateCarree(),  # Set transform to match the projection
    cmap="coolwarm",  # Set the colormap
    add_colorbar=True,  # Add a colorbar to the plot
)
# Add titles and labels
plt.title("Masked Grid with Coastlines & State Boundaries")
plt.show()
