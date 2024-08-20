# Imports
import os
from typing import Optional

import contextily  # Background tiles for basemap
import geopandas as gpd  # Spatial tabular data manipulation
import libpysal
import matplotlib.pyplot as plt  # Visulizaton
import numpy as np  # Numpy for numpy things
import pandas as pd  # Tabular data manipulation
import rioxarray  # Surface data manipulation
import seaborn as sns  # Visulization
import xarray as xr  # Surface data manipulation
from libpysal.weights import raster, weights  # Raster weighting
from matplotlib import colors  # Visulization
from pysal.explore import esda  # Exploratory spatial analysis
from pysal.lib import weights  # Spatial weights


def load_dataset(
    path: str,
    nc_array_name: Optional[str] = "AFWA_TOTPRECIP",
    reduction_dim: Optional[str] = "Time",
) -> xr.DataArray:
    # Check filetype based on extension to determine efficent way to load
    fpath, ext = os.path.splitext(path)
    if ext in [".tif", ".tiff", ".geotiff"]:
        print(f"loading {fpath}{ext}")
        return rioxarray.open_rasterio(path)
    elif ext in [".nc"]:
        # Update this cell to read in the netcdf file and get the DataArray
        print(f"loading {fpath}{ext}")
        surface = xr.open_dataset(path, engine="netcdf4")
        surface = surface[nc_array_name].mean(dim=reduction_dim)

        # Set spatial dimensions
        # Rename dimensions if necessary
        # surface = surface.rename({"south_north": "y", "west_east": "x"})
        # surface = surface.rio.set_spatial_dims(x_dim="x", y_dim="y")

        # print(surface.dims)

        # Add in the coodinate reference("EPSG:4326" is the most common)
        surface = surface.rio.write_crs("EPSG:4326")

        return surface

    else:
        print("File not supported")


def aggregate_grid(data: xr.DataArray, factor: int) -> xr.DataArray:
    """
    Aggregate grid resolution by a specified factor using block mean.
    Parameters:
        data (xr.DataArray): The original high-resolution DataArray.
        factor (int): Aggregation factor (e.g., 10 to go from 3.75 km to 37.5 km).
    Return:
        xr.DataArray: Aggregated DataArray with coarser resolution.
    """
    # Aggregate by averaging blocks of size `factor` x `factor`
    aggregated = data.coarsen(
        south_north=factor, west_east=factor, boundary="trim"
    ).mean()

    return aggregated


def sparse_weights_to_dense(
    weights_surface_sparse: libpysal.weights.weights.WSP,
) -> libpysal.weights.weights.W:
    """
    Return the sparse weights surface (which is a triangular irrigualr network
    (TIN)) as a more dense network of floats which is needed for LISA statistics.
    """

    w_surface_sp = weights_surface_sparse  # for readability

    # Building from low-level: convert sparse matrix to floats, build the WSP object, convert from WSP to W
    w_surface_all = weights.WSP2W(
        weights.WSP(
            w_surface_sp.sparse.astype(float), id_order=w_surface_sp.index.tolist()
        )
    )
    w_surface_all.index = w_surface_sp.index  # Assign index to new W

    return w_surface_all


# BUG: Coodinates are not being pulled through properly or spatial reference is not
# being set correctly. This results in contextily ploting from 0,0 (over Africa).
# This does not affect the integrity of the results, just plotting tiles.
def plot_LISA(
    lisa_data: xr.DataArray,
    surface_data: xr.DataArray,
    show_basemap: Optional[bool] = False,
) -> None:
    """Create a LISA Significance Plot Using Moran's Plot quadrants."""
    # LISA colors
    lc = {
        "ns": "lightgray",  # Values of 0
        "HH": "#d7191c",  # Values of 1, Red
        "LH": "#abd9e9",  # Values of 2, Light Blue
        "LL": "#2c7bb6",  # Values of 3, Blue
        "HL": "#fdae61",  # Values of 4, Orange
    }
    # Colors to a Listed Colormap
    lisa_cmap = colors.ListedColormap(
        [lc["ns"], lc["HH"], lc["LH"], lc["LL"], lc["HL"]]
    )

    # Set up figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # Subplot 1
    # Select pixels that do not have the 'nodata' value (where they are missing data)
    surface_data.where(surface_data != surface_data.rio.nodata).plot(
        ax=axes[0],
        add_colorbar=True,
        cbar_kwargs={"orientation": "horizontal"},
        robust=True,
    )

    # Subplot 2
    # Select pixels with no missing data and rescale to [0,1] by dividing by 4
    #     (4 is max value in lisa_da)
    (lisa_data.where(lisa_data != lisa_data.nodatavals) / 4).plot(
        cmap=lisa_cmap,
        ax=axes[1],
        add_colorbar=True,
        cbar_kwargs={"orientation": "horizontal"},
        robust=True,
    )

    # Aesthetics
    # Subplot titles
    titles = ["Surface values by cell (pixel)", "Surface value clusters"]
    # Apply the following to each of the two subplots
    for i in range(2):
        # Keep proportion of axes
        axes[i].axis("equal")
        # Remove axis
        axes[i].set_axis_off()
        # Add title
        axes[i].set_title(titles[i])
        if show_basemap:
            # Add basemap
            contextily.add_basemap(axes[i], crs=surface_data.rio.crs)

    # Finally, show the plot
    plt.show()


def main() -> None:
    # path = "./data/ghsl_sao_paulo_100m_2020.tif"  # Sao Paulo population in 2000, 100m resolution
    # surface = load_dataset(path)

    path = "/media/skye/LEAKE_250gb1/EAE790_Spring24/EOC8p5/DIST_WRFEOC8p5_99-100_WetPeriod_0p254base_15minPeriodCount_for_Annual.nc"
    surface = load_dataset(path, nc_array_name="PRECIP_PERIOD", reduction_dim="Time")

    #    min_lon = -120.00
    #    max_lon = -105.00
    #    min_lat = 30.00
    #    max_lat = 45.00

    #    surface = surface.where(
    #        (surface.lat >= min_lat)
    #        & (surface.lat <= max_lat)
    #        & (surface.lon >= min_lon)
    #        & (surface.lon <= max_lon),
    #        drop=True,
    #    )

    sig_p_val = 0.01  # Threshold for statistical significance

    print(f"{surface.rio.nodata=}")

    # Aggregate because our processes that make the data are larger than the data
    # themselves (data are overly fine for the application)i
    factor = 21  # 21 is closest to 80km
    # Perform the aggregation
    surface = aggregate_grid(surface, factor)

    # Construct the spatial weights from surface (this is a TIN)
    w_surface_sp = weights.Queen.from_xarray(surface)

    # Densify the network of spatial weights
    w_surface_all = sparse_weights_to_dense(w_surface_sp)

    # Convert the DataArray to a pd series
    surface_values = surface.to_series()

    # Subset to keep only the values that are not missing
    surface_values = surface_values[surface_values != surface.rio.nodata]

    # Final step: out w_surface_all contains a row and column for EVERY
    # value in surface, we need to filter out the weights corresponding
    # to nodata
    w_surface = weights.w_subset(w_surface_all, surface_values.index)
    w_surface.index = surface_values.index

    # Now we can run a LISA as we would with a geotable
    # NOTE: this may take a while longer to run depending on the hardware, be
    # catious with the number of cores (n_jobs) used.
    surface_lisa = esda.moran.Moran_Local(
        surface_values.astype(float), w_surface, n_jobs=-1
    )  # using row standardization (default), n_jobs = -1 is for using all cores in conditional randomization
    # NOTE: we explicitly cast the surface values as floats before computing the
    # LISA to ensure they are in line with our spatial weights

    # Surface Local Autocorrelation Visulization
    sig_surface = pd.Series(
        surface_lisa.q
        * (
            surface_lisa.p_sim < sig_p_val
        ),  # Quadrant of significance at sig_p_val (defined top of this function)
        index=surface_values.index,
    )
    # NOTE: The sig_surface object, expressed here as a 1D vector, contains the
    # information we would like to recast to an Xarrary DataArray object. For this
    # conversion we can use the w2da finction in libpysal.weights, which derives
    # the spatial configuration of each value in sig_surface from w_surface.

    # Build the DataArray from the set of values and weights
    lisa_da = raster.w2da(
        sig_surface,  # Values
        w_surface,  # Weights
        attrs={"nodatavals": [surface.rio.nodata]},  # Value to be used for missing data
    ).rio.write_crs(
        surface.rio.crs
    )  # Add in the coordinate reference in a complient manner
    # The lisa_da DataArray only contains:
    # * Missing data cells (pixels) - expressed with the same negative value as
    #     the original surface feature's no_data
    # * 0 for non significant cells (pixels)
    # * 1-4 depending on moran's plot quadrent of HH, LH, LL, HL, respectivly

    print(
        f"Counts of cells (pixels) by significance: \nnodata = {surface.rio.nodata}\nnot sig = 0 \nHH = 1 \nLH = 2 \nLL = 3 \n HL = 4"
    )
    print(lisa_da.to_series().value_counts())

    # print(surface.rio.bounds())

    plot_LISA(lisa_da, surface, show_basemap=False)


if __name__ == "__main__":
    main()
# EOF
