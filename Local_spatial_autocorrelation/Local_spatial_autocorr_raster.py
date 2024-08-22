# System Imports
import os
from typing import Optional

# Plotting Imports
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt

# Geostats Imports
import libpysal
import matplotlib.pyplot as plt

# Data Management Imports
import pandas as pd  # Tabular data manipulation
import rioxarray  # Surface data manipulation
import xarray as xr  # Surface data manipulation
from libpysal.weights import raster, weights  # Raster weighting
from matplotlib import colors, patches  # Visulization
from pysal.explore import esda  # Exploratory spatial analysis
from pysal.lib import weights  # Spatial weights

# import contextily  # Easy background tiles for fine resolution , requires data plotted in ESRI:54009


def load_dataset(
    path: str,
    nc_array_name: Optional[str] = "AFWA_TOTPRECIP",
    reduction_dim: Optional[str] = "Time",
) -> xr.DataArray:
    # Check filetype based on extension to determine efficent way to load
    fpath, ext = os.path.splitext(path)
    if ext in [".tif", ".tiff", ".geotiff"]:
        print(f"loading {fpath}{ext}")
        surface = rioxarray.open_rasterio(path)

        # Check CRS and reproject if necessary
        if surface.rio.crs != "EPSG:4326":
            print(f"Reprojecting from {surface.rio.crs} to EPSG:4326")
            surface = surface.rio.reproject("EPSG:4326")

        # Using "lat" and "lon" for y,x. rename them from .tif defaults of "y" and "x"
        surface = surface.rename({"y": "lat", "x": "lon"})

        return surface

    elif ext in [".nc"]:
        # Update this cell to read in the netcdf file and get the DataArray
        print(f"loading {fpath}{ext}")
        surface = xr.open_dataset(path, engine="netcdf4")
        surface = surface[nc_array_name].mean(dim=reduction_dim)

        # Set spatial dimensions
        # Rename dimensions if necessary
        surface = surface.rename({"south_north": "lat", "west_east": "lon"})
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
        lat=factor, lon=factor, boundary="trim"
    ).mean()  # dependent on data you may want to use .sum() instead of .mean()

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


def plot_LISA(
    lisa_data: xr.DataArray,
    surface_data: xr.DataArray,
    show_basemap: Optional[str | None] = None,
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

    # Set up figure and gridspec with Cartopy projection
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1], hspace=0.3, wspace=0.2)

    # Top row: Data plots
    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())

    # Bottom row: Colorbar and Legend
    cbar_ax = fig.add_subplot(gs[1, 0])
    legend_ax = fig.add_subplot(gs[1, 1])

    # Common extent and features for both subplots
    common_extent = [
        surface_data.lon.min(),
        surface_data.lon.max(),
        surface_data.lat.min(),
        surface_data.lat.max(),
    ]

    for ax in [ax1, ax2]:
        # Set extent to match your data
        ax.set_extent(common_extent, crs=ccrs.PlateCarree())
        # Add borders and coastlines
        ax.add_feature(cfeature.BORDERS, linewidth=1.5, edgecolor="black")
        ax.add_feature(cfeature.STATES, linewidth=0.8, edgecolor="gray")
        ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor="black")
        # Set x and y limits
        ax.set_ylim(common_extent[2], common_extent[3])
        ax.set_xlim(common_extent[0], common_extent[1])

    # Subplot 1: Surface Data
    surface_plot = surface_data.where(surface_data != surface_data.rio.nodata).plot(
        x="lon",
        y="lat",
        ax=ax1,
        add_colorbar=False,  # We’ll add a separate colorbar
        robust=True,
    )
    ax1.set_title("Surface values by cell (pixel)")

    # Subplot 2: LISA Data
    lisa_plot = (lisa_data.where(lisa_data != lisa_data.nodatavals) / 4).plot(
        x="lon",
        y="lat",
        cmap=lisa_cmap,
        ax=ax2,
        add_colorbar=False,  # Disable the default colorbar
        robust=True,
    )
    ax2.set_title("Surface value clusters")

    # Colorbar for surface data
    cbar = fig.colorbar(surface_plot, cax=cbar_ax, orientation="horizontal")
    cbar_ax.set_title("Surface Data Colorbar", fontsize=10)

    # Create a legend for LISA data
    handles = [
        patches.Patch(color=lc["ns"], label="NS (Non-Significant)"),
        patches.Patch(color=lc["HH"], label="HH (High-High)"),
        patches.Patch(color=lc["LH"], label="LH (Low-High)"),
        patches.Patch(color=lc["LL"], label="LL (Low-Low)"),
        patches.Patch(color=lc["HL"], label="HL (High-Low)"),
    ]
    legend_ax.legend(
        handles=handles,
        loc="center",
        fontsize=10,
        frameon=False,
    )
    legend_ax.axis("off")  # Hide axis lines and ticks

    # Optional: Add basemap using Cartopy's natural earth features
    if show_basemap is not None:
        if show_basemap == "tiles":
            for ax in [ax1, ax2]:
                # lisa_data = lisa_data.rio.reproject("EPSG:54009")
                # surface_data = surface_data.rep
                # contextily.add_basemap(ax, crs=lisa_data.rio.crs)
                bg_terrain = cimgt.GoogleTiles(
                    url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}.jpg",
                    cache=True,
                )
                ax.add_image(bg_terrain, 6, zorder=0)
        elif show_basemap == "mpl":
            for ax in [ax1, ax2]:
                ax.stock_img()  # Adds a low-res background
        else:
            print("Unable to fetch basemap with passed option")

    plt.show()


def main() -> None:
    # path = "./data/ghsl_sao_paulo_1000m_2020.tif"  # Sao Paulo population in 2000, 100m resolution
    # surface = load_dataset(path)

    path = "/media/skye/LEAKE_250gb1/EAE790_Spring24/EOC8p5/DIST_WRFEOC8p5_99-100_WetPeriod_0p254base_15minPeriodCount_for_MAM.nc"
    surface = load_dataset(path, nc_array_name="PRECIP_PERIOD", reduction_dim="Time")

    #    Subset Xarray Data array to rectangle based on lat/lon
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

    print(
        f"{surface.rio.nodata=}"
    )  # Print our no data variable name and value to the terminal

    # Be cautious of the scale of the data and the scale of the processes
    # creating the data
    # Aggregate because our processes that make the data are larger than the data
    # themselves (data are overly fine for the application)i
    factor = 21  # 21 is closest to 80km, 8 is 30 km for WRF-BCC
    # Perform the aggregation (needed for WRF-BCC)
    surface = aggregate_grid(surface, factor)

    # Construct the spatial weights from surface, there are other methods besides
    # Queen, although Queen will be far and away the most approperiate for raster
    w_surface_sp = weights.Queen.from_xarray(surface)

    # Densify the network of spatial weights from sparse TIN to full
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
    # conversion we can use the w2da function in libpysal.weights, which derives
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

    lisa_da.coords["lon"] = surface.coords["lon"]
    lisa_da.coords["lat"] = surface.coords["lat"]

    plot_LISA(lisa_da, surface, show_basemap="tiles")

    # NOTE: Interpreting Results of Local Indicators of Spatial Autocorrelation (LISAs):
    # Ihe Moran's statistic is an indicaton of spatial simularity - in this case based on the 8 nearest cells (Queen).
    # This means that cell size and how/if you aggregate will impact results (MAUP)
    # The 4 quadrents have differnt meanings - the colored areas are statistically significat at the chosen level (sig_p_val)
    # The measurments are therfore all relative, so if the data in an area are all sitting near some lower/upper bound w/o
    # much change results need to be interpreted carefully
    # ---
    # HH are hotspots (high values sorrounded by high values) (i.e. affluent areas for an income data raster)
    # LL are cold spots (low values sorrounded by Low values) (i.e. impoverished areas in an income data raster)
    # HL are potentially outliers (High values sorrounded by Low values) (i.e. anomoulous activity with nothing in vacinity)
    # LH are potentially outliers (Low values sorrounded by High values) (i.e. lack of activity in otherwise substantial activity)


if __name__ == "__main__":
    main()
# EOF
