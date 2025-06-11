import glob
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

def plot_qbo(data, variable='ua', level=30, cmap='bwr'):
    if variable not in data.variables:
        available = ", ".join(data.data_vars)
        raise ValueError(f"Variable '{variable}' not found; available are: {available}")

    qbo = data[variable].sel(lev=level)

    # pick vert dim
    if 'lon' in qbo.dims:
        vert = 'lon'
        ylabel = 'Longitude'
    elif 'lat' in qbo.dims:
        vert = 'lat'
        ylabel = 'Latitude'
    else:
        raise ValueError("No 'lat' or 'lon' dimension found in the data.")

    plt.figure(figsize=(10, 6))
    contour = plt.contourf(
        qbo['time'],
        qbo[vert],
        qbo.T,
        cmap=cmap
    )
    plt.colorbar(contour, label=f"{variable} at {level} hPa")
    plt.title(f"QBO {variable} at {level} hPa")
    plt.ylabel(ylabel)
    plt.xlabel("Time")
    plt.savefig(f'../figures/qbo_{variable}_level_{level}.png',dpi=300)

def plot_qbo_all_levels(ds, variable='ua', cmap='bwr'):
    if variable not in ds:
        raise ValueError(f"{variable!r} not in dataset; available: {list(ds.data_vars)}")

    # 1. Extract the 2-D time×level field:
    qbo = ds[variable]

    # If you have both lat & lon, pick equator then zonally average:
    if {'lat','lon'}.issubset(qbo.dims):
        qbo = qbo.sel(lat=0, method='nearest').mean(dim='lon')
    # Or if only one of them exists, just mean over it:
    elif 'lon' in qbo.dims:
        qbo = qbo.mean(dim='lon')
    elif 'lat' in qbo.dims:
        qbo = qbo.sel(lat=0, method='nearest')
    
    # Now qbo is (time, lev)
    plt.figure(figsize=(12,5))
    cf = plt.contourf(
        qbo['time'],
        qbo['lev'],
        qbo.T,          # transpose so that lev runs on y
        cmap=cmap,
        levels=21       # or choose your own contour levels
    )
    plt.gca().invert_yaxis()               # pressure decreases upward
    cbar = plt.colorbar(cf)
    cbar.set_label(f"{variable} (m/s)")

    plt.title(f"QBO: {variable} time–pressure diagram")
    plt.ylabel("Pressure (hPa)")
    plt.xlabel("Time")
    plt.tight_layout()
    plt.savefig(f"../figures/qbo_{variable}_all_levels.png", dpi=300)

def load_qbo_data(file_path):
    """
    Load QBO data from a NetCDF file.
    Parameters:
    - file_path: Path to the NetCDF file containing QBO data.
    Returns:
    - xarray Dataset containing the QBO data.
    """
    try:
        data = xr.open_dataset(file_path)
        return data
    except Exception as e:
        raise IOError(f"Error loading QBO data from {file_path}: {e}")

def load_multi_year_qbo(path_pattern):
    """
    Load multiple yearly QBO NetCDF files into one Dataset.
    - path_pattern: a glob string matching all your files, e.g.
      "../data/MERRA2/*_MERRA2_daily_TEM.nc"
    Returns:
    - an xarray.Dataset concatenated along the time dimension.
    """
    files = sorted(glob.glob(path_pattern))
    if not files:
        raise IOError(f"No files match {path_pattern}")
    # open_mfdataset will concatenate & merge by coordinate
    ds = xr.open_mfdataset(files, combine='by_coords', parallel=True, engine="h5netcdf")
    return ds

def plot_zonal_mean_band(
    ds,
    variable='ua',
    lat_band=(-5, 5),
    cmap='bwr_r',
    nlevels=21,
    year_range=None
):
    """
    Plot time–pressure diagram of the zonal-mean zonal wind,
    averaged over a latitude band (default 5°S–5°N), with optional year subsetting.

    Parameters
    ----------
    ds : xarray.Dataset
        Must contain `variable` on dims (time, lev, lat, lon).
    variable : str
        Name of the zonal wind variable (e.g. 'ua').
    lat_band : tuple of float
        (south, north) bounds in degrees for the latitudinal average.
    cmap : str
        Matplotlib colormap, reversed if ends in '_r'.
    nlevels : int
        Number of contour levels or bands.
    year_range : tuple of int or None
        (start_year, end_year) to subset time before plotting.
    """
    # 1) grab the data array
    da = ds[variable]

    # 2) zonal mean over longitude if present
    if 'lon' in da.dims:
        da = da.mean(dim='lon')

    # 3) select lat band and meridional mean
    da = da.sel(lat=slice(lat_band[0], lat_band[1])).mean(dim='lat')

    # 4) subset by year range if requested
    if year_range is not None:
        start_year, end_year = year_range
        start = f"{start_year}-01-01"
        end = f"{end_year}-12-31"
        da = da.sel(time=slice(start, end))

    # now da has dims (time, lev)
    T, P = da['time'], da['lev']

    # 5) plot
    fig, ax = plt.subplots(figsize=(12, 5))
    cf = ax.contourf(
        T,
        P,
        da.T,
        levels=nlevels,
        cmap=cmap
    )

    # pressure axis tweaks
    ax.set_yscale('log')
    ax.invert_yaxis()

    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label(f"{variable} (m⋅s⁻¹)")

    ax.set_title(
        f"Zonal-mean {variable} averaged {lat_band[0]}° to {lat_band[1]}°, "
        f"{year_range[0]}–{year_range[1]}" if year_range else ''
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Pressure (hPa)")
    fig.tight_layout()
    fig.savefig(
        f"../figures/{variable}_zonal_mean_{lat_band[0]}_{lat_band[1]}_" +
        (f"{year_range[0]}_{year_range[1]}" if year_range else "all") +
        ".png",
        dpi=300
    )


def main_multi():
    # adjust the glob to wherever your 1980–1999 files live
    ds = load_multi_year_qbo("../data/MERRA2/*_MERRA2_daily_TEM.nc")
    print("[DEBUG] Dataset loaded:", ds)
    # now plot, e.g. ua at 30 hPa over the full 20 years
    #plot_qbo(ds, variable='ua', level=30, cmap='bwr')
    #plot_qbo_all_levels(ds, variable='ua')
    plot_zonal_mean_band(ds, variable='ua', lat_band=(-5,5), cmap='bwr', nlevels=21, year_range=(1989, 1999))

def main():
    # Example usage
    file_path = '../data/MERRA2/1980_MERRA2_daily_TEM.nc'  # Replace with your QBO data file path
    try:
        qbo_data = load_qbo_data(file_path)
        plot_qbo(qbo_data, variable='ua', level=30)
    except Exception as e:
        print(e)
if __name__ == "__main__":
    main_multi()
# This code provides functions to load and plot Quasi-Biennial Oscillation (QBO) data.      

