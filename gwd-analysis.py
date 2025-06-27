import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Loading functions for pre-processed GWD data

def load_equatorial_monthly(path='../data/MERRA2/GWD/equatorial_monthly_gwd.nc'):
    """Load equatorial-band monthly-mean GWD DataArray."""
    ds = xr.open_dataset(path)
    return ds['DUDTGWD']


def load_climatology(path='../data/MERRA2/GWD/gwd_climatology.nc'):
    """Load equatorial monthly climatology DataArray."""
    ds = xr.open_dataset(path)
    return ds['DUDTGWD']


def load_anomaly(path='../data/MERRA2/GWD/gwd_anomaly.nc'):
    """Load deseasonalized anomaly DataArray."""
    ds = xr.open_dataset(path)
    return ds['DUDTGWD']

def load_gwd_and_u(u_nc, equatorial_gwd_nc):
    """
    Load deseasonalized QBO wind anomalies and precomputed equatorial GWD tendency.

    Parameters
    ----------
    u_nc : str
        Path to NetCDF file containing 'u_ds' (deseasonalized wind anomalies).
    equatorial_gwd_nc : str
        Path to NetCDF file containing equatorial GWD tendency.

    Returns
    -------
    u_ds : xarray.DataArray
        Deseasonalized wind anomalies (time, lev).
    gwd : xarray.DataArray
        Equatorial GWD tendency (time, lev).
    """
    # Load wind anomalies
    ds_u = xr.open_dataset(u_nc)
    u_ds = ds_u['u_ds']

    # Load precomputed equatorial GWD
    ds_gwd = xr.open_dataset(equatorial_gwd_nc)
    # Assume variable name 'DUDTGWD' carries tendency data
    gwd = ds_gwd['DUDTGWD']

    return u_ds, gwd

def interpolate_gwd_to_u_levels(ds_u, ds_gwd):
    """
    Interpolate the (coarser) GWD forcing onto the (denser) u-field levels.
    
    Parameters
    ----------
    ds_u : xarray.Dataset
        Must contain a DataArray `var_u` with coord 'lev' (pressure, hPa).
    ds_gwd : xarray.Dataset
        Must contain a DataArray `var_g` with coord 'lev'.
    var_u : str
        Name of the deseasonalised-u variable in ds_u.
    var_g : str
        Name of the GWD variable in ds_gwd.
    
    Returns
    -------
    du : xarray.DataArray
        `var_u` sorted by lev.
    dg_on_u : xarray.DataArray
        `var_g` interpolated onto du.lev, with matching coords/attrs.
    """
    # 1. sort both by lev
    du = ds_u.sortby("lev")
    dg = ds_gwd.sortby("lev")

    # 2. restrict to overlapping lev-range (avoids any extrapolation)
    lev_min, lev_max = float(du.lev.min()), float(du.lev.max())
    common_lev = ds_gwd.lev.where(
        (ds_gwd.lev >= lev_min) & (ds_gwd.lev <= lev_max),
        drop=True
    )

    # 3. interpolate GWD onto u’s levels: coarser to finer interpolation
    dg_on_u = dg.interp(lev=du.lev, method="linear")

    # 4. restore exact lev coord + metadata
    dg_on_u = dg_on_u.assign_coords(lev=du.lev)
    dg_on_u.lev.attrs.update(ds_gwd.lev.attrs)

    return du, dg_on_u

# Plotting functions

def plot_equatorial_time_pressure(
    da_eq,
    output_path='../figures/equatorial_monthly_time_pressure.png',
    dpi=150,
    lev_min=None,    # e.g. 1   (top of plot)
    lev_max=None     # e.g. 70  (bottom of plot)
):
    # Collapse lon if present
    da = da_eq.mean(dim='lon') if 'lon' in da_eq.dims else da_eq

    # Keep only strictly positive, sorted levels
    da = da.sel(lev=da.lev > 0).sortby('lev')

    # Apply a pressure window if requested
    if lev_min is not None and lev_max is not None:
        da = da.sel(lev=slice(lev_min, lev_max))

    times = da.time.values
    levs  = da.lev.values
    data  = da.values.T

    fig, ax = plt.subplots(figsize=(10,6))

    mesh = ax.pcolormesh(
        times, levs, data,
        shading='nearest',
        cmap='viridis'
    )

    # Always invert so that larger hPa is at the bottom:
    if lev_min is not None and lev_max is not None:
        ax.set_ylim(lev_min, lev_max)    # bottom = lev_max, top = lev_min
    # else, let it auto-range
    ax.invert_yaxis()                     # flips the axis direction

    ax.set_xlabel('Time')
    ax.set_ylabel('Pressure (hPa)')
    ax.set_title('Equatorial Monthly GWD Tendency')

    cbar = fig.colorbar(mesh, ax=ax, label='m s⁻²')
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved equatorial monthly Hovmöller to {output_path}")

def plot_climatology_pressure_month(
    da_clim,
    output_path='../figures/gwd_climatology_pressure_month.png',
    dpi=150
):
    """
    Plot climatology (month vs pressure) contour.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.contourf(
        da_clim['month'].values,
        da_clim['lev'].values,
        da_clim.values.T,
        levels=20
    )
    ax.invert_yaxis()
    ax.set_yscale('log')
    ax.set_xlabel('Month')
    ax.set_ylabel('Pressure (hPa)')
    ax.set_title('Climatological Seasonal Cycle of GWD')
    fig.colorbar(mesh, ax=ax, label='m s⁻²')
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved climatology plot to {output_path}")


def plot_anomaly_time_series(
    da_anom,
    lev_level=10,
    output_path='../figures/gwd_anomaly_time_series.png',
    dpi=150
):
    """
    Plot a time series of anomaly at a specific pressure level.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ts = da_anom.sel(lev=lev_level, method='nearest')
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts['time'].values, ts.values)
    ax.set_xlabel('Time')
    ax.set_ylabel('GWD Anomaly (m s⁻²)')
    ax.set_title(f'GWD Anomaly at {ts.lev.values} hPa')
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved anomaly time series to {output_path}")

def plot_time_mean_profiles(u_da, g_da,
                            figsize=(6,4),
                            title="Mean vertical profiles",
                            output_path='../figures/mean_profiles.png',
                            dpi=150):
    """
    Plot the time‐mean profiles of u and GWD vs. pressure.
    """
    mean_u = u_da.mean("time")
    mean_g = g_da.mean("time")
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(mean_u, u_da.lev, label="⟨u⟩")
    ax.plot(mean_g, g_da.lev, label="⟨GWD⟩")
    ax.invert_yaxis()
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Pressure (hPa)")
    ax.legend()
    ax.set_title(title)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved time-mean profile to {output_path}")    
    return fig, ax

def plot_scatter_profile(u_da, g_da,
                         cmap="viridis",
                         alpha=0.6,
                         s=5,
                         figsize=(5,5),
                         title="Instantaneous u vs GWD scatter",
                         output_path='../figures/scatter_u_vs_gwd.png',
                         dpi=150):
    """
    Scatter all (u, GWD) pairs, colored by pressure.
    """
    # flatten time×lev into 1D arrays
    u_flat   = u_da.values.ravel()
    g_flat   = g_da.values.ravel()
    lev_flat = np.repeat(u_da.lev.values, u_da.sizes["time"])

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(u_flat, g_flat, c=lev_flat, s=s, cmap=cmap, alpha=alpha)
    plt.colorbar(sc, label="Pressure (hPa)", ax=ax)
    ax.set_xlabel("u (deseasonalised)")
    ax.set_ylabel("GWD on u‐levels")
    ax.set_title(title)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved scatter_u vs gwd to {output_path}")
    return fig, ax

def plot_correlation_vs_level(u_da, g_da,
                              figsize=(4,5),
                              title="Levelwise correlation",
                              output_path='../figures/correlation_vs_level.png',
                              dpi=150):
    """
    Compute and plot Pearson’s r between u and GWD at each level.
    """
    corrs = [
        stats.pearsonr(u_da.isel(lev=i).values,
                       g_da.isel(lev=i).values)[0]
        for i in range(u_da.sizes["lev"])
    ]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(corrs, u_da.lev, marker="o")
    ax.invert_yaxis()
    ax.set_xlabel("Corr(u, GWD)")
    ax.set_ylabel("Pressure (hPa)")
    ax.set_title(title)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved correlation vs level to {output_path}")    
    return fig, ax

def plot_mean_difference(u_da: xr.DataArray,
                         g_da: xr.DataArray,
                         figsize=(5,4),
                         title="Mean GWD − u profile",
                         output_path='../figures/mean_difference.png',
                         dpi=150):
    """
    Plot (⟨GWD⟩ − ⟨u⟩) vs. pressure and save to file.
    """
    mu   = u_da.mean("time")
    mg   = g_da.mean("time")
    diff = mg - mu

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(diff, u_da.lev, label="⟨GWD⟩ − ⟨u⟩")
    ax.axvline(0, color="gray", linestyle="--")
    ax.invert_yaxis()
    ax.set(xlabel="Amplitude difference",
           ylabel="Pressure (hPa)",
           title=title)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved mean-difference profile to {output_path}")
    return fig, ax

def plot_seasonal_profiles(u_da: xr.DataArray,
                           g_da: xr.DataArray,
                           figsize=(6,5),
                           title="Seasonal mean profiles",
                           output_path='../figures/seasonal_profiles.png',
                           dpi=150):
    """
    Overlay u & GWD mean profiles for each meteorological season (DJF, MAM, JJA, SON).
    Works even if time is not a pandas datetime index.
    """
    # 1. compute monthly means
    u_m = u_da.groupby("time.month").mean("time")  # dim: month, lev
    g_m = g_da.groupby("time.month").mean("time")

    # 2. define season → months mapping
    season_months = {
        "DJF": [12, 1, 2],
        "MAM": [3, 4, 5],
        "JJA": [6, 7, 8],
        "SON": [9, 10, 11],
    }

    # 3. plot
    fig, ax = plt.subplots(figsize=figsize)
    for season, months in season_months.items():
        # select those months, then average over the 'month' axis
        u_s = u_m.sel(month=months).mean("month")
        g_s = g_m.sel(month=months).mean("month")

        ax.plot(u_s, u_da.lev, linestyle="--", label=f"u {season}")
        ax.plot(g_s, u_da.lev, linestyle="-", label=f"GWD {season}")

    ax.invert_yaxis()
    ax.set(
        xlabel="Amplitude",
        ylabel="Pressure (hPa)",
        title=title
    )
    ax.legend(ncol=2, fontsize="small")
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved seasonal profiles to {output_path}")
    return fig, ax

def plot_hexbin(u_da: xr.DataArray,
                g_da: xr.DataArray,
                gridsize=50,
                figsize=(5,5),
                title="Hexbin of u vs GWD",
                output_path='../figures/hexbin_u_vs_gwd.png',
                dpi=150):
    """
    2D‐histogram (hexbin) of all u vs GWD samples.
    """
    fig, ax = plt.subplots(figsize=figsize)
    hb = ax.hexbin(
        u_da.values.ravel(),
        g_da.values.ravel(),
        gridsize=gridsize,
        mincnt=1,
        cmap="Blues"
    )
    cb = fig.colorbar(hb, ax=ax, label="Counts")
    ax.set(xlabel="u (deseasonalised)",
           ylabel="GWD on u‐levels",
           title=title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved hexbin scatter to {output_path}")
    return fig, ax

def plot_monthly_correlation(u_da, g_da,
                             figsize=(7,4),
                             title="Monthly corr(u, GWD)",
                             output_path='../figures/monthly_correlation.png',
                             dpi=150):
    """
    2D‐heatmap of Pearson r by calendar month (1–12) and pressure level.
    """
    # Helper: for each u‐group, pull the matching g‐times and corr over time
    def corr_for_group(u_group):
        # select GWD at the same time stamps
        g_group = g_da.sel(time=u_group.time)
        # compute corr at each level
        return xr.corr(u_group, g_group, dim="time")

    # group u by month-of-year and apply the helper → dims: month, lev
    corr_month = u_da.groupby("time.month").apply(corr_for_group)

    months = corr_month["month"].values     # 1..12
    levs   = corr_month["lev"].values       # your pressure levels

    # transpose so array is (lev, month)
    data = corr_month.values.T

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(months, levs, data, shading="auto")
    ax.invert_yaxis()
    ax.set(xlabel="Month (1=Jan,…,12=Dec)",
           ylabel="Pressure (hPa)",
           title=title)
    cbar = fig.colorbar(im, ax=ax, label="Pearson r")
    plt.tight_layout()

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved monthly correlation heatmap to {output_path}")
    return fig, ax


if __name__ == '__main__':
    da_eq = load_equatorial_monthly()
    da_clim = load_climatology()
    da_anom = load_anomaly()
    ds_u, ds_gwd = load_gwd_and_u(
        '../data/MERRA2/TEM/u_deseasonalized.nc',
        '../data/MERRA2/GWD/equatorial_monthly_gwd.nc'
    )
    du, dg_on_u = interpolate_gwd_to_u_levels(ds_u, ds_gwd)
  
    plot_equatorial_time_pressure(da_eq, lev_min=0, lev_max=10)
    plot_climatology_pressure_month(da_clim)
    plot_anomaly_time_series(da_anom, lev_level=10)

    plot_time_mean_profiles(du, dg_on_u)
    plot_scatter_profile(du, dg_on_u)
    plot_correlation_vs_level(du, dg_on_u)

    plot_mean_difference(du, dg_on_u)
    plot_seasonal_profiles(du, dg_on_u)
    plot_hexbin(du, dg_on_u)
    plot_monthly_correlation(du, dg_on_u)