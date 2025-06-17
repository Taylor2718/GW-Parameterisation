#!/usr/bin/env python3
import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

# --- SETTINGS ---
TEM_PATTERN      = '../data/MERRA2/*_MERRA2_daily_TEM.nc'
LAT_BND_EQ       = 5    # equatorial mean for Figs 1–2
LAT_BND_SEASONAL = 30   # +/- 30° for seasonal Fig 3
OUTPUT_DIR       = '../figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 38 real pressure‐level values (hPa)
PLEV = np.array([
    0.0100, 0.0200, 0.0327, 0.0476, 0.0660, 0.0893, 0.1197, 0.1595,
    0.2113, 0.2785, 0.3650, 0.4758, 0.6168, 0.7951, 1.0194, 1.3005,
    1.6508, 2.0850, 2.6202, 3.2764, 4.0766, 5.0468, 6.2168, 7.6198,
    9.2929, 11.2769, 13.6434, 16.4571, 19.7916, 23.7304, 28.3678,
    33.8100, 40.1754, 47.6439, 56.3879, 66.6034, 78.5123, 92.3657
])

def load_and_preprocess(pattern):
    ds = xr.open_mfdataset(pattern, combine='by_coords')
    # assign real pressure levels
    ds = ds.assign_coords(lev=("lev", PLEV))
    ds.lev.attrs.update(units="hPa", long_name="Pressure")
    # equatorial means for Figs 1 & 2
    u_eq  = ds['ua'].sel(lat=slice(-LAT_BND_EQ, LAT_BND_EQ)).mean('lat') # zonal-mean zonal wind
    ut_total = (ds['utendvtem'] 
                + ds['utendwtem']
                + ds['utendepfd'] ) * 86400.0 #meridional‐advection + vertical‐advection + eddy‐pressure‐force (wave‐forcing) = total tendency
    # compute total zonal wind tendency mean across lat band centred on the equator
    ut_eq = ut_total.sel(lat=slice(-LAT_BND_EQ, LAT_BND_EQ)).mean('lat') 
    # seasonal lat‐height climatology for Fig 3
    ua_seas = ds['ua'].sel(lat=slice(-LAT_BND_SEASONAL, LAT_BND_SEASONAL))  # zonal-mean zonal wind seasonallu
    season_clim = ua_seas.groupby('time.season').mean('time')
    # compute climatology & 90s tendency
    clim      = u_eq.groupby('time.dayofyear').mean('time')
    sample90  = u_eq.sel(time=slice('1980-01-01','1999-12-31'))
    sample90_tend  = ut_eq.sel(time=slice('1980-01-01','1999-12-31'))
    return clim, sample90, sample90_tend, season_clim

def subset_qbo(sample, p_low=70, p_high=1):
    """
    Return the zonal-mean wind field restricted to the QBO layer.
    
    Parameters
    ----------
    sample : xarray.DataArray
        Zonal-mean zonal wind with dims (time, lev).
    p_low : float
        Lower pressure bound in hPa (e.g. 70).
    p_high : float
        Upper pressure bound in hPa (e.g. 1).
    
    Returns
    -------
    xarray.DataArray
        Subset of `sample` between p_low → p_high.
    """
    # ensure slice start < stop
    p_start, p_stop = sorted([p_low, p_high])
    return sample.sel(lev=slice(p_start, p_stop))

def style_pressure_axis(ax, ticks, plev=PLEV):
    """Set log–inverted pressure axis with explicit limits."""
    ax.set_yscale('log')
    # top (max) -> bottom (min)
    ax.set_ylim(plev.max(), plev.min())
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.set_ylabel('Pressure (hPa)')
    # no grid lines
    ax.grid(False)

def plot_climatology(clim):
    fig, ax = plt.subplots(figsize=(14,6))
    cs = clim.T.plot.contourf(
        ax=ax, x='dayofyear', y='lev',
        levels=11, cmap='RdBu_r', extend='both', add_colorbar=False
    )
    ax.contour(clim.dayofyear, clim.lev, clim.T, levels=[0], colors='k')
    # pressure ticks as in paper: 0.03, 0.1, 0.3, 1, 3, 10 hPa
    style_pressure_axis(ax, [0.03, 0.1, 0.3, 1, 3, 10])
    # monthly x‐axis
    months = pd.date_range(start='1990-01-01', end='1990-12-01', freq='MS')
    ax.set_xticks(months.dayofyear)
    ax.set_xticklabels(months.strftime('%b'))
    ax.set_title('Daily-Mean Climatology of Equatorial (±5°) Zonal Wind')
    cbar = fig.colorbar(cs, ax=ax, pad=0.02)
    cbar.set_label('Wind speed (m s⁻¹)')
    path = os.path.join(OUTPUT_DIR,'zw_climatology.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved Climatology Figure: {path}")

def plot_zonal_mean_zonal_wind(sample90):
    fig, ax = plt.subplots(figsize=(14,6))
    cs = sample90.T.plot.contourf(
        ax=ax, x='time', y='lev',
        levels=11, cmap='RdBu_r', extend='both', add_colorbar=False
    )
    ax.contour(sample90.time, sample90.lev, sample90.T, levels=[0], colors='k')
    # pressure ticks 0.01–70 hPa
    style_pressure_axis(ax, [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 70])
    ax.set_title('Zonal-Mean Zonal Wind (1990–1999)')
    cbar = fig.colorbar(cs, ax=ax, pad=0.02)
    cbar.set_label('m/s')
    path = os.path.join(OUTPUT_DIR,'zmzw_90s.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved Zonal-Mean Zonal Wind during 90s Figure: {path}")

def plot_tendency(sample90_tendency):
    fig, ax = plt.subplots(figsize=(14,6))
    cs = sample90_tendency.T.plot.contourf(
        ax=ax, x='time', y='lev',
        levels=11, cmap='RdBu_r', extend='both', add_colorbar=False
    )
    ax.contour(sample90_tendency.time, sample90_tendency.lev, sample90_tendency.T, levels=[0], colors='k')
    # pressure ticks 0.01–70 hPa
    style_pressure_axis(ax, [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 70])
    ax.set_title('Total Zonal-Wind Tendency (1990–1999)')
    cbar = fig.colorbar(cs, ax=ax, pad=0.02)
    cbar.set_label('m/s per day')
    path = os.path.join(OUTPUT_DIR,'tendency_90s.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved 90s tendency Figure: {path}")

def plot_latpressure(season_clim):
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharey=True, sharex=True)
    levels = np.linspace(-40, 40, 41)

    for ax, season in zip(axes, seasons):
        data = season_clim.sel(season=season)
        cf = ax.contourf(
            data.lat, data.lev, data.values,
            levels=levels, cmap='RdBu_r', extend='both'
        )
        # only zero contour
        ax.contour(
            data.lat, data.lev, data.values,
            levels=[0], colors='k', linewidths=1.2
        )
        ax.set_title(season)
        ax.set_xlabel('Latitude (°)')
        # use paper ticks (0.03–70 hPa)
        style_pressure_axis(ax, [0.03, 0.1, 0.3, 1, 3, 10, 30, 70])

    # ylabel only on first panel
    axes[0].set_ylabel('Pressure (hPa)')

    # adjust margins so colorbar has space
    plt.subplots_adjust(left=0.07, right=0.93, top=0.88, bottom=0.20, wspace=0.1)

    # single horizontal colorbar
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(cf, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Zonal-mean zonal wind (m s⁻¹)')

    fig.suptitle(
        'Seasonal Latitude–Pressure Sections of Zonal-Mean Zonal Wind\n(±30° latitude)',
        fontsize=16
    )

    out = os.path.join(OUTPUT_DIR, 'seasonal_lat_lev.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved Seasonal Lat-Pressure Zonal-Mean Zonal Wind Figure: {out}")


if __name__ == "__main__":
    clim, sample90, sample90_tend, season_clim = load_and_preprocess(TEM_PATTERN)
    sample90_qbo = subset_qbo(sample90, p_low=70, p_high=1)
    plot_zonal_mean_zonal_wind(sample90)
    plot_zonal_mean_zonal_wind(sample90_qbo)
    #plot_climatology(clim)
    #plot_zonal_mean_zonal_wind(sample90)
    #plot_tendency(sample90_tend)
    #plot_latpressure(season_clim)
