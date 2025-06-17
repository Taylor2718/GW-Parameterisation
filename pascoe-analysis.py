#!/usr/bin/env python3
import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

# --- SETTINGS ---
pattern = '../data/MERRA2/*_MERRA2_daily_TEM.nc'
lat_bnd = 5    # equatorial mean for Figs 1–2
output_dir       = '../figures'
# 38 real pressure‐level values (hPa)
plev = np.array([
    0.0100, 0.0200, 0.0327, 0.0476, 0.0660, 0.0893, 0.1197, 0.1595,
    0.2113, 0.2785, 0.3650, 0.4758, 0.6168, 0.7951, 1.0194, 1.3005,
    1.6508, 2.0850, 2.6202, 3.2764, 4.0766, 5.0468, 6.2168, 7.6198,
    9.2929, 11.2769, 13.6434, 16.4571, 19.7916, 23.7304, 28.3678,
    33.8100, 40.1754, 47.6439, 56.3879, 66.6034, 78.5123, 92.3657
])

def load_dataset(pattern):
    """
    Load MERRA-2 dataset from the specified pattern and assign pressure levels.
    Parameters
    ----------
    pattern : str
        File pattern to match MERRA-2 data files.
    Returns
    -------
    xarray.Dataset
        Dataset containing the MERRA-2 data with pressure levels assigned.
    """
    ds = xr.open_mfdataset(pattern, combine='by_coords')
    # assign real pressure levels
    ds = ds.assign_coords(lev=("lev", plev))
    ds.lev.attrs.update(units="hPa", long_name="Pressure")
    return ds

def compute_monthly_deseasonalized_u(ua, lat_bnd=5):
    """
    Given ua(time, lev, lat), return:
      - u_mon: monthly-mean zonal-mean wind over ±lat_bnd
      - u_ds : deseasonalized anomaly (u_mon minus its climatology)
    """
    # 1) Take equatorial +/-lat_bnd mean
    u_eq = ua.sel(lat=slice(-lat_bnd, lat_bnd)).mean('lat')
    
    # 2) Compute monthly means (at the start-of-month)
    u_mon = u_eq.resample(time='1MS').mean()
    
    # 3) Compute the monthly climatology (12-entry)
    clim = u_mon.groupby('time.month').mean('time') #climatology for each month
    
    # 4) Subtract the climatology for each month
    u_ds = u_mon.groupby('time.month') - clim
    
    return u_mon, u_ds

def detect_qbo_cycles(u_ds, ref_lev=10, smooth_months=5):
    """
    Detect QBO phase onset times and compute cycle periods.

    Parameters
    ----------
    u_ds : xarray.DataArray
        Deseasonalized monthly u(time, lev) anomalies.
    ref_lev : float
        Pressure level (hPa) at which to detect zero-crossings.
    smooth_months : int
        Window size for the running mean (in months).

    Returns
    -------
    onsets : pandas.DatetimeIndex
        Times of west→east onsets (u crosses from negative to positive).
    periods : pandas.Series
        Length of each cycle in months, indexed by the onset time.
    """
    # 1) select the reference level
    u_ref = u_ds.sel(lev=ref_lev, method='nearest')
    
    # 2) smooth with a centered rolling mean
    u_smooth = u_ref.rolling(time=smooth_months, center=True).mean().dropna('time')
    
    # 3) find sign changes from negative to positive
    sign = np.sign(u_smooth)
    # shift by one to compare successive points
    sign_prev = sign.shift(time=1)
    # west->east onsets: sign_prev < 0, sign >= 0
    on_w2e = u_smooth.time.where((sign_prev < 0) & (sign >= 0)).dropna('time')
    
    # 4) compute cycle lengths in months
    on = pd.DatetimeIndex(on_w2e.values) #cycle length
    # difference in months between successive onsets
    periods = pd.Series((on[1:] - on[:-1]) / np.timedelta64(1, 'W'),
                        index=on[1:]) # period in weeks
    # convert to months (approx. 4.345 weeks per month)
    periods = periods / 4.345 #periods in months, on average
    return on, periods, u_smooth

def style_pressure_axis(ax, ticks, plev=plev):
    """Set log–inverted pressure axis with explicit limits."""
    ax.set_yscale('log')
    # top (max) -> bottom (min)
    ax.set_ylim(plev.max(), plev.min())
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.set_ylabel('Pressure (hPa)')
    # no grid lines
    ax.grid(False)

def plot_field(ds, title, filename):
    fig, ax = plt.subplots(figsize=(14,6))
    cs = ds.T.plot.contourf(
        ax=ax, x='time', y='lev',
        levels=11, cmap='RdBu_r', extend='both', add_colorbar=False
    )
    ax.contour(ds.time, ds.lev, ds.T, levels=[0], colors='k')
    # pressure ticks 0.01–70 hPa
    style_pressure_axis(ax, [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 70])
    ax.set_title(title)
    cbar = fig.colorbar(cs, ax=ax, pad=0.02)
    cbar.set_label('m/s')
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved Field Figure: {path}")

def plot_onsets(u_smooth, onsets, uncertainty_months, title, filename):
    """
    Plot the smoothed reference‐level wind and mark 
    west→east onsets as vertical lines.
    
    Parameters
    ----------
    u_smooth : xarray.DataArray
      Your centered, rolling‐mean wind at ref_lev (dim time).
    onsets : pandas.DatetimeIndex
      Times of detected west→east transitions.
    """
     # approximate month as 30 days
    dt = pd.Timedelta(days=int(uncertainty_months * 30))
    
    fig, ax = plt.subplots(figsize=(12,4))
    u_smooth.plot(ax=ax, label='Smoothed wind')
    ax.axhline(0, color='gray', linestyle=':', linewidth=1)

    ymin, ymax = ax.get_ylim()
    for t in onsets:
        # shade uncertainty band
        ax.axvspan(t - dt, t + dt, color='gray', alpha=0.3)
        # dashed onset line
        ax.axvline(t, color='k', linestyle='--')
    ax.set_ylabel('u (m/s)')
    ax.set_title('Discrete QBO Onsets, plotted with +/-1-month Uncertainty')
    ax.legend()
    plt.tight_layout()

    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved Figure: {path}")

ds = load_dataset(pattern)
# extract ua
ua = ds['ua']

# get monthly and deseasonalized
u_monthly, u_deseasonalized = compute_monthly_deseasonalized_u(ua, lat_bnd=5)

# now u_monthly and u_deseasonalized have dims (time=monthly, lev)
print(u_monthly)
print(u_deseasonalized)

# u_deseasonalized is monthly deseasonalized anomalies:
onsets, cycle_periods, u_smooth = detect_qbo_cycles(u_deseasonalized, ref_lev=30, smooth_months=5)

print("Detected westerly→easterly onsets at 30 hPa:\n", onsets)
print("Cycle lengths (months):\n", cycle_periods.describe())

# raw monthly zonal‐mean wind
#plot_field(u_monthly, title='Monthly‐Mean Equatorial (+/- 5 degrees) Zonal‐Mean Zonal Wind', filename='monthly_zmzw.png')

# deseasonalized anomaly (QBO signal)
#plot_field(u_deseasonalized, title='Deseasonalized Zonal‐Mean Zonal Wind (QBO Anomaly, +/- 5 degrees)', filename='deseasonalized_qbo.png')

plot_onsets(u_smooth, onsets, uncertainty_months=1, title = 'Smoothed QBO Reference‐Level Wind with Onsets', filename = 'qbo_onsets.png')