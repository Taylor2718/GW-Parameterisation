#!/usr/bin/env python3
import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

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
    Detect QBO phase onset times and compute discrete cycle periods in months.

    Parameters
    ----------
    u_ds : xarray.DataArray
        Deseasonalized monthly u(time, lev) anomalies.
    ref_lev : float
        Pressure level (hPa) for zero-crossing detection.
    smooth_months : int
        Window size for the running mean (in months).

    Returns
    -------
    onsets : pandas.DatetimeIndex
        Timestamps of west->east onsets (always first day of month).
    periods : pandas.Series
        Integer month lengths between successive onsets, indexed by onset date.
    u_smooth : xarray.DataArray
        Smoothed reference-level wind used for detection.
    """
    # 1) select and smooth reference-level wind
    u_ref    = u_ds.sel(lev=ref_lev, method='nearest')
    u_smooth = u_ref.rolling(time=smooth_months, center=True).mean().dropna('time')

    # 2) detect discrete onsets via sign change
    sign_prev = np.sign(u_smooth).shift(time=1)
    sign      = np.sign(u_smooth)
    onset_da  = u_smooth.time.where((sign_prev < 0) & (sign >= 0)).dropna('time')
    # convert to pandas DTI for easy arithmetic
    onsets    = pd.to_datetime(onset_da.values)

    # 3) compute integer month differences between onsets
    dates   = pd.DatetimeIndex(onsets.values)
    periods = pd.Series([
        (d2.year - d1.year)*12 + (d2.month - d1.month)
        for d1, d2 in zip(dates[:-1], dates[1:])
    ], index=dates[1:])

    return onsets, periods, u_smooth

from scipy.signal import welch

def compute_vertical_spectra_welch(u_ds, dt=1.0,
                                   window='hann',
                                   nperseg=None,
                                   noverlap=None):
    """
    Vectorized Welch spectral estimate for each pressure level.

    Parameters
    ----------
    u_ds : xarray.DataArray
        Deseasonalized anomaly with dims ('time','lev').
    dt : float
        Sampling interval in time‐units (e.g. months).
    window : str or tuple or array_like
        Window specification passed to scipy.signal.welch.
    nperseg : int or None
        Length of each segment (default None → scipy chooses).
    noverlap : int or None
        Number of points to overlap between segments.

    Returns
    -------
    freqs : 1d ndarray, shape (nfreq,)
        Frequencies in cycles per dt-unit.
    power2d : 2d ndarray, shape (nfreq, nlev)
        One‐sided power spectral density at each level.
    ampl2d : 2d ndarray, shape (nfreq, nlev)
        Amplitude surface (here defined as √power, but you can
        drop the sqrt if you want a different scaling).
    """
    # 1) grab the (ntime, nlev) array and demean along time
    u = u_ds.dropna('time', how='all').values
    u = u - np.nanmean(u, axis=0, keepdims=True)

    # 2) run Welch once over axis=0
    fs = 1.0 / dt
    freqs, power2d = welch(
        u,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        axis=0,
        scaling='spectrum',
        return_onesided=True
    )
    # 3) amplitude: choose your own convention
    ampl2d = np.sqrt(power2d)

    return freqs, power2d, ampl2d

def save_spectra_netcdf(freqs, power2d, ampl2d, lev, filename):
    ds = xr.Dataset(
        {
            "power": (("freq","lev"), power2d),
            "amplitude": (("freq","lev"), ampl2d),
        },
        coords={
            "freq": freqs,
            "lev": lev
        }
    )
    ds.freq.attrs["long_name"] = "Frequency"
    ds.lev.attrs["long_name"]  = "Pressure (hPa)"
    ds.to_netcdf(filename)
    print(f"Wrote spectra to {filename}")

ds = load_dataset(pattern)
# extract ua
ua = ds['ua']

# get monthly and deseasonalized
u_monthly, u_deseasonalized = compute_monthly_deseasonalized_u(ua, lat_bnd=5)
u_deseasonalized = u_deseasonalized.sel(lev=slice(1, 70))
plev = np.array(u_deseasonalized.lev.values)
pmin = np.min(plev)
pmax = np.max(plev)
# now u_monthly and u_deseasonalized have dims (time=monthly, lev)
print(u_monthly)
print(u_deseasonalized)

# u_deseasonalized is monthly deseasonalized anomalies:
onsets, cycle_periods, u_smooth = detect_qbo_cycles(u_deseasonalized, ref_lev=30, smooth_months=5)

print("Detected westerly→easterly onsets at 30 hPa:\n", onsets)
print("Cycle lengths (months):\n", cycle_periods.describe())

freqs, power2d, ampl2d = compute_vertical_spectra_welch(u_deseasonalized)
save_spectra_netcdf(freqs, power2d, ampl2d, u_deseasonalized.lev.values,
                    "../data/qbo_fft_spectra.nc")

# If u_deseasonalized is a DataArray, wrap it in a Dataset for to_netcdf()
ds_out = xr.Dataset(
    {"u_ds": u_deseasonalized},
    coords={"time": u_deseasonalized.time, "lev": u_deseasonalized.lev}
)
ds_out.time.attrs["long_name"] = "Time"
ds_out.lev.attrs["long_name"]  = "Pressure (hPa)"
ds_out.u_ds.attrs["units"]     = ua.attrs.get("units", "")
ds_out.to_netcdf("../data/u_deseasonalized.nc")
print("Wrote deseasonalized anomalies to u_deseasonalized.nc")