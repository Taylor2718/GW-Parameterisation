#!/usr/bin/env python3
import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

# --- SETTINGS for MERRA2 analysis ---
pattern = '../data/MERRA2/TEM/*_MERRA2_daily_TEM.nc'
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

def detect_qbo_cycles_all_levels(u_ds, smooth_months=5):
    """
    Detect QBO phase onsets and compute discrete cycle periods for all pressure levels in the dataset.

    Parameters
    ----------
    u_ds : xarray.DataArray
        Deseasonalized monthly u(time, lev) anomalies.
    smooth_months : int
        Window size for the running mean (in months).

    Returns
    -------
    onsets : dict
        Keys are pressure levels (hPa), values are pandas.DatetimeIndex of west->east onsets.
    periods : dict
        Keys are pressure levels (hPa), values are pandas.Series of integer month lengths between successive onsets.
    u_smooth : xarray.DataArray
        Smoothed wind field (time x lev) used for detection.
    """
    # 1) smooth all levels in time
    u_smooth = u_ds.rolling(time=smooth_months, center=True).mean().dropna('time')

    onsets = {}
    periods = {}

    # 2) loop over each pressure level
    for lev in u_smooth.lev.values:
        # select this level
        u_ref = u_smooth.sel(lev=lev, method='nearest')

        # detect onsets: sign change from negative to non-negative
        sign_prev = np.sign(u_ref).shift(time=1)
        sign = np.sign(u_ref)
        onset_da = u_ref.time.where((sign_prev < 0) & (sign >= 0)).dropna('time')

        # convert to pandas dates
        lev_onsets = pd.to_datetime(onset_da.values)
        onsets[lev] = lev_onsets

        # 3) compute month differences between onsets
        periods[lev] = pd.Series(
            [(d2.year - d1.year) * 12 + (d2.month - d1.month)
             for d1, d2 in zip(lev_onsets[:-1], lev_onsets[1:])],
            index=pd.DatetimeIndex(lev_onsets[1:])
        )

    return onsets, periods, u_smooth

def compute_vertical_fft_fast(u_ds):
    """
    Vectorized FFT power & amplitude spectra for each level in u_ds(time, lev).
    Returns freqs (1D), power2d (freq × lev), ampl2d (freq × lev).
    """
    u = u_ds.dropna('time', how='all').values  # shape (ntime, nlev)
    # remove the mean
    u = u - np.nanmean(u, axis=0, keepdims=True)
    N = u.shape[0]
    # FFT
    fft2d = np.fft.rfft(u, axis=0)
    freqs = np.fft.rfftfreq(N, d=1.0)
    # power‐spectrum (normalized so that Parseval holds)
    power2d = (np.abs(fft2d)**2) / N
    # one‐sided amplitude spectrum
    ampl2d = 2.0 * np.abs(fft2d) / N
    return freqs, power2d, ampl2d

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

def save_qbo_to_nc(onsets, periods, u_smooth, out_dir="../data/MERRA2/GWD", suffix='onsets', attrs=None):
    """
    Save QBO onsets, periods, and smoothed wind into a single NetCDF file in a specified directory.

    Parameters
    ----------
    onsets : dict
        Mapping of pressure level to pandas.DatetimeIndex of onsets.
    periods : dict
        Mapping of pressure level to pandas.Series of cycle lengths.
    u_smooth : xarray.DataArray
        Smoothed wind field used for detection.
    out_dir : str
        Directory path where the NetCDF file will be written (default: '../data/MERRA2/GWD').
    prefix : str, optional
        Prefix for the output filename (default: 'qbo').
    attrs : dict, optional
        Global attributes to add to the NetCDF, e.g., metadata.

    Returns
    -------
    str
        The path to the saved NetCDF file.
    """
    # Prepare level dimension
    levels = np.array(list(onsets.keys()), dtype=float)
    nlev = len(levels)

    # Determine max number of onsets across levels
    max_onsets = max(len(v) for v in onsets.values())

    # Initialize arrays for onset times and periods
    onset_times = np.full((nlev, max_onsets), np.datetime64('NaT'), dtype='datetime64[ns]')
    period_vals = np.full((nlev, max_onsets), np.nan, dtype=float)

    # Fill arrays
    for i, lev in enumerate(levels):
        lev_onsets = onsets[lev]
        onset_times[i, :len(lev_onsets)] = lev_onsets.values.astype('datetime64[ns]')
        lev_periods = periods.get(lev, pd.Series(dtype=float)).values
        period_vals[i, 1:len(lev_onsets)] = lev_periods

    # Create DataArrays
    onset_da = xr.DataArray(
        onset_times,
        dims=('lev', 'onset_index'),
        coords={'lev': levels, 'onset_index': np.arange(max_onsets)},
        name='onset_times'
    )
    period_da = xr.DataArray(
        period_vals,
        dims=('lev', 'onset_index'),
        coords={'lev': levels, 'onset_index': np.arange(max_onsets)},
        name='period_months'
    )

    # Combine into Dataset
    ds = xr.Dataset({
        'onset_times': onset_da,
        'period_months': period_da,
        'u_smooth': u_smooth
    })

    # Add global attributes if provided
    if attrs:
        ds.attrs.update(attrs)

    # Construct file path and ensure directory exists
    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, f"qbo_{suffix}.nc")

    # Save to NetCDF
    ds.to_netcdf(filepath)

    return filepath

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
#onsets_all, cycle_periods_all, u_smooth_all = detect_qbo_cycles_all_levels(u_deseasonalized, smooth_months=5)
period_desc = cycle_periods.describe()
period_desc.to_csv('../data/MERRA2/TEM/period_desc_30hPa.csv')

print("Detected westerly→easterly onsets at 30 hPa:\n", onsets)
print("Cycle lengths (months):\n", period_desc)

#freqs, power2d, ampl2d = compute_vertical_spectra_welch(u_deseasonalized)
freqs, power2d, ampl2d = compute_vertical_fft_fast(u_deseasonalized)
save_spectra_netcdf(freqs, power2d, ampl2d, u_deseasonalized.lev.values,
                    "../data/MERRA2/TEM/qbo_fft_spectra.nc")

# If u_deseasonalized is a DataArray, wrap it in a Dataset for to_netcdf()
ds_out = xr.Dataset(
    {"u_ds": u_deseasonalized},
    coords={"time": u_deseasonalized.time, "lev": u_deseasonalized.lev}
)

ds_out.time.attrs["long_name"] = "Time"
ds_out.lev.attrs["long_name"]  = "Pressure (hPa)"
ds_out.u_ds.attrs["units"]     = ua.attrs.get("units", "")
ds_out.to_netcdf("../data/MERRA2/TEM/u_deseasonalized.nc")
print("Wrote deseasonalized anomalies to u_deseasonalized.nc")
"""
# Save QBO onsets and periods to NetCDF
qbo_filepath = save_qbo_to_nc(
    onsets_all, cycle_periods_all, u_smooth_all,
    out_dir="../data/MERRA2/TEM",
    suffix='onsets',
    attrs={
        "description": "QBO onsets and periods derived from MERRA-2 zonal wind anomalies",
        "source": "MERRA-2",
        "lat_bnd": lat_bnd,
        "ref_lev": 30,
        "smooth_months": 5,
        "created_by": "pascoe-preprocessing.py"
    }
)
print(f"Saved QBO onsets and periods to {qbo_filepath}")
"""