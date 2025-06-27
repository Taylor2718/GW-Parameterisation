import os
import glob
import xarray as xr
import numpy as np
import pandas as pd

def load_and_combine_gwd(
    file_pattern='../data/MERRA2/GWD/*_MERRA2_GWD.nc',
    chunks=None
):
    """
    Load all yearly GWD NetCDF files matching a glob pattern, combine into one Dataset.
    """
    files = sorted(glob.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No GWD files found matching pattern: {file_pattern}")
    ds = xr.open_mfdataset(
        files,
        combine='by_coords',
        chunks=chunks,
        decode_cf=True
    )
    return ds


def compute_full_monthly(
    ds,
    outdir='../data/MERRA2/GWD',
    outfile='full_monthly_gwd.nc'
):
    """
    Compute monthly mean over time for all latitudes & levels, save to NetCDF.
    """
    ds_full = ds['DUDTGWD'].resample(time='1MS').mean().to_dataset(name='DUDTGWD')
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, outfile)
    ds_full.to_netcdf(path)
    print(f"Full-field monthly GWD saved to {path}")
    return ds_full


def compute_monthly_average(
    ds,
    lat_bnd=5,
    outdir='../data/MERRA2/GWD',
    outfile='equatorial_monthly_gwd.nc'
):
    """
    Compute monthly mean for equatorial band (±lat_bnd), save to NetCDF.
    """
    gwd_eq = ds['DUDTGWD'].sel(lat=slice(-lat_bnd, lat_bnd)).mean('lat')
    ds_eq = gwd_eq.resample(time='1MS').mean().to_dataset(name='DUDTGWD')
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, outfile)
    ds_eq.to_netcdf(path)
    print(f"Equatorial monthly GWD saved to {path}")
    return ds_eq


def compute_monthly_climatology(
    ds_mon,
    outdir='../data/MERRA2/GWD',
    outfile='gwd_climatology.nc'
):
    """
    Compute climatological monthly mean (seasonal cycle) from monthly data, save.
    """
    clim = ds_mon['DUDTGWD'].groupby('time.month').mean('time')
    ds_clim = clim.to_dataset(name='DUDTGWD')
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, outfile)
    ds_clim.to_netcdf(path)
    print(f"GWD climatology saved to {path}")
    return ds_clim


def compute_monthly_anomaly(
    ds_mon,
    ds_clim,
    outdir='../data/MERRA2/GWD',
    outfile='gwd_anomaly.nc'
):
    """
    Compute deseasonalized anomalies from monthly data and climatology, save.
    """
    anom = ds_mon['DUDTGWD'].groupby('time.month') - ds_clim['DUDTGWD']
    ds_anom = anom.to_dataset(name='DUDTGWD')
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, outfile)
    ds_anom.to_netcdf(path)
    print(f"GWD anomalies saved to {path}")
    return ds_anom


if __name__ == '__main__':
    # load and combine all yearly GWD files
    ds = load_and_combine_gwd()

    # save combined dataset
    os.makedirs('../data/MERRA2/GWD', exist_ok=True)
    combined_path = os.path.join('../data/MERRA2/GWD', 'combined_gwd.nc')
    ds.to_netcdf(combined_path)
    print(f"Combined GWD dataset saved to {combined_path}")

    # print data resolution
    times = ds['time'].values
    dt = np.min(np.diff(times)) if times.size>1 else None
    lats = ds['lat'].values
    dlat = float(lats[1]-lats[0]) if lats.size>1 else None
    print("Original Data resolution:")
    print(f"Time resolution: {pd.to_timedelta(dt) if dt is not None else 'N/A'}")
    print(f"Latitude resolution: {dlat if dlat is not None else 'N/A'} degrees")

    # prepare and save datasets
    ds_full_mon = compute_full_monthly(ds)
    ds_eq_mon = compute_monthly_average(ds)
    ds_clim = compute_monthly_climatology(ds_eq_mon)
    ds_anom = compute_monthly_anomaly(ds_eq_mon, ds_clim)
    # print final dataset info
    print("Final datasets:")
    print(f"Full monthly GWD: {ds_full_mon}")
    print(f"Equatorial monthly GWD: {ds_eq_mon}")
    print(f"GWD climatology: {ds_clim}")
    print(f"GWD anomalies: {ds_anom}")  