import os
import glob
import xarray as xr
import numpy as np

# ----------------------------------------------------------------------------
# model_tendencies-analysis.py
# Functions to load QBOi_Control model tendency datasets and compute utendvtem/utendwtem
# ----------------------------------------------------------------------------

def _drop_bounds(da: xr.DataArray) -> xr.DataArray:
    """
    Drop CF bounds coordinate variables (e.g., time_bounds, bounds) from a DataArray.
    """
    for coord in list(da.coords):
        if coord.endswith('_bounds') or coord == 'bounds' or coord.startswith('bounds'):
            da = da.drop_vars(coord)
    return da


def load_qbo_control_data(base_dir: str) -> dict:
    """
    Read model tendency files from QBOi_Control, drop data from 1979,
    and return each as xarray.DataArray with dims ('time','lev',...).
    """
    qbo_dir = os.path.join(base_dir, 'QBOi_Control')
    patterns = {
        'ua':        os.path.join(qbo_dir, 'ua',        'ua_monZ_HadGEM3GA7-1_QBOi2Exp1_*.nc'),
        'utendepfd': os.path.join(qbo_dir, 'utendepfd', 'utendepfd_monZ_HadGEM3GA7-1_QBOi2Exp1_*.nc'),
        'utendnogw': os.path.join(qbo_dir, 'utendnogw', 'utendnogw_monZ_HadGEM3GA7-1_QBOi2Exp1_*.nc'),
        'vtem':      os.path.join(qbo_dir, 'vtem',      'vtem_monZ_HadGEM3GA7-1_QBOi2Exp1_*.nc'),
        'wtem':      os.path.join(qbo_dir, 'wtem',      'wtem_monZ_HadGEM3GA7-1_QBOi2Exp1_*.nc'),
    }
    data_arrays = {}
    for var, pat in patterns.items():
        files = sorted(glob.glob(pat))
        if not files:
            raise FileNotFoundError(f"No files found for pattern: {pat}")
        ds = xr.open_mfdataset(files, combine='by_coords')
        da = ds[var]
        # Rename pressure dim if needed
        if 'air_pressure' in da.dims:
            da = da.rename({'air_pressure': 'lev'})
        da = _drop_bounds(da)
        # drop singleton longitude dimension if present
        if 'longitude' in da.dims and da.sizes.get('longitude', 0) == 1:
            da = da.squeeze('longitude', drop=True)
        if 'time' in da.coords:
            da = da.sel(time=da['time'].dt.year != 1979)
        data_arrays[var] = da
    return data_arrays


def compute_utend_from_vtem_wtem(ua: xr.DataArray,
                                 vtem: xr.DataArray,
                                 wtem: xr.DataArray,
                                 lat: np.ndarray) -> tuple:
    """
    Compute zonal-mean tendency contributions:
      utendvtem = -vtem*(du/dlat - f)
      utendwtem = -wtem*(du/dz)
    """
    # Compute height z from pressure lev (hPa)
    lev = ua.lev.values
    z = 6950 * np.log(101325 / (lev * 100))
    # Coriolis parameter f
    Omega = 7.29e-5
    f = 2 * Omega * np.sin(np.deg2rad(lat))
    # du/dlat: convert deg->m (86400s/day, 111139 m/deg)
    du_dlat = ua.differentiate('latitude') * (86400 / 111139)
    utendv = -vtem * (du_dlat - 86400*f)
    # du/dz
    du_dz = np.gradient(ua.values, z, axis=ua.get_axis_num('lev'))
    du_dz_da = xr.DataArray(du_dz, coords=ua.coords, dims=ua.dims)
    utendw = -wtem * du_dz_da
    return utendv, utendw

def integrate_computed_tendencies(data_arrays: dict) -> dict:
    """
    Compute utendvtem and utendwtem from ua, vtem, and wtem DataArrays,
    and add them to the data_arrays dict.
    """
    ua   = data_arrays['ua']
    vtem = data_arrays['vtem']
    wtem = data_arrays['wtem']
    lat  = ua.latitude.values
    utendv, utendw = compute_utend_from_vtem_wtem(ua, vtem, wtem, lat)
    data_arrays['utendvtem'] = utendv
    data_arrays['utendwtem'] = utendw
    return data_arrays


if __name__ == '__main__':
    # Load raw tendencies
    base_dir = os.path.abspath(os.path.join(__file__, '..', '..', 'data'))
    data = load_qbo_control_data(base_dir)

    # Integrate computed tendencies
    data = integrate_computed_tendencies(data)

    # Verify that all DataArrays share the same dims
    common_dims = None
    for var, da in data.items():
        dims = da.dims
        print(f"'{var}' dims: {dims}")
        if common_dims is None:
            common_dims = dims
        elif dims != common_dims:
            raise ValueError(f"Dimension mismatch for {var}: {dims} != {common_dims}")
    print(f"All variables share dimensions: {common_dims}")
