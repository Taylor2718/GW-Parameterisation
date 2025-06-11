import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import glob

def plot_sao(data, variable='ua', level=30, cmap='bwr'):
    if variable not in data.variables:
        available = ", ".join(data.data_vars)
        raise ValueError(f"Variable '{variable}' not found; available are: {available}")

    sao = data[variable].sel(lev=level)

    # pick vert dim
    if 'lon' in sao.dims:
        vert = 'lon'
        ylabel = 'Longitude'
    elif 'lat' in sao.dims:
        vert = 'lat'
        ylabel = 'Latitude'
    else:
        raise ValueError("No 'lat' or 'lon' dimension found in the data.")

    plt.figure(figsize=(10, 6))
    contour = plt.contourf(
        sao['time'],
        sao[vert],
        sao.T,
        cmap=cmap
    )
    plt.colorbar(contour, label=f"{variable} at {level} hPa")
    plt.title(f"SAO {variable} at {level} hPa")
    plt.ylabel(ylabel)
    plt.xlabel("Time")
    plt.savefig(f'../figures/sao_{variable}_level_{level}.png', dpi=300)
def load_sao_data(file_path):
    """
    Load SAO data from a NetCDF file.
    Parameters:
    - file_path: Path to the NetCDF file containing SAO data.
    Returns:
    - xarray Dataset containing the SAO data.
    """                 
    try:
        data = xr.open_dataset(file_path)
        return data
    except Exception as e:
        raise IOError(f"Error loading SAO data from {file_path}: {e}")
def load_multi_year_sao(path_pattern):
    """
    Load multiple yearly SAO NetCDF files into one Dataset.
    - path_pattern: a glob string matching all your files, e.g.
      "../data/MERRA2/*_MERRA2_daily_TEM.nc"
    Returns:
    - an xarray.Dataset concatenated along the time dimension.
    """
    files = sorted(glob.glob(path_pattern))
    if not files:
        raise IOError(f"No files match {path_pattern}")
    
    datasets = [xr.open_dataset(f) for f in files]
    combined = xr.concat(datasets, dim='time')
    
    return combined
if __name__ == "__main__":
    # Example usage
    try:
        sao_data = load_multi_year_sao("../data/MERRA2/*_MERRA2_daily_TEM.nc")
        plot_sao(sao_data, variable='ua', level=1)
    except Exception as e:
        print(f"An error occurred: {e}")
