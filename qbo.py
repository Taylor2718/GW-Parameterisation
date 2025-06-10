import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

def plot_qbo(data, variable='u', level=30, cmap='viridis'):
    """
    Plot the Quasi-Biennial Oscillation (QBO) data.

    Parameters:
    - data: xarray Dataset containing QBO data.
    - variable: Variable to plot (default is 'u' for zonal wind).
    - level: Pressure level to plot (default is 30 hPa).
    - cmap: Colormap for the plot (default is 'viridis').
    """
    if variable not in data.variables:
        raise ValueError(f"Variable '{variable}' not found in the dataset.")

    # Select the variable and pressure level
    qbo_data = data[variable].sel(plev=level)

    # Create a contour plot
    plt.figure(figsize=(10, 6))
    contour = plt.contourf(qbo_data['lon'], qbo_data['time'], qbo_data, cmap=cmap)
    
    plt.colorbar(contour, label=f'{variable} at {level} hPa')
    plt.title(f'QBO {variable} at {level} hPa')
    plt.xlabel('Longitude')
    plt.ylabel('Time')
    
    plt.show()

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
def main():
    # Example usage
    file_path = 'data/1980_MERRA2_daily_TEM.nc'  # Replace with your QBO data file path
    try:
        qbo_data = load_qbo_data(file_path)
        plot_qbo(qbo_data, variable='u', level=30)
    except Exception as e:
        print(e)
if __name__ == "__main__":
    main()
# This code provides functions to load and plot Quasi-Biennial Oscillation (QBO) data.      

