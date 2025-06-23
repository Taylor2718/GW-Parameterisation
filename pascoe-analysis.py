import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

ds = xr.open_dataset("../data/qbo_fft_spectra.nc")
freqs     = ds.freq.values
power2d   = ds.power.values
ampl2d    = ds.amplitude.values
plev       = ds.lev.values

ds_deseasonalised = xr.open_dataset("../data/u_deseasonalized.nc")
u_ds = ds_deseasonalised.u_ds  # DataArray(time, lev)
u_ds = u_ds.sel(lev=slice(1, 70))

output_dir = "../figures"
os.makedirs(output_dir, exist_ok=True)

pmin = plev.min()
pmax = plev.max()

def style_pressure_axis(ax, ticks):
    """Set log–inverted pressure axis with explicit limits."""
    ax.set_yscale('log')
    ticks = np.array(ticks)
    # top (max) -> bottom (min)
    ax.set_ylim(np.max(ticks), np.min(ticks))
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

def plot_freq_pressure_power(freqs, power2d, filename,
                       fmin=0.02, fmax=0.075, pmin = 1, pmax = 70):
    """
    Contour-plot power2d vs frequency & pressure,
    zoomed in to [fmin, fmax] cycles/month.
    """
    fig, ax = plt.subplots(figsize=(14,6))
    cf = ax.contourf(freqs, plev, power2d.T,
                 levels=np.linspace(0, power2d.max()*0.8, 15),
                 cmap='viridis', extend='max')
    style_pressure_axis(ax, [3, 5, 10, 20, 30, 50, 70])
    ax.set_ylim(pmax, pmin)
    ax.set_xlabel('Frequency (cycles per month)')
    ax.set_title('FFT Power Spectrum: Frequency vs Pressure')
    ax.set_xlim(fmin, fmax)                    
    # find global peak:
    f_idx, lev_idx = np.unravel_index(np.nanargmax(power2d), power2d.shape)
    peak_freq = freqs[f_idx]
    peak_plev = plev[lev_idx]
    print(f"F.T. Peak frequency: {peak_freq:.4f} cycles/month at {peak_plev:.1f} hPa, Period: {1/peak_freq:.2f} months")
    ax.axvline(peak_freq, color='white', linestyle='--', linewidth=1)
    fig.colorbar(cf, ax=ax, pad=0.02, label='Power')
    fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved zoomed frequency-pressure plot: {filename}")

def plot_freq_pressure_amp(freqs, ampl2d, filename,
                       fmin=0.02, fmax=0.5, pmin = 1, pmax = 70):
    """
    Contour-plot power2d vs frequency & pressure,
    zoomed in to [fmin, fmax] cycles/month.
    """
    fig, ax = plt.subplots(figsize=(14,6))
    cf = ax.contourf(freqs, plev, ampl2d.T,
                 levels=np.linspace(0, ampl2d.max()*0.8, 5),
                 cmap='viridis', extend='max')
    style_pressure_axis(ax, [3, 5, 10, 20, 30, 50, 70])
    ax.set_ylim(pmax, pmin)
    ax.set_xlabel('Frequency (cycles per month)')
    ax.set_title('FFT Amplitude: Frequency vs Pressure')
    ax.set_xlim(fmin, fmax)                    
    # find global peak:
    f_idx, lev_idx = np.unravel_index(np.nanargmax(ampl2d), ampl2d.shape)
    peak_freq = freqs[f_idx]
    peak_plev = plev[lev_idx]
    ax.axvline(peak_freq, color='white', linestyle='--', linewidth=1)
    fig.colorbar(cf, ax=ax, pad=0.02, label='Amplitude')
    fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved zoomed frequency-pressure plot: {filename}")

# deseasonalized anomaly (QBO signal)
plot_field(u_ds, title='Deseasonalized Zonal‐Mean Zonal Wind (QBO Anomaly, +/- 5 degrees)', filename='deseasonalized_qbo.png')
plot_freq_pressure_power(freqs, power2d, 'power_freq_vs_pressure.png', pmin=pmin, pmax=pmax)
plot_freq_pressure_amp(freqs, ampl2d, 'amp_freq_vs_pressure.png', pmin=pmin, pmax=pmax)

