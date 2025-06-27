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

ds_onsets = xr.open_dataset("../data/MERRA2/TEM/qbo_onsets.nc")

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
                       fmin=0.02, fmax=0.075, pmin = 1, pmax = 70):
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

def plot_avg_amplitude_vs_pressure(freqs, ampl2d, plev, filename, fmin=0.02, fmax=0.075, ticks=[3,5,10,20,30,50,70]):

    """
    Compute the mean amplitude across a band of frequencies,
    and plot it as a profile vs pressure (inverted log axis).

    Parameters
    ----------
    freqs : 1d array
        Frequencies (cycles per month).
    ampl2d : 2d array, shape (nfreq, nlev)
        Amplitude spectrum.
    plev : 1d array
        Pressure levels (hPa).
    filename : str
        Output file name (saved into output_dir).
    fmin, fmax : float or None
        If given, restrict averaging to freqs in [fmin, fmax].
        If None, average over all frequencies.
    ticks : list of floats
        Pressure ticks for the y-axis.
    """
    # 1) pick frequency band
    if fmin is None: fmin = freqs.min()
    if fmax is None: fmax = freqs.max()
    mask = (freqs >= fmin) & (freqs <= fmax)

    # 2) mean across that band
    avg_amp = ampl2d[mask, :].mean(axis=0)

    # 3) plot
    fig, ax = plt.subplots(figsize=(6,8))
    ax.plot(avg_amp, plev, '-o', lw=2)
    ax.set_xlabel('Mean Amplitude')
    ax.set_title(f'Mean FFT Amplitude\n({fmin:.3f}–{fmax:.3f} cpm)')

    # invert & log–scale pressure axis
    ax.set_yscale('log')
    ax.set_ylim(plev.max(), plev.min())
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.set_ylabel('Pressure (hPa)')
    ax.grid(False)

    # save
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved average amplitude profile: {path}")

def plot_qbo_onset_heatmap(ds, filename, ax=None, cmap='viridis'):
    """
    Create a heatmap of QBO onset events: time on x-axis, pressure level on y-axis.
    Uses pcolormesh for robust plotting without xarray aspect issues.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing 'onset_times', 'u_smooth', etc.
    filename : str
        Output filename for saving the figure under the default output directory.
    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes for plotting; created if None.
    cmap : str, optional
        Colormap for the heatmap.

    Returns
    -------
    matplotlib.axes.Axes
        Axes with the heatmap.
    """
    # Extract data
    onset = ds['onset_times']  # dims: lev x onset_index
    times = ds['u_smooth'].time.values
    levels = ds['lev'].values
    # Build binary mask: time x lev
    mask = np.zeros((len(times), len(levels)), dtype=int)
    for i, lev in enumerate(levels):
        t_vals = onset.sel(lev=lev).values
        for t in t_vals:
            if not np.isnat(t):
                # find time index
                ti = np.searchsorted(times, np.datetime64(t))
                if 0 <= ti < len(times):
                    mask[ti, i] = 1
    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    # use pcolormesh
    # extent: [time_min, time_max, lev_max, lev_min] for inverted y-axis
    t0, t1 = times[0], times[-1]
    v0, v1 = levels[0], levels[-1]
    mesh = ax.pcolormesh(times, levels, mask.T,
                         cmap=cmap, shading='auto')
    ax.invert_yaxis()
    ax.set_title('QBO West→East Onset Heatmap')
    ax.set_xlabel('Time')
    ax.set_ylabel('Pressure Level (hPa)')
    # colorbar
    cbar = plt.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['', 'Onset'])
    # save
    output_dir = "../figures"
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig = ax.get_figure()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved onset heatmap: {path}")
    return ax

def plot_onset_histogram_by_level(ds, filename, bins=30, ax=None):
    """
    Plot a series of histograms showing onset date distribution for each pressure level stacked vertically.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing 'onset_times' and 'lev'.
    filename : str
        Filename to save the resulting figure (in ../figures).
    bins : int or sequence
        Bins for the date histogram (applied uniformly across panels).
    ax : array-like of matplotlib.axes.Axes, optional
        Pre-created axes grid matching number of levels. If None, axes will be created.

    Returns
    -------
    array of Axes
        The axes array with histograms per level.
    """
    onset = ds['onset_times']  # dims: lev x onset_index
    levels = ds['lev'].values
    nlev = len(levels)

    fig, axes = plt.subplots(nlev, 1, sharex=True, figsize=(8, 2*nlev))

    for i, lev in enumerate(levels):
        ax_i = axes[i]
        dates = onset.sel(lev=lev).values
        dates = pd.to_datetime(dates[~pd.isnull(dates)])
        ax_i.hist(dates, bins=bins, color='tab:blue', alpha=0.7)
        ax_i.set_ylabel(f'{int(lev)} hPa')
        ax_i.grid(False)
        if i < nlev-1:
            ax_i.set_xticklabels([])
    axes[-1].set_xlabel('Onset Date')
    fig.suptitle('QBO Onset Date Distributions by Pressure Level', y=1.02)
    fig.tight_layout()

    # save
    output_dir = "../figures"
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved onset histograms by level: {path}")
    return axes


def plot_onset_level_histogram(ds, filename, ax=None):
    """
    Plot a histogram of counts of onsets per pressure level to show vertical profile of onset frequency.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing 'onset_times' and 'lev'.
    filename : str
        Filename to save the figure (in ../figures).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.

    Returns
    -------
    Axes
    """
    onset = ds['onset_times']
    levels = ds['lev'].values
    counts = []
    for lev in levels:
        vals = onset.sel(lev=lev).values
        counts.append(np.count_nonzero(~np.isnat(vals)))

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,8))
    ax.barh(levels, counts, height=np.diff(levels).mean()*0.8)
    ax.invert_yaxis()
    ax.set_xlabel('Number of Onsets')
    ax.set_ylabel('Pressure Level (hPa)')
    ax.set_title('Frequency of QBO Onsets by Level')
    ax.grid(False)

    # save
    output_dir = "../figures"
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig = ax.get_figure()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved onset count profile: {path}")
    return ax

def plot_period_histogram_all_levels(ds, filename, bins=20, ax=None):
    """
    Plot a histogram of all QBO period lengths aggregated across all pressure levels.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing 'period_months'.
    filename : str
        Filename to save figure in ../figures.
    bins : int or sequence
        Number or edges of histogram bins.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.

    Returns
    -------
    Axes
    """
    period_da = ds['period_months']
    # flatten and drop NaNs
    periods = period_da.values.ravel()
    periods = periods[~np.isnan(periods)]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(periods, bins=bins, color='tab:orange', alpha=0.8)
    ax.set_xlabel('Cycle Period (months)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of QBO Cycle Periods (All Levels)')
    ax.grid(False)
    output_dir = "../figures"
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig = ax.get_figure()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved aggregated period histogram: {path}")
    return ax


def plot_period_distribution_heatmap(ds, filename, bins=20, ax=None, cmap='plasma'):
    """
    Create a heatmap of period-length distributions by pressure level.
    Rows: pressure levels; columns: period-length bins; color: counts.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing 'period_months' and 'lev'.
    filename : str
        Filename to save figure in ../figures.
    bins : int or sequence
        Bins for the histogram along the period axis.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    cmap : str
        Colormap for heatmap.

    Returns
    -------
    Axes
    """
    period_da = ds['period_months']
    levels = ds['lev'].values
    # define bins
    counts, bin_edges = np.histogram(period_da.values.ravel()[~np.isnan(period_da.values.ravel())], bins=bins)
    # actually want per level: compute 2D
    bin_edges = np.array(bin_edges)
    centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
    hist2d = np.zeros((len(levels), len(centers)))
    for i, lev in enumerate(levels):
        vals = period_da.sel(lev=lev).values
        vals = vals[~np.isnan(vals)]
        if vals.size:
            h, _ = np.histogram(vals, bins=bin_edges)
        else:
            h = np.zeros(len(centers), dtype=int)
        hist2d[i] = h
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,6))
    mesh = ax.pcolormesh(centers, levels, hist2d, cmap=cmap, shading='auto')
    ax.invert_yaxis()
    ax.set_xlabel('Cycle Period (months)')
    ax.set_ylabel('Pressure Level (hPa)')
    ax.set_title('QBO Period-Length Distribution by Level')
    cbar = plt.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label('Count')
    output_dir = "../figures"
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig = ax.get_figure()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved period distribution heatmap: {path}")
    return ax

csv_path = '../data/MERRA2/TEM/period_desc_30hPa.csv'
# Read it in, using the first column as the index
df = pd.read_csv(csv_path, index_col=0)
# Since there's only one data column (named “0”), grab it as a Series
stats = df[df.columns[0]]
# Extract mean and std
mean = stats.loc['mean']
std  = stats.loc['std']
print(f"Mean cycle length: {mean}")
print(f"Std dev cycle length: {std}")
mean_f = 1 / mean
std_f  = std / (mean**2)  # std of 1/x
fmin = mean_f - std_f
fmax = mean_f + std_f
print(f"Mean frequency: {mean_f:.4f} cpm, std dev: {std_f:.4f} cpm")
print(f"Frequency range: [{fmin:.4f}, {fmax:.4f}] cpm")
# deseasonalized anomaly (QBO signal)
plot_field(u_ds, title='Deseasonalized Zonal‐Mean Zonal Wind (QBO Anomaly, +/- 5 degrees)', filename='deseasonalized_qbo.png')
plot_freq_pressure_power(freqs, power2d, 'power_freq_vs_pressure.png', pmin=pmin, pmax=pmax)
plot_freq_pressure_amp(freqs, ampl2d, 'amp_freq_vs_pressure.png', pmin=pmin, pmax=pmax)
plot_avg_amplitude_vs_pressure(freqs, ampl2d, plev, filename='avg_amp_vs_pressure.png', fmin=fmin, fmax=fmax)
plot_qbo_onset_heatmap(ds_onsets, 'qbo_onset_heatmap.png')
plot_onset_histogram_by_level(ds_onsets, 'qbo_onset_histograms_by_level.png', bins=30)
plot_onset_level_histogram(ds_onsets, 'qbo_onset_count_profile.png')
plot_period_histogram_all_levels(ds_onsets, 'qbo_period_histogram_all_levels.png', bins=30)
plot_period_distribution_heatmap(ds_onsets, 'qbo_period_distribution_heatmap.png', bins=30)
