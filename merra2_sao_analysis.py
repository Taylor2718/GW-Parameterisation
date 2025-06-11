#!/usr/bin/env python3
import os
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd

# --- SETTINGS ---
TEM_PATTERN = '../data/MERRA2/*_MERRA2_daily_TEM.nc'
LAT_BND = 5  # degrees latitude
OUTPUT_DIR = '../figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LOAD DATA ---
print("Opening datasets")
dset = xr.open_mfdataset(TEM_PATTERN, combine='by_coords')

u = dset['ua']  # zonal‐mean zonal wind (m/s)
ut = dset['du_dt']  # total zonal‐wind tendency
u_eq = ut.sel(lat=slice(-LAT_BND, LAT_BND)).mean(dim='lat') # Compute equatorial mean
# --- SELECT 1990–1999 PERIOD ---
ut_sample = u_eq.sel(time=slice('1990-01-01', '1999-12-31'))

# --- DAILY MEAN CLIMATOLOGY ---
# Group by day of year and average over all years
u_clim = u_eq.groupby('time.dayofyear').mean(dim='time')

# --- PLOT AND SAVE FIGURE 1 ---
fig, ax = plt.subplots(figsize=(14, 6))
# Contour‐fill of climatology
cs = u_clim.T.plot.contourf(
    ax=ax,
    x='dayofyear',
    y='lev',
    levels=11,
    cmap='RdBu_r',
    extend='both',
    add_colorbar=False
)
# Zero‐wind contour
ax.contour(
    u_clim.dayofyear, u_clim.lev, u_clim.T,
    levels=[0], colors='k', linewidths=1.0
)

# Log‐scale pressure axis, inverted so 1000 hPa at bottom
ax.set_yscale('log')
ax.invert_yaxis()
ax.set_ylabel('Pressure levels')

# Label the x‐axis with month names
# Find day‐of‐year for start of each month (non‐leap)
months = pd.date_range('2001-01-01', '2001-12-01', freq='MS')
mdoy = months.dayofyear
ax.set_xticks(mdoy)
ax.set_xticklabels(months.strftime('%b'))
ax.set_xlabel('Day of year')

ax.set_title('Figure 1. Daily‐Mean Climatology of Equatorial (±5°) Zonal-Mean Zonal Wind\nMERRA-2')
# Colorbar
cbar = fig.colorbar(cs, ax=ax, pad=0.02)
cbar.set_label('Wind speed (m s⁻¹)')

# Save
output_path = os.path.join(OUTPUT_DIR, 'figure1_zonal_wind_climatology_MERRA2.png')
fig.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved Figure 1 to {output_path}")

# --- PLOT AND SAVE FIGURE 2 ---
fig, ax = plt.subplots(figsize=(14,6))  # widened figure
cs = ut_sample.T.plot.contourf(
    ax=ax,
    x='time',
    y='lev',
    levels=11,
    cmap='RdBu_r',
    extend='both',
    add_colorbar=False
)
# Add zero contour as a black solid line
t0 = ax.contour(
    ut_sample.time, ut_sample.lev, ut_sample.T,
    levels=[0], colors='k', linewidths=1.0, linestyles='solid'
)

ax.set_yscale('log')
ax.invert_yaxis()
# Set y-ticks to actual pressure levels (hPa)
pressures = ut_sample.lev.values
# Choose a subset of ticks for readability, e.g., every 5th level
yticks = pressures[::5]
ax.set_yticks(yticks)
ax.set_yticklabels([f"{int(p)}" for p in yticks])
ax.set_title('Total Zonal-Wind Tendency (1990–1999)')
ax.set_ylabel('Pressure levels')
cbar = fig.colorbar(ax.collections[0], ax=ax)
cbar.set_label('m/s/day')

output_path = os.path.join(OUTPUT_DIR, 'figure2_total_tendency_1990_1999.png')
fig.savefig(output_path, dpi=150)
print(f"Saved Figure 2 to {output_path}")
