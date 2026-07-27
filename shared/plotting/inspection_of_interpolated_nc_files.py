#!/usr/bin/env python3
"""
thorough_nc_analyzer.py
Deep inspection of one NetCDF file for mapping / interpolation issues.
"""

import xarray as xr
import numpy as np
from pathlib import Path

# ====================== CONFIG ======================
nc_path = r"C:\Users\stefa\Documents\UNI\Master\Masterarbeit\Daten\IDW\Output\interpolated_maps\full\res_500m\precip_sum\precip_sum_500m.nc"
# nc_path = r"C:\Users\stefa\Documents\UNI\Master\Masterarbeit\Daten\IDW\Output\idw_maps\full\precip_sum_2000m.nc"
# ===================================================

print("=" * 80)
print(f"THOROUGH NETCDF ANALYSIS")
print(f"File: {nc_path}")
print("=" * 80)

ds = xr.open_dataset(nc_path)

print("\n" + "=" * 40)
print("1. DATASET OVERVIEW")
print("=" * 40)
print(f"Dimensions: {dict(ds.dims)}")
print(f"Data variables: {list(ds.data_vars)}")
print(f"Coordinates:    {list(ds.coords)}")
print(f"Attributes (dataset):")
for k, v in ds.attrs.items():
    print(f"  {k}: {v}")

# Find main data variable
data_vars = [v for v in ds.data_vars if v not in {"time", "x", "y", "elev", "mask"}]
if not data_vars:
    print("WARNING: No obvious main data variable found!")
    main_var = list(ds.data_vars)[0]
else:
    main_var = data_vars[0]

print(f"\nMain data variable chosen: '{main_var}'")

da = ds[main_var]

print("\n" + "=" * 40)
print("2. MAIN VARIABLE PROPERTIES")
print("=" * 40)
print(f"Full dims: {da.dims}")
print(f"Full shape: {da.shape}")
print(f"dtype: {da.dtype}")

# Check for _FillValue or missing_value attribute
fill_value = da.attrs.get("_FillValue", da.attrs.get("missing_value", None))
if fill_value is not None:
    print(f"_FillValue / missing_value attribute: {fill_value}")

print("\nVariable attributes:")
for k, v in da.attrs.items():
    print(f"  {k}: {v}")

print("\n" + "=" * 40)
print("3. SPATIAL STRUCTURE ANALYSIS")
print("=" * 40)

has_time = "time" in da.dims
if has_time:
    da0 = da.isel(time=0)
    print(f"Time dimension present with {len(ds['time'])} timesteps")
else:
    da0 = da
    print("No time dimension")

print(f"Spatial dims order: {da0.dims}")
print(f"Spatial shape:      {da0.shape}")

# Check if we have a regular grid or unstructured 'cell' format
if "cell" in da0.dims:
    print("\n>>> FORMAT: UNSTRUCTURED / POINT-BASED ('cell' dimension)")
    is_structured = False
else:
    print("\n>>> FORMAT: REGULAR GRID")
    is_structured = True

print("\n" + "=" * 40)
print("4. COORDINATE ANALYSIS")
print("=" * 40)

for coord_name in ["x", "y"]:
    if coord_name in ds:
        coord = ds[coord_name]
        values = coord.values
        print(f"\n--- {coord_name.upper()} coordinate ---")
        print(f"  dims: {coord.dims}")
        print(f"  length: {len(values)}")
        print(f"  min: {values.min():.2f}")
        print(f"  max: {values.max():.2f}")
        print(f"  first 3: {values[:3]}")
        print(f"  last 3:  {values[-3:]}")

        # Check if regularly spaced
        if len(values) > 1:
            diffs = np.diff(values)
            unique_diffs = np.unique(np.round(diffs, decimals=6))
            if len(unique_diffs) == 1:
                print(f"  spacing: {unique_diffs[0]:.2f} (regular grid)")
            else:
                print(f"  spacing: IRREGULAR (unique steps: {unique_diffs[:5]}...)")

print("\n" + "=" * 40)
print("5. DATA VALUES & MISSING DATA ANALYSIS")
print("=" * 40)

data = da0.values.astype(float)   # ensure float for NaN handling

total_cells = data.size
nan_count = np.isnan(data).sum()
valid_count = total_cells - nan_count
nan_percent = (nan_count / total_cells) * 100

print(f"Total cells:     {total_cells:,}")
print(f"Valid cells:     {valid_count:,} ({100 - nan_percent:.2f}%)")
print(f"NaN / missing:   {nan_count:,} ({nan_percent:.2f}%)")

if nan_percent > 30:
    print(">>> WARNING: High percentage of missing values — this is likely why plots look half-empty!")

if valid_count > 0:
    valid_data = data[~np.isnan(data)]
    print(f"\nValid data statistics:")
    print(f"  min:    {np.min(valid_data):.4f}")
    print(f"  max:    {np.max(valid_data):.4f}")
    print(f"  mean:   {np.mean(valid_data):.4f}")
    print(f"  median: {np.median(valid_data):.4f}")
    print(f"  std:    {np.std(valid_data):.4f}")
else:
    print(">>> CRITICAL: No valid data at all in this timestep!")

# Check for extreme values that could cause white plots
if valid_count > 0:
    print("\n" + "=" * 40)
    print("6. POTENTIAL VISUALISATION ISSUES")
    print("=" * 40)
    
    if np.nanmax(data) > 1e6 or np.nanmin(data) < -1e6:
        print(">>> Possible extreme values detected — check units or scaling.")

    # Check if data is mostly zero or constant
    if np.allclose(valid_data, valid_data[0], atol=1e-6):
        print(">>> Data appears almost constant — check if this is expected.")

print("\n" + "=" * 40)
print("7. ELEVATION / MASK CHECK (if present)")
print("=" * 40)

for extra in ["elev", "mask"]:
    if extra in ds:
        extra_da = ds[extra]
        if has_time and "time" in extra_da.dims:
            extra0 = extra_da.isel(time=0).values
        else:
            extra0 = extra_da.values
        extra_nan = np.isnan(extra0).sum()
        print(f"{extra}: shape={extra0.shape}, NaNs={extra_nan}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

ds.close()