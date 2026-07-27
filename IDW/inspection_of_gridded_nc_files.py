#!/usr/bin/env python3
"""
inspect_master_grids.py
Simple standalone inspector for the new master grid .nc files.

Run this on your Windows machine to see:
- Do the files exist at the new master/ path?
- What variables/coordinates are inside?
- Is 'mask' present and how many True cells does it have?
- Is 'elev' present and reasonable?

Usage:
    python inspect_master_grids.py
    python inspect_master_grids.py --resolutions 2000 1000 500 100
    python inspect_master_grids.py --method IDW --base "C:/Users/stefa/Documents/UNI/Master/Masterarbeit/Daten"
"""

import argparse
from pathlib import Path
import xarray as xr
import numpy as np


def inspect_grid(grid_path: Path):
    print(f"\n{'='*70}")
    print(f"Inspecting: {grid_path}")
    print(f"{'='*70}")

    if not grid_path.exists():
        print("  [ERROR] File does not exist!")
        return

    try:
        ds = xr.open_dataset(grid_path)
    except Exception as e:
        print(f"  [ERROR] Could not open with xarray: {e}")
        return

    print(f"  Dimensions: {dict(ds.dims)}")
    print(f"  Coordinates: {list(ds.coords)}")
    print(f"  Data variables: {list(ds.data_vars)}")
    print(f"  Attributes: {dict(ds.attrs)}")

    print("\n  --- Critical variables for interpolation ---")

    if "mask" in ds:
        mask = ds["mask"].values
        n_true = int(np.sum(mask))
        n_total = mask.size
        print(f"  'mask' found: shape={mask.shape}, dtype={mask.dtype}")
        print(f"    Valid cells (True): {n_true} / {n_total} ({100*n_true/n_total:.2f}%)")
        if n_true == 0:
            print("    >>> WARNING: ZERO valid cells! This is why production crashes.")
    else:
        print("  'mask' NOT FOUND in the file!")

    if "elev" in ds:
        elev = ds["elev"].values
        print(f"  'elev' found: shape={elev.shape}, dtype={elev.dtype}")
        if "mask" in ds:
            valid_mask = ds["mask"].values
            valid_elev = elev[valid_mask]
            if len(valid_elev) > 0:
                print(f"    Valid elev range: {np.nanmin(valid_elev):.2f} ... {np.nanmax(valid_elev):.2f}")
                print(f"    NaNs in valid elev: {np.sum(np.isnan(valid_elev))}")
            else:
                print("    No valid cells to compute elev stats.")
    else:
        print("  'elev' NOT FOUND in the file!")

    if "x" in ds.coords or "x" in ds:
        x = ds["x"].values
        print(f"  'x' coordinate: len={len(x)}, range=[{x.min():.1f}, {x.max():.1f}]")
    if "y" in ds.coords or "y" in ds:
        y = ds["y"].values
        print(f"  'y' coordinate: len={len(y)}, range=[{y.min():.1f}, {y.max():.1f}]")

    ds.close()
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="IDW")
    parser.add_argument("--base", default=None,
                        help="Base data dir (e.g. C:/.../Daten). If omitted, uses paths.py")
    parser.add_argument("--resolutions", nargs="+", type=int,
                        default=[2000, 1000, 500, 200, 100, 50])
    args = parser.parse_args()

    if args.base:
        base = Path(args.base)
        grid_dir = base / args.method / "Output" / "grids" / "master"
    else:
        try:
            import sys
            sys.path.append(str(Path(__file__).resolve().parent))
            from paths import get_master_grid_path
            print("Using central paths.py ...")
        except Exception:
            print("Could not import paths.py — please use --base")
            return

    print(f"Method: {args.method}")
    print(f"Resolutions: {args.resolutions}\n")

    for res in args.resolutions:
        if args.base:
            p = grid_dir / f"res_{res}m" / "grid.nc"
        else:
            p = get_master_grid_path(args.method, res)
        inspect_grid(p)

    print("Inspection finished.")


if __name__ == "__main__":
    main()