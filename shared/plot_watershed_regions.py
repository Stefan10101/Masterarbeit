#!/usr/bin/env python3
"""
Visualise the watershed region divider.

Loads the master-grid elev (or the cached watershed base), builds the
n_regions merge, and writes a PNG of the region labels.

Usage (from CODE/):
  python shared/plot_watershed_regions.py
  python shared/plot_watershed_regions.py --n-regions 6 --res 1000 --out plots/watershed_regions.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-regions", type=int, default=6)
    p.add_argument("--res", type=int, default=1000)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--method", type=str, default=None, help="grid method folder (default: from config)")
    args = p.parse_args()

    from paths import get_master_grid_path, get_watershed_base_path, get_project_root
    import yaml

    root = get_project_root()
    cfg_path = root / "shared" / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    method = args.method or cfg.get("grid_method", "grids")

    # Prefer cached base (already has raw basins)
    cache = get_watershed_base_path(args.res)
    if cache.exists():
        print(f"loading cached base {cache}")
        z = np.load(cache)
        raw = z["raw"]
        xs, ys = z["xs"], z["ys"]
        dem = None  # merge_basins needs dem; load from master if missing
    else:
        raw = None

    path = get_master_grid_path(method, args.res)
    if not path.exists():
        raise SystemExit(f"master grid missing: {path}")
    import xarray as xr
    g = xr.open_dataset(path)
    dem = g["elev"].values.astype(np.float64)
    xs = g["x"].values.astype(np.float64)
    ys = g["y"].values.astype(np.float64)

    from shared.watersheds import WatershedConfig, build_pack, merge_basins

    if raw is None:
        print("building pack from DEM …")
        wcfg = WatershedConfig(n_regions=args.n_regions)
        pack = build_pack(dem, xs, ys, wcfg)
        regions = pack["regions"]
    else:
        print(f"merging cached basins to n_regions={args.n_regions}")
        regions = merge_basins(raw, dem, xs, ys, args.n_regions)

    n = int(regions.max())
    print(f"regions: {n}  shape={regions.shape}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    # categorical colours
    cmap = plt.get_cmap("tab20", n)
    im = ax.imshow(
        regions,
        origin="lower",
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        cmap=cmap,
        vmin=0.5,
        vmax=n + 0.5,
        interpolation="nearest",
    )
    ax.set_title(f"Watershed regions  n={n}  res={args.res}m")
    ax.set_xlabel("Easting (EPSG:31287)")
    ax.set_ylabel("Northing")
    cbar = fig.colorbar(im, ax=ax, ticks=range(1, n + 1))
    cbar.set_label("region id")
    fig.tight_layout()

    out = Path(args.out) if args.out else root / "Plots" / f"watershed_regions_n{n}_{args.res}m.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    plt.close()


if __name__ == "__main__":
    main()
