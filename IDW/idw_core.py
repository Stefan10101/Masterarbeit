# ============================================================
# idw_core.py  (full updated file)
# ============================================================
#!/usr/bin/env python3
"""
idw_core.py
Core Modified IDW implementation (efficient + chunked).

Implements modified Inverse Distance Weighting with separate 
horizontal (XY) and vertical (Z) distance weighting.
"""

import numpy as np
from sklearn.neighbors import KDTree
from pathlib import Path
import xarray as xr
from typing import Optional, Tuple


def two_step_var(var: str) -> bool:
    v = str(var).lower()
    return v.startswith("precip") or v.startswith("snow")


def clip_var(var: str, pred) -> np.ndarray:
    v = str(var).lower()
    out = np.asarray(pred, dtype=np.float64)
    if v.startswith("precip") or v.startswith("snow") or v.startswith("wind"):
        return np.maximum(out, 0.0)
    if v.startswith("rh"):
        return np.clip(out, 0.0, 100.0)
    return out


def power_weight(d: np.ndarray, p: float) -> np.ndarray:
    """Power weight function: w = 1 / d^p"""
    with np.errstate(divide='ignore', invalid='ignore'):
        w = np.where(d > 0, d ** (-p), 0.0)
    return w


def modified_idw(
    station_values: np.ndarray,
    station_coords: np.ndarray,
    station_elev: np.ndarray,
    target_coords: np.ndarray,
    target_elev: np.ndarray,
    tree: KDTree,
    p: float = 2.0,
    Fz: float = 0.3,
    k: int = 12,
    min_dist: float = 1e-6,
    chunk_size: Optional[int] = None
) -> np.ndarray:
    """
    Modified IDW with separate horizontal and vertical distance weighting.
    Supports chunked processing for large grids (100m / 50m).
    """
    n_targets = len(target_coords)
    if n_targets == 0:
        return np.array([], dtype=np.float32)

    interpolated = np.full(n_targets, np.nan, dtype=np.float32)

    if chunk_size is None or chunk_size >= n_targets:
        distances, indices = tree.query(target_coords, k=k)
        distances = np.maximum(distances, min_dist)

        for i in range(n_targets):
            idx = indices[i]
            d_xy = distances[i]
            d_z = np.abs(station_elev[idx] - target_elev[i])

            w_xy = power_weight(d_xy, p)
            w_z = power_weight(d_z, p)
            F_xy = 1.0 - Fz
            w = (F_xy * w_xy**2 + Fz * w_z**2) / 0.5

            vals = station_values[idx]
            valid = ~np.isnan(vals)
            if np.any(valid) and w[valid].sum() > 0:
                interpolated[i] = np.average(vals[valid], weights=w[valid])
    else:
        for start in range(0, n_targets, chunk_size):
            end = min(start + chunk_size, n_targets)
            chunk_coords = target_coords[start:end]
            chunk_elev = target_elev[start:end]

            distances, indices = tree.query(chunk_coords, k=k)
            distances = np.maximum(distances, min_dist)

            for j in range(len(chunk_coords)):
                i = start + j
                idx = indices[j]
                d_xy = distances[j]
                d_z = np.abs(station_elev[idx] - chunk_elev[j])

                w_xy = power_weight(d_xy, p)
                w_z = power_weight(d_z, p)
                F_xy = 1.0 - Fz
                w = (F_xy * w_xy**2 + Fz * w_z**2) / 0.5

                vals = station_values[idx]
                valid = ~np.isnan(vals)
                if np.any(valid) and w[valid].sum() > 0:
                    interpolated[i] = np.average(vals[valid], weights=w[valid])

    return interpolated


def interpolate_idw(
    station_values, station_coords, station_elev,
    target_coords, target_elev, tree,
    p=2.0, Fz=0.3, k=12, min_dist=1e-6, chunk_size=None,
    var: str = "", trace: float = 1.0, tau_wet: float = 0.5,
) -> np.ndarray:
    vals = np.asarray(station_values, dtype=np.float64)
    if two_step_var(var):
        wet = (vals > trace).astype(np.float64)
        p_wet = modified_idw(
            wet, station_coords, station_elev, target_coords, target_elev,
            tree, p=p, Fz=Fz, k=k, min_dist=min_dist, chunk_size=chunk_size,
        )
        amt = modified_idw(
            vals, station_coords, station_elev, target_coords, target_elev,
            tree, p=p, Fz=Fz, k=k, min_dist=min_dist, chunk_size=chunk_size,
        )
        out = np.where(np.clip(p_wet, 0.0, 1.0) >= tau_wet, np.maximum(amt, 0.0), 0.0)
        return clip_var(var, out)
    return clip_var(var, modified_idw(
        vals, station_coords, station_elev, target_coords, target_elev,
        tree, p=p, Fz=Fz, k=k, min_dist=min_dist, chunk_size=chunk_size,
    ))


def load_grid(resolution_m: int, method: str = "IDW") -> xr.Dataset:
    """
    Load precomputed grid using central paths.py.
    This replaces the old manual grid_cache_dir approach.
    """
    from paths import get_domain_grid_path  # import here to avoid circular import issues

    # Note: domain must be passed from the calling script for full generality.
    # For now we keep it simple — domain is handled in produce_idw_maps.py
    raise NotImplementedError(
        "Use get_domain_grid_path() from paths.py directly in the production script. "
        "This function is kept for backward compatibility only."
    )


def get_valid_targets(grid: xr.Dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return flattened coordinates and elevation for all valid grid cells.

    Expects the master grid.nc to contain:
      - 2D boolean (or castable) variable 'mask'  (True = valid interpolation target)
      - 2D variable 'elev'
      - coordinates / variables 'x' and 'y'
    If mask has zero True cells, returns empty target_coords (shape (0,2)).
    Caller (produce script) should check and skip such resolutions.
    """
    if "mask" not in grid:
        raise KeyError("Master grid is missing required 'mask' variable. "
                       "Regenerate grids with an explicit boolean mask of valid cells.")
    if "elev" not in grid:
        raise KeyError("Master grid is missing required 'elev' variable.")

    mask = np.asarray(grid["mask"].values).astype(bool)
    elev = grid["elev"].values

    y_idx, x_idx = np.where(mask)
    x_coords = grid["x"].values[x_idx]
    y_coords = grid["y"].values[y_idx]

    target_coords = np.column_stack([x_coords, y_coords])
    target_elev = elev[y_idx, x_idx]

    return target_coords, target_elev, mask