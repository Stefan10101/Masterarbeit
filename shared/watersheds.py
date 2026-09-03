#!/usr/bin/env python3
"""
DEM basins, TPI flags, valley-axis cost.

Used by Frei (subregions + residual metric) and TPS (per-basin αz).
n_regions is a DEV free parameter. Everything else is derived from the DEM.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist


@dataclass
class WatershedConfig:
    n_regions: int = 6
    accum_pct: float = 96.0          # pour-point threshold as accumulation percentile
    tpi_radius_m: float = 5000.0
    summit_tpi: float = 80.0         # m above neighbourhood mean
    coldpool_tpi: float = -60.0
    min_basin_cells: int = 50


D8 = [
    (-1, 0), (-1, 1), (0, 1), (1, 1),
    (1, 0), (1, -1), (0, -1), (-1, -1),
]


def d8_flow(dem: np.ndarray) -> np.ndarray:
    """Flow direction index into D8, -1 = sink / flat."""
    ny, nx = dem.shape
    direc = np.full((ny, nx), -1, dtype=np.int8)
    pad = np.pad(dem, 1, mode="edge")
    for i, (di, dj) in enumerate(D8):
        sl = pad[1 + di : ny + 1 + di, 1 + dj : nx + 1 + dj]
        drop = dem - sl
        better = drop > 0
        if i == 0:
            best = np.where(better, drop, -np.inf)
            direc[better] = 0
        else:
            win = better & (drop > best)
            best = np.where(win, drop, best)
            direc[win] = i
    return direc


def flow_accum(direc: np.ndarray) -> np.ndarray:
    ny, nx = direc.shape
    acc = np.ones((ny, nx), dtype=np.float64)
    # poor-man's topological pass: iterate downhill updates
    for _ in range(8):
        add = np.zeros_like(acc)
        for i, (di, dj) in enumerate(D8):
            src = direc == i
            if not src.any():
                continue
            yi, xi = np.where(src)
            yj = np.clip(yi + di, 0, ny - 1)
            xj = np.clip(xi + dj, 0, nx - 1)
            np.add.at(add, (yj, xj), acc[yi, xi])
        acc = 1.0 + add
    return acc


def label_basins(direc: np.ndarray, acc: np.ndarray, accum_pct: float, min_cells: int) -> np.ndarray:
    """Upstream labels from high-accumulation pour points + domain edge."""
    ny, nx = direc.shape
    thr = np.percentile(acc, accum_pct)
    pour = (acc >= thr).copy()
    pour[0, :] = pour[-1, :] = pour[:, 0] = pour[:, -1] = True
    labels = np.zeros((ny, nx), dtype=np.int32)
    ys, xs = np.where(pour)
    # keep strongest pour points
    order = np.argsort(acc[ys, xs])[::-1]
    next_id = 1
    for k in order:
        y0, x0 = int(ys[k]), int(xs[k])
        if labels[y0, x0] != 0:
            continue
        labels[y0, x0] = next_id
        next_id += 1
    # walk every cell downhill to a labelled pour
    for y in range(ny):
        for x in range(nx):
            if labels[y, x] != 0:
                continue
            path = []
            cy, cx = y, x
            seen = set()
            lab = 0
            for _ in range(ny + nx):
                if (cy, cx) in seen:
                    break
                seen.add((cy, cx))
                path.append((cy, cx))
                if labels[cy, cx] != 0:
                    lab = labels[cy, cx]
                    break
                d = int(direc[cy, cx])
                if d < 0:
                    break
                di, dj = D8[d]
                cy = min(max(cy + di, 0), ny - 1)
                cx = min(max(cx + dj, 0), nx - 1)
            if lab == 0:
                lab = next_id
                next_id += 1
                labels[cy, cx] = lab
            for py, px in path:
                if labels[py, px] == 0:
                    labels[py, px] = lab
    # drop tiny basins into neighbour
    for lab in np.unique(labels):
        m = labels == lab
        if m.sum() < min_cells:
            dilated = ndi.binary_dilation(m)
            ring = dilated & ~m
            if ring.any():
                labels[m] = int(np.bincount(labels[ring]).argmax())
    return labels


def merge_basins(labels: np.ndarray, dem: np.ndarray, xs: np.ndarray, ys: np.ndarray, n_regions: int) -> np.ndarray:
    labs = [int(v) for v in np.unique(labels) if v != 0]
    if len(labs) <= n_regions:
        # compact to 1..K
        out = np.zeros_like(labels)
        for i, lab in enumerate(labs, start=1):
            out[labels == lab] = i
        return out
    feats = []
    for lab in labs:
        m = labels == lab
        yy, xx = np.where(m)
        feats.append([
            float(xs[xx].mean()),
            float(ys[yy].mean()),
            float(dem[m].mean()),
            float(dem[m].min()),
        ])
    feats = np.asarray(feats)
    feats = (feats - feats.mean(0)) / np.clip(feats.std(0), 1e-6, None)
    z = linkage(pdist(feats), method="ward")
    grp = fcluster(z, t=int(n_regions), criterion="maxclust")
    out = np.zeros_like(labels)
    for lab, g in zip(labs, grp):
        out[labels == lab] = int(g)
    return out


def tpi(dem: np.ndarray, radius_cells: int) -> np.ndarray:
    r = max(int(radius_cells), 1)
    kern = np.ones((2 * r + 1, 2 * r + 1), dtype=np.float64)
    kern[r, r] = 0.0
    s = ndi.convolve(dem, kern, mode="nearest")
    c = ndi.convolve(np.ones_like(dem), kern, mode="nearest")
    return dem - s / np.clip(c, 1.0, None)


def flags_from_tpi(tpi_map: np.ndarray, summit_tpi: float, cold_tpi: float):
    return tpi_map >= float(summit_tpi), tpi_map <= float(cold_tpi)


def valley_unit(direc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit vector along flow (down-valley)."""
    uy = np.zeros(direc.shape, dtype=np.float64)
    ux = np.zeros(direc.shape, dtype=np.float64)
    for i, (di, dj) in enumerate(D8):
        n = np.hypot(di, dj)
        uy[direc == i] = di / n
        ux[direc == i] = dj / n
    return uy, ux


def cost_between(y0, x0, y1, x1, uy, ux, tpi_map, across_w: float = 4.0) -> np.ndarray:
    """
    Anisotropic path cost. Cheap along valley axis, expensive across ridges.
    Straight-line sample of the vector field; not full Fast Marching.
    """
    y0 = np.asarray(y0, dtype=np.float64)
    x0 = np.asarray(x0, dtype=np.float64)
    y1 = np.asarray(y1, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    ny, nx = tpi_map.shape
    dy = y1 - y0
    dx = x1 - x0
    length = np.hypot(dy * 1.0, dx * 1.0) + 1e-6
    sy, sx = dy / length, dx / length
    steps = 8
    cost = np.zeros(y0.shape[0], dtype=np.float64)
    for t in np.linspace(0.0, 1.0, steps):
        y = np.clip((y0 + t * dy).astype(int), 0, ny - 1)
        x = np.clip((x0 + t * dx).astype(int), 0, nx - 1)
        # alignment with valley axis: |seg × valley|
        cross = np.abs(sx * uy[y, x] - sy * ux[y, x])
        ridge = np.maximum(tpi_map[y, x], 0.0) / 100.0
        cost += 1.0 + across_w * cross + ridge
    return cost * length / steps


def sample_region(labels: np.ndarray, xs: np.ndarray, ys: np.ndarray, x, y) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    dx = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
    dy = float(ys[1] - ys[0]) if len(ys) > 1 else 1.0
    ix = np.clip(((x - xs[0]) / dx).astype(int), 0, len(xs) - 1)
    iy = np.clip(((y - ys[0]) / dy).astype(int), 0, len(ys) - 1)
    return labels[iy, ix]


def build_pack(dem: np.ndarray, xs: np.ndarray, ys: np.ndarray, cfg: WatershedConfig) -> dict:
    dx = abs(float(xs[1] - xs[0])) if len(xs) > 1 else 1000.0
    direc = d8_flow(dem)
    acc = flow_accum(direc)
    raw = label_basins(direc, acc, cfg.accum_pct, cfg.min_basin_cells)
    regions = merge_basins(raw, dem, xs, ys, cfg.n_regions)
    radius = max(int(cfg.tpi_radius_m / max(dx, 1.0)), 1)
    tpi_map = tpi(dem, radius)
    summit, cold = flags_from_tpi(tpi_map, cfg.summit_tpi, cfg.coldpool_tpi)
    uy, ux = valley_unit(direc)
    return {
        "regions": regions,
        "tpi": tpi_map,
        "summit": summit,
        "coldpool": cold,
        "uy": uy,
        "ux": ux,
        "xs": np.asarray(xs, dtype=np.float64),
        "ys": np.asarray(ys, dtype=np.float64),
        "n_regions": int(regions.max()),
        "cfg": cfg,
    }
