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
    """Upstream labels from high-accumulation pour points + domain edge.

    Vectorized: seed pour points, then propagate labels downhill in array
    steps instead of a Python walk over every cell.
    """
    ny, nx = direc.shape
    thr = np.percentile(acc, accum_pct)
    pour = (acc >= thr).copy()
    pour[0, :] = pour[-1, :] = pour[:, 0] = pour[:, -1] = True
    labels = np.zeros((ny, nx), dtype=np.int32)
    ys, xs = np.where(pour)
    order = np.argsort(acc[ys, xs])[::-1]
    labels[ys[order], xs[order]] = np.arange(1, len(order) + 1, dtype=np.int32)
    next_id = len(order) + 1
    print(f"      seeded {len(order):,} pour points", flush=True)

    yy, xx = np.indices((ny, nx))
    di = np.array([d[0] for d in D8], dtype=np.int32)
    dj = np.array([d[1] for d in D8], dtype=np.int32)
    d = np.clip(direc, 0, 7)
    flows = direc >= 0
    down_y = np.clip(yy + di[d], 0, ny - 1)
    down_x = np.clip(xx + dj[d], 0, nx - 1)
    down = (down_y * nx + down_x).ravel()
    self_idx = (yy * nx + xx).ravel()
    down = np.where(flows.ravel(), down, self_idx)

    flat = labels.ravel()
    n = flat.size
    pos = np.arange(n, dtype=np.int64)
    # Labelled cells are roots. Everyone else follows downstream.
    # log2(n) pointer jumps reach the root instead of one step per cell.
    nxt = np.where(flat != 0, pos, down.astype(np.int64))
    jumps = int(np.ceil(np.log2(max(n, 2)))) + 1
    print(f"      basins pointer-jump x{jumps} on {n:,} cells", flush=True)
    for j in range(jumps):
        nxt = nxt[nxt]
    flat = flat[nxt]
    # leftovers flowed to an unlabelled sink: one id per sink
    left = flat == 0
    if left.any():
        roots = nxt[left]
        uniq, inv = np.unique(roots, return_inverse=True)
        flat[left] = next_id + inv
    labels = flat.reshape(ny, nx)
    print(f"      basins labelled  n={len(np.unique(labels)):,}", flush=True)

    # drop tiny basins into the dominant 4-neighbour (vectorized, one pass)
    if min_cells > 1:
        counts = np.bincount(labels.ravel())
        small = counts < min_cells
        if small.any() and small.size > 1:
            small[0] = False
            # neighbour label: prefer downslope cell, else right/below
            nb = labels.ravel()[down]
            replace = small[labels]
            labels[replace] = nb.reshape(ny, nx)[replace]
            print(f"      merged {int(small.sum())} tiny basins", flush=True)
    return labels


def merge_basins(labels: np.ndarray, dem: np.ndarray, xs: np.ndarray, ys: np.ndarray, n_regions: int) -> np.ndarray:
    """Cluster basins to n_regions. Vectorized; pre-bins if there are too many for linkage."""
    flat = np.asarray(labels).ravel()
    labs, inv = np.unique(flat, return_inverse=True)
    keep = labs != 0
    if not keep.all():
        # drop background by mapping 0-bins out; Frei grids are fully labelled
        pass
    n = len(labs)
    print(f"      merge: {n:,} basins -> {n_regions}", flush=True)
    if n <= int(n_regions):
        return (inv + 1).reshape(labels.shape).astype(np.int32)

    yy, xx = np.indices(labels.shape)
    xcoord = np.asarray(xs, dtype=np.float64)[xx]
    ycoord = np.asarray(ys, dtype=np.float64)[yy]
    counts = np.bincount(inv, minlength=n).astype(np.float64)
    counts = np.clip(counts, 1.0, None)
    cx = np.bincount(inv, weights=xcoord.ravel(), minlength=n) / counts
    cy = np.bincount(inv, weights=ycoord.ravel(), minlength=n) / counts
    cz = np.bincount(inv, weights=np.asarray(dem, dtype=np.float64).ravel(), minlength=n) / counts
    min_z = np.full(n, np.inf)
    np.minimum.at(min_z, inv, np.asarray(dem, dtype=np.float64).ravel())

    # Too many basins for a full distance matrix: bin centroids on a coarse grid first.
    if n > 1500:
        target = max(int(n_regions) * 40, 200)
        side = int(np.ceil(np.sqrt(target)))
        ix = np.clip((cx - cx.min()) / (np.ptp(cx) + 1.0) * side, 0, side - 1).astype(np.int32)
        iy = np.clip((cy - cy.min()) / (np.ptp(cy) + 1.0) * side, 0, side - 1).astype(np.int32)
        bin_id = iy * side + ix
        _, bin_inv = np.unique(bin_id, return_inverse=True)
        print(f"      merge: pre-bin {n:,} -> {bin_inv.max()+1} spatial bins", flush=True)
        bcounts = np.bincount(bin_inv).astype(np.float64)
        bcounts = np.clip(bcounts, 1.0, None)
        bcx = np.bincount(bin_inv, weights=cx) / bcounts
        bcy = np.bincount(bin_inv, weights=cy) / bcounts
        bcz = np.bincount(bin_inv, weights=cz) / bcounts
        bzmin = np.full(len(bcounts), np.inf)
        np.minimum.at(bzmin, bin_inv, min_z)
        feats = np.column_stack([bcx, bcy, bcz, bzmin])
        back = bin_inv
    else:
        feats = np.column_stack([cx, cy, cz, min_z])
        back = inv

    feats = (feats - feats.mean(0)) / np.clip(feats.std(0), 1e-6, None)
    print(f"      merge: linkage on {len(feats)}", flush=True)
    z = linkage(pdist(feats), method="ward")
    grp = fcluster(z, t=int(n_regions), criterion="maxclust").astype(np.int32)
    out = grp[back]
    if back is not inv:
        out = out[inv]
    return out.reshape(labels.shape)


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


def _index_xy(xs, ys, x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    dx = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
    dy = float(ys[1] - ys[0]) if len(ys) > 1 else 1.0
    ix = np.clip(np.round((x - xs[0]) / dx).astype(int), 0, len(xs) - 1)
    iy = np.clip(np.round((y - ys[0]) / dy).astype(int), 0, len(ys) - 1)
    return ix, iy


def sample_region(labels: np.ndarray, xs: np.ndarray, ys: np.ndarray, x, y) -> np.ndarray:
    ix, iy = _index_xy(xs, ys, x, y)
    return labels[iy, ix]


def region_distance(regions: np.ndarray) -> np.ndarray:
    """Distance in cells from every pixel to each region. Shape (n_regions, ny, nx)."""
    n_r = int(np.max(regions)) if regions.size else 0
    if n_r < 1:
        return np.zeros((1,) + regions.shape, dtype=np.float32)
    out = np.empty((n_r,) + regions.shape, dtype=np.float32)
    for r in range(1, n_r + 1):
        out[r - 1] = ndi.distance_transform_edt(regions != r)
    return out


def sample_blend_weights(dist: np.ndarray, xs, ys, x, y, blend_cells: float) -> np.ndarray:
    """Soft region membership. Inside a basin w~1; decays across the divide."""
    ix, iy = _index_xy(xs, ys, x, y)
    d = dist[:, iy, ix].T  # (n_pts, n_r)
    b = max(float(blend_cells), 1e-6)
    w = np.clip(1.0 - d / b, 0.0, 1.0)
    empty = w.sum(axis=1) <= 0
    if empty.any():
        nearest = np.argmin(d[empty], axis=1)
        w[empty] = 0.0
        w[np.where(empty)[0], nearest] = 1.0
    w = w / w.sum(axis=1, keepdims=True)
    return w


def build_pack(dem: np.ndarray, xs: np.ndarray, ys: np.ndarray, cfg: WatershedConfig) -> dict:
    print(f"    watershed pack: grid {dem.shape} n_regions={cfg.n_regions}", flush=True)
    dx = abs(float(xs[1] - xs[0])) if len(xs) > 1 else 1000.0
    print("    D8 flow …", flush=True)
    direc = d8_flow(dem)
    print("    accumulation …", flush=True)
    acc = flow_accum(direc)
    print("    basins …", flush=True)
    raw = label_basins(direc, acc, cfg.accum_pct, cfg.min_basin_cells)
    print("    merge to n_regions …", flush=True)
    regions = merge_basins(raw, dem, xs, ys, cfg.n_regions)
    print("    TPI / valley axis …", flush=True)
    radius = max(int(cfg.tpi_radius_m / max(dx, 1.0)), 1)
    tpi_map = tpi(dem, radius)
    summit, cold = flags_from_tpi(tpi_map, cfg.summit_tpi, cfg.coldpool_tpi)
    uy, ux = valley_unit(direc)
    print(f"    pack done ({int(max(int(regions.max()), 1))} regions)", flush=True)
    return {
        "regions": regions,
        "region_dist": region_distance(regions),
        "tpi": tpi_map,
        "summit": summit,
        "coldpool": cold,
        "uy": uy,
        "ux": ux,
        "xs": np.asarray(xs, dtype=np.float64),
        "ys": np.asarray(ys, dtype=np.float64),
        "dx": float(dx),
        "n_regions": int(max(int(regions.max()), 1)),
        "cfg": cfg,
    }
