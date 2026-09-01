#!/usr/bin/env python3
"""kNN graph in (x, y, alpha_z * z). Only unmasked nodes send messages."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def metric_coords(x, y, z, alpha_z: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 0:
        z = np.full_like(x, float(z))
    z = np.nan_to_num(z, nan=0.0)
    return np.column_stack([x, y, float(alpha_z) * z])


def knn_edges(
    x,
    y,
    z,
    sender_mask: np.ndarray,
    k: int,
    alpha_z: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Directed edges sender -> receiver.

    Every node (including masked / query) receives from its k nearest
    senders in the (x, y, alpha_z z) metric. Self-loops are dropped.
    edge_attr columns: horizontal distance (m), elevation difference (m).
    """
    n = len(x)
    sender_idx = np.flatnonzero(np.asarray(sender_mask, dtype=bool))
    if sender_idx.size == 0:
        return (
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0, 2), dtype=np.float32),
        )

    xyz = metric_coords(x, y, z, alpha_z)
    tree = cKDTree(xyz[sender_idx])
    k_eff = min(int(k) + 1, sender_idx.size)
    dist, loc = tree.query(xyz, k=k_eff)
    if k_eff == 1:
        dist = dist.reshape(-1, 1)
        loc = loc.reshape(-1, 1)

    src_list = []
    dst_list = []
    horiz = []
    dz = []
    z = np.asarray(z, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    for dst in range(n):
        for j in range(loc.shape[1]):
            src = int(sender_idx[int(loc[dst, j])])
            if src == dst:
                continue
            src_list.append(src)
            dst_list.append(dst)
            dx = x[dst] - x[src]
            dy = y[dst] - y[src]
            horiz.append(float(np.hypot(dx, dy)))
            dz.append(float(z[dst] - z[src]))
            if len(src_list) >= 1 and loc.shape[1] > k:
                # already skipped self; stop at k real neighbours
                n_for_dst = sum(1 for d in dst_list if d == dst)
                if n_for_dst >= k:
                    break

    if not src_list:
        return (
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0, 2), dtype=np.float32),
        )
    edge_index = np.vstack([
        np.asarray(src_list, dtype=np.int64),
        np.asarray(dst_list, dtype=np.int64),
    ])
    edge_attr = np.column_stack([
        np.asarray(horiz, dtype=np.float32),
        np.asarray(dz, dtype=np.float32),
    ])
    return edge_index, edge_attr


def knn_edges_fast(
    x,
    y,
    z,
    sender_mask: np.ndarray,
    k: int,
    alpha_z: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised variant of knn_edges."""
    n = len(x)
    sender_idx = np.flatnonzero(np.asarray(sender_mask, dtype=bool))
    if sender_idx.size == 0:
        return (
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0, 2), dtype=np.float32),
        )

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.nan_to_num(np.asarray(z, dtype=np.float64), nan=0.0)
    xyz = metric_coords(x, y, z, alpha_z)
    tree = cKDTree(xyz[sender_idx])
    k_eff = min(int(k) + 1, sender_idx.size)
    _dist, loc = tree.query(xyz, k=k_eff)
    if k_eff == 1:
        loc = loc.reshape(-1, 1)

    n_q, n_k = loc.shape
    dst = np.repeat(np.arange(n_q), n_k)
    src = sender_idx[loc.reshape(-1)]
    keep = src != dst
    # keep at most k edges per destination after dropping self
    order = np.argsort(dst, kind="mergesort")
    src = src[order]
    dst = dst[order]
    keep = keep[order]
    src = src[keep]
    dst = dst[keep]
    # cap at k per dst
    _, starts = np.unique(dst, return_index=True)
    take = np.zeros(len(dst), dtype=bool)
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(dst)
        take[s:min(e, s + int(k))] = True
    src = src[take]
    dst = dst[take]
    horiz = np.hypot(x[dst] - x[src], y[dst] - y[src]).astype(np.float32)
    dz = (z[dst] - z[src]).astype(np.float32)
    edge_index = np.vstack([src.astype(np.int64), dst.astype(np.int64)])
    edge_attr = np.column_stack([horiz, dz])
    return edge_index, edge_attr
