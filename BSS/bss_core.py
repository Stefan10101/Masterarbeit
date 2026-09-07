#!/usr/bin/env python3
"""
bss_core.py
Core implementation of Bilinear Surface Smoothing (BSS) and BSSE.

This module contains only the mathematical core.
It has no file I/O and no hardcoded paths.

BSSE elevation is internally in kilometres (elev / ELEV_SCALE) so the
d-surface and e-surface are on comparable scales. Retrain cluster
parameters after this change - old tau_e values are not transferable.
"""

from __future__ import annotations

from typing import Dict, Tuple
import warnings

import numpy as np
from scipy.linalg import solve

warnings.filterwarnings("ignore", category=RuntimeWarning)

# metres -> km. Keeps tau_e comparable across timestamps / clusters.
ELEV_SCALE = 1000.0


def create_knot_grid(xmin: float, xmax: float, ymin: float, ymax: float,
                     n_segments: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Create a regular knot grid with small margin around the domain."""
    margin_x = (xmax - xmin) * 0.02
    margin_y = (ymax - ymin) * 0.02
    knot_x = np.linspace(xmin - margin_x, xmax + margin_x, n_segments + 1)
    knot_y = np.linspace(ymin - margin_y, ymax + margin_y, n_segments + 1)
    return knot_x, knot_y


def _bilinear_weights(x: float, y: float, x0: float, x1: float,
                      y0: float, y1: float) -> np.ndarray:
    if x1 == x0 or y1 == y0:
        return np.zeros(4)
    dx = (x - x0) / (x1 - x0)
    dy = (y - y0) / (y1 - y0)
    return np.array([(1 - dx) * (1 - dy), dx * (1 - dy), dx * dy, (1 - dx) * dy])


def build_design_matrix(coords: np.ndarray,
                        knot_x: np.ndarray,
                        knot_y: np.ndarray) -> np.ndarray:
    """Bilinear basis. Vectorized over points; cells outside the knot grid stay 0."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must be (n, 2)")
    x = coords[:, 0]
    y = coords[:, 1]
    n_points = x.size
    n_knots_x = len(knot_x)
    n_knots_y = len(knot_y)
    n_knots = n_knots_x * n_knots_y
    Pi = np.zeros((n_points, n_knots), dtype=np.float64)

    ix = np.searchsorted(knot_x, x) - 1
    iy = np.searchsorted(knot_y, y) - 1
    valid = (ix >= 0) & (ix < n_knots_x - 1) & (iy >= 0) & (iy < n_knots_y - 1)
    dropped = int((~valid).sum())
    if dropped and dropped == n_points:
        warnings.warn(
            f"{dropped}/{n_points} stations lie outside the knot grid and were ignored "
            f"(consider increasing margin or checking coordinate alignment).",
            UserWarning, stacklevel=2,
        )
        return Pi
    if not valid.any():
        return Pi

    ixv = ix[valid]
    iyv = iy[valid]
    x0 = knot_x[ixv]
    x1 = knot_x[ixv + 1]
    y0 = knot_y[iyv]
    y1 = knot_y[iyv + 1]
    dx = np.divide(x[valid] - x0, x1 - x0, out=np.zeros(ixv.size), where=(x1 != x0))
    dy = np.divide(y[valid] - y0, y1 - y0, out=np.zeros(iyv.size), where=(y1 != y0))
    w00 = (1.0 - dx) * (1.0 - dy)
    w10 = dx * (1.0 - dy)
    w11 = dx * dy
    w01 = (1.0 - dx) * dy

    rows = np.where(valid)[0]
    k00 = iyv * n_knots_x + ixv
    k10 = iyv * n_knots_x + ixv + 1
    k11 = (iyv + 1) * n_knots_x + ixv + 1
    k01 = (iyv + 1) * n_knots_x + ixv
    Pi[rows, k00] = w00
    Pi[rows, k10] = w10
    Pi[rows, k11] = w11
    Pi[rows, k01] = w01
    return Pi


def build_penalty_matrices(knot_x: np.ndarray, knot_y: np.ndarray,
                           tau_x: float = 0.1, tau_y: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    nx = len(knot_x)
    ny = len(knot_y)
    n_knots = nx * ny

    Psi_x = np.zeros((n_knots, n_knots))
    for iy in range(ny):
        for ix in range(nx - 2):
            k0 = iy * nx + ix
            k1 = iy * nx + ix + 1
            k2 = iy * nx + ix + 2
            Psi_x[k0, k0] += 1.0
            Psi_x[k0, k1] += -2.0
            Psi_x[k0, k2] += 1.0
            Psi_x[k1, k0] += -2.0
            Psi_x[k1, k1] += 4.0
            Psi_x[k1, k2] += -2.0
            Psi_x[k2, k0] += 1.0
            Psi_x[k2, k1] += -2.0
            Psi_x[k2, k2] += 1.0

    Psi_y = np.zeros((n_knots, n_knots))
    for ix in range(nx):
        for iy in range(ny - 2):
            k0 = iy * nx + ix
            k1 = (iy + 1) * nx + ix
            k2 = (iy + 2) * nx + ix
            Psi_y[k0, k0] += 1.0
            Psi_y[k0, k1] += -2.0
            Psi_y[k0, k2] += 1.0
            Psi_y[k1, k0] += -2.0
            Psi_y[k1, k1] += 4.0
            Psi_y[k1, k2] += -2.0
            Psi_y[k2, k0] += 1.0
            Psi_y[k2, k1] += -2.0
            Psi_y[k2, k2] += 1.0

    Psi_x *= tau_x
    Psi_y *= tau_y
    return Psi_x, Psi_y


def compute_gcv(z: np.ndarray = None, Pi: np.ndarray = None, d_hat: np.ndarray = None,
                effective_df: float = None, rss: float = None, n: int = None) -> float:
    """Compute GCV score.

    Can be called in two ways:
    - Old style (BSS): compute_gcv(z, Pi, d_hat, effective_df)
    - New style (recommended): compute_gcv(rss=rss, effective_df=effective_df, n=n)
    """
    if rss is None:
        if z is None or Pi is None or d_hat is None:
            raise ValueError("Either provide rss or (z, Pi, d_hat)")
        residuals = z - Pi @ d_hat
        rss = np.sum(residuals ** 2)
        if n is None:
            n = len(z)
    else:
        if n is None:
            raise ValueError("When passing rss you must also pass n")

    if effective_df is None:
        raise ValueError("effective_df is required")

    if effective_df >= n - 1e-6:
        return np.inf
    return (rss / n) / ((1 - effective_df / n) ** 2)


def _hat_trace(A: np.ndarray, XtX: np.ndarray) -> float:
    """tr(H) = tr(A^{-1} X^T X) without forming the n x n hat matrix."""
    try:
        return float(np.trace(solve(A, XtX, assume_a="pos")))
    except np.linalg.LinAlgError:
        try:
            return float(np.trace(np.linalg.solve(A, XtX)))
        except np.linalg.LinAlgError:
            return float(np.trace(np.linalg.pinv(A) @ XtX))


def fit_bss(station_values: np.ndarray, station_coords: np.ndarray,
            knot_x: np.ndarray, knot_y: np.ndarray,
            tau_x: float = 0.1, tau_y: float = 0.1) -> Dict:
    Pi = build_design_matrix(station_coords, knot_x, knot_y)
    Psi_x, Psi_y = build_penalty_matrices(knot_x, knot_y, tau_x, tau_y)
    XtX = Pi.T @ Pi
    A = XtX + Psi_x + Psi_y
    b = Pi.T @ station_values

    try:
        d = solve(A, b, assume_a="pos")
    except np.linalg.LinAlgError:
        d = np.linalg.lstsq(A, b, rcond=None)[0]

    hat_trace = _hat_trace(A, XtX)
    gcv = compute_gcv(station_values, Pi, d, hat_trace)
    return {
        "d": d,
        "gcv": gcv,
        "effective_df": hat_trace,
        "Pi": Pi,
        "knot_x": knot_x,
        "knot_y": knot_y,
        "tau_x": tau_x,
        "tau_y": tau_y,
    }


def _scaled_elev(elev: np.ndarray) -> np.ndarray:
    return np.asarray(elev, dtype=np.float64) / ELEV_SCALE


def fit_bsse(station_values: np.ndarray, station_coords: np.ndarray, station_elev: np.ndarray,
             knot_x: np.ndarray, knot_y: np.ndarray,
             tau_d: float = 0.1, tau_e: float = 0.1) -> Dict:
    """Fit BSSE with separate smoothing for the main surface (d) and elevation surface (e).

    Elevation is divided by ELEV_SCALE (km) inside the fit.
    tau_d and tau_e are applied equally to x and y directions within each surface.
    """
    Pi = build_design_matrix(station_coords, knot_x, knot_y)
    elev_s = _scaled_elev(station_elev)
    Pi_T = elev_s[:, None] * Pi
    n_knots = Pi.shape[1]

    Psi_x_d, Psi_y_d = build_penalty_matrices(knot_x, knot_y, tau_d, tau_d)
    Psi_x_e, Psi_y_e = build_penalty_matrices(knot_x, knot_y, tau_e, tau_e)

    A11 = Pi.T @ Pi + Psi_x_d + Psi_y_d
    A12 = Pi.T @ Pi_T
    A21 = Pi_T.T @ Pi
    A22 = Pi_T.T @ Pi_T + Psi_x_e + Psi_y_e
    A = np.block([[A11, A12], [A21, A22]])
    b = np.concatenate([Pi.T @ station_values, Pi_T.T @ station_values])
    A = A + 1e-10 * np.eye(A.shape[0])

    try:
        coeffs = solve(A, b, assume_a="pos")
    except np.linalg.LinAlgError:
        coeffs = np.linalg.lstsq(A, b, rcond=None)[0]

    d = coeffs[:n_knots]
    e = coeffs[n_knots:]

    X_aug = np.hstack([Pi, Pi_T])
    hat_trace = _hat_trace(A, X_aug.T @ X_aug)

    z_hat = Pi @ d + elev_s * (Pi @ e)
    rss = np.sum((station_values - z_hat) ** 2)
    n = len(station_values)

    gcv = compute_gcv(rss=rss, effective_df=hat_trace, n=n)
    return {
        "d": d,
        "e": e,
        "gcv": gcv,
        "effective_df": hat_trace,
        "Pi": Pi,
        "knot_x": knot_x,
        "knot_y": knot_y,
        "tau_d": tau_d,
        "tau_e": tau_e,
        "elev_scale": ELEV_SCALE,
    }


def predict_surface(coeffs: np.ndarray, target_coords: np.ndarray,
                    knot_x: np.ndarray, knot_y: np.ndarray) -> np.ndarray:
    Pi_target = build_design_matrix(target_coords, knot_x, knot_y)
    return Pi_target @ coeffs


def predict_bsse(d: np.ndarray, e: np.ndarray, target_coords: np.ndarray,
                 target_elev: np.ndarray, knot_x: np.ndarray, knot_y: np.ndarray,
                 elev_scale: float = ELEV_SCALE) -> np.ndarray:
    scale = float(elev_scale) if elev_scale else ELEV_SCALE
    return (
        predict_surface(d, target_coords, knot_x, knot_y)
        + (np.asarray(target_elev, dtype=np.float64) / scale)
        * predict_surface(e, target_coords, knot_x, knot_y)
    )


def optimize_bss_gcv(station_values: np.ndarray,
                     station_coords: np.ndarray,
                     station_elev: np.ndarray,
                     xmin: float, xmax: float, ymin: float, ymax: float,
                     segment_range: range,
                     tau_values: list = None,
                     tau_d_values: list = None,
                     tau_e_values: list = None,
                     method: str = "bsse") -> Dict:
    if method == "bss":
        if tau_values is None:
            tau_values = [0.1]
        tau_d_list = tau_values
        tau_e_list = tau_values
    else:
        if tau_d_values is not None and tau_e_values is not None:
            tau_d_list = tau_d_values
            tau_e_list = tau_e_values
        elif tau_values is not None:
            tau_d_list = tau_values
            tau_e_list = tau_values
        else:
            tau_d_list = [0.1]
            tau_e_list = [0.1]

    best_gcv = np.inf
    best_result = None

    for n_seg in segment_range:
        knot_x, knot_y = create_knot_grid(xmin, xmax, ymin, ymax, n_seg)
        for tau_d in tau_d_list:
            for tau_e in tau_e_list:
                if method == "bss":
                    res = fit_bss(station_values, station_coords, knot_x, knot_y, tau_d, tau_d)
                else:
                    res = fit_bsse(station_values, station_coords, station_elev, knot_x, knot_y,
                                   tau_d=tau_d, tau_e=tau_e)

                if res["gcv"] < best_gcv:
                    best_gcv = res["gcv"]
                    best_result = res
                    if method == "bss":
                        best_result["best_params"] = {
                            "n_segments": n_seg,
                            "tau_x": tau_d,
                            "tau_y": tau_d,
                            "gcv": res["gcv"],
                        }
                    else:
                        best_result["best_params"] = {
                            "n_segments": n_seg,
                            "tau_d": tau_d,
                            "tau_e": tau_e,
                            "gcv": res["gcv"],
                        }

    if best_result is None:
        raise RuntimeError("BSS/BSSE GCV search produced no valid fit")
    best_result["method"] = method
    return best_result
