#!/usr/bin/env python3
"""
bss_core.py
Core implementation of Bilinear Surface Smoothing (BSS) and BSSE.

This module contains only the mathematical core.
It has no file I/O and no hardcoded paths.
"""

import numpy as np
from scipy.linalg import solve
from typing import Tuple, Dict
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


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
    n_points = coords.shape[0]
    n_knots_x = len(knot_x)
    n_knots_y = len(knot_y)
    n_knots = n_knots_x * n_knots_y
    Pi = np.zeros((n_points, n_knots), dtype=np.float64)

    dropped = 0
    for i, (x, y) in enumerate(coords):
        ix = np.searchsorted(knot_x, x) - 1
        iy = np.searchsorted(knot_y, y) - 1
        if ix < 0 or ix >= n_knots_x - 1 or iy < 0 or iy >= n_knots_y - 1:
            dropped += 1
            continue
        x0, x1 = knot_x[ix], knot_x[ix + 1]
        y0, y1 = knot_y[iy], knot_y[iy + 1]
        weights = _bilinear_weights(x, y, x0, x1, y0, y1)
        k00 = iy * n_knots_x + ix
        k10 = iy * n_knots_x + ix + 1
        k11 = (iy + 1) * n_knots_x + ix + 1
        k01 = (iy + 1) * n_knots_x + ix
        Pi[i, k00] = weights[0]
        Pi[i, k10] = weights[1]
        Pi[i, k11] = weights[2]
        Pi[i, k01] = weights[3]

    if dropped > 0:
        import warnings
        warnings.warn(f"{dropped}/{n_points} stations lie outside the knot grid and were ignored "
                      f"(consider increasing margin or checking coordinate alignment).",
                      UserWarning, stacklevel=2)
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
            for k in [k0, k1, k2]:
                Psi_x[k0, k] += [1, -2, 1][[k0, k1, k2].index(k)]
                Psi_x[k1, k] += [-2, 4, -2][[k0, k1, k2].index(k)]
                Psi_x[k2, k] += [1, -2, 1][[k0, k1, k2].index(k)]

    Psi_y = np.zeros((n_knots, n_knots))
    for ix in range(nx):
        for iy in range(ny - 2):
            k0 = iy * nx + ix
            k1 = (iy + 1) * nx + ix
            k2 = (iy + 2) * nx + ix
            for k in [k0, k1, k2]:
                Psi_y[k0, k] += [1, -2, 1][[k0, k1, k2].index(k)]
                Psi_y[k1, k] += [-2, 4, -2][[k0, k1, k2].index(k)]
                Psi_y[k2, k] += [1, -2, 1][[k0, k1, k2].index(k)]

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


def fit_bss(station_values: np.ndarray, station_coords: np.ndarray,
            knot_x: np.ndarray, knot_y: np.ndarray,
            tau_x: float = 0.1, tau_y: float = 0.1) -> Dict:
    Pi = build_design_matrix(station_coords, knot_x, knot_y)
    Psi_x, Psi_y = build_penalty_matrices(knot_x, knot_y, tau_x, tau_y)
    A = Pi.T @ Pi + Psi_x + Psi_y
    b = Pi.T @ station_values

    try:
        d = solve(A, b, assume_a='pos')
    except np.linalg.LinAlgError:
        d = np.linalg.lstsq(A, b, rcond=None)[0]

    try:
        hat_trace = np.trace(Pi @ np.linalg.inv(A) @ Pi.T)
    except np.linalg.LinAlgError:
        hat_trace = np.sum(np.diag(Pi @ np.linalg.pinv(A) @ Pi.T))

    gcv = compute_gcv(station_values, Pi, d, hat_trace)
    return {
        "d": d,
        "gcv": gcv,
        "effective_df": hat_trace,
        "Pi": Pi,
        "knot_x": knot_x,
        "knot_y": knot_y,
        "tau_x": tau_x,
        "tau_y": tau_y
    }


def fit_bsse(station_values: np.ndarray, station_coords: np.ndarray, station_elev: np.ndarray,
             knot_x: np.ndarray, knot_y: np.ndarray,
             tau_d: float = 0.1, tau_e: float = 0.1) -> Dict:
    """Fit BSSE with separate smoothing for the main surface (d) and elevation surface (e).

    tau_d and tau_e are applied equally to x and y directions within each surface.
    """
    Pi = build_design_matrix(station_coords, knot_x, knot_y)
    T = np.diag(station_elev)
    Pi_T = T @ Pi
    n_knots = Pi.shape[1]

    # Build penalties with separate tau for d and e surfaces
    Psi_x_d, Psi_y_d = build_penalty_matrices(knot_x, knot_y, tau_d, tau_d)
    Psi_x_e, Psi_y_e = build_penalty_matrices(knot_x, knot_y, tau_e, tau_e)

    A11 = Pi.T @ Pi + Psi_x_d + Psi_y_d
    A12 = Pi.T @ Pi_T
    A21 = Pi_T.T @ Pi
    A22 = Pi_T.T @ Pi_T + Psi_x_e + Psi_y_e
    A = np.block([[A11, A12], [A21, A22]])
    b = np.concatenate([Pi.T @ station_values, Pi_T.T @ station_values])

    # Small ridge for numerical stability when tau is very small (Change 7)
    if min(tau_d, tau_e) < 1e-5:
        ridge = 1e-10 * np.eye(A.shape[0])
        A = A + ridge

    try:
        coeffs = solve(A, b, assume_a='pos')
    except np.linalg.LinAlgError:
        coeffs = np.linalg.lstsq(A, b, rcond=None)[0]

    d = coeffs[:n_knots]
    e = coeffs[n_knots:]

    # Proper effective df: trace of full augmented hat matrix (Change 2)
    try:
        A_inv = np.linalg.inv(A)
        X_aug = np.hstack([Pi, Pi_T])
        H = X_aug @ A_inv @ X_aug.T
        hat_trace = np.trace(H)
    except np.linalg.LinAlgError:
        hat_trace = 1.8 * n_knots   # conservative fallback

    # Proper residuals for BSSE (Change 1)
    z_hat = Pi @ d + station_elev * (Pi @ e)
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
        "tau_e": tau_e
    }


def predict_surface(coeffs: np.ndarray, target_coords: np.ndarray,
                    knot_x: np.ndarray, knot_y: np.ndarray) -> np.ndarray:
    Pi_target = build_design_matrix(target_coords, knot_x, knot_y)
    return Pi_target @ coeffs


def predict_bsse(d: np.ndarray, e: np.ndarray, target_coords: np.ndarray,
                 target_elev: np.ndarray, knot_x: np.ndarray, knot_y: np.ndarray) -> np.ndarray:
    return predict_surface(d, target_coords, knot_x, knot_y) + target_elev * predict_surface(e, target_coords, knot_x, knot_y)


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
        # BSSE
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
                            "gcv": res["gcv"]
                        }
                    else:
                        best_result["best_params"] = {
                            "n_segments": n_seg,
                            "tau_d": tau_d,
                            "tau_e": tau_e,
                            "gcv": res["gcv"]
                        }

    best_result["method"] = method
    return best_result