#!/usr/bin/env python3
"""Shared synthetic panel generator for method tests.

Produces a small panel that mirrors the real data structure and statistics
(from variable_profile.csv daily summaries) but is fast to run.

Columns match the real loaders:
  station_name, time, x, y, elev, smooth_elev, tpi, slope, aspect,
  northness, dist_to_coast, signed_dist_to_ridge, clc_level2,
  value, is_heldout, split

Terrain–variable relationships vary across the domain and across fields.
Mild temporal correlation is present on the residual fields.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Daily summary statistics from variable_profile.csv (subset=all)
# Used to set means, scales, wet fractions and approximate terrain correlations.
STATS = {
    "temp_mean": {
        "mean": 7.68, "std": 8.26, "skew": -0.03,
        "corr_elev": -0.47, "wet": None,
    },
    "precip_sum": {
        "mean": 3.49, "std": 8.80, "skew": 5.62,
        "corr_elev": 0.05, "wet": 0.43,
    },
    "snow_mean": {
        "mean": 15.57, "std": 38.94, "skew": 3.50,
        "corr_elev": 0.49, "wet": 0.26,
    },
    "wind_mean": {
        "mean": 2.34, "std": 1.80, "skew": 2.53,
        "corr_elev": 0.29, "corr_tpi": 0.43, "wet": None,
    },
    "rh_mean": {
        "mean": 73.58, "std": 15.93, "skew": -0.82,
        "corr_elev": -0.17, "wet": None,
    },
}


def _exp_cov(dist: np.ndarray, range_m: float, nugget: float = 0.0) -> np.ndarray:
    """Exponential covariance matrix."""
    c = np.exp(-dist / range_m)
    np.fill_diagonal(c, 1.0 + nugget)
    return c


def _sample_grf(coords: np.ndarray, range_m: float, nugget: float, rng: np.random.Generator) -> np.ndarray:
    """One realisation of a unit-variance exponential GRF at the given points."""
    n = len(coords)
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    cov = _exp_cov(dist, range_m, nugget)
    # Cholesky; add tiny jitter for numerical stability
    try:
        L = np.linalg.cholesky(cov + 1e-8 * np.eye(n))
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(cov + 1e-4 * np.eye(n))
    return L @ rng.standard_normal(n)


def _make_terrain(n_stations: int, side_m: float, rng: np.random.Generator) -> pd.DataFrame:
    """Synthetic stations with full terrain covariates."""
    # Cluster stations a bit (more in valleys / mid elevations)
    x = rng.uniform(0, side_m, n_stations)
    y = rng.uniform(0, side_m, n_stations)
    # Elevation: mixture of low and high so the distribution is skewed high
    elev_base = rng.normal(800, 400, n_stations)
    high = rng.normal(2200, 600, n_stations // 4)
    elev_base[: len(high)] = high
    elev = np.clip(elev_base, 200, 3500)

    # Smooth elevation (longer-range field)
    smooth_coords = np.column_stack([x, y])
    smooth_resid = _sample_grf(smooth_coords, range_m=25_000, nugget=0.0, rng=rng)
    smooth_elev = elev + 150 * smooth_resid
    smooth_elev = np.clip(smooth_elev, 100, 3600)

    # Local roughness for TPI, slope
    rough = _sample_grf(smooth_coords, range_m=4_000, nugget=0.1, rng=rng)
    tpi = 80 * rough
    slope = np.clip(8 + 12 * np.abs(rough) + rng.uniform(0, 5, n_stations), 0, 45)
    aspect = rng.uniform(0, 360, n_stations)
    northness = np.cos(np.deg2rad(aspect))

    # Distance to coast: simple linear gradient + noise
    dist_to_coast = np.clip(20_000 + 0.4 * x + 15_000 * rng.normal(0, 1, n_stations), 1_000, 150_000)
    # Signed distance to ridge: correlated with TPI
    signed_dist_to_ridge = -40 * tpi + 200 * rng.normal(0, 1, n_stations)

    # CLC level-2 (15 classes, simplified frequencies)
    clc_classes = np.array([11, 12, 21, 22, 23, 24, 31, 32, 33, 41, 42, 51])
    clc_probs = np.array([0.05, 0.04, 0.12, 0.03, 0.08, 0.06, 0.25, 0.15, 0.08, 0.05, 0.04, 0.05])
    clc_probs /= clc_probs.sum()
    clc = rng.choice(clc_classes, n_stations, p=clc_probs)

    return pd.DataFrame({
        "station_name": [f"s{i:03d}" for i in range(n_stations)],
        "x": x, "y": y,
        "elev": elev,
        "smooth_elev": smooth_elev,
        "tpi": tpi,
        "slope": slope,
        "aspect": aspect,
        "northness": northness,
        "dist_to_coast": dist_to_coast,
        "signed_dist_to_ridge": signed_dist_to_ridge,
        "clc_level2": clc,
    })


def _varying_coeff(terrain: pd.DataFrame, base: float, amplitude: float,
                   range_m: float, rng: np.random.Generator) -> np.ndarray:
    """Spatially varying coefficient (smooth field)."""
    coords = terrain[["x", "y"]].to_numpy()
    field = _sample_grf(coords, range_m=range_m, nugget=0.0, rng=rng)
    return base + amplitude * field


def generate_synthetic_panel(
    variable: str = "temp_mean",
    n_stations: int = 120,
    n_fields: int = 24,
    side_m: float = 100_000,
    holdout_frac: float = 0.15,
    seed: int = 42,
    temporal_ar: float = 0.45,
) -> pd.DataFrame:
    """Generate a synthetic panel for one variable.

    Returns a DataFrame with the same columns and split labels as the real data.
    """
    if variable not in STATS:
        raise ValueError(f"unknown variable {variable}")
    rng = np.random.default_rng(seed)
    st = STATS[variable]

    terrain = _make_terrain(n_stations, side_m, rng)
    coords = terrain[["x", "y"]].to_numpy()

    # Held-out stations (stratified roughly by elevation)
    n_hold = max(8, int(round(holdout_frac * n_stations)))
    elev_rank = terrain["elev"].rank(method="first").to_numpy()
    hold_idx = np.argsort(elev_rank)[:: max(1, n_stations // n_hold)][:n_hold]
    held = set(terrain["station_name"].iloc[hold_idx])

    # Splits: 16 train / 4 tune / 4 test (same proportion as real quarters)
    n_train = int(0.67 * n_fields)
    n_tune = int(0.17 * n_fields)
    splits = (["train"] * n_train + ["tune"] * n_tune +
              ["test"] * (n_fields - n_train - n_tune))

    # Spatially varying base coefficients
    if variable == "temp_mean":
        lapse = _varying_coeff(terrain, -0.0065, 0.0015, 30_000, rng)  # °C/m
        resid_range = 35_000
    elif variable == "precip_sum":
        oro = _varying_coeff(terrain, 0.0004, 0.0003, 25_000, rng)  # mm per m
        resid_range = 18_000
    elif variable == "snow_mean":
        elev_sens = _varying_coeff(terrain, 0.025, 0.01, 28_000, rng)
        resid_range = 22_000
    elif variable == "wind_mean":
        tpi_sens = _varying_coeff(terrain, 0.012, 0.006, 15_000, rng)
        resid_range = 12_000
    else:  # rh
        elev_sens = _varying_coeff(terrain, -0.008, 0.004, 32_000, rng)
        resid_range = 30_000

    rows = []
    prev_resid = np.zeros(n_stations)
    for t, split in enumerate(splits):
        # Mild temporal AR on the residual
        innov = _sample_grf(coords, resid_range, nugget=0.05, rng=rng)
        resid = temporal_ar * prev_resid + np.sqrt(1 - temporal_ar ** 2) * innov
        prev_resid = resid

        # Time-varying amplitude of the terrain effect (smooth random walk)
        amp = 1.0 + 0.25 * np.sin(2 * np.pi * t / n_fields) + 0.1 * rng.normal()

        if variable == "temp_mean":
            mu = st["mean"] + amp * lapse * (terrain["elev"] - 1200)
            value = mu + st["std"] * 0.45 * resid
            value = np.clip(value, -25, 35)
        elif variable == "precip_sum":
            # Smoother occurrence: threshold a spatially correlated latent, not a noisy quantile
            p_wet = np.clip(st["wet"] + 0.08 * (terrain["elev"] - 1200) / 2000, 0.25, 0.65)
            occ_latent = 0.7 * resid + 0.3 * _sample_grf(coords, 40_000, 0.0, rng)
            # Station-wise threshold so wet fraction stays near target
            thr = np.quantile(occ_latent, 1.0 - float(np.mean(p_wet)))
            wet = occ_latent > thr
            amount_mu = np.log(max(st["mean"] / max(st["wet"], 0.05), 0.5)) + amp * oro * (terrain["elev"] - 1200)
            amount = np.exp(np.clip(amount_mu + 0.6 * resid, -2, 4))
            value = np.where(wet, np.maximum(amount, 0.2), 0.0)
        elif variable == "snow_mean":
            p_wet = np.clip(st["wet"] + 0.15 * (terrain["elev"] - 1500) / 2000, 0.12, 0.55)
            occ_latent = 0.8 * resid + 0.2 * _sample_grf(coords, 30_000, 0.0, rng)
            thr = np.quantile(occ_latent, 1.0 - float(np.mean(p_wet)))
            wet = occ_latent > thr
            amount = np.exp(np.clip(np.log(30) + amp * elev_sens * (terrain["elev"] - 1800) / 1000 + 0.5 * resid, -1, 5))
            value = np.where(wet, np.maximum(amount, 1.0), 0.0)
        elif variable == "wind_mean":
            # Guard: mu must stay positive before log
            mu = st["mean"] + amp * tpi_sens * terrain["tpi"] + 0.00025 * (terrain["elev"] - 1000)
            mu = np.maximum(mu, 0.3)
            value = np.exp(np.log(mu) + 0.35 * resid)
            value = np.clip(value, 0.2, 20)
        else:  # rh_mean
            mu = st["mean"] + amp * elev_sens * (terrain["elev"] - 1200)
            value = mu + st["std"] * 0.5 * resid
            value = np.clip(value, 20, 98)

        for i, name in enumerate(terrain["station_name"]):
            rows.append({
                "station_name": name,
                "time": pd.Timestamp("2020-01-01") + pd.Timedelta(days=t),
                "x": float(terrain["x"].iloc[i]),
                "y": float(terrain["y"].iloc[i]),
                "elev": float(terrain["elev"].iloc[i]),
                "smooth_elev": float(terrain["smooth_elev"].iloc[i]),
                "tpi": float(terrain["tpi"].iloc[i]),
                "slope": float(terrain["slope"].iloc[i]),
                "aspect": float(terrain["aspect"].iloc[i]),
                "northness": float(terrain["northness"].iloc[i]),
                "dist_to_coast": float(terrain["dist_to_coast"].iloc[i]),
                "signed_dist_to_ridge": float(terrain["signed_dist_to_ridge"].iloc[i]),
                "clc_level2": int(terrain["clc_level2"].iloc[i]),
                "clc_code": int(terrain["clc_level2"].iloc[i]),  # alias for RGI/RFSI
                "clc_group": int(terrain["clc_level2"].iloc[i]),
                "value": float(value[i]),
                variable: float(value[i]),  # also under the variable name
                "is_heldout": name in held,
                "split": split,
            })

    return pd.DataFrame(rows)


def save_panel(panel: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(path, index=False)


if __name__ == "__main__":
    # Quick smoke test
    p = generate_synthetic_panel("temp_mean", n_stations=40, n_fields=8, seed=1)
    print(p.shape, p["value"].mean(), p["split"].value_counts().to_dict())
    print("held-out", p["is_heldout"].sum() // 8)
    print("columns", list(p.columns))
