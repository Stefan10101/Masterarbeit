#!/usr/bin/env python3
"""
rfsi_optimizer.py
Leave-Location-Out Cross-Validation optimizer for RFSI.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from typing import Dict, Tuple, List, Optional
from rfsi_core import RFSI


def compute_metrics(obs: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Compute RMSE, MAE, NSE and KGE."""
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    valid = ~np.isnan(obs) & ~np.isnan(pred)
    o = obs[valid]
    p = pred[valid]
    n = len(o)

    if n < 2:
        return {"rmse": np.nan, "mae": np.nan, "nse": np.nan, "kge": np.nan}

    rmse = float(np.sqrt(np.mean((o - p) ** 2)))
    mae = float(np.mean(np.abs(o - p)))

    ss_res = np.sum((o - p) ** 2)
    ss_tot = np.sum((o - np.mean(o)) ** 2)
    nse = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    if n > 1:
        r = np.corrcoef(o, p)[0, 1]
        alpha = np.std(p) / np.std(o) if np.std(o) > 0 else np.nan
        beta = np.mean(p) / np.mean(o) if np.mean(o) != 0 else np.nan
        kge = float(1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2))
    else:
        kge = np.nan

    return {"rmse": rmse, "mae": mae, "nse": nse, "kge": kge}


def _evaluate_combination(
    valid_df: pd.DataFrame,
    var_name: str,
    n_obs: int,
    rf_params: dict,
    time_col: str = "time"
) -> Dict:
    """Run full Leave-Location-Out CV for one parameter combination."""
    n = len(valid_df)
    if n < max(5, n_obs + 1):
        return {"n_obs": n_obs, **rf_params,
                "rmse": np.nan, "mae": np.nan, "nse": np.nan, "kge": np.nan}

    predictions = np.full(n, np.nan, dtype=np.float32)
    coords = valid_df[["x", "y"]].values.astype(np.float64)
    z_vals = valid_df[var_name].values.astype(np.float32)

    # Identify covariate columns
    exclude_cols = {"station_name", time_col, var_name, "x", "y"}
    cov_cols = [c for c in valid_df.columns if c not in exclude_cols]
    X_cov = valid_df[cov_cols].values.astype(np.float32) if cov_cols else None

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        model = RFSI(n_obs=n_obs, rf_params=rf_params)
        model.fit(
            coords=coords[mask],
            z=z_vals[mask],
            X_cov=X_cov[mask] if X_cov is not None else None
        )
        pred = model.predict(
            coords_pred=coords[[i]],
            X_cov_pred=X_cov[[i]] if X_cov is not None else None
        )
        predictions[i] = pred[0]

    metrics = compute_metrics(z_vals, predictions)
    return {"n_obs": n_obs, **rf_params, **metrics}


def optimize_rfsi_params_loocv(
    valid_df: pd.DataFrame,
    var_name: str,
    n_obs_list: List[int],
    rf_fixed: dict,
    rf_tunable: dict,
    primary_metric: str = "rmse",
    n_jobs: int = -1,
    verbose: bool = True,
    time_col: str = "time"
) -> Tuple[Dict, pd.DataFrame]:
    """
    Optimize RFSI hyperparameters using full Leave-Location-Out Cross-Validation.
    """
    from itertools import product

    param_names = list(rf_tunable.keys())
    param_values = [rf_tunable[name] for name in param_names]

    combinations = []
    for n_obs in n_obs_list:
        for values in product(*param_values):
            rf_params = rf_fixed.copy()
            rf_params.update(dict(zip(param_names, values)))
            combinations.append((n_obs, rf_params))

    if verbose:
        print(f"Optimizing {var_name} — {len(combinations)} combinations (full LLOCV)")

    results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(_evaluate_combination)(valid_df, var_name, n_obs, rf_params, time_col)
        for n_obs, rf_params in combinations
    )

    results_df = pd.DataFrame(results)

    if results_df[primary_metric].isna().all():
        print(f"[Warning] All combinations returned NaN for {var_name}.")
        fallback = {"n_obs": n_obs_list[0], **rf_fixed}
        return fallback, results_df

    # Select best
    if primary_metric in ["kge", "nse"]:
        best_idx = results_df[primary_metric].idxmax()
    else:
        best_idx = results_df[primary_metric].idxmin()

    best_row = results_df.loc[best_idx]

    best_params = {"n_obs": int(best_row["n_obs"])}
    best_params.update({k: best_row[k] for k in rf_fixed.keys()})
    best_params.update({k: best_row[k] for k in rf_tunable.keys()})

    best_scores = {
        "rmse": float(best_row["rmse"]),
        "mae": float(best_row["mae"]),
        "nse": float(best_row["nse"]),
        "kge": float(best_row["kge"])
    }

    if verbose:
        print(f"Best parameters for {var_name}: {best_params}")

    return best_params, best_scores