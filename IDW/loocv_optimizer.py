#!/usr/bin/env python3
"""
loocv_optimizer.py
LOOCV Parameter Optimiser for Modified IDW.

Tunes p, Fz, and k using leave-one-out cross-validation.
Default primary metric is now RMSE (more suitable for spatial interpolation).
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from idw_core import modified_idw
from joblib import Parallel, delayed
from typing import Dict, Tuple


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

    rmse = np.sqrt(np.mean((o - p) ** 2))
    mae = np.mean(np.abs(o - p))

    ss_res = np.sum((o - p) ** 2)
    ss_tot = np.sum((o - np.mean(o)) ** 2)
    nse = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    r = np.corrcoef(o, p)[0, 1] if n > 1 else np.nan
    alpha = np.std(p) / np.std(o) if np.std(o) > 0 else np.nan
    beta = np.mean(p) / np.mean(o) if np.mean(o) != 0 else np.nan
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    return {"rmse": rmse, "mae": mae, "nse": nse, "kge": kge}


def loocv_for_params(valid_df, var_name, p, Fz, k):
    n = len(valid_df)
    if n < 3:
        return {"rmse": np.nan, "mae": np.nan, "nse": np.nan, "kge": np.nan}

    predictions = np.full(n, np.nan)
    station_coords = valid_df[["x", "y"]].values
    station_elev = valid_df["elev"].values
    values = valid_df[var_name].values

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        loo_coords = station_coords[mask]
        loo_elev = station_elev[mask]
        loo_values = values[mask]

        if len(loo_values) < k:
            continue

        tree_loo = KDTree(loo_coords)
        pred = modified_idw(
            station_values=loo_values,
            station_coords=loo_coords,
            station_elev=loo_elev,
            target_coords=station_coords[[i]],
            target_elev=station_elev[[i]],
            tree=tree_loo,
            p=p, Fz=Fz, k=min(k, len(loo_values))
        )
        predictions[i] = pred[0]

    return compute_metrics(values, predictions)


def _evaluate_one_combination(args):
    valid_df, var_name, p, Fz, k = args
    scores = loocv_for_params(valid_df, var_name, p, Fz, k)
    return {"p": p, "Fz": Fz, "k": k, **scores}


def optimize_idw_params_loocv(
    valid_df: pd.DataFrame,
    var_name: str,
    param_grid: Dict,
    primary_metric: str = "rmse",   # Changed default to rmse
    verbose: bool = True,
    n_jobs: int = -1
) -> Tuple[Dict, Dict]:

    p_list = param_grid.get("p", [2.0])
    Fz_list = param_grid.get("Fz", [0.0, 0.3])
    k_list = param_grid.get("k_neighbors", [12])

    combinations = [
        (valid_df, var_name, p, Fz, k)
        for p in p_list for Fz in Fz_list for k in k_list
    ]

    if verbose:
        print(f"Optimising {var_name} with {len(combinations)} combinations...")

    results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(_evaluate_one_combination)(comb) for comb in combinations
    )

    results_df = pd.DataFrame(results)

    if results_df[primary_metric].isna().all():
        print(f"  [Warning] All combinations returned NaN for {var_name}. Using fallback.")
        return {"p": 2.0, "Fz": 0.3, "k": 12}, {"rmse": np.nan, "mae": np.nan, "nse": np.nan, "kge": np.nan}

    if primary_metric in ["kge", "nse"]:
        best_idx = results_df[primary_metric].idxmax()
    else:
        best_idx = results_df[primary_metric].idxmin()

    best_row = results_df.loc[best_idx]
    best_params = {"p": best_row["p"], "Fz": best_row["Fz"], "k": int(best_row["k"])}
    best_scores = {
        "rmse": best_row["rmse"],
        "mae": best_row["mae"],
        "nse": best_row["nse"],
        "kge": best_row["kge"]
    }

    if verbose:
        print(f"Best parameters for {var_name}: p={best_params['p']}, Fz={best_params['Fz']}, k={best_params['k']}")

    return best_params, best_scores


if __name__ == "__main__":
    print("loocv_optimizer.py ready.")

def loocv_predictions(valid_df, var_name, p, Fz, k):
    """Leave-one-out predictions for fixed params. Returns DataFrame."""
    import numpy as np
    import pandas as pd
    from sklearn.neighbors import KDTree
    from idw_core import modified_idw

    n = len(valid_df)
    if n < 3:
        return pd.DataFrame(columns=["station_name", "x", "y", "observed", "predicted"])
    station_coords = valid_df[["x", "y"]].values
    station_elev = valid_df["elev"].values
    values = valid_df[var_name].values
    names = valid_df["station_name"].values if "station_name" in valid_df.columns else np.arange(n)
    records = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        loo_coords = station_coords[mask]
        loo_elev = station_elev[mask]
        loo_values = values[mask]
        if len(loo_values) < 2:
            continue
        k_use = min(k, len(loo_values))
        tree_loo = KDTree(loo_coords)
        pred = modified_idw(
            station_values=loo_values, station_coords=loo_coords, station_elev=loo_elev,
            target_coords=station_coords[[i]], target_elev=station_elev[[i]],
            tree=tree_loo, p=p, Fz=Fz, k=k_use,
        )
        records.append({
            "station_name": names[i],
            "x": float(station_coords[i, 0]),
            "y": float(station_coords[i, 1]),
            "observed": float(values[i]),
            "predicted": float(pred[0]),
        })
    return pd.DataFrame(records)
