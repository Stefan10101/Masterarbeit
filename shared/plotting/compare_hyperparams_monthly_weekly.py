#!/usr/bin/env python3
"""
Compare tuned hyperparameters across monthly, weekly and daily resolutions.

Goal: decide whether hyperparameters found at coarser resolutions can be
reused for finer ones (daily / half-hourly), or whether they diverge enough
that retuning is required.

Loads the yaml files written by the tune_* scripts via paths.py helpers,
extracts the searched hyperparameters, and writes:

  - full long table of every pairwise comparison (CSV)
  - summary per method x variable x pair (CSV)
  - figures (heatmaps of fraction differing and max relative difference)

Run from anywhere; it locates the project root via paths.get_project_root().
Missing files for a given resolution are reported and that pair is skipped.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

# Locate CODE so imports work (script may live in shared/plotting/ or elsewhere)
CODE = Path(__file__).resolve().parent
while CODE != CODE.parent and not (CODE / "paths.py").exists():
    CODE = CODE.parent
if not (CODE / "paths.py").exists():
    raise RuntimeError("Could not locate paths.py relative to the script")
sys.path.insert(0, str(CODE))

from paths import (
    get_idw_tuned_params_path,
    get_frei_tuned_params_path,
    get_kriging_tuned_params_path,
    get_gam_tuned_params_path,
    get_rfsi_tuned_params_path,
    get_cnn_tuned_params_path,
    get_rgi_tuned_params_path,
    get_smacnp_tuned_params_path,
    get_project_root,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
METHODS = ["IDW", "Frei", "Kriging", "GAM", "RFSI", "CNN", "RGI", "SMACNP"]
VARS = ["temp_mean", "precip_sum", "wind_mean", "rh_mean", "snow_mean"]
TIMES = ["monthly", "weekly", "daily"]
PAIRS = [("monthly", "weekly"), ("monthly", "daily"), ("weekly", "daily")]

# Hyperparameters that are actually searched (ignore scores, metadata, etc.)
PARAM_KEYS = {
    "IDW": ["p", "Fz", "k"],
    "Frei": ["n_sections", "shrink", "across_w", "p", "Fz", "k"],
    "Kriging": ["k", "k_indicator", "family", "theta_deg", "aniso_ratio", "alpha_z", "interactions", "trend"],
    "GAM": ["formula", "n_splines", "tau_wet"],
    "RFSI": ["n_obs", "max_depth", "min_samples_leaf", "max_features"],
    "CNN": ["n_levels", "base_ch"],
    "RGI": ["k", "n_layers", "alpha", "alpha_z"],
    "SMACNP": ["k_localized", "hidden_size"],
}

SKIP_KEYS = {
    "variable", "time_resolution", "rmse", "mae", "nse", "kge", "ccc",
    "dev_rmse", "dev_mae", "dev_nse", "dev_kge", "dev_ccc", "val_rmse",
    "n", "finite", "sec", "search_seconds", "tune_months", "tune_mode",
    "tune_folds", "architecture", "split", "cluster_method", "n_medoid_months",
    "n_estimators", "n_estimators_search", "primary_metric", "boundary",
    "rh_t_mode", "trace", "dev", "dev_mse",
}

OUT_DIR = get_project_root() / "Plots" / "hyperparam_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_params(method: str, var: str, time_res: str) -> dict | None:
    """Return the dict written by the corresponding tune script, or None if missing."""
    getters = {
        "IDW": get_idw_tuned_params_path,
        "Frei": get_frei_tuned_params_path,
        "Kriging": get_kriging_tuned_params_path,
        "GAM": get_gam_tuned_params_path,
        "RFSI": get_rfsi_tuned_params_path,
        "CNN": get_cnn_tuned_params_path,
        "RGI": get_rgi_tuned_params_path,
        "SMACNP": get_smacnp_tuned_params_path,
    }
    path = getters[method](var, time_res)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def is_numeric(v) -> bool:
    return isinstance(v, (int, float, np.number)) and np.isfinite(v)


def relative_diff(a, b) -> float | None:
    """Symmetric relative difference for numeric values. None if not both numeric."""
    if not (is_numeric(a) and is_numeric(b)):
        return None
    a, b = float(a), float(b)
    denom = max(abs(a), abs(b), 1e-8)
    return abs(a - b) / denom


def collect() -> pd.DataFrame:
    rows = []
    missing = []
    for method in METHODS:
        keys = PARAM_KEYS.get(method, [])
        for var in VARS:
            loaded = {t: load_params(method, var, t) for t in TIMES}
            available = [t for t, d in loaded.items() if d is not None]
            if len(available) < 2:
                missing.append((method, var, available))
                continue
            for t1, t2 in PAIRS:
                d1, d2 = loaded[t1], loaded[t2]
                if d1 is None or d2 is None:
                    continue
                present = set(keys)
                for d in (d1, d2):
                    present.update(k for k in d if k not in SKIP_KEYS and not k.startswith("dev_"))
                for key in sorted(present):
                    v1 = d1.get(key)
                    v2 = d2.get(key)
                    if v1 is None and v2 is None:
                        continue
                    same = (v1 == v2) or (
                        is_numeric(v1) and is_numeric(v2) and abs(float(v1) - float(v2)) < 1e-6
                    )
                    rel = relative_diff(v1, v2)
                    rows.append({
                        "method": method,
                        "variable": var,
                        "pair": f"{t1}_vs_{t2}",
                        "param": key,
                        t1: v1,
                        t2: v2,
                        "same": same,
                        "rel_diff": rel,
                        "abs_diff": abs(float(v1) - float(v2)) if is_numeric(v1) and is_numeric(v2) else None,
                    })
    if missing:
        print("Insufficient resolutions (need at least two of monthly/weekly/daily):")
        for m, v, avail in missing:
            print(f"  {m} / {v}: {avail or 'none'}")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    def _agg(g):
        n = len(g)
        n_diff = int((~g["same"]).sum())
        rels = g["rel_diff"].dropna()
        return pd.Series({
            "n_params": n,
            "n_differ": n_diff,
            "frac_differ": n_diff / n if n else 0,
            "max_rel_diff": float(rels.max()) if len(rels) else np.nan,
            "mean_rel_diff": float(rels.mean()) if len(rels) else np.nan,
        })
    return df.groupby(["method", "variable", "pair"]).apply(_agg).reset_index()


def plot_heatmaps(summary: pd.DataFrame):
    if summary.empty:
        return
    for pair in summary["pair"].unique():
        sub = summary[summary["pair"] == pair]
        for metric, vmax, fname in [
            ("frac_differ", 1.0, f"frac_differ_{pair}.png"),
            ("max_rel_diff", None, f"max_rel_diff_{pair}.png"),
        ]:
            pivot = sub.pivot(index="method", columns="variable", values=metric)
            plt.figure(figsize=(8, 5))
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd",
                        vmin=0, vmax=vmax, cbar_kws={"label": metric})
            plt.title(f"{metric} ({pair.replace('_', ' ')})")
            plt.tight_layout()
            plt.savefig(OUT_DIR / fname, dpi=150)
            plt.close()


def plot_param_bars(df: pd.DataFrame):
    """One figure per method x pair: relative difference of each numeric param."""
    if df.empty:
        return
    for method in df["method"].unique():
        for pair in df["pair"].unique():
            sub = df[(df["method"] == method) & (df["pair"] == pair) & df["rel_diff"].notna()]
            if sub.empty:
                continue
            pivot = sub.pivot(index="param", columns="variable", values="rel_diff")
            pivot.plot(kind="bar", figsize=(9, 4), legend=True)
            plt.title(f"{method}: relative difference ({pair.replace('_', ' ')})")
            plt.ylabel("symmetric relative difference")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            safe = pair.replace("_", "-")
            plt.savefig(OUT_DIR / f"{method}_rel_diff_{safe}.png", dpi=150)
            plt.close()


def main():
    print("Collecting hyperparameters across monthly / weekly / daily...")
    df = collect()
    if df.empty:
        print("No overlapping resolution pairs found. Check Methods/*/Output/.")
        return

    full_path = OUT_DIR / "hyperparam_comparison.csv"
    df.to_csv(full_path, index=False)
    print(f"Wrote {full_path}  ({len(df)} rows)")

    summary = summarize(df)
    sum_path = OUT_DIR / "hyperparam_summary.csv"
    summary.to_csv(sum_path, index=False)
    print(f"Wrote {sum_path}")

    print("\nFraction of hyperparameters that differ (by pair):")
    for pair in summary["pair"].unique():
        print(f"\n{pair}:")
        print(summary[summary["pair"] == pair]
              .pivot(index="method", columns="variable", values="frac_differ")
              .round(2).to_string())

    plot_heatmaps(summary)
    plot_param_bars(df)

    md = OUT_DIR / "summary.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("# Hyperparameter comparison across resolutions\n\n")
        for pair in summary["pair"].unique():
            sub = summary[summary["pair"] == pair]
            f.write(f"## {pair}\n\n")
            f.write("Fraction differing:\n\n```\n")
            f.write(sub.pivot(index="method", columns="variable", values="frac_differ").round(2).to_string())
            f.write("\n```\n\nMax relative difference:\n\n```\n")
            f.write(sub.pivot(index="method", columns="variable", values="max_rel_diff").round(2).to_string())
            f.write("\n```\n\n")
    print(f"Wrote {md}")
    print(f"\nFigures and tables in: {OUT_DIR}")


if __name__ == "__main__":
    main()
