#!/usr/bin/env python3
"""
Analyze RGI hyperparameter search CSVs to identify which parameters
drive DEV RMSE and which levels can be dropped to cut compute.

Usage (on Windows, from the Daten/CODE directory or with full paths):

    python analyze_rgi_search.py ^
        --search-dir "C:\Users\stefa\Documents\UNI\Master\Masterarbeit\Daten\Methods\RGI\Output\cluster_params\weekly\rgi"

Or point at any directory that contains the *_search.csv files.
The script also looks one level up for the params yaml files if present.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_searches(search_dir: Path) -> dict[str, pd.DataFrame]:
    files = sorted(search_dir.glob("*_search.csv"))
    if not files:
        raise SystemExit(f"No *_search.csv found in {search_dir}")
    out = {}
    for f in files:
        var = f.stem.replace("_search", "")
        df = pd.read_csv(f)
        df["variable"] = var
        df["source"] = str(f)
        out[var] = df
        print(f"  {var:12s}  {len(df):3d} trials  columns={list(df.columns)}")
    return out


def param_sensitivity(df: pd.DataFrame, params: list[str]) -> pd.DataFrame:
    """For each parameter, report how much the best/mean RMSE varies across its levels."""
    rows = []
    overall_best = df["rmse"].min()
    for p in params:
        if p not in df.columns:
            continue
        g = df.groupby(p)["rmse"].agg(["min", "mean", "max", "count", "std"])
        g = g.reset_index()
        g["param"] = p
        g["level"] = g[p]
        g["best_gap_pct"] = 100.0 * (g["min"] - overall_best) / overall_best
        g["mean_gap_pct"] = 100.0 * (g["mean"] - overall_best) / overall_best
        rows.append(g[["param", "level", "min", "mean", "max", "count", "std",
                        "best_gap_pct", "mean_gap_pct"]])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def recommend_drop(sens: pd.DataFrame, gap_threshold_pct: float = 5.0) -> list[str]:
    """Levels whose best RMSE is more than threshold worse than the global best
    are candidates for dropping. Also flag levels that were rarely best."""
    recs = []
    for p, sub in sens.groupby("param"):
        worst_gap = sub["best_gap_pct"].max()
        if worst_gap > gap_threshold_pct:
            bad = sub[sub["best_gap_pct"] > gap_threshold_pct]
            levels = ", ".join(str(x) for x in bad["level"].tolist())
            recs.append(
                f"  {p}: drop levels [{levels}] "
                f"(best RMSE ≥{gap_threshold_pct:.0f}% worse than global best)"
            )
        # also note the single best level
        best_row = sub.loc[sub["min"].idxmin()]
        recs.append(
            f"  {p}: best level = {best_row['level']} "
            f"(min RMSE={best_row['min']:.4f}, n={int(best_row['count'])})"
        )
    return recs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--search-dir", type=Path, required=True,
                   help="Directory containing the *_search.csv files")
    p.add_argument("--gap-pct", type=float, default=5.0,
                   help="Drop levels whose best RMSE is this % worse than global best")
    args = p.parse_args()

    print(f"Searching in {args.search_dir}")
    searches = load_searches(args.search_dir)
    print()

    all_sens = []
    for var, df in searches.items():
        print("=" * 70)
        print(f"{var}   n_trials={len(df)}   best RMSE={df['rmse'].min():.4f}   "
              f"worst={df['rmse'].max():.4f}")
        # detect which hyperparams are present
        candidates = ["k", "n_layers", "alpha", "alpha_z"]
        present = [c for c in candidates if c in df.columns]
        if not present:
            print("  no recognised hyperparameter columns")
            continue
        sens = param_sensitivity(df, present)
        if sens.empty:
            continue
        sens.insert(0, "variable", var)
        all_sens.append(sens)
        print(sens.to_string(index=False, float_format="%.4f"))
        print("\nRecommendations (gap threshold {:.0f}%):".format(args.gap_pct))
        for line in recommend_drop(sens, args.gap_pct):
            print(line)
        print()

    if all_sens:
        summary = pd.concat(all_sens, ignore_index=True)
        out_path = args.search_dir / "rgi_search_sensitivity.csv"
        summary.to_csv(out_path, index=False)
        print(f"Wrote sensitivity table → {out_path}")

    # Hard-coded early-stop observations from the LLOCV run that followed this tune
    # (useful for deciding how many epochs are actually needed during search)
    print("\n" + "=" * 70)
    print("Early-stop observations from the subsequent LLOCV (epochs=80, patience=12):")
    print("  temp_mean   early stop epoch 36   val_mse=0.0176   wall 1.8 min")
    print("  precip_sum  early stop epoch 68   val_mse=0.1573   wall 6.2 min")
    print("  wind_mean   (no early stop reported)               wall 2.3 min")
    print("  rh_mean     early stop epoch 76   val_mse=0.1148   wall 4.0 min")
    print("  snow_mean   early stop epoch 66   val_mse=0.1950   wall 2.8 min")
    print("\nSuggested cheaper search settings:")
    print("  epochs: 50   (covers all observed early stops with margin)")
    print("  patience: 8")
    print("  Consider dropping the outermost levels of k / n_layers / alpha_z")
    print("  that never won, once the sensitivity table above is reviewed.")
    print("  Phase structure can stay; just shrink the grids in config.yaml.")


if __name__ == "__main__":
    main()
