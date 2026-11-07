#!/usr/bin/env python3
"""Scatter + metrics from a nested or plain LLOCV parquet (not frozen-model output).

Resolves paths for both get_nested_llocv_path (Frei, GAM, Kriging, RFSI, ...) and
get_llocv_path (IDW). For Kriging, tries the plain variable name and the common
_rk / _ok suffixes automatically.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_DIR))

from paths import (
    get_nested_llocv_path,
    get_llocv_path,
    get_method_output_dir,
    get_llocv_compare_dir,
)
from shared.metrics import compute_metrics, format_eval

CORE_VARS = ("temp_mean", "precip_sum", "wind_mean", "rh_mean", "snow_mean")
# Thesis suite order; missing methods are skipped.
DEFAULT_METHODS = ("IDW", "Kriging", "Frei", "GAM", "RFSI", "CNN", "SMACNP", "RGI")


def _resolve_llocv_path(method: str, variable: str, time_resolution: str) -> Path:
    """Return the first existing LLOCV parquet for this method / variable / timescale."""
    candidates: list[Path] = []

    # 1. Nested path (most methods)
    candidates.append(get_nested_llocv_path(method, variable, time_resolution))

    # 2. Kriging tags (writes {var}_rk or {var}_ok)
    if method.lower() == "kriging" and not variable.lower().endswith(("_rk", "_ok")):
        for tag in ("rk", "ok"):
            candidates.append(
                get_nested_llocv_path(method, f"{variable}_{tag}", time_resolution)
            )

    # 3. Plain LLOCV path used by IDW (and possibly others)
    # Common period embedded in filename by produce/llocv scripts
    date_pairs = [
        (None, None),
        ("2020-01-01", "2025-12-31"),
    ]
    for start, end in date_pairs:
        candidates.append(
            get_llocv_path(
                method, "full", variable, 1000,
                start_date=start, end_date=end, time_resolution=time_resolution,
            )
        )

    # 4. Glob fallback under the method's llocv tree (catches unexpected naming)
    method_dir = get_method_output_dir(method) / "llocv"
    if method_dir.exists():
        patterns = [
            f"**/{variable}*{time_resolution}*nested_llocv.parquet",
            f"**/{variable}*{time_resolution}*llocv.parquet",
            f"**/*{variable}*{time_resolution}*.parquet",
        ]
        for pat in patterns:
            for p in method_dir.glob(pat):
                if p.is_file() and p not in candidates:
                    candidates.append(p)

    for p in candidates:
        if p.exists():
            return p

    # Build a helpful error
    tried = "\n  ".join(str(c) for c in candidates[:12])
    raise FileNotFoundError(
        f"No LLOCV parquet found for method={method!r} variable={variable!r} "
        f"time_resolution={time_resolution!r}.\nTried:\n  {tried}"
    )


def plot_one(
    df: pd.DataFrame,
    title: str,
    out: Path,
    variable: str,
    time_resolution: str,
):
    met = compute_metrics(
        df["observed"].to_numpy(),
        df["predicted"].to_numpy(),
        variable=variable,
        time_resolution=time_resolution,
    )
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.scatter(df["observed"], df["predicted"], s=6, alpha=0.25, linewidths=0)
    lo = np.nanmin([df["observed"].min(), df["predicted"].min()])
    hi = np.nanmax([df["observed"].max(), df["predicted"].max()])
    ax.plot([lo, hi], [lo, hi], color="k", lw=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("observed")
    ax.set_ylabel("predicted (station out)")
    ax.set_title(title)

    # Compact multi-line annotation using the shared metric set
    def _fmt(key: str, fmt: str = ".3f") -> str:
        v = met.get(key, np.nan)
        if v is None or not (v == v):
            return "nan"
        return format(float(v), fmt)

    lines = [
        f"n={int(met.get('n', len(df)))}",
        f"ME={_fmt('me')}",
        f"MAE={_fmt('mae')}",
        f"RMSE={_fmt('rmse')}",
        f"RMSE/std={_fmt('rmse_std')}",
        f"NSE={_fmt('nse')}",
        f"KGE={_fmt('kge')}",
        f"CCC={_fmt('ccc')}",
        f"r={_fmt('r')}",
    ]
    if "rmse_wet" in met and met.get("rmse_wet") == met.get("rmse_wet"):
        lines += [
            f"wetRMSE={_fmt('rmse_wet')}",
            f"freq={_fmt('freq_bias')}",
            f"POD={_fmt('pod')}",
            f"FAR={_fmt('far')}",
            f"ETS={_fmt('ets')}",
        ]
    txt = "\n".join(lines)
    ax.text(
        0.04, 0.96, txt, transform=ax.transAxes, va="top",
        fontsize=8, family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, lw=0),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  {out.name}")
    print(format_eval(met, prefix="  "))


def _load_llocv(method: str, variable: str, time_resolution: str, split: str) -> pd.DataFrame | None:
    """Load and filter an LLOCV parquet. Returns None if missing or empty."""
    try:
        src = _resolve_llocv_path(method, variable, time_resolution)
    except FileNotFoundError:
        return None
    df = pd.read_parquet(src)
    if "split" in df.columns and split != "all":
        df = df[df["split"] == split]
    if df.empty or "observed" not in df.columns or "predicted" not in df.columns:
        return None
    return df


def plot_methods(
    frames: dict[str, pd.DataFrame],
    variable: str,
    time_resolution: str,
    split: str,
    out: Path,
):
    """One figure, one subplot per method, shared axis limits."""
    methods = list(frames)
    n = len(methods)
    if n == 0:
        return
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows), squeeze=False)

    # Shared limits across methods
    all_obs = np.concatenate([f["observed"].to_numpy() for f in frames.values()])
    all_pred = np.concatenate([f["predicted"].to_numpy() for f in frames.values()])
    lo = float(np.nanmin([all_obs.min(), all_pred.min()]))
    hi = float(np.nanmax([all_obs.max(), all_pred.max()]))

    for ax, method in zip(axes.flat, methods):
        df = frames[method]
        met = compute_metrics(
            df["observed"].to_numpy(),
            df["predicted"].to_numpy(),
            variable=variable,
            time_resolution=time_resolution,
        )
        ax.scatter(df["observed"], df["predicted"], s=4, alpha=0.2, linewidths=0)
        ax.plot([lo, hi], [lo, hi], color="k", lw=0.7)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(method, fontsize=10)
        ax.set_xlabel("observed", fontsize=8)
        ax.set_ylabel("predicted", fontsize=8)

        def _fmt(key: str) -> str:
            v = met.get(key, np.nan)
            if v is None or not (v == v):
                return "nan"
            return f"{float(v):.2f}"

        txt = (
            f"n={int(met.get('n', 0))}\n"
            f"RMSE={_fmt('rmse')}\n"
            f"NSE={_fmt('nse')}\n"
            f"KGE={_fmt('kge')}\n"
            f"r={_fmt('r')}"
        )
        ax.text(
            0.03, 0.97, txt, transform=ax.transAxes, va="top",
            fontsize=7, family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, lw=0),
        )

    # Hide unused axes
    for ax in axes.flat[n:]:
        ax.set_visible(False)

    fig.suptitle(f"{variable}  {time_resolution}  {split}", fontsize=12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  {out}")


def run_compare(methods: list[str], variables: list[str], time_resolution: str, split: str):
    """Write one multi-method scatter per variable into Plots/llocv_compare/."""
    for var in variables:
        frames = {}
        for method in methods:
            df = _load_llocv(method, var, time_resolution, split)
            if df is None:
                print(f"[skip] {method} {var}")
                continue
            frames[method] = df
            print(f"  loaded {method} {var}  n={len(df)}")
        if not frames:
            print(f"[skip] no methods for {var}")
            continue
        out = (
            get_llocv_compare_dir(time_resolution, var)
            / f"{var}_{time_resolution}_{split}_methods_scatter.png"
        )
        plot_methods(frames, var, time_resolution, split, out)
    print("\nCompare plots written under Plots/llocv_compare/")


def main():
    p = argparse.ArgumentParser(
        description="Scatter plot + full metrics from an LLOCV parquet."
    )
    p.add_argument("--method", default="RFSI", help="Method name (IDW, Kriging, Frei, GAM, ...)")
    p.add_argument(
        "--methods", default=None,
        help="Comma-separated methods for --compare (default: finished suite). "
             "Example: IDW,Kriging,Frei,GAM",
    )
    p.add_argument(
        "--variable", default="all",
        help="Variable code (temp_mean, precip_sum, ...) or 'all' for the five core variables",
    )
    p.add_argument("--time-resolution", default="monthly", help="weekly | daily | monthly | half_hourly")
    p.add_argument(
        "--split", default="all",
        help="all | train | tune | dev | test  (filters the 'split' column if present)",
    )
    p.add_argument(
        "--compare", action="store_true",
        help="One figure per variable with a subplot for each available method. "
             "Written to Plots/llocv_compare/{time_resolution}/{variable}/",
    )
    args = p.parse_args()

    if args.variable.lower() == "all":
        variables = list(CORE_VARS)
    else:
        variables = [args.variable]

    if args.compare:
        if args.methods:
            methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        else:
            methods = list(DEFAULT_METHODS)
        run_compare(methods, variables, args.time_resolution, args.split)
        return

    out_dir = get_method_output_dir(args.method) / "plots" / "llocv_nested"
    tag = args.split
    ok, missing = [], []

    for var in variables:
        try:
            src = _resolve_llocv_path(args.method, var, args.time_resolution)
        except FileNotFoundError as e:
            print(f"[skip] {var}: {e}")
            missing.append(var)
            continue

        print(f"Reading {src}")
        df = pd.read_parquet(src)

        if "split" in df.columns and args.split != "all":
            df = df[df["split"] == args.split]
        if df.empty:
            print(f"[skip] {var}: no rows for split={args.split}")
            missing.append(var)
            continue

        for col in ("observed", "predicted"):
            if col not in df.columns:
                print(f"[skip] {var}: missing column {col!r}")
                missing.append(var)
                break
        else:
            out = out_dir / f"{var}_{args.time_resolution}_{tag}_scatter.png"
            plot_one(
                df,
                f"{args.method} LLOCV | {var} | {args.time_resolution} | {tag}",
                out,
                variable=var,
                time_resolution=args.time_resolution,
            )
            ok.append(var)

    print(f"\nDone. plotted={ok}  missing={missing}")


if __name__ == "__main__":
    main()
