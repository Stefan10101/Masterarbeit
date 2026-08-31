#!/usr/bin/env python3
"""Scatter + metrics from a nested LLOCV parquet (not frozen-model output)."""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[2]))
from paths import get_nested_llocv_path, get_method_output_dir

sys.path.append(str(Path(__file__).resolve().parents[2] / "RFSI"))
from rfsi_optimizer import compute_metrics


def plot_one(df: pd.DataFrame, title: str, out: Path):
    met = compute_metrics(df["observed"].to_numpy(), df["predicted"].to_numpy())
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.scatter(df["observed"], df["predicted"], s=6, alpha=0.25, linewidths=0)
    lo = np.nanmin([df["observed"].min(), df["predicted"].min()])
    hi = np.nanmax([df["observed"].max(), df["predicted"].max()])
    ax.plot([lo, hi], [lo, hi], color="k", lw=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("observed")
    ax.set_ylabel("predicted (station out)")
    ax.set_title(title)
    txt = (
        f"n={len(df)}\n"
        f"RMSE={met['rmse']:.3f}\n"
        f"MAE={met['mae']:.3f}\n"
        f"NSE={met['nse']:.3f}\n"
        f"KGE={met['kge']:.3f}\n"
        f"CCC={met['ccc']:.3f}"
    )
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, lw=0))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  {out.name}  RMSE={met['rmse']:.3f}  NSE={met['nse']:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="RFSI")
    p.add_argument("--variable", required=True)
    p.add_argument("--time-resolution", default="monthly")
    p.add_argument("--split", default="all",
                   help="all | train | dev | test")
    args = p.parse_args()

    src = get_nested_llocv_path(args.method, args.variable, args.time_resolution)
    if not src.exists():
        raise FileNotFoundError(src)
    df = pd.read_parquet(src)
    if args.split != "all":
        df = df[df["split"] == args.split]
    if df.empty:
        raise RuntimeError(f"no rows for split={args.split}")

    out_dir = get_method_output_dir(args.method) / "plots" / "llocv_nested"
    tag = args.split
    plot_one(
        df,
        f"{args.method} nested LLOCV | {args.variable} | {args.time_resolution} | {tag}",
        out_dir / f"{args.variable}_{args.time_resolution}_{tag}_scatter.png",
    )


if __name__ == "__main__":
    main()
