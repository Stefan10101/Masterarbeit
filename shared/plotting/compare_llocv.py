#!/usr/bin/env python3
"""Compare nested / produce LLOCV parquet files across interpolation methods.

Metrics follow the usual climate-interpolation set
(Hofstra et al. 2008, Haylock et al. 2008, Frei 2014, Sekulić et al. 2020):
  ME, MAE, RMSE, NSE, KGE, CCC, Pearson r
plus skill vs a reference (IDW if present), meteorological seasons,
elevation bands, and precip wet-day / frequency scores.

Usage (from CODE/):
  python shared/plotting/compare_llocv.py --variable temp_mean
  python shared/plotting/compare_llocv.py --variable precip_sum --split test
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

from paths import (  # noqa: E402
    get_domain_stations_path,
    get_llocv_compare_dir,
    get_llocv_path,
    get_method_output_dir,
    get_nested_llocv_path,
)

DEFAULT_METHODS = [
    "IDW", "BSS", "RFSI", "RGI", "Kriging", "GAM", "TPS", "CNN", "Frei",
]
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
ELEV_BINS = [0, 500, 1000, 1500, 2000, 10_000]
ELEV_LABELS = ["<500", "500-1000", "1000-1500", "1500-2000", ">2000"]
TRACE_MM = {"half_hourly": 0.05, "daily": 0.1, "weekly": 0.7, "monthly": 1.0, "seasonal": 3.0}
COLORS = {
    "IDW": "#7f7f7f", "BSS": "#8c564b", "RFSI": "#1f77b4", "RGI": "#17becf",
    "Kriging": "#2ca02c", "Kriging-OK": "#98df8a", "Kriging-RK": "#2ca02c",
    "GAM": "#d62728", "TPS": "#ff7f0e", "CNN": "#9467bd", "Frei": "#e377c2",
}


def compute_metrics(obs, pred) -> dict:
    o = np.asarray(obs, dtype=float)
    p = np.asarray(pred, dtype=float)
    m = np.isfinite(o) & np.isfinite(p)
    o, p = o[m], p[m]
    n = int(len(o))
    empty = {k: np.nan for k in ("n", "me", "mae", "rmse", "nse", "kge", "ccc", "r")}
    empty["n"] = n
    if n < 2:
        return empty
    err = p - o
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    me = float(np.mean(err))
    ss_tot = float(np.sum((o - o.mean()) ** 2))
    nse = float(1.0 - np.sum((o - p) ** 2) / ss_tot) if ss_tot > 0 else np.nan
    r = float(np.corrcoef(o, p)[0, 1])
    so, sp = float(np.std(o)), float(np.std(p))
    alpha = float(sp / so) if so > 0 else np.nan
    beta = float(np.mean(p) / np.mean(o)) if np.mean(o) != 0 else np.nan
    kge = float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
    mx, my = float(o.mean()), float(p.mean())
    sxx = float(np.mean((o - mx) ** 2))
    syy = float(np.mean((p - my) ** 2))
    sxy = float(np.mean((o - mx) * (p - my)))
    den = sxx + syy + (mx - my) ** 2
    ccc = float(2.0 * sxy / den) if den > 0 else np.nan
    return {"n": n, "me": me, "mae": mae, "rmse": rmse, "nse": nse, "kge": kge, "ccc": ccc, "r": r}


def precip_extra(obs, pred, trace: float) -> dict:
    o = np.asarray(obs, dtype=float)
    p = np.asarray(pred, dtype=float)
    m = np.isfinite(o) & np.isfinite(p)
    o, p = o[m], p[m]
    ow, pw = o >= trace, p >= trace
    n = len(o)
    hits = int(np.sum(ow & pw))
    fa = int(np.sum(~ow & pw))
    miss = int(np.sum(ow & ~pw))
    wet = ow
    rmse_wet = float(np.sqrt(np.mean((p[wet] - o[wet]) ** 2))) if wet.any() else np.nan
    freq_bias = float(pw.mean() / ow.mean()) if ow.any() else np.nan
    denom = hits + fa + miss
    ets = np.nan
    if n and denom:
        hits_rand = (hits + miss) * (hits + fa) / n
        ets = float((hits - hits_rand) / (denom - hits_rand)) if (denom - hits_rand) else np.nan
    return {"rmse_wet": rmse_wet, "freq_bias": freq_bias, "pod": hits / (hits + miss) if (hits + miss) else np.nan,
            "far": fa / (hits + fa) if (hits + fa) else np.nan, "ets": ets}


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    cols = {c.lower(): c for c in df.columns}
    for want, aliases in {
        "observed": ("observed", "obs", "actual", "y_true"),
        "predicted": ("predicted", "pred", "yhat", "y_pred"),
        "station_name": ("station_name", "station", "name"),
        "time": ("time", "timestamp", "date"),
        "split": ("split",),
        "elev": ("elev", "elevation", "z", "height"),
    }.items():
        for a in aliases:
            if a in cols:
                rename[cols[a]] = want
                break
    out = df.rename(columns=rename).copy()
    if "time" in out.columns:
        out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce").dt.tz_localize(None)
    return out


def _label_from_path(method: str, path: Path) -> str:
    name = path.name.upper()
    if method == "Kriging":
        if "_RK_" in name or name.startswith("RK_") or "_RK." in name:
            return "Kriging-RK"
        if "_OK_" in name or name.startswith("OK_"):
            return "Kriging-OK"
    return method


def find_parquets(method: str, var: str, time_res: str, domain: str, resolution: int,
                  start: str, end: str) -> list[Path]:
    hits: list[Path] = []
    nested = get_nested_llocv_path(method, var, time_res, domain)
    if nested.exists():
        hits.append(nested)
    ndir = nested.parent
    if ndir.exists():
        hits.extend(sorted(ndir.glob(f"{var}*{time_res}*nested_llocv.parquet")))
        hits.extend(sorted(ndir.glob(f"*_{var}_{time_res}_nested_llocv.parquet")))
    prod = get_llocv_path(method, domain, var, resolution, start, end, time_res)
    if prod.exists():
        hits.append(prod)
    pdir = get_method_output_dir(method) / "llocv" / domain / f"res_{resolution}m" / var
    if pdir.exists():
        hits.extend(sorted(pdir.glob("*llocv.parquet")))
    seen, out = set(), []
    for p in hits:
        key = str(p.resolve()) if p.exists() else None
        if key and key not in seen:
            seen.add(key)
            out.append(p)
    return out


def attach_elev(df: pd.DataFrame) -> pd.DataFrame:
    if "elev" in df.columns and df["elev"].notna().any():
        return df
    try:
        sta = pd.read_parquet(get_domain_stations_path("RFSI", "full"))
    except Exception:
        return df
    col = next((c for c in ("elev", "elevation", "z") if c in sta.columns), None)
    if col is None or "station_name" not in sta.columns:
        return df
    meta = sta[["station_name", col]].drop_duplicates("station_name").rename(columns={col: "elev"})
    return df.merge(meta, on="station_name", how="left")


def load_all(methods, var, time_res, domain, resolution, start, end) -> pd.DataFrame:
    frames = []
    for method in methods:
        paths = find_parquets(method, var, time_res, domain, resolution, start, end)
        if not paths:
            print(f"  skip {method}: no LLOCV parquet", flush=True)
            continue
        for path in paths:
            raw = pd.read_parquet(path)
            df = _norm_cols(raw)
            if "observed" not in df.columns or "predicted" not in df.columns:
                print(f"  skip {path.name}: missing observed/predicted", flush=True)
                continue
            label = _label_from_path(method, path)
            df["method"] = label
            df["source"] = str(path)
            if "split" not in df.columns:
                df["split"] = "all"
            df = attach_elev(df)
            print(f"  {label:12} {len(df):7d} rows  {path}", flush=True)
            frames.append(df)
    if not frames:
        raise FileNotFoundError("no LLOCV parquet found for any method")
    return pd.concat(frames, ignore_index=True)


def season_of(ts: pd.Series) -> pd.Series:
    m = pd.to_datetime(ts).dt.month
    out = pd.Series(index=ts.index, dtype=object)
    for name, months in SEASONS.items():
        out[m.isin(months)] = name
    return out


def metrics_table(df: pd.DataFrame, var: str, time_res: str) -> pd.DataFrame:
    rows = []
    is_p = var.lower().startswith("precip") or var.lower().startswith("snow")
    trace = TRACE_MM.get(time_res, 1.0)
    for (method, split), g in df.groupby(["method", "split"], sort=False):
        row = {"method": method, "split": split, "subset": "all", **compute_metrics(g.observed, g.predicted)}
        if is_p:
            row.update(precip_extra(g.observed, g.predicted, trace))
        rows.append(row)
    for method, g0 in df.groupby("method", sort=False):
        row = {"method": method, "split": "all", "subset": "all", **compute_metrics(g0.observed, g0.predicted)}
        if is_p:
            row.update(precip_extra(g0.observed, g0.predicted, trace))
        rows.append(row)
        g0 = g0.copy()
        g0["season"] = season_of(g0["time"])
        for season, gs in g0.groupby("season"):
            row = {"method": method, "split": "all", "subset": str(season),
                   **compute_metrics(gs.observed, gs.predicted)}
            rows.append(row)
        if "elev" in g0.columns and g0["elev"].notna().any():
            g0["eband"] = pd.cut(g0["elev"], ELEV_BINS, labels=ELEV_LABELS, include_lowest=True)
            for band, gb in g0.groupby("eband", observed=True):
                row = {"method": method, "split": "all", "subset": f"elev:{band}",
                       **compute_metrics(gb.observed, gb.predicted)}
                rows.append(row)
    return pd.DataFrame(rows)


def _color(name: str):
    return COLORS.get(name, "#333333")


def plot_bars(tab: pd.DataFrame, subset: str, split: str, metrics: list[str], title: str, out: Path):
    sub = tab[(tab["subset"] == subset) & (tab["split"] == split)].copy()
    if sub.empty:
        sub = tab[(tab["subset"] == subset) & (tab["split"] == "all")].copy()
        split = "all"
    if sub.empty:
        return
    sub = sub.drop_duplicates("method")
    methods = list(sub["method"])
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 4.2), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    x = np.arange(len(methods))
    for ax, met in zip(axes, metrics):
        vals = sub[met].to_numpy(float)
        ax.bar(x, vals, color=[_color(m) for m in methods], width=0.72)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.set_title(met.upper())
        ax.grid(axis="y", alpha=0.3)
        if met in ("nse", "kge", "ccc", "r"):
            ax.set_ylim(min(0.0, np.nanmin(vals) - 0.05) if np.isfinite(vals).any() else 0, 1.02)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_grouped(tab: pd.DataFrame, subset_prefix: str, metric: str, title: str, out: Path):
    sub = tab[tab["subset"].astype(str).str.startswith(subset_prefix) | (tab["subset"].isin(SEASONS))].copy()
    if subset_prefix == "elev:":
        sub = tab[tab["subset"].astype(str).str.startswith("elev:")].copy()
        groups = [f"elev:{b}" for b in ELEV_LABELS]
        labels = ELEV_LABELS
    else:
        groups = list(SEASONS)
        labels = groups
        sub = tab[tab["subset"].isin(groups) & (tab["split"] == "all")].copy()
    if sub.empty:
        return
    methods = list(dict.fromkeys(sub["method"]))
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    x = np.arange(len(groups))
    width = 0.8 / max(len(methods), 1)
    for i, method in enumerate(methods):
        vals = []
        for g in groups:
            hit = sub[(sub["method"] == method) & (sub["subset"] == g)]
            vals.append(float(hit[metric].iloc[0]) if len(hit) else np.nan)
        ax.bar(x + i * width - 0.4 + width / 2, vals, width=width, label=method, color=_color(method))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.legend(ncol=min(5, len(methods)), fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_taylor(df: pd.DataFrame, split: str, title: str, out: Path):
    part = df[df["split"] == split] if split != "all" and "split" in df.columns else df
    fig = plt.figure(figsize=(6.4, 6.2))
    ax = fig.add_subplot(111, polar=True)
    ref_std = None
    for method, g in part.groupby("method"):
        m = np.isfinite(g.observed) & np.isfinite(g.predicted)
        o, p = g.observed[m].to_numpy(), g.predicted[m].to_numpy()
        if len(o) < 2:
            continue
        so, sp = float(np.std(o)), float(np.std(p))
        r = float(np.corrcoef(o, p)[0, 1])
        if ref_std is None:
            ref_std = so
        theta = np.arccos(np.clip(r, -1, 1))
        ax.plot(theta, sp, "o", ms=8, color=_color(method), label=method)
    if ref_std:
        ax.plot(0.0, ref_std, "k*", ms=12, label="obs")
        ax.set_ylim(0, max(ref_std * 1.6, ax.get_ylim()[1]))
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_title(title, pad=16)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.08), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_skill(tab: pd.DataFrame, ref: str, split: str, title: str, out: Path):
    sub = tab[(tab["subset"] == "all") & (tab["split"] == split)]
    if sub.empty:
        sub = tab[(tab["subset"] == "all") & (tab["split"] == "all")]
    refs = sub[sub["method"] == ref]
    if refs.empty:
        return
    mse_ref = float(refs["rmse"].iloc[0]) ** 2
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    methods, scores = [], []
    for _, row in sub.iterrows():
        methods.append(row["method"])
        mse = float(row["rmse"]) ** 2
        scores.append(1.0 - mse / mse_ref if mse_ref > 0 else np.nan)
    x = np.arange(len(methods))
    ax.bar(x, scores, color=[_color(m) for m in methods])
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel(f"1 - MSE / MSE_{ref}")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_station_box(df: pd.DataFrame, split: str, title: str, out: Path):
    part = df[df["split"] == split] if split != "all" else df
    recs = []
    for (method, sta), g in part.groupby(["method", "station_name"]):
        recs.append({"method": method, "station_name": sta,
                     "rmse": float(np.sqrt(np.mean((g.predicted - g.observed) ** 2)))})
    box = pd.DataFrame(recs)
    if box.empty:
        return
    methods = list(dict.fromkeys(box["method"]))
    data = [box.loc[box["method"] == m, "rmse"].to_numpy() for m in methods]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    bp = ax.boxplot(data, labels=methods, showfliers=False, patch_artist=True)
    for patch, method in zip(bp["boxes"], methods):
        patch.set_facecolor(_color(method))
        patch.set_alpha(0.7)
    ax.set_ylabel("station RMSE")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_scatter_grid(df: pd.DataFrame, split: str, title: str, out: Path):
    part = df[df["split"] == split] if split != "all" else df
    methods = list(dict.fromkeys(part["method"]))
    n = len(methods)
    if n == 0:
        return
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.0 * nrow), squeeze=False)
    for ax, method in zip(axes.ravel(), methods):
        g = part[part["method"] == method]
        ax.scatter(g.observed, g.predicted, s=4, alpha=0.2, linewidths=0, c=_color(method))
        lo = np.nanmin([g.observed.min(), g.predicted.min()])
        hi = np.nanmax([g.observed.max(), g.predicted.max()])
        ax.plot([lo, hi], [lo, hi], "k", lw=0.7)
        met = compute_metrics(g.observed, g.predicted)
        ax.set_title(f"{method}  RMSE={met['rmse']:.3f}")
        ax.set_xlabel("observed")
        ax.set_ylabel("predicted")
        ax.set_aspect("equal", adjustable="box")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_monthly_rmse(df: pd.DataFrame, split: str, title: str, out: Path):
    part = df[df["split"] == split] if split != "all" else df
    part = part.copy()
    part["month"] = pd.to_datetime(part["time"]).dt.to_period("M").dt.to_timestamp()
    fig, ax = plt.subplots(figsize=(11.0, 4.2))
    for method, g in part.groupby("method"):
        ser = g.groupby("month").apply(
            lambda x: np.sqrt(np.mean((x.predicted - x.observed) ** 2)), include_groups=False
        )
        ax.plot(ser.index, ser.values, marker="o", ms=3, lw=1.2, label=method, color=_color(method))
    ax.legend(ncol=4, fontsize=8, frameon=False)
    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def print_table(tab: pd.DataFrame, split: str):
    sub = tab[(tab["subset"] == "all") & (tab["split"] == split)].drop_duplicates("method")
    if sub.empty:
        sub = tab[(tab["subset"] == "all") & (tab["split"] == "all")].drop_duplicates("method")
        split = "all"
    cols = [c for c in ("method", "n", "me", "mae", "rmse", "nse", "kge", "ccc", "r",
                        "rmse_wet", "freq_bias", "ets") if c in sub.columns]
    print(f"\n{split} scores")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.max_columns", 20):
        print(sub[cols].to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variable", required=True)
    p.add_argument("--time-resolution", default="monthly")
    p.add_argument("--domain", default="full")
    p.add_argument("--resolution", type=int, default=1000)
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--split", default="test", help="primary split for ranking plots (test|dev|all)")
    p.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    p.add_argument("--ref", default="IDW")
    args = p.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    var, time_res, split = args.variable, args.time_resolution, args.split
    print(f"LLOCV compare {var} {time_res} split={split}", flush=True)
    df = load_all(methods, var, time_res, args.domain, args.resolution, args.start, args.end)
    tab = metrics_table(df, var, time_res)
    out = get_llocv_compare_dir(time_res, var)
    tab.to_csv(out / "metrics.csv", index=False)
    print_table(tab, split)
    print_table(tab, "all")

    plot_bars(tab, "all", split, ["rmse", "mae", "me"],
              f"{var} {time_res} | {split} | error", out / f"{split}_error_bars.png")
    plot_bars(tab, "all", split, ["nse", "kge", "ccc"],
              f"{var} {time_res} | {split} | skill", out / f"{split}_skill_bars.png")
    plot_grouped(tab, "season", "rmse", f"{var} RMSE by season", out / "season_rmse.png")
    plot_grouped(tab, "season", "mae", f"{var} MAE by season", out / "season_mae.png")
    plot_grouped(tab, "elev:", "rmse", f"{var} RMSE by elevation", out / "elev_rmse.png")
    plot_taylor(df, split, f"Taylor | {var} | {split}", out / f"{split}_taylor.png")
    ref = args.ref if args.ref in set(df["method"]) else df["method"].iloc[0]
    plot_skill(tab, ref, split, f"Skill vs {ref} | {var} | {split}", out / f"{split}_skill_vs_{ref}.png")
    plot_station_box(df, split, f"Station RMSE | {var} | {split}", out / f"{split}_station_rmse_box.png")
    plot_scatter_grid(df, split, f"Obs vs pred | {var} | {split}", out / f"{split}_scatter.png")
    plot_monthly_rmse(df, split, f"Monthly RMSE | {var} | {split}", out / f"{split}_monthly_rmse.png")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
