#!/usr/bin/env python3
"""Compare nested / produce LLOCV parquet files across interpolation methods.

Metrics from shared.metrics (core + precip/snow extras).
Writes per-variable diagnostics and cross-variable heatmaps to
Plots/llocv_compare/{time_resolution}/{variable}/ and .../_overview/.

Variable-specific comparisons use the best method (lowest RMSE) as reference.
Model-specific comparisons (by_method/) use IDW as reference.

Usage (from CODE/):
  python shared/plotting/compare_llocv.py --variable all --time-resolution weekly
  python shared/plotting/compare_llocv.py --variable temp_mean --split test
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

from shared.metrics import (
    compute_metrics as shared_metrics,
    get_trace,
    precip_scores,
    rmse_over_std,
)
from paths import (  # noqa: E402
    get_dem_path,
    get_llocv_compare_dir,
    get_llocv_path,
    get_method_output_dir,
    get_nested_llocv_path,
    get_station_terrain_path,
    get_metadata_path,
)

# Active thesis suite. BSS and TPS dropped 2026-10-20 / 2026-11-04.
DEFAULT_METHODS = [
    "IDW", "Kriging", "Frei", "GAM", "RFSI", "CNN", "SMACNP", "RGI",
]
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
ELEV_BINS = [0, 500, 1000, 1500, 2000, 10_000]
ELEV_LABELS = ["<500", "500-1000", "1000-1500", "1500-2000", ">2000"]
COLORS = {
    "IDW": "#7f7f7f", "RFSI": "#1f77b4", "RGI": "#17becf",
    "Kriging": "#2ca02c", "Kriging-OK": "#98df8a", "Kriging-RK": "#2ca02c",
    "GAM": "#d62728", "CNN": "#9467bd", "Frei": "#e377c2", "SMACNP": "#bcbd22",
}


def compute_metrics(obs, pred) -> dict:
    """Delegate to the shared evaluation set (core scores only)."""
    return shared_metrics(obs, pred)


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
    # Active suite is residual kriging only; keep the method name so heatmaps match.
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
        # Restrict to the requested timescale; the folder holds monthly/weekly/daily together.
        hits.extend(sorted(pdir.glob(f"*{time_res}*llocv.parquet")))
    # SMACNP writes Output/llocv/{time_res}/{var}_heldout.parquet (no nested/ layout).
    held = get_method_output_dir(method) / "llocv" / time_res / f"{var}_heldout.parquet"
    if held.exists():
        hits.append(held)
    hdir = get_method_output_dir(method) / "llocv" / time_res
    if hdir.exists():
        hits.extend(sorted(hdir.glob(f"{var}*heldout*.parquet")))
    seen, out = set(), []
    for path in hits:
        key = str(path.resolve()) if path.exists() else None
        if key and key not in seen:
            seen.add(key)
            out.append(path)
    return out


def attach_elev(df: pd.DataFrame) -> pd.DataFrame:
    """Attach station elevation from the shared terrain cache (fallback: overview)."""
    if "elev" in df.columns and df["elev"].notna().any():
        return df
    if "station_name" not in df.columns:
        return df
    meta = None
    terrain = get_station_terrain_path()
    if terrain.exists():
        try:
            sta = pd.read_parquet(terrain, columns=["station_name", "elev"])
            meta = sta.drop_duplicates("station_name")
        except Exception:
            meta = None
    if meta is None:
        overview = get_metadata_path()
        if overview.exists():
            try:
                sta = pd.read_csv(overview, usecols=lambda c: c in ("station_name", "elev", "elevation"))
                col = "elev" if "elev" in sta.columns else "elevation"
                if col in sta.columns and "station_name" in sta.columns:
                    meta = sta[["station_name", col]].drop_duplicates("station_name").rename(columns={col: "elev"})
            except Exception:
                meta = None
    if meta is None:
        return df
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
                # SMACNP heldout parquet is the test evaluation; it has no split column.
                df["split"] = "test" if "heldout" in path.name.lower() else "all"
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
    trace = get_trace(var, time_res) if is_p else None
    for (method, split), g in df.groupby(["method", "split"], sort=False):
        row = {"method": method, "split": split, "subset": "all",
               **shared_metrics(g.observed, g.predicted, var, time_res, trace=trace)}
        rows.append(row)
    for method, g0 in df.groupby("method", sort=False):
        row = {"method": method, "split": "all", "subset": "all",
               **shared_metrics(g0.observed, g0.predicted, var, time_res, trace=trace)}
        rows.append(row)
        g0 = g0.copy()
        g0["season"] = season_of(g0["time"])
        for season, gs in g0.groupby("season"):
            row = {"method": method, "split": "all", "subset": str(season),
                   **shared_metrics(gs.observed, gs.predicted, var, time_res, trace=trace)}
            rows.append(row)
        if "elev" in g0.columns and g0["elev"].notna().any():
            g0["eband"] = pd.cut(g0["elev"], ELEV_BINS, labels=ELEV_LABELS, include_lowest=True)
            for band, gb in g0.groupby("eband", observed=True):
                row = {"method": method, "split": "all", "subset": f"elev:{band}",
                       **shared_metrics(gb.observed, gb.predicted, var, time_res, trace=trace)}
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
    cols = [c for c in ("method", "n", "me", "mae", "rmse", "rmse_std", "nse", "kge", "ccc", "r",
                        "rmse_wet", "freq_bias", "ets") if c in sub.columns]
    print(f"\n{split} scores")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.max_columns", 20):
        print(sub[cols].to_string(index=False))


def _split(df, split):
    return df[df["split"] == split] if split != "all" and "split" in df.columns else df


def best_method(tab: pd.DataFrame, split: str, metric: str = "rmse") -> str:
    sub = tab[(tab["subset"] == "all") & (tab["split"] == split)]
    if sub.empty:
        sub = tab[(tab["subset"] == "all") & (tab["split"] == "all")]
    if sub.empty or metric not in sub.columns:
        return ""
    return str(sub.loc[sub[metric].idxmin(), "method"])


def attach_xy(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing x/y from any method that has them (usually IDW)."""
    if "x" in df.columns and df["x"].notna().any():
        src = df.loc[df["x"].notna(), ["station_name", "x", "y"]].drop_duplicates("station_name")
        if src.empty:
            return df
        base = df.drop(columns=[c for c in ("x", "y") if c in df.columns])
        return base.merge(src, on="station_name", how="left")
    return df


def _load_regions():
    """11 HydroBASINS polygons, numbered 1..11 from west to east by centroid longitude."""
    from paths import get_source_root
    import geopandas as gpd
    basins_path = get_source_root() / "Basins" / "basins_n11.geojson"
    if not basins_path.exists():
        return None
    basins = gpd.read_file(basins_path).to_crs("EPSG:4326").reset_index(drop=True)
    basins["_lon"] = basins.geometry.centroid.x
    basins = basins.sort_values("_lon").reset_index(drop=True)
    basins["_lab"] = [str(i + 1) for i in range(len(basins))]
    return basins


def _region_labels(df: pd.DataFrame) -> pd.Series:
    """Assign each row to region 1..11 (west to east)."""
    import geopandas as gpd
    empty = pd.Series("outside", index=df.index)
    basins = _load_regions()
    if basins is None or "lon" not in df.columns or not df["lon"].notna().any():
        print("  region bins skipped: basins_n11.geojson missing or no coordinates", flush=True)
        return empty
    pts = gpd.GeoDataFrame(
        {"_i": np.arange(len(df))},
        geometry=gpd.points_from_xy(df["lon"].to_numpy(), df["lat"].to_numpy()),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(pts, basins[["_lab", "geometry"]], how="left", predicate="within")
    lab = joined.drop_duplicates("_i").set_index("_i")["_lab"]
    return pd.Series(lab.reindex(np.arange(len(df))).fillna("outside").to_numpy(), index=df.index)


def plot_region_map(out: Path):
    """Overview map of the 11 regions numbered west to east."""
    import geopandas as gpd
    import cartopy.crs as ccrs
    basins = _load_regions()
    if basins is None:
        return
    peri_path = Path(__file__).resolve().parent / "alpine_perimeter.geojson"
    peri = gpd.read_file(peri_path).to_crs("EPSG:4326") if peri_path.exists() else None
    peri_m = (peri if peri is not None else basins).to_crs("EPSG:31287")
    buffered = peri_m.buffer(100_000).union_all()
    buf_wgs = gpd.GeoDataFrame(geometry=[buffered], crs="EPSG:31287").to_crs("EPSG:4326")
    minx, miny, maxx, maxy = buf_wgs.total_bounds
    extent = (minx - 0.35, maxx + 0.35, miny - 0.35, maxy + 0.35)
    fig = plt.figure(figsize=(12.0, 8.0))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection=ccrs.PlateCarree())
    if peri is not None:
        _base_rmse_map(ax, peri, extent)
    else:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    basins.plot(ax=ax, facecolor="none", edgecolor="#c0392b", linewidth=0.8,
                transform=ccrs.PlateCarree(), zorder=6)
    for _, row in basins.iterrows():
        c = row.geometry.centroid
        ax.text(c.x, c.y, row["_lab"], ha="center", va="center", fontsize=11,
                fontweight="bold", color="black", transform=ccrs.PlateCarree(), zorder=7,
                bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="black", lw=0.4))
    ax.set_title("Regions 1–11, west to east")
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_heatmap(tab: pd.DataFrame, metric: str, title: str, out: Path, methods, variables):
    sub = tab[(tab["subset"] == "all")].copy()
    if sub.empty or metric not in sub.columns:
        return
    piv = sub.pivot_table(index="method", columns="variable", values=metric, aggfunc="first")
    piv = piv.reindex(index=[m for m in methods if m in set(piv.index)],
                      columns=[v for v in variables if v in piv.columns])
    if piv.empty:
        return
    show = piv.to_numpy().astype(float)
    # RMSE and ME live on different scales per variable. Colour within each
    # column so one colourbar stays readable; the cell text keeps the raw value.
    if metric == "rmse":
        scale = np.nanmin(show, axis=0)
        scale = np.where(scale > 0, scale, np.nan)
        show = show / scale
        cmap, vmin, vmax = "viridis_r", 1.0, None
        cbar = "RMSE / best in variable"
    elif metric == "me":
        scale = np.nanmax(np.abs(show), axis=0)
        scale = np.where(scale > 0, scale, np.nan)
        show = show / scale
        cmap, vmin, vmax = "RdBu_r", -1.0, 1.0
        cbar = "ME / max|ME| in variable"
    else:
        cmap = "RdYlGn" if metric in ("nse", "kge", "ccc") else "viridis_r"
        vmin = 0.0 if metric in ("nse", "kge", "ccc") else None
        vmax = 1.0 if metric in ("nse", "kge", "ccc") else None
        cbar = metric.upper()
    fig, ax = plt.subplots(figsize=(1.6 * piv.shape[1] + 2.5, 0.55 * piv.shape[0] + 2.0))
    im = ax.imshow(show, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label=cbar)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_rank_heatmap(tab: pd.DataFrame, title: str, out: Path, methods, variables):
    sub = tab[(tab["subset"] == "all")].copy()
    if sub.empty or "rmse" not in sub.columns:
        return
    ranks = []
    for var, g in sub.groupby("variable"):
        gg = g.copy()
        gg["rank"] = gg["rmse"].rank(method="min")
        ranks.append(gg)
    rk = pd.concat(ranks)
    piv = rk.pivot_table(index="method", columns="variable", values="rank", aggfunc="first")
    piv = piv.reindex(index=[m for m in methods if m in piv.index],
                      columns=[v for v in variables if v in piv.columns])
    fig, ax = plt.subplots(figsize=(1.6 * piv.shape[1] + 2.5, 0.55 * piv.shape[0] + 2.0))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{int(val)}", ha="center", va="center", fontsize=9, color="white")
    fig.colorbar(im, ax=ax, label="rank (1 = best RMSE)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_wet_bars(tab: pd.DataFrame, split: str, title: str, out: Path):
    metrics = [m for m in ("rmse_wet", "pod", "far", "ets", "freq_bias") if m in tab.columns]
    if not metrics:
        return
    plot_bars(tab, "all", split, metrics, title, out)


def plot_best_vs_idw(df, split, best, out: Path, title: str):
    part = _split(df, split)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    for method, color, alpha in (( "IDW", "#7f7f7f", 0.25), (best, _color(best), 0.45)):
        g = part[part["method"] == method]
        if g.empty:
            continue
        ax.scatter(g.observed, g.predicted, s=8, alpha=alpha, linewidths=0, c=color, label=method)
    lo = np.nanmin(part[["observed", "predicted"]].to_numpy())
    hi = np.nanmax(part[["observed", "predicted"]].to_numpy())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.set_xlabel("observed")
    ax.set_ylabel("predicted")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_residual_hist_vs(df, split, ref, out: Path, title: str):
    part = _split(df, split)
    methods = [m for m in dict.fromkeys(part["method"]) if m != ref]
    if not methods or part[part["method"] == ref].empty:
        return
    ncol = min(3, len(methods))
    nrow = int(np.ceil(len(methods) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow), squeeze=False)
    ref_r = (part.loc[part["method"] == ref, "predicted"] - part.loc[part["method"] == ref, "observed"])
    for ax, method in zip(axes.ravel(), methods):
        g = part[part["method"] == method]
        ax.hist(ref_r, bins=40, density=True, alpha=0.45, color="#7f7f7f", label=ref)
        ax.hist(g.predicted - g.observed, bins=40, density=True, alpha=0.55, color=_color(method), label=method)
        ax.axvline(0, color="k", lw=0.7, ls="--")
        ax.set_title(method)
        ax.legend(fontsize=8, frameon=False)
    for ax in axes.ravel()[len(methods):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_residual_vs_obs(df, split, ref, out: Path, title: str):
    part = _split(df, split)
    methods = [m for m in dict.fromkeys(part["method"]) if m != ref]
    if not methods:
        return
    ncol = min(3, len(methods))
    nrow = int(np.ceil(len(methods) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow), squeeze=False)
    ref_g = part[part["method"] == ref]
    for ax, method in zip(axes.ravel(), methods):
        g = part[part["method"] == method]
        if not ref_g.empty:
            ax.scatter(ref_g.observed, ref_g.predicted - ref_g.observed, s=6, alpha=0.2,
                       c="#7f7f7f", linewidths=0, label=ref)
        ax.scatter(g.observed, g.predicted - g.observed, s=6, alpha=0.35,
                   c=_color(method), linewidths=0, label=method)
        ax.axhline(0, color="k", lw=0.7, ls="--")
        ax.set_xlabel("observed")
        ax.set_ylabel("residual")
        ax.set_title(method)
        ax.legend(fontsize=8, frameon=False)
    for ax in axes.ravel()[len(methods):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_scatter_vs(df, split, ref, out: Path, title: str):
    part = _split(df, split)
    methods = [m for m in dict.fromkeys(part["method"]) if m != ref]
    if not methods:
        return
    ncol = min(3, len(methods))
    nrow = int(np.ceil(len(methods) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.0 * nrow), squeeze=False)
    ref_g = part[part["method"] == ref]
    for ax, method in zip(axes.ravel(), methods):
        g = part[part["method"] == method]
        if not ref_g.empty:
            ax.scatter(ref_g.observed, ref_g.predicted, s=6, alpha=0.2, c="#7f7f7f", linewidths=0, label=ref)
        ax.scatter(g.observed, g.predicted, s=6, alpha=0.35, c=_color(method), linewidths=0, label=method)
        lo = np.nanmin(part[["observed", "predicted"]].to_numpy())
        hi = np.nanmax(part[["observed", "predicted"]].to_numpy())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.7)
        ax.set_title(method)
        ax.legend(fontsize=8, frameon=False)
        ax.set_aspect("equal", adjustable="box")
    for ax in axes.ravel()[len(methods):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_by_calendar_month(df, split, metric, title, out: Path):
    part = _split(df, split).copy()
    part["cal_month"] = pd.to_datetime(part["time"]).dt.month
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    for method, g in part.groupby("method"):
        rows = []
        for m, gm in g.groupby("cal_month"):
            if metric == "rmse":
                val = float(np.sqrt(np.mean((gm.predicted - gm.observed) ** 2)))
            else:
                val = float(np.mean(gm.predicted - gm.observed))
            rows.append((m, val))
        if not rows:
            continue
        xs, ys = zip(*sorted(rows))
        ax.plot(xs, ys, marker="o", ms=4, lw=1.3, label=method, color=_color(method))
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("month")
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.legend(ncol=4, fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_by_year(df, split, title, out: Path):
    part = _split(df, split).copy()
    part["year"] = pd.to_datetime(part["time"]).dt.year
    years = sorted(part["year"].dropna().unique())
    if len(years) < 2:
        return
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for method, g in part.groupby("method"):
        rows = []
        for y, gy in g.groupby("year"):
            rows.append((y, float(np.sqrt(np.mean((gy.predicted - gy.observed) ** 2)))))
        xs, ys = zip(*sorted(rows))
        ax.plot(xs, ys, marker="o", ms=5, lw=1.3, label=method, color=_color(method))
    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.legend(ncol=4, fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_elev_curve(df, split, title, out: Path):
    part = _split(df, split)
    if "elev" not in part.columns or not part["elev"].notna().any():
        return
    part = part.dropna(subset=["elev"]).copy()
    part["eband"] = pd.cut(part["elev"], ELEV_BINS, labels=ELEV_LABELS, include_lowest=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for method, g in part.groupby("method"):
        rows = []
        for band, gb in g.groupby("eband", observed=True):
            rows.append((str(band), float(np.sqrt(np.mean((gb.predicted - gb.observed) ** 2)))))
        if not rows:
            continue
        ax.plot(range(len(rows)), [r[1] for r in rows], marker="o", lw=1.3, label=method, color=_color(method))
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([r[0] for r in rows], rotation=30, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.legend(ncol=4, fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_residual_vs_elev(df, split, ref, out: Path, title: str):
    part = _split(df, split)
    if "elev" not in part.columns or not part["elev"].notna().any():
        return
    methods = [m for m in dict.fromkeys(part["method"]) if m != ref]
    if not methods:
        return
    ncol = min(3, len(methods))
    nrow = int(np.ceil(len(methods) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow), squeeze=False)
    ref_g = part[part["method"] == ref]
    for ax, method in zip(axes.ravel(), methods):
        g = part[part["method"] == method]
        if not ref_g.empty:
            ax.scatter(ref_g.elev, ref_g.predicted - ref_g.observed, s=6, alpha=0.2,
                       c="#7f7f7f", linewidths=0, label=ref)
        ax.scatter(g.elev, g.predicted - g.observed, s=6, alpha=0.35,
                   c=_color(method), linewidths=0, label=method)
        ax.axhline(0, color="k", lw=0.7, ls="--")
        ax.set_xlabel("elevation (m)")
        ax.set_ylabel("residual")
        ax.set_title(method)
        ax.legend(fontsize=8, frameon=False)
    for ax in axes.ravel()[len(methods):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_valley_mountain(df, split, title, out: Path):
    part = _split(df, split)
    if "elev" not in part.columns or not part["elev"].notna().any():
        return
    part = part.dropna(subset=["elev"]).copy()
    part["zone"] = np.where(part["elev"] >= 1500, "mountain", "valley")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, metric, ylab in ((axes[0], "rmse", "RMSE"), (axes[1], "me", "ME")):
        methods = list(dict.fromkeys(part["method"]))
        x = np.arange(len(methods))
        w = 0.35
        for i, zone in enumerate(("valley", "mountain")):
            vals = []
            for m in methods:
                g = part[(part["method"] == m) & (part["zone"] == zone)]
                if g.empty:
                    vals.append(np.nan)
                elif metric == "rmse":
                    vals.append(float(np.sqrt(np.mean((g.predicted - g.observed) ** 2))))
                else:
                    vals.append(float(np.mean(g.predicted - g.observed)))
            ax.bar(x + (i - 0.5) * w, vals, w, label=zone)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.set_ylabel(ylab)
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _station_lonlat(df: pd.DataFrame) -> pd.DataFrame:
    """Station lon/lat for mapping. Prefer overview columns; else project x/y from EPSG:31287."""
    if "lon" in df.columns and "lat" in df.columns and df["lon"].notna().any():
        return df
    overview = get_metadata_path()
    if overview.exists() and "station_name" in df.columns:
        try:
            sta = pd.read_csv(overview)
            cols = {c.lower(): c for c in sta.columns}
            lon_c = cols.get("lon") or cols.get("longitude")
            lat_c = cols.get("lat") or cols.get("latitude")
            name_c = cols.get("station_name")
            if lon_c and lat_c and name_c:
                meta = sta[[name_c, lon_c, lat_c]].drop_duplicates(name_c)
                meta = meta.rename(columns={name_c: "station_name", lon_c: "lon", lat_c: "lat"})
                out = df.drop(columns=[c for c in ("lon", "lat") if c in df.columns])
                return out.merge(meta, on="station_name", how="left")
        except Exception:
            pass
    if "x" in df.columns and "y" in df.columns and df["x"].notna().any():
        try:
            from pyproj import Transformer
            tr = Transformer.from_crs("EPSG:31287", "EPSG:4326", always_xy=True)
            lon, lat = tr.transform(df["x"].to_numpy(), df["y"].to_numpy())
            out = df.copy()
            out["lon"] = lon
            out["lat"] = lat
            return out
        except Exception:
            return df
    return df


def _base_rmse_map(ax, peri, extent):
    """Blackscale DEM + Alpine perimeter, matching the overview station maps."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    import rasterio

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    mean_lat = (extent[2] + extent[3]) / 2
    ax.set_aspect(1.0 / np.cos(np.radians(mean_lat)))
    ax.set_facecolor("#f5f5f5")
    dem_path = get_dem_path()
    if dem_path.exists():
        with rasterio.open(dem_path) as src:
            width = max(900, min(int((extent[1] - extent[0]) / 0.0018), 4800))
            height = max(700, min(int((extent[3] - extent[2]) / 0.0018), 3800))
            transform = rasterio.transform.from_bounds(extent[0], extent[2], extent[1], extent[3], width, height)
            arr = np.full((height, width), np.nan, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1), destination=arr,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=transform, dst_crs="EPSG:4326",
                resampling=Resampling.nearest,
                src_nodata=src.nodata if src.nodata is not None else 0,
                dst_nodata=np.nan,
            )
        arr = np.where(np.isnan(arr) | (arr < -200), np.nan, arr)
        ax.imshow(arr, origin="upper", extent=(extent[0], extent[1], extent[2], extent[3]),
                  cmap="gray_r", vmin=0, vmax=3200, transform=ccrs.PlateCarree(),
                  zorder=1, interpolation="nearest")
    peri_m = peri.to_crs("EPSG:31287")
    buf = peri_m.buffer(100_000).union_all()
    import geopandas as gpd
    buf_gdf = gpd.GeoDataFrame(geometry=[buf], crs="EPSG:31287").to_crs("EPSG:4326")
    buf_gdf.plot(ax=ax, facecolor="none", edgecolor="#7aa2d4", linewidth=0.6,
                 linestyle="--", transform=ccrs.PlateCarree(), zorder=4)
    peri.plot(ax=ax, facecolor="none", edgecolor="#1a365d", linewidth=0.9,
              transform=ccrs.PlateCarree(), zorder=5)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.3, edgecolor="0.4", zorder=3)
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#e6f3ff", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.RIVERS.with_scale("10m"), linewidth=0.28, edgecolor="#4a90d9", zorder=2)
    ax.add_feature(cfeature.LAKES.with_scale("10m"), facecolor="#a8d5e5", edgecolor="#4a90d9",
                   linewidth=0.22, zorder=2)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.5", alpha=0.45, linestyle="-")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}


def plot_spatial_rmse(df, split, title, out: Path):
    """One overview-style map per method. Stations coloured by station RMSE."""
    part = _split(df, split)
    part = _station_lonlat(part)
    if "lon" not in part.columns or not part["lon"].notna().any():
        print("  spatial RMSE skipped: no lon/lat", flush=True)
        return
    try:
        import cartopy.crs as ccrs
        import geopandas as gpd
    except ImportError:
        print("  spatial RMSE skipped: cartopy/geopandas missing", flush=True)
        return
    peri_path = Path(__file__).resolve().parent / "alpine_perimeter.geojson"
    if not peri_path.exists():
        print(f"  spatial RMSE skipped: {peri_path.name} missing", flush=True)
        return
    peri = gpd.read_file(peri_path).to_crs("EPSG:4326")
    peri_m = peri.to_crs("EPSG:31287")
    buffered = peri_m.buffer(100_000).union_all()
    buf_wgs = gpd.GeoDataFrame(geometry=[buffered], crs="EPSG:31287").to_crs("EPSG:4326")
    minx, miny, maxx, maxy = buf_wgs.total_bounds
    pad = 0.35
    extent = (minx - pad, maxx + pad, miny - pad, maxy + pad)
    methods = list(dict.fromkeys(part["method"]))
    # Shared colour scale so methods are comparable.
    station_rmses = []
    per_method = {}
    for method in methods:
        g = part[part["method"] == method]
        recs = []
        for sta, gs in g.groupby("station_name"):
            if not np.isfinite(gs["lon"].iloc[0]) or not np.isfinite(gs["lat"].iloc[0]):
                continue
            recs.append((gs["lon"].iloc[0], gs["lat"].iloc[0],
                         float(np.sqrt(np.mean((gs.predicted - gs.observed) ** 2))),
                         float(gs["elev"].iloc[0]) if "elev" in gs.columns and np.isfinite(gs["elev"].iloc[0]) else np.nan))
        per_method[method] = recs
        station_rmses.extend(r[2] for r in recs)
    if not station_rmses:
        print("  spatial RMSE skipped: no station scores", flush=True)
        return
    vmin, vmax = float(np.nanpercentile(station_rmses, 5)), float(np.nanpercentile(station_rmses, 95))
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = out.stem
    for method, recs in per_method.items():
        if not recs:
            continue
        fig = plt.figure(figsize=(12.0, 8.0))
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection=ccrs.PlateCarree())
        _base_rmse_map(ax, peri, extent)
        lons, lats, rms, elevs = zip(*recs)
        elevs = np.asarray(elevs)
        valley = ~np.isfinite(elevs) | (elevs < 1500)
        sc = ax.scatter(np.asarray(lons)[valley], np.asarray(lats)[valley], c=np.asarray(rms)[valley],
                        s=16, marker="o", cmap="viridis_r", vmin=vmin, vmax=vmax,
                        edgecolors="black", linewidths=0.3, transform=ccrs.PlateCarree(), zorder=6)
        if (~valley).any():
            ax.scatter(np.asarray(lons)[~valley], np.asarray(lats)[~valley], c=np.asarray(rms)[~valley],
                       s=22, marker="^", cmap="viridis_r", vmin=vmin, vmax=vmax,
                       edgecolors="black", linewidths=0.3, transform=ccrs.PlateCarree(), zorder=6)
        cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label("station RMSE")
        ax.set_title(f"{title} | {method}", fontsize=11)
        fig.savefig(out.with_name(f"{stem}_{method}.png"), dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)



def plot_by_obs_bin(df, split, metric, title, out: Path):
    part = _station_lonlat(_split(df, split).copy())
    part["obin"] = _region_labels(part)
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    methods = list(dict.fromkeys(part["method"]))
    bins = [b for b in sorted(part["obin"].dropna().unique(), key=lambda s: int(s) if str(s).isdigit() else 99) if b != "outside"]
    x = np.arange(len(bins))
    w = 0.8 / max(len(methods), 1)
    for i, method in enumerate(methods):
        vals = []
        g = part[part["method"] == method]
        for b in bins:
            gb = g[g["obin"] == b]
            if gb.empty:
                vals.append(np.nan)
            elif metric == "rmse":
                vals.append(float(np.sqrt(np.mean((gb.predicted - gb.observed) ** 2))))
            else:
                vals.append(float(np.mean(gb.predicted - gb.observed)))
        ax.bar(x + i * w, vals, w, label=method, color=_color(method))
    ax.set_xticks(x + w * (len(methods) - 1) / 2)
    ax.set_xticklabels([str(b) for b in bins], rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("region")
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.legend(ncol=4, fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_binned_residual(df, split, ref, out: Path, title: str):
    part = _station_lonlat(_split(df, split).copy())
    part["obin"] = _region_labels(part)
    methods = [m for m in dict.fromkeys(part["method"]) if m != ref]
    if not methods:
        return
    bins = [b for b in sorted(part["obin"].dropna().unique(), key=lambda s: int(s) if str(s).isdigit() else 99) if b != "outside"]
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    for method in [ref] + methods:
        g = part[part["method"] == method]
        xs, ys, ye = [], [], []
        for i, b in enumerate(bins):
            gb = g[g["obin"] == b]
            if gb.empty:
                continue
            r = gb.predicted - gb.observed
            xs.append(i)
            ys.append(float(r.mean()))
            ye.append(float(r.std()))
        if not xs:
            continue
        ax.errorbar(xs, ys, yerr=ye, marker="o", lw=1.2, label=method, color=_color(method), capsize=2)
    ax.axhline(0, color="k", lw=0.7, ls="--")
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([str(b) for b in bins], rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("region")
    ax.set_ylabel("mean residual ± std")
    ax.set_title(title)
    ax.legend(ncol=4, fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_method_vs_idw_set(df, split, method, out_dir: Path, var: str):
    """Model-specific diagnostics: method against IDW."""
    if method == "IDW":
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    part = _split(df, split)
    sub = part[part["method"].isin([method, "IDW"])]
    if sub[sub["method"] == method].empty or sub[sub["method"] == "IDW"].empty:
        return
    plot_residual_hist_vs(sub, split, "IDW", out_dir / "residual_hist_vs_idw.png",
                          f"{var} residual | {method} vs IDW")
    plot_scatter_vs(sub, split, "IDW", out_dir / "scatter_vs_idw.png",
                    f"{var} obs-pred | {method} vs IDW")
    plot_residual_vs_obs(sub, split, "IDW", out_dir / "residual_vs_obs_idw.png",
                         f"{var} residual vs obs | {method} vs IDW")
    plot_by_calendar_month(sub, split, "me", f"{var} ME by month | {method} vs IDW",
                           out_dir / "me_by_month_vs_idw.png")
    if "elev" in sub.columns:
        plot_grouped_pair = sub
        # elevation RMSE bars for the pair
        plot_elev_curve(plot_grouped_pair, split, f"{var} RMSE by elevation | {method} vs IDW",
                        out_dir / "rmse_by_elev_vs_idw.png")
    plot_by_obs_bin(sub, split, "rmse", f"{var} RMSE by region | {method} vs IDW",
                    out_dir / "rmse_by_region_vs_idw.png")



def plot_station_timeseries(df, split, method, var, out_dir: Path, n: int = 5):
    """Observed vs predicted time series for the n best and n worst stations by RMSE."""
    part = _split(df, split)
    g = part[part["method"] == method]
    if g.empty:
        return
    scores = []
    for sta, gs in g.groupby("station_name"):
        if len(gs) < 3:
            continue
        rmse = float(np.sqrt(np.mean((gs.predicted - gs.observed) ** 2)))
        scores.append((rmse, sta))
    if not scores:
        return
    scores.sort()
    best = [s for _, s in scores[:n]]
    worst = [s for _, s in scores[-n:]]
    picks = [("best", best), ("worst", worst)]
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 5.6), sharex=False, squeeze=False)
    for row, (tag, stations) in enumerate(picks):
        for col in range(n):
            ax = axes[row, col]
            if col >= len(stations):
                ax.axis("off")
                continue
            sta = stations[col]
            gs = g[g["station_name"] == sta].sort_values("time")
            ax.plot(gs["time"], gs["observed"], color="black", lw=1.0, label="observed")
            ax.plot(gs["time"], gs["predicted"], color=_color(method), lw=1.0, label="predicted")
            rmse = float(np.sqrt(np.mean((gs.predicted - gs.observed) ** 2)))
            ax.set_title(f"{tag} {col+1}: {sta}\nRMSE={rmse:.2f}", fontsize=8)
            ax.tick_params(labelsize=7)
            if col == 0:
                ax.set_ylabel(tag)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f"{var} | {method} | best/worst stations")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{split}_station_timeseries.png", dpi=140, bbox_inches="tight")
    plt.close(fig)



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variable", default="all", help="variable code or 'all' for the five core variables")
    p.add_argument("--time-resolution", default="monthly")
    p.add_argument("--domain", default="full")
    p.add_argument("--resolution", type=int, default=1000)
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--split", default="test", help="primary split for ranking plots (test|dev|all)")
    p.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    p.add_argument("--ref", default="IDW")
    args = p.parse_args()

    CORE = ("temp_mean", "precip_sum", "wind_mean", "rh_mean", "snow_mean")
    variables = list(CORE) if args.variable.lower() == "all" else [args.variable]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    all_tabs = []
    for var in variables:
        time_res, split = args.time_resolution, args.split
        print(f"LLOCV compare {var} {time_res} split={split}", flush=True)
        try:
            df = load_all(methods, var, time_res, args.domain, args.resolution, args.start, args.end)
            df = attach_xy(df)
        except FileNotFoundError as e:
            print(f"  skip {var}: {e}", flush=True)
            continue
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
        plot_grouped(tab, "season", "me", f"{var} ME by season", out / "season_me.png")
        plot_grouped(tab, "season", "nse", f"{var} NSE by season", out / "season_nse.png")
        plot_grouped(tab, "elev:", "rmse", f"{var} RMSE by elevation", out / "elev_rmse.png")
        plot_grouped(tab, "elev:", "mae", f"{var} MAE by elevation", out / "elev_mae.png")
        plot_grouped(tab, "elev:", "me", f"{var} ME by elevation", out / "elev_me.png")
        plot_grouped(tab, "elev:", "nse", f"{var} NSE by elevation", out / "elev_nse.png")
        plot_taylor(df, split, f"Taylor | {var} | {split}", out / f"{split}_taylor.png")
        ref = args.ref if args.ref in set(df["method"]) else df["method"].iloc[0]
        plot_skill(tab, ref, split, f"Skill vs {ref} | {var} | {split}", out / f"{split}_skill_vs_{ref}.png")
        plot_station_box(df, split, f"Station RMSE | {var} | {split}", out / f"{split}_station_rmse_box.png")
        plot_scatter_grid(df, split, f"Obs vs pred | {var} | {split}", out / f"{split}_scatter.png")
        plot_monthly_rmse(df, split, f"Monthly RMSE | {var} | {split}", out / f"{split}_monthly_rmse.png")

        # Variable-specific expansions: reference is the best method on this variable.
        best = best_method(tab, split, "rmse")
        is_wet = var.lower().startswith("precip") or var.lower().startswith("snow")
        if is_wet:
            plot_wet_bars(tab, split, f"{var} wet scores | {split}", out / f"{split}_wet_bars.png")
        if best:
            plot_best_vs_idw(df, split, best, out / f"{split}_best_vs_idw.png",
                             f"{var} best ({best}) vs IDW")
            plot_residual_hist_vs(df, split, best, out / f"{split}_residual_hist_vs_best.png",
                                  f"{var} residual vs best ({best})")
            plot_residual_vs_obs(df, split, best, out / f"{split}_residual_vs_obs_best.png",
                                 f"{var} residual vs obs | best={best}")
            plot_scatter_vs(df, split, best, out / f"{split}_scatter_vs_best.png",
                            f"{var} obs-pred vs best ({best})")
            plot_residual_vs_elev(df, split, best, out / f"{split}_residual_vs_elev_best.png",
                                  f"{var} residual vs elev | best={best}")
            plot_binned_residual(df, split, best, out / f"{split}_residual_by_region_vs_best.png",
                                 f"{var} residual by region | best={best}")
        plot_by_calendar_month(df, split, "rmse", f"{var} RMSE by month", out / f"{split}_rmse_by_month.png")
        plot_by_calendar_month(df, split, "me", f"{var} ME by month", out / f"{split}_me_by_month.png")
        plot_by_year(df, split, f"{var} RMSE by year", out / f"{split}_rmse_by_year.png")
        plot_elev_curve(df, split, f"{var} RMSE by elevation", out / f"{split}_elev_rmse_curve.png")
        plot_valley_mountain(df, split, f"{var} valley vs mountain", out / f"{split}_valley_mountain.png")
        plot_spatial_rmse(df, split, f"{var} station RMSE", out / f"{split}_spatial_rmse.png")
        plot_by_obs_bin(df, split, "rmse", f"{var} RMSE by region", out / f"{split}_rmse_by_region.png")
        plot_by_obs_bin(df, split, "me", f"{var} ME by region", out / f"{split}_me_by_region.png")

        # Model-specific diagnostics: each method against IDW.
        for method in methods:
            plot_method_vs_idw_set(df, split, method, out / "by_method" / method, var)
            plot_station_timeseries(df, split, method, var, out / "by_method" / method)

        # Rankings and subset tables.
        rank_rows = []
        sub = tab[(tab["subset"] == "all") & (tab["split"] == split)]
        if not sub.empty:
            ss = sub.copy()
            ss["rank_rmse"] = ss["rmse"].rank(method="min")
            ss["variable"] = var
            rank_rows.append(ss)
            ss.to_csv(out / "rankings.csv", index=False)
        tab.to_csv(out / "metrics_by_subset.csv", index=False)
        all_tabs.append(tab.assign(variable=var))
        print(f"wrote {out}", flush=True)

    if all_tabs:
        cross = pd.concat(all_tabs, ignore_index=True)
        cross = cross[cross["subset"] == "all"]
        wanted = cross[cross["split"] == args.split]
        missing = set(cross["method"]) - set(wanted["method"])
        if missing:
            wanted = pd.concat([wanted, cross[(cross["method"].isin(missing)) & (cross["split"] == "all")]],
                               ignore_index=True)
        cross = wanted
        cross_dir = get_llocv_compare_dir(args.time_resolution, "_overview")
        cross.to_csv(cross_dir / "metrics_all.csv", index=False)
        if not cross.empty:
            best_rows = []
            for var, g in cross.groupby("variable"):
                i = g["rmse"].idxmin()
                best_rows.append(g.loc[i, ["variable", "method", "rmse", "nse", "kge", "ccc", "me"]])
            pd.DataFrame(best_rows).to_csv(cross_dir / "best_by_variable.csv", index=False)
            plot_heatmap(cross, "rmse", f"RMSE | {args.time_resolution}", cross_dir / "rmse_heatmap.png", methods, variables)
            plot_heatmap(cross, "nse", f"NSE | {args.time_resolution}", cross_dir / "nse_heatmap.png", methods, variables)
            plot_heatmap(cross, "kge", f"KGE | {args.time_resolution}", cross_dir / "kge_heatmap.png", methods, variables)
            plot_heatmap(cross, "ccc", f"CCC | {args.time_resolution}", cross_dir / "ccc_heatmap.png", methods, variables)
            plot_heatmap(cross, "me", f"ME | {args.time_resolution}", cross_dir / "me_heatmap.png", methods, variables)
            plot_rank_heatmap(cross, f"RMSE rank | {args.time_resolution}", cross_dir / "rank_heatmap.png", methods, variables)
            plot_region_map(cross_dir / "regions_west_east.png")
            print(f"wrote overview {cross_dir}", flush=True)


if __name__ == "__main__":
    main()
