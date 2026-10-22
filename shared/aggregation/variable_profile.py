#!/usr/bin/env python3
"""
Variable x timescale profile. Full panel, streamed.

Exact on the whole aggregated file. Half-hourly is read in chunks so the
panel is never held in memory. Quantiles use every finite value.

Subsets: all / train (pool x train quarters) / dev (pool x tune) / test (held-out x test).

Writes Source/profile/variable_profile.csv.

Run from CODE/:
  python shared/aggregation/variable_profile.py
  python shared/aggregation/variable_profile.py --time-resolution monthly
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_DIR))

from paths import (
    get_aggregated_data_path,
    get_domain_stations_path,
    get_source_root,
    get_station_terrain_path,
    ensure_dir,
)
from shared.clc import level2_index, N_LEVEL2
from shared.splits.evaluation_sets import load_holdout, load_quarter_splits, quarter_of
from shared.time_res import TIME_RESOLUTIONS, add_time_res_arg

VARS = ("temp_mean", "precip_sum", "wind_mean", "rh_mean", "snow_mean")
TRACE = {
    "precip_sum": {"half_hourly": 0.05, "daily": 0.1, "weekly": 0.7, "monthly": 1.0},
    "snow_mean": {"half_hourly": 0.5, "daily": 0.5, "weekly": 1.0, "monthly": 1.0},
}
TERRAIN_COLS = (
    "elev", "slope", "tpi", "northness",
    "dist_to_coast", "signed_dist_to_ridge",
)
SUBSETS = ("all", "train", "dev", "test")
K_NEIGH = 8
CHUNK = 400_000


def _time_col(names, time_res: str) -> str:
    for c in ("time", "timestamp", "date", "year_month", "year_week"):
        if c in names:
            return c
    raise ValueError(f"no time column for {time_res}")


def _parse_time(series: pd.Series, time_res: str) -> pd.Series:
    if time_res == "weekly":
        return pd.to_datetime(series.astype(str) + "-1", format="%Y-W%W-%w", utc=True, errors="coerce")
    if time_res == "monthly":
        return pd.to_datetime(series.astype(str) + "-01", utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


class Acc:
    """Streaming moments, wet count, terrain and CLC correlations, value store."""

    def __init__(self, trace, terrain_cols, n_clc):
        self.trace = trace
        self.terrain_cols = terrain_cols
        self.n = 0
        self.wet = 0
        self.sum = 0.0
        self.sum2 = 0.0
        self.sum3 = 0.0
        self.stations = set()
        self.times = set()
        self.t_n = {c: 0 for c in terrain_cols}
        self.t_sy = {c: 0.0 for c in terrain_cols}
        self.t_sy2 = {c: 0.0 for c in terrain_cols}
        self.t_sx = {c: 0.0 for c in terrain_cols}
        self.t_sx2 = {c: 0.0 for c in terrain_cols}
        self.t_sxy = {c: 0.0 for c in terrain_cols}
        self.clc_n = np.zeros(n_clc)
        self.clc_sy = np.zeros(n_clc)
        self.clc_sy2 = np.zeros(n_clc)
        self.values = []
        self.ac_raw = []
        self.ac_resid = []

    def add_values(self, y, stations, times):
        y = np.asarray(y, dtype=np.float64)
        m = np.isfinite(y)
        if not m.any():
            return
        v = y[m]
        self.n += int(v.size)
        self.sum += float(v.sum())
        self.sum2 += float(np.dot(v, v))
        self.sum3 += float(np.dot(v, v * v))
        if self.trace is not None:
            self.wet += int((v > self.trace).sum())
        self.stations.update(stations[m])
        self.times.update(times[m])
        self.values.append(v.astype(np.float32))

    def add_terrain(self, y, frame):
        y = np.asarray(y, dtype=np.float64)
        for c in self.terrain_cols:
            if c not in frame.columns:
                continue
            x = frame[c].to_numpy(dtype=np.float64)
            m = np.isfinite(y) & np.isfinite(x)
            if not m.any():
                continue
            yy, xx = y[m], x[m]
            self.t_n[c] += int(yy.size)
            self.t_sy[c] += float(yy.sum())
            self.t_sy2[c] += float(np.dot(yy, yy))
            self.t_sx[c] += float(xx.sum())
            self.t_sx2[c] += float(np.dot(xx, xx))
            self.t_sxy[c] += float(np.dot(yy, xx))

    def add_clc(self, y, clc):
        y = np.asarray(y, dtype=np.float64)
        idx = level2_index(np.nan_to_num(clc, nan=0.0))
        m = np.isfinite(y)
        yy, ii = y[m], idx[m]
        for c in range(len(self.clc_n)):
            sel = ii == c
            if not sel.any():
                continue
            v = yy[sel]
            self.clc_n[c] += int(v.size)
            self.clc_sy[c] += float(v.sum())
            self.clc_sy2[c] += float(np.dot(v, v))

    def add_ac(self, raw, resid):
        if np.isfinite(raw):
            self.ac_raw.append(float(raw))
        if np.isfinite(resid):
            self.ac_resid.append(float(resid))

    def _corr_from(self, n, sy, sy2, sx, sx2, sxy):
        if n < 8:
            return np.nan
        vy = sy2 - sy * sy / n
        vx = sx2 - sx * sx / n
        if vy <= 0 or vx <= 0:
            return np.nan
        cov = sxy - sy * sx / n
        return float(cov / np.sqrt(vy * vx))

    def finish(self, time_res, var, subset):
        n = self.n
        mean = self.sum / n if n else np.nan
        var_ = self.sum2 / n - mean * mean if n else np.nan
        std = float(np.sqrt(var_)) if var_ > 0 else 0.0
        if n >= 8 and std > 1e-8:
            m3 = self.sum3 / n - 3 * mean * self.sum2 / n + 2 * mean ** 3
            skew = float(m3 / std ** 3)
        else:
            skew = np.nan
        if self.values:
            cat = np.concatenate(self.values)
            p50 = float(np.quantile(cat, 0.5))
            p90 = float(np.quantile(cat, 0.9))
        else:
            cat = np.empty(0, dtype=np.float32)
            p50 = p90 = np.nan
        row = {
            "time_resolution": time_res,
            "variable": var,
            "subset": subset,
            "n_rows": int(n),
            "n_stations": len(self.stations),
            "n_fields": len(self.times),
            "mean": float(mean) if n else np.nan,
            "std": std,
            "skew": skew,
            "p50": p50,
            "p90": p90,
            "cv": std / mean if mean not in (0, np.nan) else np.nan,
            "wet_frac": self.wet / n if self.trace is not None and n else np.nan,
            "full_panel": True,
        }
        for c in self.terrain_cols:
            row[f"corr_{c}"] = self._corr_from(
                self.t_n[c], self.t_sy[c], self.t_sy2[c],
                self.t_sx[c], self.t_sx2[c], self.t_sxy[c],
            )
        best, best_c = np.nan, -1
        # CLC corr from class counts against the overall moments is not a
        # dummy correlation. Compute it from stored class sums and overall.
        sy, sy2 = self.sum, self.sum2
        for c in range(len(self.clc_n)):
            nc = self.clc_n[c]
            if nc < 8 or n - nc < 8:
                continue
            # point-biserial
            mean_c = self.clc_sy[c] / nc
            r = (mean_c - mean) * np.sqrt(nc * (n - nc) / n) / std if std > 0 else np.nan
            if np.isfinite(r) and (not np.isfinite(best) or abs(r) > abs(best)):
                best, best_c = float(r), c
        row["clc_max_corr"] = best
        row["clc_max_class"] = best_c
        row["ac_raw"] = float(np.mean(self.ac_raw)) if self.ac_raw else np.nan
        row["ac_resid"] = float(np.mean(self.ac_resid)) if self.ac_resid else np.nan
        row["n_fields_ac"] = len(self.ac_raw)
        return row, cat


def _local_autocorr(xy, val, k=K_NEIGH):
    from scipy.spatial import cKDTree

    m = np.isfinite(val)
    if m.sum() < k + 2:
        return np.nan
    pts, v = xy[m], val[m]
    tree = cKDTree(pts)
    kk = min(k + 1, len(v))
    _, ix = tree.query(pts, k=kk)
    ix = np.atleast_2d(ix)
    return _corr_pair(v, v[ix[:, 1:]].mean(axis=1))


def _idw_residual(xy, val, k=K_NEIGH):
    from scipy.spatial import cKDTree

    out = np.full(len(val), np.nan)
    m = np.isfinite(val)
    if m.sum() < k + 2:
        return out
    pts, v = xy[m], val[m]
    tree = cKDTree(pts)
    kk = min(k + 1, len(v))
    d, ix = tree.query(pts, k=kk)
    d = np.atleast_2d(np.asarray(d, dtype=float))
    ix = np.atleast_2d(ix)
    d = np.maximum(d[:, 1:], 1.0)
    w = 1.0 / d
    pred = (w * v[ix[:, 1:]]).sum(axis=1) / w.sum(axis=1)
    out[np.flatnonzero(m)] = v - pred
    return out


def _corr_pair(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 8:
        return np.nan
    if np.std(a[m]) < 1e-8 or np.std(b[m]) < 1e-8:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _ks(a, b):
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 8 or len(b) < 8:
        return np.nan
    grid = np.sort(np.unique(np.concatenate([a, b])))
    if len(grid) < 2:
        return 0.0
    ca = np.searchsorted(np.sort(a), grid, side="right") / len(a)
    cb = np.searchsorted(np.sort(b), grid, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


def _subset_mask(held, split_name):
    if split_name == "all":
        return np.ones(len(held), dtype=bool)
    if split_name == "train":
        return (~held) & (split_name == "train")  # placeholder, replaced below
    return np.ones(len(held), dtype=bool)


def load_station_table():
    stations = pd.read_parquet(get_domain_stations_path(None, "full"))
    keep = ["station_name"]
    for c in ("x", "y", "elev"):
        if c in stations.columns:
            keep.append(c)
    stations = stations[keep].drop_duplicates("station_name")
    terrain_path = get_station_terrain_path()
    if terrain_path.exists():
        terrain = pd.read_parquet(terrain_path)
        tkeep = ["station_name"] + [c for c in TERRAIN_COLS + ("clc_level2",) if c in terrain.columns]
        terrain = terrain[tkeep].drop_duplicates("station_name")
        stations = stations.merge(terrain, on="station_name", how="left", suffixes=("", "_t"))
    return stations.set_index("station_name")


def quarter_split_map():
    raw = load_quarter_splits()
    out = {}
    for name, qs in raw.items():
        for q in qs:
            out[q] = name
    return out


def run_variable(time_res, var, stations, held_set, qmap):
    path = get_aggregated_data_path(None, time_res)
    import fastparquet
    pf = fastparquet.ParquetFile(str(path))
    names = set(pf.columns)
    tcol = _time_col(names, time_res)
    cols = [c for c in ("station_name", tcol, var) if c in names]
    trace = TRACE.get(var, {}).get(time_res)
    acc = {s: Acc(trace, TERRAIN_COLS, N_LEVEL2) for s in SUBSETS}
    buf_time = None
    buf = []

    def flush():
        if not buf:
            return
        sl = pd.concat(buf, ignore_index=True)
        xy = sl[["x", "y"]].to_numpy(dtype=float)
        v = sl[var].to_numpy(dtype=float)
        raw = _local_autocorr(xy, v)
        resid = _local_autocorr(xy, _idw_residual(xy, v))
        for name in SUBSETS:
            if name == "all" or (sl["_sub"] == name).any():
                acc[name].add_ac(raw, resid)
        buf.clear()

    seen = 0
    for df in pf.iter_row_groups(columns=cols):
        df = df.copy()
        df["time"] = _parse_time(df[tcol], time_res)
        df = df.dropna(subset=["time", var])
        if df.empty:
            continue
        df = df.join(stations, on="station_name")
        held = df["station_name"].astype(str).isin(held_set).to_numpy()
        quarters = df["time"].map(quarter_of)
        split = quarters.map(qmap).fillna("outside")
        df["_held"] = held
        df["_split"] = split.to_numpy()
        df["_sub"] = ""
        is_train = (~df["_held"]) & (df["_split"] == "train")
        is_dev = (~df["_held"]) & (df["_split"] == "tune")
        is_test = df["_held"] & (df["_split"] == "test")
        df.loc[is_train.to_numpy(), "_sub"] = "train"
        df.loc[is_dev.to_numpy(), "_sub"] = "dev"
        df.loc[is_test.to_numpy(), "_sub"] = "test"

        y = df[var].to_numpy(dtype=float)
        st = df["station_name"].astype(str).to_numpy()
        tm = df["time"].to_numpy()
        acc["all"].add_values(y, st, tm)
        acc["all"].add_terrain(y, df)
        if "clc_level2" in df.columns:
            acc["all"].add_clc(y, df["clc_level2"].to_numpy())
        for name, mask in (("train", is_train), ("dev", is_dev), ("test", is_test)):
            if not mask.any():
                continue
            m = mask.to_numpy()
            acc[name].add_values(y[m], st[m], tm[m])
            acc[name].add_terrain(y[m], df.loc[m])
            if "clc_level2" in df.columns:
                acc[name].add_clc(y[m], df.loc[m, "clc_level2"].to_numpy())

        # field buffer for AC, in time order within the chunk
        if "x" in df.columns:
            df = df.sort_values("time")
            for t, g in df.groupby("time", sort=False):
                if buf_time is None:
                    buf_time = t
                if t != buf_time:
                    flush()
                    buf_time = t
                buf.append(g[["x", "y", var, "_sub"]])
        seen += len(df)
        if seen % 2_000_000 < CHUNK:
            print(f"    {seen:,} rows", flush=True)
    flush()

    rows = []
    stored = {}
    for name in SUBSETS:
        row, cat = acc[name].finish(time_res, var, name)
        rows.append(row)
        stored[name] = cat
        print(
            f"  {name:5s}  n={row['n_rows']:9d}  st={row['n_stations']:4d}  "
            f"fields={row['n_fields']:6d}  wet={row['wet_frac'] if np.isfinite(row['wet_frac']) else float('nan'):.2f}  "
            f"skew={row['skew'] if np.isfinite(row['skew']) else float('nan'):.2f}  "
            f"ac_raw={row['ac_raw'] if np.isfinite(row['ac_raw']) else float('nan'):.2f}  "
            f"ac_resid={row['ac_resid'] if np.isfinite(row['ac_resid']) else float('nan'):.2f}",
            flush=True,
        )
    train = stored["train"]
    for row in rows:
        if row["subset"] == "train":
            row["ks_vs_train"] = 0.0
        else:
            row["ks_vs_train"] = _ks(stored[row["subset"]], train)
    del acc, stored
    gc.collect()
    return rows


def main():
    p = argparse.ArgumentParser(description="Variable x timescale profile, full panel")
    add_time_res_arg(p)
    p.add_argument("--variables", nargs="*", default=list(VARS))
    args = p.parse_args()
    wanted = [args.time_resolution] if args.time_resolution else list(TIME_RESOLUTIONS)

    print("loading station and terrain table", flush=True)
    stations = load_station_table()
    held, _ = load_holdout()
    held_set = set(map(str, held))
    qmap = quarter_split_map()

    rows = []
    for time_res in wanted:
        for var in args.variables:
            print(f"\n{time_res}  {var}", flush=True)
            try:
                rows.extend(run_variable(time_res, var, stations, held_set, qmap))
            except Exception as exc:
                print(f"  failed: {exc}")
            gc.collect()

    out = pd.DataFrame(rows)
    dest = ensure_dir(get_source_root() / "profile") / "variable_profile.csv"
    out.to_csv(dest, index=False)
    print(f"\nwrote {dest}  rows={len(out)}")


if __name__ == "__main__":
    main()
