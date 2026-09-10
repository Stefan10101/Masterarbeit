#!/usr/bin/env python3
"""Station-field completeness at weekly / daily / half-hourly / monthly.

Run from CODE/:
  python shared/aggregation/completeness_report.py
  python shared/aggregation/completeness_report.py --time-resolution weekly daily
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_DIR))

from paths import get_aggregated_data_path, get_plots_root, ensure_dir
from shared.time_res import TIME_RESOLUTIONS, add_time_res_arg

VARS = ("temp_mean", "precip_sum", "wind_mean", "rh_mean", "snow_mean")
MIN_STATIONS = 10
TRACE = {
    "precip_sum": {
        "half_hourly": 0.05,
        "daily": 0.1,
        "weekly": 0.7,
        "monthly": 1.0,
    },
    "snow_mean": {
        "half_hourly": 0.5,
        "daily": 0.5,
        "weekly": 1.0,
        "monthly": 1.0,
    },
}


def parse_times(df: pd.DataFrame, time_res: str) -> pd.Series:
    if "time" in df.columns:
        return pd.to_datetime(df["time"], utc=True, errors="coerce")
    col = {
        "weekly": "year_week",
        "daily": "date",
        "monthly": "year_month",
        "half_hourly": "timestamp",
    }.get(time_res)
    if col and col in df.columns:
        if time_res == "weekly":
            return pd.to_datetime(df[col].astype(str) + "-1", format="%Y-W%W-%w", utc=True, errors="coerce")
        if time_res == "monthly":
            return pd.to_datetime(df[col].astype(str) + "-01", utc=True, errors="coerce")
        return pd.to_datetime(df[col], utc=True, errors="coerce")
    for c in ("timestamp", "date", "year_month", "year_week"):
        if c in df.columns:
            return pd.to_datetime(df[c], utc=True, errors="coerce")
    raise ValueError(f"no time column in {list(df.columns)}")


def station_col(df: pd.DataFrame) -> str:
    for c in ("station_name", "station_id", "name"):
        if c in df.columns:
            return c
    raise ValueError("no station column")


def summarise(time_res: str, min_stations: int) -> pd.DataFrame:
    path = get_aggregated_data_path(None, time_res)
    if not path.exists():
        print(f"MISSING  {time_res}  {path}")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    t = parse_times(df, time_res)
    sid = station_col(df)
    rows = []
    print(f"\n{time_res}  file={path}  rows={len(df):,}  stations={df[sid].nunique()}")
    print(f"  columns={list(df.columns)}")
    tmin, tmax = t.min(), t.max()
    print(f"  span={tmin} → {tmax}")
    for var in VARS:
        if var not in df.columns:
            print(f"  {var:12s}  COLUMN ABSENT")
            continue
        sub = df.loc[t.notna() & df[var].notna(), [sid]].copy()
        sub["time"] = t.loc[sub.index]
        if sub.empty:
            print(f"  {var:12s}  no finite values")
            continue
        counts = sub.groupby("time")[sid].nunique()
        n_t = int(len(counts))
        med = float(counts.median())
        p10 = float(counts.quantile(0.10))
        frac_thin = float((counts < min_stations).mean())
        vals = df.loc[t.notna() & df[var].notna(), var].to_numpy(dtype=float)
        wet = np.nan
        if var in TRACE:
            tr = TRACE[var][time_res]
            wet = float((vals > tr).mean())
        rows.append({
            "time_resolution": time_res,
            "variable": var,
            "n_rows": int(df[var].notna().sum()),
            "n_timestamps": n_t,
            "n_stations_any": int(sub[sid].nunique()),
            "median_stations": med,
            "p10_stations": p10,
            "frac_below_min_stations": frac_thin,
            "min_stations": min_stations,
            "wet_frac": wet,
            "t_start": str(pd.Timestamp(counts.index.min())),
            "t_end": str(pd.Timestamp(counts.index.max())),
        })
        wet_s = f"  wet={wet:.3f}" if np.isfinite(wet) else ""
        print(
            f"  {var:12s}  times={n_t:7d}  st_any={sub[sid].nunique():4d}  "
            f"med={med:6.1f}  p10={p10:6.1f}  thin={frac_thin:5.1%}{wet_s}"
        )
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description="Aggregated-panel completeness")
    add_time_res_arg(p)
    p.add_argument("--resolutions", nargs="*", default=None)
    p.add_argument("--min-stations", type=int, default=MIN_STATIONS)
    args = p.parse_args()
    wanted = args.resolutions or ([args.time_resolution] if args.time_resolution else list(TIME_RESOLUTIONS))
    frames = [summarise(r, args.min_stations) for r in wanted]
    out = pd.concat([f for f in frames if len(f)], ignore_index=True)
    if out.empty:
        print("no aggregated files found")
        return
    dest = ensure_dir(get_plots_root() / "completeness") / "station_field_completeness.csv"
    try:
        out.to_csv(dest, index=False)
        print(f"\nwrote {dest}")
    except Exception as exc:
        print(f"\ncould not write plots csv ({exc}); printing table only")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
