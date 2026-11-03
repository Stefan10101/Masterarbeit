#!/usr/bin/env python3
"""Aggregate raw station series into Source/aggregated/.

Adds an inside_domain column (1 = inside Alpine Convention perimeter + 100 km, 0 = outside).
All stations are retained so the flag can be verified.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

CODE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_DIR))

from paths import get_aggregated_data_path, get_metadata_path, get_raw_data_dir, ensure_dir
from aggregation_core import aggregate_all_stations, get_time_col_name
from shared.splits.evaluation_sets import stations_inside_domain

DEFAULT_CFG = CODE_DIR / "shared" / "source" / "config.yaml"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resolution", required=True,
                   choices=["daily", "weekly", "monthly", "half_hourly", "seasonal"])
    p.add_argument("--seasonal_mode", default="across_years",
                   choices=["across_years", "per_year"])
    p.add_argument("--filter", default="all", choices=["all", "day", "night"])
    p.add_argument("--start-date", type=str, default=None)
    p.add_argument("--end-date", type=str, default=None)
    p.add_argument("--config", default=str(DEFAULT_CFG))
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    resolution = args.resolution
    out_file = get_aggregated_data_path(None, resolution)
    if args.filter != "all":
        out_file = out_file.with_name(f"{args.filter}_{out_file.name}")
    ensure_dir(out_file.parent)

    print(f"aggregate {resolution} -> {out_file}", flush=True)
    metadata = pd.read_csv(get_metadata_path())
    # Flag stations inside Alpine Convention perimeter + 100 km; keep everyone.
    inside = stations_inside_domain(metadata, buffer_km=100.0)
    inside_ids = set(inside["station_name"].astype(str))
    print(
        f"stations inside domain (perimeter + 100 km): {len(inside_ids)} / {len(metadata)}",
        flush=True,
    )
    combined = aggregate_all_stations(
        metadata_df=metadata,
        data_dir=get_raw_data_dir(),
        targets=cfg["targets"],
        resolution=resolution,
        verbose=True,
    )
    if combined.empty:
        print("No data was aggregated.")
        return

    if args.filter != "all" and "is_daylight" in combined.columns:
        flag = args.filter == "day"
        combined = combined[combined["is_daylight"] == flag]

    time_col = get_time_col_name(resolution)
    if args.start_date or args.end_date:
        combined[time_col] = pd.to_datetime(combined[time_col], errors="coerce")
        if args.start_date:
            combined = combined[combined[time_col] >= args.start_date]
        if args.end_date:
            combined = combined[combined[time_col] <= args.end_date]

    combined["inside_domain"] = (
        combined["station_name"].astype(str).isin(inside_ids).astype("int8")
    )

    combined.to_parquet(out_file, index=False, compression="zstd")
    print(f"Saved {out_file} rows={len(combined):,}", flush=True)
    print(
        f"inside_domain=1: {int(combined['inside_domain'].sum()):,} rows "
        f"({combined.loc[combined['inside_domain']==1, 'station_name'].nunique()} stations)",
        flush=True,
    )


if __name__ == "__main__":
    main()
