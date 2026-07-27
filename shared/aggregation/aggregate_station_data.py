#!/usr/bin/env python3
"""
aggregate_station_data.py
"""

import argparse
import pandas as pd
import sys
from pathlib import Path
import yaml

sys.path.append(str(Path(__file__).resolve().parents[2]))
from paths import get_aggregated_dir, get_metadata_path, get_raw_data_dir
from aggregation_core import aggregate_all_stations, get_time_col_name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["BSS", "IDW", "RFSI"])
    parser.add_argument("--resolution", required=True,
                        choices=["daily", "weekly", "monthly", "half_hourly", "seasonal"])
    parser.add_argument("--seasonal_mode", default="across_years",
                        choices=["across_years", "per_year"])
    parser.add_argument("--filter", default="all", choices=["all", "day", "night"])
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    method = args.method.upper()
    resolution = args.resolution

    aggregated_dir = get_aggregated_dir(method)
    aggregated_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*75}")
    print(f"Method: {method} | Resolution: {resolution}")
    if resolution == "seasonal":
        print(f"Seasonal Mode: {args.seasonal_mode}")
    print(f"Output: {aggregated_dir}")
    print(f"{'='*75}\n")

    # Use paths.py as single source of truth
    meta = pd.read_csv(get_metadata_path())
    data_dir = get_raw_data_dir()
    targets = cfg["targets"]

    combined = aggregate_all_stations(
        metadata_df=meta,
        data_dir=data_dir,
        targets=targets,
        resolution=resolution,
        verbose=True
    )

    if combined.empty:
        print("No data was aggregated.")
        return

    # Day/Night filter
    if args.filter != "all" and "is_daylight" in combined.columns:
        if args.filter == "day":
            combined = combined[combined["is_daylight"] == True]
        elif args.filter == "night":
            combined = combined[combined["is_daylight"] == False]

    # Custom date range
    time_col = get_time_col_name(resolution)
    if args.start_date or args.end_date:
        combined[time_col] = pd.to_datetime(combined[time_col], errors="coerce")
        if args.start_date:
            combined = combined[combined[time_col] >= args.start_date]
        if args.end_date:
            combined = combined[combined[time_col] <= args.end_date]

    # Save
    filename = f"{resolution}_station_data.parquet"
    if args.filter != "all":
        filename = f"{args.filter}_{resolution}_station_data.parquet"

    out_file = aggregated_dir / filename
    combined.to_parquet(out_file, index=False, compression="zstd")

    print(f"\nSaved: {out_file}")
    print(f"Total rows: {len(combined):,}")


if __name__ == "__main__":
    main()