#!/usr/bin/env python3
"""
aggregation_core.py
Central functions for station data aggregation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional


def safe_agg(series: pd.Series, how: str) -> float:
    if series.isna().all():
        return np.nan
    if how == "sum":
        return series.sum(min_count=1)
    elif how == "mean":
        return series.mean()
    elif how == "min":
        return series.min()
    elif how == "max":
        return series.max()
    else:
        raise ValueError(f"Unknown aggregation method: {how}")


def get_time_col_name(resolution: str) -> str:
    mapping = {
        "daily": "date",
        "weekly": "year_week",
        "monthly": "year_month",
        "half_hourly": "timestamp",
        "seasonal": "season"
    }
    return mapping.get(resolution, "time")


def get_season(month: int) -> str:
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Autumn"
    return "Unknown"


MIN_PERIOD_COVERAGE = 0.80  # GPCC/REGEN use 0.70; lock >=0.80 as requested


def process_station_parquet(
    parquet_path: Path,
    station_name: str,
    targets: Dict,
    resolution: str,
    time_col_name: Optional[str] = None,
    seasonal_mode: str = "across_years",
    min_coverage: float = MIN_PERIOD_COVERAGE,
) -> pd.DataFrame:

    if time_col_name is None:
        time_col_name = get_time_col_name(resolution)

    try:
        df = pd.read_parquet(parquet_path)
    except Exception:
        return pd.DataFrame()

    if "timestamp" not in df.columns:
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    available_cols = set(df.columns)
    rows = []

    # Seasonal
    if resolution == "seasonal":
        df = df.reset_index()
        df["month"] = df["timestamp"].dt.month
        df["season"] = df["month"].apply(get_season)

        if seasonal_mode == "per_year":
            df["year"] = df["timestamp"].dt.year
            df["season_period"] = df["year"].astype(str) + "-" + df["season"]
            group_col = "season_period"
        else:
            group_col = "season"

        for target_name, spec in targets.items():
            raw_col = spec["raw_col"]
            if raw_col not in available_cols:
                continue

            for stat in spec["stats"]:
                col_name = spec["output_names"][spec["stats"].index(stat)]

                if stat == "sum":
                    grouped = df.groupby(group_col)[raw_col].sum(min_count=1)
                elif stat == "mean":
                    grouped = df.groupby(group_col)[raw_col].mean()
                elif stat == "min":
                    grouped = df.groupby(group_col)[raw_col].min()
                elif stat == "max":
                    grouped = df.groupby(group_col)[raw_col].max()
                else:
                    continue

                agg_df = grouped.to_frame(name=col_name)
                agg_df["station_name"] = station_name
                agg_df[time_col_name] = agg_df.index
                agg_df = agg_df.reset_index(drop=True)
                rows.append(agg_df)

        if not rows:
            return pd.DataFrame()

        result = pd.concat(rows, axis=1)
        result = result.loc[:, ~result.columns.duplicated()]
        return result

    # Standard resampling
    resample_rule = {
        "daily": "D",
        "weekly": "W",
        "monthly": "ME",
        "half_hourly": "30min"
    }.get(resolution)

    if resample_rule is None:
        raise ValueError(f"Unsupported resolution: {resolution}")

    for target_name, spec in targets.items():
        raw_col = spec["raw_col"]
        if raw_col not in available_cols:
            continue

        for stat in spec["stats"]:
            col_name = spec["output_names"][spec["stats"].index(stat)]

            src = df[raw_col]
            count = src.resample(resample_rule).count()
            expected = src.resample(resample_rule).size()
            enough = (count / expected.replace(0, np.nan)) >= min_coverage
            if resolution == "half_hourly":
                enough[:] = True

            if stat == "sum":
                resampled = src.resample(resample_rule).sum(min_count=1)
            elif stat == "mean":
                resampled = src.resample(resample_rule).mean()
            elif stat == "min":
                resampled = src.resample(resample_rule).min()
            elif stat == "max":
                resampled = src.resample(resample_rule).max()
            else:
                continue
            resampled = resampled.where(enough)

            agg_df = resampled.to_frame(name=col_name)
            agg_df["station_name"] = station_name

            if resolution == "daily":
                agg_df[time_col_name] = agg_df.index.date.astype(str)
            elif resolution == "weekly":
                agg_df[time_col_name] = agg_df.index.strftime("%Y-W%W")
            elif resolution == "monthly":
                agg_df[time_col_name] = agg_df.index.strftime("%Y-%m")
            elif resolution == "half_hourly":
                agg_df[time_col_name] = agg_df.index

            agg_df = agg_df.reset_index(drop=True)
            rows.append(agg_df)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, axis=1)
    result = result.loc[:, ~result.columns.duplicated()]
    return result


def aggregate_all_stations(
    metadata_df: pd.DataFrame,
    data_dir: Path,
    targets: Dict,
    resolution: str,
    verbose: bool = True
) -> pd.DataFrame:

    all_data = []
    time_col = get_time_col_name(resolution)

    for _, row in metadata_df.iterrows():
        station = row["station_name"]
        parquet_path = data_dir / row["parquet"]

        if not parquet_path.exists():
            if verbose:
                print(f"  [SKIP] {station} - file not found")
            continue

        if verbose:
            print(f"Processing: {station} ...", end=" ")

        df = process_station_parquet(
            parquet_path=parquet_path,
            station_name=station,
            targets=targets,
            resolution=resolution
        )

        if df.empty:
            if verbose:
                print("no usable data")
            continue

        all_data.append(df)
        if verbose:
            print(f"ok ({len(df)} rows)")

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)

    id_cols = ["station_name", time_col]
    value_cols = [c for c in combined.columns if c not in id_cols]
    combined = combined[id_cols + value_cols]
    combined = combined.sort_values(["station_name", time_col]).reset_index(drop=True)

    return combined