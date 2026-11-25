#!/usr/bin/env python3
"""Aggregate raw station series into Source/aggregated/.

Adds an inside_domain column (1 = inside Alpine Convention perimeter + 100 km).
Only inside-domain stations are processed. Stations are aggregated and written
in batches so the full panel never sits in memory.

Writer preference: fastparquet append (no pyarrow import). Falls back to
per-batch parquet files if fastparquet is unavailable.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CODE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_DIR))

from paths import get_aggregated_data_path, get_metadata_path, get_raw_data_dir, ensure_dir
from aggregation_core import process_station_parquet, get_time_col_name
from shared.splits.evaluation_sets import stations_inside_domain

DEFAULT_CFG = CODE_DIR / "shared" / "source" / "config.yaml"
BATCH_SIZE = 40


def _expected_columns(targets: dict, time_col: str) -> list[str]:
    """Fixed schema so every batch has the same columns for fastparquet append."""
    cols = ["station_id", "station_name", time_col]
    for spec in targets.values():
        cols.extend(spec["output_names"])
    cols.append("inside_domain")
    # preserve order, drop dups
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _parquet_engine() -> str:
    try:
        import fastparquet  # noqa: F401
        return "fastparquet"
    except ImportError:
        raise SystemExit(
            "fastparquet is required (pyarrow DLL is broken in this env).\n"
            "Install with:  conda install -n Master fastparquet\n"
            "or:            pip install fastparquet"
        )



def _agg_one(task: tuple) -> pd.DataFrame:
    """Process exactly one station. Returns empty frame on skip/failure."""
    parquet_path, station_name, station_id, targets, resolution, seasonal_mode = task
    try:
        return process_station_parquet(
            parquet_path=Path(parquet_path),
            station_name=station_name,
            station_id=station_id,
            targets=targets,
            resolution=resolution,
            seasonal_mode=seasonal_mode,
        )
    except Exception as exc:
        print(f"  [ERROR] {station_name}: {exc}", flush=True)
        return pd.DataFrame()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resolution", required=True,
                   choices=["daily", "weekly", "monthly", "six_hourly", "half_hourly", "seasonal"])
    p.add_argument("--seasonal_mode", default="across_years",
                   choices=["across_years", "per_year"])
    p.add_argument("--filter", default="all", choices=["all", "day", "night"])
    p.add_argument("--start-date", type=str, default=None)
    p.add_argument("--end-date", type=str, default=None)
    p.add_argument("--config", default=str(DEFAULT_CFG))
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                   help="Stations per write batch (default 40)")
    return p.parse_args()


def _apply_filters(df: pd.DataFrame, args, time_col: str) -> pd.DataFrame:
    if df.empty:
        return df
    if args.filter != "all" and "is_daylight" in df.columns:
        flag = args.filter == "day"
        df = df[df["is_daylight"] == flag]
    if args.start_date or args.end_date:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        if args.start_date:
            df = df[df[time_col] >= args.start_date]
        if args.end_date:
            df = df[df[time_col] <= args.end_date]
    return df


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    resolution = args.resolution
    out_file = get_aggregated_data_path(None, resolution)
    if args.filter != "all":
        out_file = out_file.with_name(f"{args.filter}_{out_file.name}")
    ensure_dir(out_file.parent)

    engine = _parquet_engine()
    print(f"aggregate {resolution} -> {out_file} (engine={engine})", flush=True)

    metadata = pd.read_csv(get_metadata_path())
    n_all = len(metadata)
    inside = stations_inside_domain(metadata, buffer_km=100.0)
    inside_ids = set(inside["station_name"].astype(str))
    metadata = metadata[metadata["station_name"].astype(str).isin(inside_ids)].reset_index(drop=True)
    print(
        f"stations inside domain (perimeter + 100 km): {len(metadata)} / {n_all}",
        flush=True,
    )

    data_dir = get_raw_data_dir()
    time_col = get_time_col_name(resolution)
    targets = cfg["targets"]
    schema_cols = _expected_columns(targets, time_col)
    batch_size = max(1, args.batch_size)

    # Clean previous output / parts
    if out_file.exists():
        out_file.unlink()
    parts_dir = out_file.with_suffix(".parts")
    if parts_dir.exists():
        for p in parts_dir.glob("*.parquet"):
            p.unlink()
    parts_dir.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    n_stations = 0
    batch: list[pd.DataFrame] = []
    part_idx = 0
    first_write = True

    def flush():
        nonlocal n_rows, batch, part_idx, first_write
        if not batch:
            return
        df = pd.concat(batch, ignore_index=True)
        batch = []
        df = _apply_filters(df, args, time_col)
        if df.empty:
            return
        df["inside_domain"] = np.int8(1)
        # Align to the full schema. Stations that lack a variable contribute NaN.
        for c in schema_cols:
            if c not in df.columns:
                df[c] = np.nan
        df = df[schema_cols]

        if engine == "fastparquet":
            df.to_parquet(
                out_file,
                engine="fastparquet",
                compression="zstd",
                index=False,
                append=not first_write,
            )
            first_write = False
        else:
            raise RuntimeError("only fastparquet is supported in this script")

        n_rows += len(df)
        print(f"  wrote batch: {len(df):,} rows (total {n_rows:,})", flush=True)

    # One row per physical station. Prefer station_id; fall back to station_name.
    key = "station_id" if "station_id" in metadata.columns else "station_name"
    before = len(metadata)
    metadata = metadata.drop_duplicates(subset=[key], keep="first").reset_index(drop=True)
    if len(metadata) < before:
        print(f"deduped overview on {key}: {before} -> {len(metadata)}", flush=True)

    tasks = []
    for _, row in metadata.iterrows():
        station = str(row["station_name"])
        parquet_path = data_dir / row["parquet"]
        if not parquet_path.exists():
            print(f"  [SKIP] {station} - file not found", flush=True)
            continue
        sid = str(row["station_id"]) if "station_id" in row and pd.notna(row["station_id"]) else None
        tasks.append((str(parquet_path), station, sid, targets, resolution, args.seasonal_mode))

    n_workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"aggregating {len(tasks)} unique stations with {n_workers} workers", flush=True)

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_agg_one, task) for task in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            df = fut.result()
            if df is None or df.empty:
                continue
            batch.append(df)
            n_stations += 1
            if len(batch) >= batch_size:
                flush()
            if i % 100 == 0:
                print(f"  processed {i}/{len(tasks)}", flush=True)

    flush()

    if n_rows == 0:
        print("No data was aggregated.")
        return

    if engine != "fastparquet":
        # Concatenate parts without holding the full frame: stream via pandas
        # in order, writing a single file with fastparquet if it becomes available,
        # otherwise leave the parts directory and print the path.
        print(
            f"pyarrow unavailable and fastparquet missing — "
            f"parts written to {parts_dir}",
            flush=True,
        )
        print("Install fastparquet and re-run, or concat the parts yourself.", flush=True)
        return

    # Remove empty parts dir
    try:
        parts_dir.rmdir()
    except OSError:
        pass

    print(
        f"Saved {out_file} rows={n_rows:,} "
        f"({n_stations} stations, all inside_domain=1)",
        flush=True,
    )


if __name__ == "__main__":
    main()
