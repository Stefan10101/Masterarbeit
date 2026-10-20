#!/usr/bin/env python3
"""
One-time generation of the shared evaluation sets.

Usage (from the repo root or anywhere):
    python CODE/shared/splits/make_evaluation_sets.py
    python CODE/shared/splits/make_evaluation_sets.py --fraction 0.12 --seed 7

Writes
    CODE/shared/splits/holdout_stations.yaml
    CODE/shared/splits/quarter_splits.yaml

Every method then loads these two files; the held-out stations are never
used for training or tuning.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make CODE importable
CODE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_DIR))

from paths import get_metadata_path, get_domain_stations_path, get_station_terrain_path
from shared.splits.evaluation_sets import (
    select_stratified_holdout,
    assign_quarter_splits,
    write_holdout,
    write_quarter_splits,
    HOLDOUT_PATH,
    QUARTER_SPLITS_PATH,
)


def load_station_table() -> pd.DataFrame:
    """
    Prefer the rich overview / terrain table; fall back to the projected
    domain stations if necessary.  Normalises column names to
    station_name, lon, lat, elev.
    """
    candidates = []

    # 1. QC/stations_overview.csv (source of truth for plotting)
    meta = get_metadata_path()
    if meta.exists():
        candidates.append(("overview", pd.read_csv(meta)))

    # 2. Shared terrain features (has elev and coordinates if already built)
    terrain = get_station_terrain_path()
    if terrain.exists():
        candidates.append(("terrain", pd.read_parquet(terrain)))

    # 3. Domain stations parquet (x/y in EPSG:31287; may lack lon/lat)
    try:
        dom = get_domain_stations_path("IDW", "full")
        if dom.exists():
            candidates.append(("domain", pd.read_parquet(dom)))
    except Exception:
        pass

    if not candidates:
        raise FileNotFoundError(
            "No station table found. Expected QC/stations_overview.csv "
            "or Source/stations/full/stations_projected.parquet"
        )

    # Pick the first that has usable lon/lat
    for name, df in candidates:
        cols = {c.lower(): c for c in df.columns}
        lon = next((cols[c] for c in ("lon", "longitude", "long") if c in cols), None)
        lat = next((cols[c] for c in ("lat", "latitude") if c in cols), None)
        sid = next((cols[c] for c in ("station_name", "station", "name", "id") if c in cols), None)
        if lon and lat and sid:
            elev = next(
                (cols[c] for c in ("elev", "elev_dem", "hoehe", "elevation", "height") if c in cols),
                None,
            )
            out = pd.DataFrame({
                "station_name": df[sid].astype(str),
                "lon": pd.to_numeric(df[lon], errors="coerce"),
                "lat": pd.to_numeric(df[lat], errors="coerce"),
            })
            if elev:
                out["elev"] = pd.to_numeric(df[elev], errors="coerce")
            else:
                out["elev"] = np.nan
            out = out.dropna(subset=["lon", "lat"]).drop_duplicates("station_name")
            print(f"loaded {len(out)} stations from {name} ({meta if name=='overview' else 'other'})")
            return out

    raise RuntimeError("Station tables exist but none contain lon/lat/station_name")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate shared hold-out stations and quarter splits")
    p.add_argument("--fraction", type=float, default=0.15,
                   help="Fraction of Alpine stations to hold out (default 0.15)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--elev-bins", type=int, default=4, help="Number of elevation quantile bins")
    p.add_argument("--spatial-step", type=float, default=0.75,
                   help="Lon/lat grid step in degrees for spatial stratification")
    p.add_argument("--start-year", type=int, default=2020)
    p.add_argument("--end-year", type=int, default=2025)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--tune-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--no-alpine", action="store_true",
                   help="Do not restrict to the Alpine Convention perimeter")
    args = p.parse_args()

    stations = load_station_table()

    held, pool = select_stratified_holdout(
        stations,
        fraction=args.fraction,
        seed=args.seed,
        n_elev_bins=args.elev_bins,
        spatial_step_deg=args.spatial_step,
        alpine_only=not args.no_alpine,
    )

    hold_meta = {
        "fraction": args.fraction,
        "seed": args.seed,
        "n_elev_bins": args.elev_bins,
        "spatial_step_deg": args.spatial_step,
        "alpine_only": not args.no_alpine,
        "source": "stratified (space × elevation) inside Alpine Convention perimeter",
    }
    write_holdout(held, pool, hold_meta, HOLDOUT_PATH)
    print(f"wrote {len(held)} held-out / {len(pool)} pool stations → {HOLDOUT_PATH}")

    qsplits = assign_quarter_splits(
        start_year=args.start_year,
        end_year=args.end_year,
        train_frac=args.train_frac,
        tune_frac=args.tune_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    q_meta = {
        "start_year": args.start_year,
        "end_year": args.end_year,
        "train_frac": args.train_frac,
        "tune_frac": args.tune_frac,
        "test_frac": args.test_frac,
        "seed": args.seed,
        "note": "Whole year-quarters; every season appears in each split when possible",
    }
    write_quarter_splits(qsplits, q_meta, QUARTER_SPLITS_PATH)
    print(f"wrote quarter splits → {QUARTER_SPLITS_PATH}")
    for name, qs in qsplits.items():
        print(f"  {name:5s}: {len(qs):2d} quarters  {qs}")


if __name__ == "__main__":
    main()
