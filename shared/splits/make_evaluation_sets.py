#!/usr/bin/env python3
"""
One-time generation of the shared evaluation sets.

Usage (from the repo root or anywhere):
    python CODE/shared/splits/make_evaluation_sets.py
    python CODE/shared/splits/make_evaluation_sets.py --fraction 0.12 --seed 7

Writes, for each overview variable (temperature, precipitation, wind_speed,
relative_humidity, snow_height):
    CODE/shared/splits/holdout_stations_{variable}.yaml
    CODE/shared/splits/tune_targets_{variable}.yaml

and the global
    CODE/shared/splits/quarter_splits.yaml

Every method loads the variable-specific station lists (pass variable= to
apply_evaluation_protocol / load_holdout). Held-out stations are never used
as donors and lie strictly inside the Alpine Convention perimeter;
tune-targets are held out only during hyperparameter scoring.
Pool stations (donors) may lie in the perimeter + 100 km domain.
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

from paths import get_metadata_path, get_domain_stations_path, get_station_terrain_path, get_aggregated_data_path
from shared.splits.evaluation_sets import (
    select_stratified_holdout,
    select_tune_targets,
    assign_quarter_splits,
    write_holdout,
    write_quarter_splits,
    write_tune_targets,
    holdout_path,
    tune_targets_path,
    stations_measuring,
    stations_inside_domain,
    stations_inside_perimeter,
    OVERVIEW_VARS,
    QUARTER_SPLITS_PATH,
)


def load_station_table() -> pd.DataFrame:
    """
    Load QC/stations_overview.csv (preferred) and keep the variables column.
    Normalises to station_name, lon, lat, elev, variables.
    """
    meta = get_metadata_path()
    if not meta.exists():
        raise FileNotFoundError(f"stations_overview.csv not found at {meta}")

    df = pd.read_csv(meta, encoding="utf-8")
    cols = {c.lower(): c for c in df.columns}
    lon = next((cols[c] for c in ("lon", "longitude", "long") if c in cols), None)
    lat = next((cols[c] for c in ("lat", "latitude") if c in cols), None)
    sid = next((cols[c] for c in ("station_name", "station", "name", "id") if c in cols), None)
    if not (lon and lat and sid):
        raise RuntimeError("overview lacks lon/lat/station_name")

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

    var_col = next((cols[c] for c in ("variables", "variable", "vars") if c in cols), None)
    if var_col:
        out["variables"] = df[var_col].astype(str)
    else:
        raise RuntimeError("overview has no 'variables' column")

    out = out.dropna(subset=["lon", "lat"]).drop_duplicates("station_name")
    print(f"loaded {len(out)} stations from overview ({meta})")
    return out



def stations_in_aggregate(time_res: str = "monthly") -> set:
    """Station names that appear in the aggregated parquet (any non-null row)."""
    path = get_aggregated_data_path("Source", time_res)
    if not path.exists():
        print(f"  warning: aggregate file not found at {path}; no station filter applied")
        return set()
    df = pd.read_parquet(path, columns=["station_name"])
    return set(df["station_name"].astype(str).str.strip().unique())



def main() -> None:
    p = argparse.ArgumentParser(description="Generate variable-specific hold-out stations and quarter splits")
    p.add_argument("--fraction", type=float, default=0.15,
                   help="Fraction of domain stations to hold out (default 0.15)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--elev-bins", type=int, default=4, help="Number of elevation quantile bins")
    p.add_argument("--spatial-step", type=float, default=0.75,
                   help="Lon/lat grid step in degrees for spatial stratification")
    p.add_argument("--start-year", type=int, default=2020)
    p.add_argument("--end-year", type=int, default=2025)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--tune-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    args = p.parse_args()

    stations = load_station_table()

    # Domain = Alpine Convention perimeter + 100 km (matches plot_maps.alpine_mask).
    # Donors may come from the full domain.
    # Held-out stations must lie strictly inside the Alpine Convention perimeter
    # (union of all polygons, including any exclave).
    domain_stations = stations_inside_domain(stations)
    perimeter_stations = stations_inside_perimeter(domain_stations)
    print(f"stations inside interpolation domain: {len(domain_stations)}")
    print(f"stations strictly inside Alpine Convention perimeter: {len(perimeter_stations)}")

    for var in OVERVIEW_VARS:
        measuring = stations_measuring(domain_stations, var)
        measuring_peri = stations_measuring(perimeter_stations, var)
        print(f"\n{var}: {len(measuring)} in domain, {len(measuring_peri)} inside perimeter")
        if measuring_peri.empty:
            print(f"  skipped (no stations inside perimeter)")
            continue

        # Stratified hold-out drawn only from stations inside the perimeter.
        held, _ = select_stratified_holdout(
            measuring_peri,
            fraction=args.fraction,
            seed=args.seed,
            n_elev_bins=args.elev_bins,
            spatial_step_deg=args.spatial_step,
            domain_only=False,
            alpine_only=False,
        )
        # Pool = every domain station measuring the variable, minus the held-out.
        held_set = set(held)
        pool = [sid for sid in measuring["station_name"].astype(str) if sid not in held_set]

        # Keep only stations that actually appear in the aggregate file.
        in_agg = stations_in_aggregate("monthly")
        if in_agg:
            before_h, before_p = len(held), len(pool)
            held = [sid for sid in held if str(sid).strip() in in_agg]
            pool = [sid for sid in pool if str(sid).strip() in in_agg]
            print(f"  aggregate filter: held-out {before_h} → {len(held)}, pool {before_p} → {len(pool)}")

        hold_meta = {
            "variable": var,
            "fraction": args.fraction,
            "seed": args.seed,
            "n_elev_bins": args.elev_bins,
            "spatial_step_deg": args.spatial_step,
            "domain": "pool = Alpine Convention polygon + 100 km; held-out = strictly inside Alpine Convention perimeter (all polygons / exclave included)",
            "source": "stations_overview.csv variables column; domain-filtered, hold-out perimeter-filtered",
        }
        hpath = holdout_path(var)
        write_holdout(held, pool, hold_meta, hpath)
        print(f"  wrote {len(held)} held-out / {len(pool)} pool → {hpath.name}")

        pool_df = measuring[measuring["station_name"].isin(pool)].copy()
        tune_targets, donors_during = select_tune_targets(
            pool_df,
            fraction=args.fraction,
            seed=args.seed + 1,
            n_elev_bins=args.elev_bins,
            spatial_step_deg=args.spatial_step,
        )
        tt_meta = {
            "variable": var,
            "fraction": args.fraction,
            "seed": args.seed + 1,
            "n_elev_bins": args.elev_bins,
            "spatial_step_deg": args.spatial_step,
            "note": "Fixed 15 % of the pool; held out only during hyperparameter scoring",
        }
        tpath = tune_targets_path(var)
        write_tune_targets(tune_targets, donors_during, tt_meta, tpath)
        print(f"  wrote {len(tune_targets)} tune-targets / {len(donors_during)} temporary donors → {tpath.name}")

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
        "note": "Contiguous chronological blocks: train 2020-Q1..2023-Q4, tune 2024-Q1..2024-Q4, test 2025-Q1..2025-Q4",
    }
    write_quarter_splits(qsplits, q_meta, QUARTER_SPLITS_PATH)
    print(f"\nwrote quarter splits → {QUARTER_SPLITS_PATH}")
    for name, qs in qsplits.items():
        print(f"  {name:5s}: {len(qs):2d} quarters  {qs}")


if __name__ == "__main__":
    main()
