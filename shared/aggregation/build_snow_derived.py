#!/usr/bin/env python3
"""Compute slr / swe / snow_density and merge into Source/aggregated parquet.

SLR: rolling 6 h net HS change / P on QC 30-min masters. Event iff
HN>=2 cm and P>=1 mm. Daily = mean of valid 6 h events that day.
hh = 6 h window ending at the timestamp. No weekly SLR.

SWE/density: ΔSNOW on daily snow_mean (cm), Alpine nixmass parameters.
Weekly = mean of daily. hh = daily value held onto the timestamp.

Does not rebuild the rest of the aggregated files. Existing columns stay.
Run from CODE/:
  python shared/aggregation/build_snow_derived.py
  python shared/aggregation/build_snow_derived.py --time-resolution daily
  python shared/aggregation/build_snow_derived.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_DIR))

from paths import (  # noqa: E402
    get_aggregated_data_path,
    get_metadata_path,
    get_raw_data_dir,
)
from shared.snowpack.deltasnow import deltasnow_daily_cm  # noqa: E402
from shared.snowpack.slr import daily_slr_mean, slr_from_half_hourly  # noqa: E402
from shared.time_res import TIME_RESOLUTIONS  # noqa: E402

DERIVED_COLS = ("slr", "swe", "snow_density")
HS_COL = "snow_height"
P_COL = "precipitation"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--time-resolution",
        nargs="*",
        default=["daily", "weekly", "half_hourly"],
        choices=[r for r in TIME_RESOLUTIONS if r != "monthly"],
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-stations", type=int, default=None)
    return p.parse_args()


def _station_col(meta: pd.DataFrame) -> str:
    for c in ("station_name", "station", "name"):
        if c in meta.columns:
            return c
    raise ValueError(f"no station column in metadata: {list(meta.columns)}")


def _resolve_parquet(raw_root: Path, row: pd.Series) -> Path | None:
    for key in ("parquet", "file_path", "path"):
        if key in row.index and pd.notna(row[key]):
            p = Path(str(row[key]))
            if not p.is_absolute():
                p = raw_root / p
            if p.exists():
                return p
    name = str(row.get("station_name") or row.get("station") or "")
    if name:
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
        cand = raw_root / f"{safe}_full_2020_2025_master.parquet"
        if cand.exists():
            return cand
    return None


def load_qc_station(path: Path) -> pd.DataFrame:
    cols = ["timestamp", HS_COL, P_COL]
    have = pd.read_parquet(path, columns=None)
    use = [c for c in cols if c in have.columns]
    if "timestamp" not in use:
        return pd.DataFrame()
    df = have[use].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df


def slr_for_station(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (hh slr rows, daily slr rows). Empty if P or HS missing."""
    if HS_COL not in df.columns or P_COL not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    hh = slr_from_half_hourly(df["timestamp"], df[HS_COL], df[P_COL])
    if hh.empty:
        return hh, pd.DataFrame()
    daily = daily_slr_mean(hh)
    return hh[["time", "slr"]], daily


def swe_from_daily_panel(daily: pd.DataFrame) -> pd.DataFrame:
    """ΔSNOW per station on snow_mean [cm]. Skip stations with no real HS."""
    if daily.empty or "snow_mean" not in daily.columns:
        return pd.DataFrame(columns=["station_name", "date", "swe", "snow_density"])
    rows = []
    n_skip = 0
    for station, g in daily.groupby("station_name", sort=False):
        hs = pd.to_numeric(g["snow_mean"], errors="coerce")
        if not (hs > 0.5).any():
            n_skip += 1
            continue
        g = g.assign(_hs=hs).sort_values("date")
        swe, dens = deltasnow_daily_cm(g["_hs"].to_numpy())
        rows.append(pd.DataFrame({
            "station_name": station,
            "date": g["date"].astype(str).to_numpy(),
            "swe": swe,
            "snow_density": dens,
        }))
    print(f"ΔSNOW skipped {n_skip} stations with no HS > 0.5 cm", flush=True)
    if not rows:
        return pd.DataFrame(columns=["station_name", "date", "swe", "snow_density"])
    return pd.concat(rows, ignore_index=True)


def match_timestamp(series, template: pd.Series) -> pd.Series:
    """Same tz and datetime unit as template so merge keys line up."""
    t = pd.to_datetime(series, utc=True, errors="coerce")
    if getattr(template.dtype, "tz", None) is not None:
        if getattr(t.dtype, "tz", None) is None:
            t = t.dt.tz_localize("UTC")
        else:
            t = t.dt.tz_convert("UTC")
    else:
        if getattr(t.dtype, "tz", None) is not None:
            t = t.dt.tz_convert("UTC").dt.tz_localize(None)
    try:
        return t.astype(template.dtype)
    except (TypeError, ValueError):
        return t


def week_key(dates) -> pd.Series:
    t = pd.to_datetime(dates, errors="coerce")
    return t.dt.strftime("%Y-W%W")


def merge_columns(base: pd.DataFrame, extra: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    drop = [c for c in DERIVED_COLS if c in base.columns]
    if drop:
        base = base.drop(columns=drop)
    keep = keys + [c for c in DERIVED_COLS if c in extra.columns]
    extra = extra[keep].drop_duplicates(keys)
    return base.merge(extra, on=keys, how="left")


def write_parquet(path: Path, df: pd.DataFrame, dry: bool):
    if dry:
        print(f"  dry-run skip write {path} rows={len(df):,}", flush=True)
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)
    print(f"  wrote {path} rows={len(df):,}", flush=True)


def main():
    args = parse_args()
    meta_path = get_metadata_path()
    raw_root = get_raw_data_dir()
    print(f"metadata={meta_path}", flush=True)
    print(f"qc={raw_root}", flush=True)

    slr_hh_parts = []
    slr_day_parts = []
    n_hs_p = 0
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        sid = _station_col(meta)
        n = 0
        for _, row in meta.iterrows():
            if args.max_stations is not None and n >= args.max_stations:
                break
            path = _resolve_parquet(raw_root, row)
            if path is None:
                continue
            df = load_qc_station(path)
            if df.empty or HS_COL not in df.columns or P_COL not in df.columns:
                continue
            if df[HS_COL].notna().sum() == 0 or df[P_COL].notna().sum() == 0:
                continue
            n += 1
            n_hs_p += 1
            hh, day = slr_for_station(df)
            name = str(row[sid])
            if not hh.empty:
                hh = hh.rename(columns={"time": "timestamp"})
                hh["station_name"] = name
                slr_hh_parts.append(hh)
            if not day.empty:
                day["station_name"] = name
                slr_day_parts.append(day)
            if n % 25 == 0:
                print(f"  slr stations {n}", flush=True)
        print(f"SLR co-located stations={n_hs_p}", flush=True)
    else:
        print("WARNING no stations_overview.csv — SLR skipped, SWE still from daily parquet", flush=True)

    slr_hh = pd.concat(slr_hh_parts, ignore_index=True) if slr_hh_parts else pd.DataFrame()
    slr_day = pd.concat(slr_day_parts, ignore_index=True) if slr_day_parts else pd.DataFrame()
    if not slr_hh.empty:
        n_evt = int(slr_hh["slr"].notna().sum())
        print(
            f"SLR hh rows={len(slr_hh):,} events={n_evt:,} "
            f"median={slr_hh['slr'].median(skipna=True):.2f}",
            flush=True,
        )
    if not slr_day.empty:
        print(
            f"SLR daily event-days={slr_day['slr'].notna().sum():,} "
            f"stations={slr_day['station_name'].nunique()}",
            flush=True,
        )

    daily_path = get_aggregated_data_path(None, "daily")
    swe_day = pd.DataFrame()
    if daily_path.exists():
        daily_src = pd.read_parquet(daily_path, columns=None)
        if "snow_mean" in daily_src.columns:
            npos = daily_src.loc[pd.to_numeric(daily_src["snow_mean"], errors="coerce") > 0.5, "station_name"].nunique()
            print(f"ΔSNOW candidates={npos} (HS > 0.5 cm at least once)", flush=True)
            swe_day = swe_from_daily_panel(daily_src)
            ok = swe_day["swe"].notna().sum()
            cover = int((swe_day["snow_density"].notna()).sum()) if not swe_day.empty else 0
            print(
                f"SWE daily finite={ok:,} pack-days={cover:,} "
                f"stations={swe_day['station_name'].nunique() if not swe_day.empty else 0}",
                flush=True,
            )
        else:
            print("daily parquet has no snow_mean — SWE skipped", flush=True)
    else:
        print(f"MISSING {daily_path} — SWE skipped", flush=True)

    for res in args.time_resolution:
        path = get_aggregated_data_path(None, res)
        print(f"\n{res}  {path}", flush=True)
        if not path.exists():
            print("  MISSING, skip", flush=True)
            continue
        base = pd.read_parquet(path)
        extra = pd.DataFrame()

        if res == "daily":
            extra = pd.DataFrame({"station_name": base["station_name"], "date": base["date"].astype(str)})
            extra = extra.drop_duplicates()
            if not slr_day.empty:
                extra = extra.merge(slr_day, on=["station_name", "date"], how="left")
            if not swe_day.empty:
                extra = extra.merge(swe_day, on=["station_name", "date"], how="left")
            out = merge_columns(base, extra, ["station_name", "date"])

        elif res == "weekly":
            extra = pd.DataFrame({
                "station_name": base["station_name"],
                "year_week": base["year_week"].astype(str),
            }).drop_duplicates()
            if not swe_day.empty:
                w = swe_day.copy()
                w["year_week"] = week_key(w["date"])
                w = w.groupby(["station_name", "year_week"], as_index=False)[["swe", "snow_density"]].mean()
                extra = extra.merge(w, on=["station_name", "year_week"], how="left")
            extra["slr"] = np.nan
            out = merge_columns(base, extra, ["station_name", "year_week"])

        elif res == "half_hourly":
            tcol = "timestamp" if "timestamp" in base.columns else "time"
            extra = base[["station_name", tcol]].drop_duplicates()
            extra[tcol] = match_timestamp(extra[tcol], base[tcol])
            if not slr_hh.empty:
                sl = slr_hh.rename(columns={"timestamp": tcol})
                sl = sl.dropna(subset=["slr"])
                sl[tcol] = match_timestamp(sl[tcol], base[tcol])
                extra = extra.merge(sl, on=["station_name", tcol], how="left")
            if not swe_day.empty:
                extra["_date"] = pd.to_datetime(extra[tcol], utc=True).dt.strftime("%Y-%m-%d")
                extra = extra.merge(
                    swe_day.rename(columns={"date": "_date"}),
                    on=["station_name", "_date"],
                    how="left",
                )
                extra = extra.drop(columns=["_date"])
            extra[tcol] = match_timestamp(extra[tcol], base[tcol])
            out = merge_columns(base, extra, ["station_name", tcol])
        else:
            print("  not in lock, skip", flush=True)
            continue

        for c in DERIVED_COLS:
            nfin = int(out[c].notna().sum()) if c in out.columns else 0
            print(f"  {c} finite={nfin:,}", flush=True)
        write_parquet(path, out, args.dry_run)


if __name__ == "__main__":
    main()
