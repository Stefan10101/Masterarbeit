#!/usr/bin/env python3
"""Panel, metrics and time cuts for GAM. Self-contained; no other method imports."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

SEASONAL4_MONTHS = (2, 5, 8, 11)
DEFAULT_TRACE = {"half_hourly": 0.05, "daily": 0.1, "weekly": 0.7, "monthly": 1.0}
DEFAULT_TRACE_SNOW = {"half_hourly": 0.5, "daily": 0.5, "weekly": 1.0, "monthly": 1.0}


def load_gam_config(path: Path | None = None) -> dict:
    path = path or Path(__file__).resolve().parent / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_sources(cfg) -> tuple[str, str, str]:
    return ("Source", "Source", "Source")


def precip_trace(cfg, time_res: str, var: str = "precip_sum") -> float:
    block = cfg.get("gam", {})
    if str(var).lower().startswith("snow"):
        table = dict(DEFAULT_TRACE_SNOW)
        table.update(block.get("trace_snow", {}))
        return float(table.get(time_res, 1.0))
    table = dict(DEFAULT_TRACE)
    table.update(block.get("trace_mm", {}))
    return float(table.get(time_res, 0.1))


def clc_group(codes) -> np.ndarray:
    from shared.clc import to_level2
    return to_level2(codes)


def _time_col(cfg, time_res: str) -> str:
    try:
        return cfg["aggregation"][time_res]["time_col"]
    except Exception:
        return {"weekly": "year_week", "daily": "date",
                "monthly": "year_month", "half_hourly": "timestamp"}.get(time_res, "time")


def _parse_time(series: pd.Series, time_res: str) -> pd.Series:
    if time_res == "weekly":
        return pd.to_datetime(series.astype(str) + "-1", format="%Y-W%W-%w", utc=True)
    if time_res == "monthly":
        s = series.astype(str)
        s = s.where(s.str.len() > 7, s + "-01")
        return pd.to_datetime(s, utc=True)
    return pd.to_datetime(series, utc=True)


def _station_terrain() -> pd.DataFrame:
    from paths import get_station_terrain_path
    path = get_station_terrain_path()
    if not path.exists():
        return pd.DataFrame(columns=["station_name", "slope", "sinasp", "cosasp", "clc_group"])
    t = pd.read_parquet(path)
    keep = ["station_name"]
    for c in ("slope", "sinasp", "cosasp", "clc_level2"):
        if c in t.columns:
            keep.append(c)
    t = t[keep].drop_duplicates("station_name")
    if "clc_level2" in t.columns:
        t = t.rename(columns={"clc_level2": "clc_group"})
    return t


def load_panel(cfg: dict, var: str) -> pd.DataFrame:
    from paths import get_aggregated_data_path, get_domain_stations_path
    from shared.time_res import apply_hour_cut

    time_res = cfg["time_resolution"]
    path = get_aggregated_data_path("GAM", time_res)
    if not path.exists():
        raise FileNotFoundError(f"Aggregated file not found: {path}")
    df = pd.read_parquet(path)
    if var not in df.columns:
        raise KeyError(f"column {var} not in {path.name}")
    tcol = _time_col(cfg, time_res)
    if tcol not in df.columns:
        tcol = next(c for c in ("time", "date", "timestamp", "year_month", "year_week") if c in df.columns)
    df["time"] = _parse_time(df[tcol], time_res)
    start = pd.Timestamp(cfg["start_date"])
    end = pd.Timestamp(cfg["end_date"])
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    df = df[(df["time"] >= start) & (df["time"] <= end)]

    stations = pd.read_parquet(get_domain_stations_path("GAM", cfg.get("domain", {}).get("preset", "full")))
    if "elev" not in stations.columns:
        if "elev_dem" in stations.columns:
            stations = stations.rename(columns={"elev_dem": "elev"})
        elif "hoehe" in stations.columns:
            stations = stations.rename(columns={"hoehe": "elev"})
    stations = stations[["station_name", "x", "y", "elev"]].drop_duplicates("station_name")
    panel = (
        df[["station_name", "time", var]].dropna(subset=[var])
        .merge(stations, on="station_name", how="inner")
        .dropna(subset=["x", "y", "elev", var])
    )
    panel = panel.merge(_station_terrain(), on="station_name", how="left")
    for c, fill in (("slope", 0.0), ("sinasp", 0.0), ("cosasp", 1.0), ("clc_group", 0)):
        if c not in panel.columns:
            panel[c] = fill
        else:
            panel[c] = panel[c].fillna(fill)
    panel, _ = apply_hour_cut(panel, time_res)
    return panel.reset_index(drop=True)


def attach_temperature(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if "temp_mean" in panel.columns:
        return panel
    tpanel = load_panel(cfg, "temp_mean")[["station_name", "time", "temp_mean"]]
    return panel.merge(tpanel, on=["station_name", "time"], how="left")


def attach_pack_terrain(panel: pd.DataFrame, pack=None) -> pd.DataFrame:
    """Terrain comes from the shared station cache in load_panel. Pack is unused."""
    return panel


def pack_from_master(cfg, n_regions: int = 6):
    """GAM does not use watershed regions for temp/precip. Kept so callers compile."""
    return None


def subset_times(panel: pd.DataFrame, spec) -> pd.DataFrame:
    if spec is None or spec == "" or spec == "all":
        return panel
    times = pd.to_datetime(panel["time"])
    spec = str(spec).strip().lower()
    if spec == "seasonal4":
        return panel.loc[times.dt.month.isin(SEASONAL4_MONTHS)].copy()
    keys = {s.strip() for s in spec.split(",") if s.strip()}
    stamp = times.dt.strftime("%Y-%m")
    return panel.loc[stamp.isin(keys)].copy()


def subset_years(panel: pd.DataFrame, years) -> pd.DataFrame:
    if not years:
        return panel
    if isinstance(years, str):
        years = [y.strip() for y in years.split(",") if y.strip()]
    want = {int(y) for y in years}
    return panel.loc[pd.to_datetime(panel["time"]).dt.year.isin(want)].copy()


def subset_splits(panel: pd.DataFrame, splits) -> pd.DataFrame:
    if splits is None or splits == "all" or splits == {"all"}:
        return panel
    if isinstance(splits, str):
        splits = {s.strip() for s in splits.split(",") if s.strip()}
    if not splits or splits == {"all"}:
        return panel
    return panel.loc[panel["split"].isin(splits)].copy()


def compute_block(cfg: dict) -> dict:
    block = dict(cfg.get("gam", {}).get("compute", {}) or {})
    block.setdefault("tune_months", "seasonal4")
    block.setdefault("tune_folds", 5)
    block.setdefault("produce_splits", "all")
    return block


def compute_metrics(obs, pred) -> dict:
    o = np.asarray(obs, dtype=float)
    p = np.asarray(pred, dtype=float)
    m = np.isfinite(o) & np.isfinite(p)
    o, p = o[m], p[m]
    if len(o) < 2:
        return {k: np.nan for k in ("rmse", "mae", "nse", "kge", "ccc", "r2")}
    rmse = float(np.sqrt(np.mean((o - p) ** 2)))
    mae = float(np.mean(np.abs(o - p)))
    ss_tot = float(np.sum((o - o.mean()) ** 2))
    nse = float(1 - np.sum((o - p) ** 2) / ss_tot) if ss_tot > 0 else np.nan
    r = float(np.corrcoef(o, p)[0, 1]) if len(o) > 2 else np.nan
    alpha = float(np.std(p) / np.std(o)) if np.std(o) > 0 else np.nan
    beta = float(np.mean(p) / np.mean(o)) if np.mean(o) != 0 else np.nan
    kge = float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)) \
        if np.isfinite(r) and np.isfinite(alpha) and np.isfinite(beta) else np.nan
    return {"rmse": rmse, "mae": mae, "nse": nse, "kge": kge, "ccc": np.nan, "r2": nse}


def print_split_metrics(pred_df: pd.DataFrame) -> None:
    for split_name, part in pred_df.groupby("split"):
        met = compute_metrics(part["observed"], part["predicted"])
        print(f"  {split_name:7} n={len(part):6d}  RMSE={met['rmse']:.3f}  "
              f"MAE={met['mae']:.3f}  NSE={met['nse']:.3f}  KGE={met['kge']:.3f}")
    met = compute_metrics(pred_df["observed"], pred_df["predicted"])
    print(f"  {'all':7} n={len(pred_df):6d}  RMSE={met['rmse']:.3f}  "
          f"MAE={met['mae']:.3f}  NSE={met['nse']:.3f}  KGE={met['kge']:.3f}")
