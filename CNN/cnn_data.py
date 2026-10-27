#!/usr/bin/env python3
"""Panel, metrics and time cuts for CNN. Self-contained; no other method imports."""

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


def load_cnn_config(path: Path | None = None) -> dict:
    path = path or Path(__file__).resolve().parent / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: Path | None = None) -> dict:
    return load_cnn_config(path)


def data_sources(cfg) -> tuple[str, str, str]:
    return ("Source", "Source", "Source")


def precip_trace(cfg, time_res: str, var: str = "precip_sum") -> float:
    block = cfg.get("cnn", {})
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


def load_panel(cfg: dict, var: str) -> pd.DataFrame:
    from paths import get_aggregated_data_path, get_domain_stations_path
    from shared.time_res import apply_hour_cut

    time_res = cfg["time_resolution"]
    path = get_aggregated_data_path("CNN", time_res)
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

    stations = pd.read_parquet(get_domain_stations_path("CNN", cfg.get("domain", {}).get("preset", "full")))
    from shared.terrain import attach_enabled_terrain
    stations = attach_enabled_terrain(stations, cfg)
    if "elev" not in stations.columns:
        if "elev_dem" in stations.columns:
            stations = stations.rename(columns={"elev_dem": "elev"})
        elif "hoehe" in stations.columns:
            stations = stations.rename(columns={"hoehe": "elev"})
    stations = stations[["station_name", "x", "y"] + [c for c in stations.columns if c not in ("station_name", "x", "y")]].drop_duplicates("station_name")
    panel = (
        df[["station_name", "time", var]].dropna(subset=[var])
        .merge(stations, on="station_name", how="inner")
        .dropna(subset=["x", "y", "elev", var])
    )
    panel, _ = apply_hour_cut(panel, time_res)
    return panel.reset_index(drop=True)


def attach_temperature(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Merge observed temp_mean for RH covariate. Stations without T get the panel mean."""
    if "temp_mean" in panel.columns:
        return panel
    tpanel = load_panel(cfg, "temp_mean")[["station_name", "time", "temp_mean"]]
    out = panel.merge(tpanel, on=["station_name", "time"], how="left")
    fill = float(out["temp_mean"].mean()) if out["temp_mean"].notna().any() else 0.0
    out["temp_mean"] = out["temp_mean"].fillna(fill)
    return out


def attach_station_tpi(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach TPI from the shared station terrain cache (for wind)."""
    from paths import get_station_terrain_path
    path = get_station_terrain_path()
    if not path.exists() or panel.empty:
        panel = panel.copy()
        panel["tpi"] = 0.0
        return panel
    t = pd.read_parquet(path)
    if "tpi" not in t.columns:
        panel = panel.copy()
        panel["tpi"] = 0.0
        return panel
    keep = t[["station_name", "tpi"]].drop_duplicates("station_name")
    out = panel.merge(keep, on="station_name", how="left")
    out["tpi"] = out["tpi"].fillna(0.0)
    return out
    if spec is None or spec == "" or spec == "all":
        return panel
    times = pd.to_datetime(panel["time"])
    if str(spec).strip().lower() == "seasonal4":
        return panel.loc[times.dt.month.isin(SEASONAL4_MONTHS)].copy()
    keys = {s.strip() for s in str(spec).split(",") if s.strip()}
    return panel.loc[times.dt.strftime("%Y-%m").isin(keys)].copy()


def compute_metrics(obs, pred) -> dict:
    o = np.asarray(obs, dtype=float)
    p = np.asarray(pred, dtype=float)
    m = np.isfinite(o) & np.isfinite(p)
    o, p = o[m], p[m]
    if len(o) < 2:
        return {k: np.nan for k in ("rmse", "mae", "nse", "kge")}
    rmse = float(np.sqrt(np.mean((o - p) ** 2)))
    mae = float(np.mean(np.abs(o - p)))
    ss_tot = float(np.sum((o - o.mean()) ** 2))
    nse = float(1 - np.sum((o - p) ** 2) / ss_tot) if ss_tot > 0 else np.nan
    if len(o) > 2 and np.std(o) > 1e-8 and np.std(p) > 1e-8:
        r = float(np.corrcoef(o, p)[0, 1])
    else:
        r = np.nan
    alpha = float(np.std(p) / np.std(o)) if np.std(o) > 0 else np.nan
    beta = float(np.mean(p) / np.mean(o)) if np.mean(o) != 0 else np.nan
    kge = float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)) \
        if np.isfinite(r) and np.isfinite(alpha) and np.isfinite(beta) else np.nan
    return {"rmse": rmse, "mae": mae, "nse": nse, "kge": kge}


def print_split_metrics(pred_df: pd.DataFrame) -> None:
    for split_name, part in pred_df.groupby("split"):
        met = compute_metrics(part["observed"], part["predicted"])
        print(f"  {split_name:7} n={len(part):6d}  RMSE={met['rmse']:.3f}  "
              f"MAE={met['mae']:.3f}  NSE={met['nse']:.3f}  KGE={met['kge']:.3f}")
