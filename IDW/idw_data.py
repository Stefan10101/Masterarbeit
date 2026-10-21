#!/usr/bin/env python3
"""
idw_data.py
Panel loading, station tables, and global parameter files for IDW.

No regimes. One (p, Fz, k) per variable, tuned on the designated subset
and stored under IDW/Output/global_params/{time_resolution}/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent

FALLBACK_PARAMS = {
    "p": 2.0, "Fz": 0.3, "k": 12,
    "rmse": float("nan"), "mae": float("nan"),
    "nse": float("nan"), "kge": float("nan"),
}


def load_config(path: Optional[Path] = None) -> dict:
    path = Path(path) if path else SCRIPT_DIR / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _time_col(df: pd.DataFrame, time_res: str) -> str:
    candidates = {
        "monthly": ["year_month", "time", "date"],
        "weekly": ["year_week", "time", "date"],
        "daily": ["date", "time"],
        "half_hourly": ["timestamp", "time"],
    }.get(time_res, ["time", "date", "timestamp"])
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(f"No time column in aggregated file. Columns: {list(df.columns)}")


def _parse_time(series: pd.Series, time_res: str) -> pd.Series:
    if time_res == "weekly":
        return pd.to_datetime(series.astype(str) + "-1", format="%Y-W%W-%w", utc=True)
    if time_res == "monthly":
        s = series.astype(str)
        # accept YYYY-MM or YYYY-MM-DD
        s = s.where(s.str.len() > 7, s + "-01")
        return pd.to_datetime(s, utc=True)
    return pd.to_datetime(series, utc=True)


def load_stations(domain: str = "full") -> pd.DataFrame:
    from paths import get_domain_stations_path
    path = get_domain_stations_path("IDW", domain)
    if not path.exists():
        raise FileNotFoundError(f"Stations file not found: {path}")
    stations = pd.read_parquet(path)
    if "elev" not in stations.columns:
        if "elev_dem" in stations.columns:
            stations = stations.rename(columns={"elev_dem": "elev"})
        elif "hoehe" in stations.columns:
            stations = stations.rename(columns={"hoehe": "elev"})
    need = {"station_name", "x", "y", "elev"}
    missing = need - set(stations.columns)
    if missing:
        raise RuntimeError(f"Stations file missing columns: {missing}")
    return stations[["station_name", "x", "y", "elev"]].drop_duplicates("station_name")


def load_panel(cfg: dict, var: str) -> pd.DataFrame:
    """Station panel for one variable, restricted to config start/end.

    Columns: station_name, time, value, x, y, elev.
    """
    from paths import get_aggregated_data_path
    from shared.time_res import apply_hour_cut

    time_res = cfg["time_resolution"]
    path = get_aggregated_data_path("IDW", time_res)
    if not path.exists():
        raise FileNotFoundError(f"Aggregated file not found: {path}")
    df = pd.read_parquet(path)
    if var not in df.columns:
        raise KeyError(f"column {var} not in {path.name}")

    tcol = _time_col(df, time_res)
    df["time"] = _parse_time(df[tcol], time_res)
    start = pd.Timestamp(cfg["start_date"])
    end = pd.Timestamp(cfg["end_date"])
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    df = df[(df["time"] >= start) & (df["time"] <= end)]

    stations = load_stations(cfg.get("domain", {}).get("preset", "full"))
    panel = (
        df[["station_name", "time", var]]
        .dropna(subset=[var])
        .merge(stations, on="station_name", how="inner")
        .dropna(subset=["x", "y", "elev", var])
        .rename(columns={var: "value"})
    )
    panel, _ = apply_hour_cut(panel, time_res)
    return panel.reset_index(drop=True)


def build_field(
    panel: pd.DataFrame,
    timestamp,
    min_stations: int = 10,
    exclude: Optional[set] = None,
) -> Optional[pd.DataFrame]:
    """One timestamp as a station table, or None if too few stations."""
    t = pd.Timestamp(timestamp)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    mask = panel["time"] == t
    if not mask.any():
        mask = (panel["time"] - t).abs() <= pd.Timedelta("1min")
    if not mask.any():
        return None
    df = panel.loc[mask, ["station_name", "value", "x", "y", "elev"]].copy()
    if exclude:
        df = df[~df["station_name"].isin(exclude)]
    df = df.dropna().drop_duplicates("station_name")
    if len(df) < min_stations:
        return None
    return df


def global_params_path(time_res: str, var: str) -> Path:
    from paths import get_idw_tuned_params_path
    return get_idw_tuned_params_path(var, time_res)


def load_global_params(time_res: str, var: str) -> dict:
    path = global_params_path(time_res, var)
    if not path.exists():
        print(f"  [WARN] no global params for {var} — using fallback p=2 Fz=0.3 k=12")
        return dict(FALLBACK_PARAMS)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    out = dict(FALLBACK_PARAMS)
    for k in ("p", "Fz", "k", "rmse", "mae", "nse", "kge"):
        if k in raw:
            out[k] = raw[k]
    out["k"] = int(out["k"])
    print(f"  global params {var}: p={out['p']} Fz={out['Fz']} k={out['k']}")
    return out


def save_global_params(time_res: str, var: str, params: dict) -> Path:
    path = global_params_path(time_res, var)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(params, f, sort_keys=False)
    return path


def compute_metrics(obs, pred) -> dict:
    o = np.asarray(obs, dtype=float)
    p = np.asarray(pred, dtype=float)
    m = np.isfinite(o) & np.isfinite(p)
    o, p = o[m], p[m]
    n = len(o)
    if n < 2:
        return {"rmse": np.nan, "mae": np.nan, "nse": np.nan, "kge": np.nan, "n": n}
    err = p - o
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    var_o = float(np.var(o))
    nse = float(1.0 - np.mean(err ** 2) / var_o) if var_o > 0 else np.nan
    r = float(np.corrcoef(o, p)[0, 1]) if n > 2 else np.nan
    alpha = float(np.std(p) / np.std(o)) if np.std(o) > 0 else np.nan
    beta = float(np.mean(p) / np.mean(o)) if np.mean(o) != 0 else np.nan
    kge = float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)) \
        if np.isfinite(r) and np.isfinite(alpha) and np.isfinite(beta) else np.nan
    return {"rmse": rmse, "mae": mae, "nse": nse, "kge": kge, "n": n}


def print_metrics(pred_df: pd.DataFrame, by: str = "quarter") -> None:
    met = compute_metrics(pred_df["observed"], pred_df["predicted"])
    print(
        f"  all     n={met['n']:6d}  RMSE={met['rmse']:.3f}  "
        f"MAE={met['mae']:.3f}  NSE={met['nse']:.3f}  KGE={met['kge']:.3f}"
    )
    if by in pred_df.columns:
        for key, part in pred_df.groupby(by):
            m = compute_metrics(part["observed"], part["predicted"])
            print(
                f"  {str(key):7} n={m['n']:6d}  RMSE={m['rmse']:.3f}  "
                f"MAE={m['mae']:.3f}  NSE={m['nse']:.3f}  KGE={m['kge']:.3f}"
            )


SEASONAL4_MONTHS = (2, 5, 8, 11)


def subset_times(panel: pd.DataFrame, spec) -> pd.DataFrame:
    """all | seasonal4 | YYYY-MM,YYYY-MM | list of those."""
    if spec is None or spec == "" or spec == "all":
        return panel
    if isinstance(spec, (list, tuple)):
        keys = {str(s).strip() for s in spec}
        if keys == {"seasonal4"}:
            spec = "seasonal4"
        else:
            stamp = pd.to_datetime(panel["time"]).dt.strftime("%Y-%m")
            return panel.loc[stamp.isin(keys)].copy()
    spec = str(spec).strip().lower()
    if spec in ("all", "none"):
        return panel
    times = pd.to_datetime(panel["time"])
    if spec == "seasonal4":
        return panel.loc[times.dt.month.isin(SEASONAL4_MONTHS)].copy()
    keys = {s.strip() for s in spec.split(",") if s.strip()}
    stamp = times.dt.strftime("%Y-%m")
    return panel.loc[stamp.isin(keys)].copy()
