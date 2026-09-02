#!/usr/bin/env python3
"""Shared panel loading, CLC grouping, metrics, regime labels for Kriging."""

from __future__ import annotations

from pathlib import Path
import importlib.util as _ilu
import sys

import numpy as np
import pandas as pd
import yaml

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

VAR_TO_REGIME = {
    "temp_mean": "temperature",
    "temp_min": "temperature",
    "temp_max": "temperature",
    "precip_sum": "precipitation",
    "wind_mean": "wind_speed",
    "rh_mean": "relative_humidity",
    "snow_mean": "snow_height",
}

DEFAULT_TRACE = {
    "half_hourly": 0.05,
    "daily": 0.1,
    "weekly": 0.7,
    "monthly": 1.0,
    "seasonal": 3.0,
}

# snow height units follow the aggregated column (cm in the station pipeline)
DEFAULT_TRACE_SNOW = {
    "half_hourly": 0.5,
    "daily": 0.5,
    "weekly": 1.0,
    "monthly": 1.0,
    "seasonal": 2.0,
}


def _label_times(times):
    _splits_path = CODE_DIR / "shared" / "splits" / "splits.py"
    _spec = _ilu.spec_from_file_location("thesis_time_splits", _splits_path)
    _splits_mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_splits_mod)
    return _splits_mod.label_times(times)


def load_config(path: Path | None = None) -> dict:
    path = path or Path(__file__).resolve().parent / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def time_col_name(cfg, time_res):
    try:
        return cfg["aggregation"][time_res]["time_col"]
    except Exception:
        return {
            "weekly": "year_week",
            "daily": "date",
            "monthly": "year_month",
            "half_hourly": "timestamp",
        }.get(time_res, "time")


def as_naive_utc(values) -> np.ndarray:
    idx = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    return idx.tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[ns]")


def data_sources(cfg) -> tuple[str, str, str]:
    p = cfg.get("paths", {})
    return (
        p.get("aggregated_from", "RFSI"),
        p.get("stations_from", "RFSI"),
        p.get("grids_from", "RFSI"),
    )


def clc_group(codes) -> np.ndarray:
    """CLC2018 → level-1 (1 urban … 5 water). 0 = missing/other."""
    c = np.asarray(codes, dtype=np.int32)
    out = np.zeros(c.shape, dtype=np.int32)
    high = c >= 100
    out[high] = c[high] // 100
    mid = (c >= 1) & (c <= 5)
    out[mid] = c[mid]
    out[(out < 0) | (out > 5)] = 0
    return out


def merge_rare_clc(groups: np.ndarray, min_count: int) -> tuple[np.ndarray, dict[int, int]]:
    """Map groups with fewer than min_count rows to 0. Returns remapped array + map."""
    mapping = {0: 0}
    for g in sorted(set(groups.tolist())):
        if g == 0:
            continue
        mapping[int(g)] = int(g) if int((groups == g).sum()) >= min_count else 0
    remapped = np.array([mapping.get(int(g), 0) for g in groups], dtype=np.int32)
    return remapped, mapping


def apply_clc_map(groups: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    return np.array([mapping.get(int(g), 0) for g in groups], dtype=np.int32)


def uses_two_step(var: str) -> bool:
    """Zero-inflated fields: precip and snow height (present / amount)."""
    v = var.lower()
    return v.startswith("precip") or v.startswith("snow")


def is_precip(var: str) -> bool:
    return uses_two_step(var)


def precip_trace(cfg, time_res: str, var: str = "precip_sum") -> float:
    kcfg = cfg.get("kriging", {})
    if str(var).lower().startswith("snow"):
        table = dict(DEFAULT_TRACE_SNOW)
        table.update(kcfg.get("trace_snow", {}))
        return float(table.get(time_res, 1.0))
    table = dict(DEFAULT_TRACE)
    table.update(kcfg.get("trace_mm", {}))
    return float(table.get(time_res, 0.1))


def load_panel(cfg, var: str) -> pd.DataFrame:
    time_res = cfg["time_resolution"]
    start = pd.Timestamp(cfg["start_date"]).tz_localize(None)
    end = pd.Timestamp(cfg["end_date"]).tz_localize(None)
    from paths import get_aggregated_data_path, get_domain_stations_path

    agg_method, sta_method, _ = data_sources(cfg)
    df = pd.read_parquet(get_aggregated_data_path(agg_method, time_res))
    col = time_col_name(cfg, time_res)
    if time_res == "weekly":
        raw = pd.to_datetime(df[col] + "-1", format="%Y-W%W-%w", utc=True)
    elif time_res == "monthly":
        raw = pd.to_datetime(df[col].astype(str) + "-01", utc=True)
    else:
        raw = pd.to_datetime(df[col], utc=True)
    df["time"] = as_naive_utc(raw)
    df = df[(df["time"] >= start) & (df["time"] <= end)]

    stations = pd.read_parquet(get_domain_stations_path(sta_method, "full"))
    if "elev" not in stations.columns:
        if "elev_dem" in stations.columns:
            stations = stations.rename(columns={"elev_dem": "elev"})
        elif "hoehe" in stations.columns:
            stations = stations.rename(columns={"hoehe": "elev"})
    keep = ["station_name", "x", "y"]
    for extra in ("elev", "clc_code"):
        if extra in stations.columns:
            keep.append(extra)
    stations = stations[keep].drop_duplicates("station_name")

    valid = df[["station_name", "time", var]].dropna()
    valid = valid.merge(stations, on="station_name", how="inner")
    valid = valid.dropna(subset=["x", "y", var])
    if "elev" in valid.columns:
        valid = valid.dropna(subset=["elev"])
    else:
        valid["elev"] = 0.0
    if "clc_code" not in valid.columns:
        valid["clc_code"] = 0
    valid["clc_code"] = valid["clc_code"].fillna(0).astype(np.int32)
    valid["clc_group"] = clc_group(valid["clc_code"].to_numpy())
    valid["split"] = _label_times(valid["time"]).to_numpy()
    valid["cluster_id"] = attach_regimes(cfg, valid["time"].to_numpy(), var)
    return valid


def attach_regimes(cfg, times: np.ndarray, var: str) -> np.ndarray:
    """Join existing GMM assignments when present. Else cluster_id=0."""
    time_res = cfg["time_resolution"]
    fine = time_res in ("weekly", "daily", "half_hourly")
    if not fine:
        return np.zeros(len(times), dtype=np.int32)
    kcfg = cfg.get("kriging", {})
    method = kcfg.get("regimes_from", "RFSI")
    cluster_method = kcfg.get("cluster_method", "gmm")
    from paths import get_clusters_dir

    canon = VAR_TO_REGIME.get(var, var)
    path = get_clusters_dir(method) / time_res / cluster_method / f"{canon}_assignments.parquet"
    if not path.exists():
        return np.zeros(len(times), dtype=np.int32)
    asg = pd.read_parquet(path)
    tcol = "timestamp" if "timestamp" in asg.columns else "time"
    asg["time"] = as_naive_utc(asg[tcol])
    asg = asg.drop_duplicates("time")
    src = pd.DataFrame({"time": times})
    merged = src.merge(asg[["time", "cluster_id"]], on="time", how="left")
    return merged["cluster_id"].fillna(0).astype(np.int32).to_numpy()


def compute_metrics(obs, pred) -> dict:
    o = np.asarray(obs, dtype=float)
    p = np.asarray(pred, dtype=float)
    m = np.isfinite(o) & np.isfinite(p)
    o, p = o[m], p[m]
    n = len(o)
    if n < 2:
        return {k: np.nan for k in ("rmse", "mae", "nse", "kge", "ccc", "r2")}
    rmse = float(np.sqrt(np.mean((o - p) ** 2)))
    mae = float(np.mean(np.abs(o - p)))
    ss_tot = float(np.sum((o - o.mean()) ** 2))
    nse = float(1 - np.sum((o - p) ** 2) / ss_tot) if ss_tot > 0 else np.nan
    r = float(np.corrcoef(o, p)[0, 1])
    alpha = float(np.std(p) / np.std(o)) if np.std(o) > 0 else np.nan
    beta = float(np.mean(p) / np.mean(o)) if np.mean(o) != 0 else np.nan
    kge = float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
    mx, my = float(o.mean()), float(p.mean())
    sxx = float(np.mean((o - mx) ** 2))
    syy = float(np.mean((p - my) ** 2))
    sxy = float(np.mean((o - mx) * (p - my)))
    den = sxx + syy + (mx - my) ** 2
    ccc = float(2 * sxy / den) if den > 0 else np.nan
    return {"rmse": rmse, "mae": mae, "nse": nse, "kge": kge, "ccc": ccc, "r2": nse}


def print_split_metrics(pred_df: pd.DataFrame) -> None:
    for split_name, part in pred_df.groupby("split"):
        met = compute_metrics(part["observed"], part["predicted"])
        print(
            f"  {split_name:7} n={len(part):6d}  "
            f"RMSE={met['rmse']:.3f}  MAE={met['mae']:.3f}  "
            f"NSE={met['nse']:.3f}  KGE={met['kge']:.3f}  CCC={met['ccc']:.3f}"
        )
    met_all = compute_metrics(pred_df["observed"], pred_df["predicted"])
    print(
        f"  {'all':7} n={len(pred_df):6d}  "
        f"RMSE={met_all['rmse']:.3f}  MAE={met_all['mae']:.3f}  "
        f"NSE={met_all['nse']:.3f}  KGE={met_all['kge']:.3f}  CCC={met_all['ccc']:.3f}"
    )
