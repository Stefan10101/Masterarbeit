#!/usr/bin/env python3
"""Shared panel loading, metrics, and feature scaling for RGI."""

from __future__ import annotations

from pathlib import Path
import importlib.util as _ilu
import sys

import numpy as np
import pandas as pd
import yaml

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))


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


def as_naive_utc(values, time_res: str | None = None) -> np.ndarray:
    from shared.time_res import to_naive_utc
    return to_naive_utc(values, time_res)


def data_sources(cfg) -> tuple[str, str, str]:
    p = cfg.get("paths", {})
    return (
        p.get("aggregated_from", "RFSI"),
        p.get("stations_from", "RFSI"),
        p.get("grids_from", "RFSI"),
    )


def load_panel(cfg, var: str) -> pd.DataFrame:
    time_res = cfg["time_resolution"]
    start = pd.Timestamp(cfg["start_date"]).tz_localize(None)
    end = pd.Timestamp(cfg["end_date"]).tz_localize(None)
    from paths import get_aggregated_data_path, get_domain_stations_path

    agg_method, sta_method, _ = data_sources(cfg)
    df = pd.read_parquet(get_aggregated_data_path(agg_method, time_res))
    col = time_col_name(cfg, time_res)
    df["time"] = as_naive_utc(df[col], time_res)
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
    valid["split"] = _label_times(valid["time"]).to_numpy()
    return valid


def extra_cols_for_var(var: str, cfg: dict) -> tuple[str, ...]:
    r = cfg.get("rgi", {})
    v = str(var).lower()
    cols = []
    if v.startswith("snow") and r.get("snow_terrain", True):
        cols.extend(["slope", "northness"])
    if v.startswith("wind") and r.get("wind_tpi", True):
        cols.append("tpi")
    if v.startswith("rh") and str(r.get("rh_t_mode", "predicted")) in ("predicted", "observed"):
        cols.append("temp_mean")
    return tuple(cols)


def attach_extras(panel: pd.DataFrame, cfg: dict, var: str) -> pd.DataFrame:
    cols = extra_cols_for_var(var, cfg)
    if "temp_mean" in cols and "temp_mean" not in panel.columns:
        t = load_panel(cfg, "temp_mean")[["station_name", "time", "temp_mean"]]
        panel = panel.merge(t, on=["station_name", "time"], how="left")
    need_pack = any(c in cols for c in ("slope", "northness", "tpi"))
    if need_pack:
        try:
            from Kriging.kriging_data import attach_pack_terrain, pack_from_master
            pack = pack_from_master(cfg, int(cfg.get("rgi", {}).get("n_regions", 6)))
            panel = attach_pack_terrain(panel, pack)
        except Exception as exc:
            print(f"  RGI pack attach failed: {exc}")
    return panel


def compute_metrics(obs, pred) -> dict:
    o = np.asarray(obs, dtype=float)
    p = np.asarray(pred, dtype=float)
    m = ~np.isnan(o) & ~np.isnan(p)
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


class FeatureScaler:
    """z-score for value, x, y, elev. CLC kept as integer codes."""

    keys = ("value", "x", "y", "elev")
    extra_keys: tuple = ()

    def __init__(self):
        self.mean = {k: 0.0 for k in self.keys}
        self.std = {k: 1.0 for k in self.keys}
        self.clc_codes: list[int] = [0]

    def fit(self, df: pd.DataFrame, var: str) -> "FeatureScaler":
        for col, key in ((var, "value"), ("x", "x"), ("y", "y"), ("elev", "elev")):
            v = df[col].to_numpy(dtype=np.float64)
            self.mean[key] = float(np.nanmean(v))
            s = float(np.nanstd(v))
            self.std[key] = s if s > 1e-8 else 1.0
        codes = sorted(set(int(c) for c in df["clc_code"].to_numpy().tolist()) | {0})
        self.clc_codes = codes
        for key in self.extra_keys:
            if key not in df.columns:
                self.mean[key] = 0.0
                self.std[key] = 1.0
                continue
            v = df[key].to_numpy(dtype=np.float64)
            self.mean[key] = float(np.nanmean(v)) if np.isfinite(v).any() else 0.0
            s = float(np.nanstd(v)) if np.isfinite(v).any() else 1.0
            self.std[key] = s if s > 1e-8 else 1.0
        return self

    def transform_value(self, v) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32)
        return ((v - self.mean["value"]) / self.std["value"]).astype(np.float32)

    def inverse_value(self, v) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32)
        return (v * self.std["value"] + self.mean["value"]).astype(np.float32)

    def coord_block(self, x, y, elev) -> np.ndarray:
        x = (np.asarray(x, dtype=np.float32) - self.mean["x"]) / self.std["x"]
        y = (np.asarray(y, dtype=np.float32) - self.mean["y"]) / self.std["y"]
        e = (np.asarray(elev, dtype=np.float32) - self.mean["elev"]) / self.std["elev"]
        return np.column_stack([x, y, e]).astype(np.float32)

    def extra_block(self, df: pd.DataFrame) -> np.ndarray | None:
        if not self.extra_keys:
            return None
        cols = []
        n = len(df)
        for key in self.extra_keys:
            if key in df.columns:
                v = np.asarray(df[key], dtype=np.float32)
            else:
                v = np.zeros(n, dtype=np.float32)
            v = np.where(np.isfinite(v), v, self.mean.get(key, 0.0))
            cols.append((v - self.mean.get(key, 0.0)) / self.std.get(key, 1.0))
        return np.column_stack(cols).astype(np.float32)

    def clc_index(self, codes) -> np.ndarray:
        table = {c: i for i, c in enumerate(self.clc_codes)}
        out = np.array([table.get(int(c), 0) for c in np.asarray(codes).ravel()], dtype=np.int64)
        return out

    def state_dict(self) -> dict:
        return {
            "mean": self.mean,
            "std": self.std,
            "clc_codes": self.clc_codes,
            "extra_keys": list(self.extra_keys),
        }

    @classmethod
    def from_state(cls, state: dict) -> "FeatureScaler":
        obj = cls()
        obj.mean = dict(state["mean"])
        obj.std = dict(state["std"])
        obj.clc_codes = list(state["clc_codes"])
        obj.extra_keys = tuple(state.get("extra_keys") or ())
        return obj
