#!/usr/bin/env python3
"""Data adapter for SMACNP.

Loads station panels, builds DEM features (elev, dist_to_coast, signed_dist_to_ridge),
applies paper-style normalization, and prepares context/target tensors.
Held-out stations are fixed and never used as context during training.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util as _ilu
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))


def load_config(path: Path | None = None) -> dict:
    path = path or Path(__file__).resolve().parent / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def time_col_name(cfg: dict, time_res: str) -> str:
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


def data_sources(cfg: dict) -> Tuple[str, str, str]:
    p = cfg.get("paths", {})
    return (
        p.get("aggregated_from", "RFSI"),
        p.get("stations_from", "RFSI"),
        p.get("grids_from", "RFSI"),
    )


def load_panel(cfg: dict, var: str) -> pd.DataFrame:
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
    for extra in ("elev",):
        if extra in stations.columns:
            keep.append(extra)
    stations = stations[keep].drop_duplicates("station_name")
    df = df.merge(stations, on="station_name", how="left")
    return df


def compute_dem_features(stations: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Attach precomputed static terrain features from the shared cache.

    Loads Source/terrain/station_features.parquet and merges the columns
    requested by the config. No DEM sampling at runtime.
    """
    from paths import get_station_terrain_path

    cache = pd.read_parquet(get_station_terrain_path())
    feats = cfg.get("smacnp", {}).get("features", {})
    want = ["station_name"]
    if feats.get("elev", True):
        want.append("elev")
    if feats.get("smooth_elev", False):
        want.append("smooth_elev")
    if feats.get("tpi", False):
        want.append("tpi")
    if feats.get("dist_to_coast", True):
        want.append("dist_to_coast")
    if feats.get("signed_dist_to_ridge", True):
        want.append("signed_dist_to_ridge")
    # Optional extras that may be useful later
    for extra in ("slope", "aspect", "sinasp", "cosasp", "northness", "clc_level2"):
        if extra in cache.columns and extra not in want:
            want.append(extra)

    keep = [c for c in want if c in cache.columns]
    merged = stations.drop(columns=[c for c in keep if c != "station_name" and c in stations.columns], errors="ignore")
    return merged.merge(cache[keep], on="station_name", how="left")


def feature_columns(cfg: dict) -> List[str]:
    feats = cfg.get("smacnp", {}).get("features", {})
    cols = ["x", "y"]
    if feats.get("elev", True):
        cols.append("elev")
    if feats.get("smooth_elev", False):
        cols.append("smooth_elev")
    if feats.get("tpi", False):
        cols.append("tpi")
    if feats.get("dist_to_coast", True):
        cols.append("dist_to_coast")
    if feats.get("signed_dist_to_ridge", True):
        cols.append("signed_dist_to_ridge")
    if feats.get("clc", True):
        # Level-2 one-hot column names; filled by attach_clc_onehot
        cols.extend([f"clc_l2_{i}" for i in range(16)])
    return cols


def attach_clc_onehot(df: pd.DataFrame) -> pd.DataFrame:
    """Add CLC level-2 one-hot columns (clc_l2_0 .. clc_l2_15)."""
    from shared.clc import one_hot_level2
    src = None
    for c in ("clc_level2", "clc_code", "clc"):
        if c in df.columns:
            src = df[c].to_numpy()
            break
    if src is None:
        src = np.zeros(len(df), dtype=np.int32)
    oh = one_hot_level2(src)
    out = df.copy()
    for i in range(oh.shape[1]):
        out[f"clc_l2_{i}"] = oh[:, i]
    return out


class FeatureScaler:
    """Per-column standardization (mean/std) for static features.

    Fit on training-pool stations only. Saved in the checkpoint so
    evaluation and map production use the same transform.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.columns_ = None

    def fit(self, df: pd.DataFrame, columns: List[str]) -> "FeatureScaler":
        self.columns_ = list(columns)
        vals = df[columns].to_numpy(dtype=np.float64)
        self.mean_ = np.nanmean(vals, axis=0)
        self.std_ = np.nanstd(vals, axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        vals = out[self.columns_].to_numpy(dtype=np.float64)
        out[self.columns_] = (vals - self.mean_) / self.std_
        return out

    def state_dict(self) -> dict:
        return {"mean": self.mean_, "std": self.std_, "columns": self.columns_}

    def load_state_dict(self, state: dict) -> "FeatureScaler":
        self.mean_ = np.asarray(state["mean"], dtype=np.float64)
        self.std_ = np.asarray(state["std"], dtype=np.float64)
        self.columns_ = list(state["columns"])
        return self


class PaperScaler:
    """Paper normalization: min-max [0,1] for continuous; log1p + min-max for two-step amount."""

    def __init__(self, two_step: bool = False):
        self.two_step = two_step
        self.min_ = None
        self.max_ = None
        self.log_min_ = None
        self.log_max_ = None

    def fit(self, y: np.ndarray):
        y = np.asarray(y, dtype=np.float64)
        if self.two_step:
            # Fit on positive amounts only for the log transform
            pos = y[y > 0]
            if len(pos) == 0:
                pos = np.array([1.0])
            log_pos = np.log1p(pos)
            self.log_min_ = float(log_pos.min())
            self.log_max_ = float(log_pos.max())
            if self.log_max_ == self.log_min_:
                self.log_max_ = self.log_min_ + 1.0
        else:
            self.min_ = float(np.nanmin(y))
            self.max_ = float(np.nanmax(y))
            if self.max_ == self.min_:
                self.max_ = self.min_ + 1.0
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        if self.two_step:
            log_y = np.log1p(np.maximum(y, 0.0))
            return (log_y - self.log_min_) / (self.log_max_ - self.log_min_)
        else:
            return (y - self.min_) / (self.max_ - self.min_)

    def inverse(self, y_norm: np.ndarray) -> np.ndarray:
        y_norm = np.asarray(y_norm, dtype=np.float64)
        if self.two_step:
            log_y = y_norm * (self.log_max_ - self.log_min_) + self.log_min_
            return np.expm1(log_y)
        else:
            return y_norm * (self.max_ - self.min_) + self.min_


def split_held_out(station_names: np.ndarray, fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Random station split. Held-out stations are never used as context."""
    rng = np.random.default_rng(seed)
    n = len(station_names)
    n_hold = max(1, int(round(n * fraction)))
    idx = rng.choice(n, n_hold, replace=False)
    held = station_names[idx]
    train = np.setdiff1d(station_names, held)
    return train, held


def split_days_random(day_ids: np.ndarray, train_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Random day split (no temporal ordering). Returns (train_days, val_days)."""
    rng = np.random.default_rng(seed)
    n = len(day_ids)
    n_train = max(1, int(round(n * train_fraction)))
    idx = rng.choice(n, n_train, replace=False)
    train = day_ids[idx]
    val = np.setdiff1d(day_ids, train)
    return train, val


def prepare_day_tensors(
    df_day: pd.DataFrame,
    feature_cols: List[str],
    scaler: PaperScaler,
    train_stations: np.ndarray,
    on_grid: bool,
    context_fraction: float,
    rng: np.random.Generator,
) -> Optional[Dict[str, np.ndarray]]:
    """Build one day's context/target tensors. Held-out stations are never context."""
    df_day = df_day.dropna(subset=feature_cols + ["value"])
    if len(df_day) < 2:
        return None
    # Only stations that are in the training pool can be context
    mask_train = df_day["station_name"].isin(train_stations)
    df_ctx_pool = df_day[mask_train]
    if len(df_ctx_pool) < 1:
        return None
    n_ctx = max(1, int(len(df_ctx_pool) * context_fraction))
    ctx_idx = rng.choice(len(df_ctx_pool), n_ctx, replace=False)
    df_ctx = df_ctx_pool.iloc[ctx_idx]

    if on_grid:
        # Target = all stations present this day (including context)
        df_tgt = df_day
    else:
        # Target = stations not used as context
        ctx_names = set(df_ctx["station_name"])
        df_tgt = df_day[~df_day["station_name"].isin(ctx_names)]
        if len(df_tgt) == 0:
            return None

    x_ctx = df_ctx[feature_cols].to_numpy(dtype=np.float32)
    y_ctx = scaler.transform(df_ctx["value"].to_numpy(dtype=np.float64)).astype(np.float32)
    x_tgt = df_tgt[feature_cols].to_numpy(dtype=np.float32)
    y_tgt = scaler.transform(df_tgt["value"].to_numpy(dtype=np.float64)).astype(np.float32)
    return {
        "x_context": x_ctx,
        "y_context": y_ctx[:, None],
        "x_target": x_tgt,
        "y_target": y_tgt[:, None],
    }
