#!/usr/bin/env python3
"""Spatial consistency QC: k-nearest neighbour median ± MAD check.

No lapse-rate correction. Neighbours are restricted by a hard elevation
difference cap. Applied after per-station QC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorised haversine distance in km."""
    r = 6371.0
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def build_neighbour_map(
    stations: list[dict],
    k: int = 10,
    elev_cap_m: float = 800.0,
) -> dict[str, list[str]]:
    """Return {station_id: [neighbour_id, ...]} limited to k nearest within elev_cap_m.

    Each station dict must contain: id, lat, lon, elev, path.
    """
    n = len(stations)
    if n == 0:
        return {}
    ids = [s["id"] for s in stations]
    lats = np.array([s["lat"] for s in stations], dtype=float)
    lons = np.array([s["lon"] for s in stations], dtype=float)
    elevs = np.array([s["elev"] if s["elev"] is not None else np.nan for s in stations], dtype=float)
    paths = {s["id"]: s["path"] for s in stations}

    neighbour_map: dict[str, list[str]] = {}
    for i, sid in enumerate(ids):
        if np.isnan(lats[i]) or np.isnan(lons[i]):
            neighbour_map[sid] = []
            continue
        dist = haversine_km(lats[i], lons[i], lats, lons)
        dist[i] = np.inf
        elev_ok = np.abs(elevs - elevs[i]) <= elev_cap_m
        elev_ok[i] = False
        # stations with missing elev are excluded from being neighbours
        elev_ok &= np.isfinite(elevs)
        candidates = np.where(elev_ok & np.isfinite(dist))[0]
        if len(candidates) == 0:
            neighbour_map[sid] = []
            continue
        order = candidates[np.argsort(dist[candidates])]
        chosen = order[:k]
        neighbour_map[sid] = [ids[j] for j in chosen]
    return neighbour_map, paths


def _mad(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def spatial_outlier_mask(
    own: pd.Series,
    neighbour_series: list[pd.Series],
    mad_multiplier: float = 4.0,
    min_valid: int = 7,
    precip_mode: bool = False,
    precip_wet_threshold: float = 1.0,
    precip_max_wet_neighbours: int = 3,
) -> pd.Series:
    """Return boolean Series (True = outlier) aligned to own.index. Vectorised."""
    if own.empty or not neighbour_series:
        return pd.Series(False, index=own.index)

    aligned = [s.reindex(own.index) for s in neighbour_series if s is not None and not s.empty]
    if not aligned:
        return pd.Series(False, index=own.index)

    panel = pd.concat(aligned, axis=1)
    vals = panel.to_numpy(dtype=float)
    own_vals = own.to_numpy(dtype=float)
    n_valid = np.sum(np.isfinite(vals), axis=1)
    enough = n_valid >= min_valid

    if precip_mode:
        wet = np.nansum(vals > precip_wet_threshold, axis=1)
        flag = enough & (own_vals > precip_wet_threshold) & (wet < precip_max_wet_neighbours)
        return pd.Series(flag, index=own.index)

    # median and MAD along neighbours
    with np.errstate(all="ignore"):
        med = np.nanmedian(vals, axis=1)
        mad = np.nanmedian(np.abs(vals - med[:, None]), axis=1)
    mad = np.where((~np.isfinite(mad)) | (mad == 0.0), 1e-6, mad)
    flag = enough & np.isfinite(own_vals) & (np.abs(own_vals - med) > mad_multiplier * mad)
    return pd.Series(flag, index=own.index)


def apply_spatial_qc_to_directory(
    qc_dir: Path,
    variable: str,
    k: int = 10,
    elev_cap_m: float = 800.0,
    min_valid: int = 7,
    mad_multiplier: float | None = None,
    precip_mode: bool = False,
    logger=None,
) -> int:
    """Post-pass: load all *_final_qc.parquet in qc_dir, apply neighbour check, overwrite.

    Returns number of values set to NaN.
    """
    from pathlib import Path as _P
    qc_dir = _P(qc_dir)
    files = sorted(qc_dir.glob("*_final_qc.parquet"))
    if not files:
        if logger:
            logger.warning(f"spatial QC: no files in {qc_dir}")
        return 0

    stations = []
    series_cache: dict[str, pd.Series] = {}
    for f in files:
        try:
            df = pd.read_parquet(f, columns=["timestamp", "value", "lat", "lon", "height", "station"])
        except Exception:
            continue
        if df.empty:
            continue
        sid = str(df["station"].iloc[0]) if "station" in df.columns else f.stem
        lat = float(df["lat"].iloc[0]) if pd.notna(df["lat"].iloc[0]) else np.nan
        lon = float(df["lon"].iloc[0]) if pd.notna(df["lon"].iloc[0]) else np.nan
        elev = float(df["height"].iloc[0]) if "height" in df.columns and pd.notna(df["height"].iloc[0]) else np.nan
        stations.append({"id": sid, "lat": lat, "lon": lon, "elev": elev, "path": f})
        s = df.set_index(pd.to_datetime(df["timestamp"], utc=True))["value"]
        series_cache[sid] = s

    neighbour_map, _ = build_neighbour_map(stations, k=k, elev_cap_m=elev_cap_m)

    if mad_multiplier is None:
        mad_multiplier = 5.0 if variable.lower() in ("wind", "wind_mean") else 4.0

    total_flagged = 0
    for st in stations:
        sid = st["id"]
        own = series_cache.get(sid)
        if own is None or own.empty:
            continue
        neigh_ids = neighbour_map.get(sid, [])
        neigh_series = [series_cache[n] for n in neigh_ids if n in series_cache]
        mask = spatial_outlier_mask(
            own,
            neigh_series,
            mad_multiplier=mad_multiplier,
            min_valid=min_valid,
            precip_mode=precip_mode,
        )
        n_flag = int(mask.sum())
        if n_flag == 0:
            continue
        total_flagged += n_flag
        # rewrite the parquet with flagged values set to NaN
        full = pd.read_parquet(st["path"])
        full["timestamp"] = pd.to_datetime(full["timestamp"], utc=True)
        full = full.set_index("timestamp")
        full.loc[mask[mask].index, "value"] = np.nan
        if "is_missing" in full.columns:
            full.loc[mask[mask].index, "is_missing"] = True
        full = full.reset_index()
        full.to_parquet(st["path"], compression="snappy", index=False)
        if logger:
            logger.info(f"  spatial QC | {sid}: {n_flag} values set to NaN")

    if logger:
        logger.info(f"spatial QC finished: {total_flagged} values flagged across {len(stations)} stations")
    return total_flagged


def apply_snow_flatline_neighbour_rule(
    qc_dir: Path,
    k: int = 5,
    elev_cap_m: float = 800.0,
    min_days: float = 3.0,
    min_value: float = 2.0,
    zero_tol: float = 0.5,
    logger=None,
) -> int:
    """Set stuck non-zero flatlines (>= min_days) to 0 when a majority of the
    k nearest neighbours (elev cap) are ~0 over the same window.

    Returns number of values set to 0.
    """
    from pathlib import Path as _P
    qc_dir = _P(qc_dir)
    files = sorted(qc_dir.glob("*_final_qc.parquet"))
    if not files:
        return 0

    stations = []
    series_cache: dict[str, pd.Series] = {}
    for f in files:
        try:
            df = pd.read_parquet(f, columns=["timestamp", "value", "lat", "lon", "height", "station"])
        except Exception:
            continue
        if df.empty:
            continue
        sid = str(df["station"].iloc[0]) if "station" in df.columns else f.stem
        lat = float(df["lat"].iloc[0]) if pd.notna(df["lat"].iloc[0]) else np.nan
        lon = float(df["lon"].iloc[0]) if pd.notna(df["lon"].iloc[0]) else np.nan
        elev = float(df["height"].iloc[0]) if "height" in df.columns and pd.notna(df["height"].iloc[0]) else np.nan
        stations.append({"id": sid, "lat": lat, "lon": lon, "elev": elev, "path": f})
        s = df.set_index(pd.to_datetime(df["timestamp"], utc=True))["value"].sort_index()
        series_cache[sid] = s

    neighbour_map, _ = build_neighbour_map(stations, k=k, elev_cap_m=elev_cap_m)
    min_points = int(min_days * 48)
    total_set = 0

    for st in stations:
        sid = st["id"]
        own = series_cache.get(sid)
        if own is None or own.empty:
            continue
        # identify flat non-zero runs
        s = own
        groups = (s != s.shift()).cumsum()
        group_sizes = s.groupby(groups).transform("size")
        group_values = s.groupby(groups).transform("first")
        is_flat = (group_sizes >= min_points) & (group_values > min_value) & s.notna()
        if not is_flat.any():
            continue

        neigh_ids = neighbour_map.get(sid, [])
        if not neigh_ids:
            continue
        neigh_series = [series_cache[n] for n in neigh_ids if n in series_cache]
        if not neigh_series:
            continue

        # for each flat run, check if majority of neighbours are ~0
        run_ids = groups[is_flat].unique()
        to_zero = []
        for rid in run_ids:
            idx = s.index[groups == rid]
            if len(idx) < min_points:
                continue
            n_zero = 0
            n_checked = 0
            for ns in neigh_series:
                aligned = ns.reindex(idx)
                valid = aligned.dropna()
                if len(valid) < max(10, len(idx) // 4):
                    continue
                n_checked += 1
                if (valid <= zero_tol).mean() >= 0.8:
                    n_zero += 1
            if n_checked >= 3 and n_zero > n_checked / 2:
                to_zero.extend(idx.tolist())

        if not to_zero:
            continue
        total_set += len(to_zero)
        full = pd.read_parquet(st["path"])
        full["timestamp"] = pd.to_datetime(full["timestamp"], utc=True)
        full = full.set_index("timestamp")
        full.loc[to_zero, "value"] = 0.0
        if "is_missing" in full.columns:
            full.loc[to_zero, "is_missing"] = False
        full = full.reset_index()
        full.to_parquet(st["path"], compression="snappy", index=False)
        if logger:
            logger.info(f"  flatline neighbour | {sid}: {len(to_zero)} points set to 0")

    if logger:
        logger.info(f"Snow flatline neighbour rule: {total_set} values set to 0")
    return total_set
