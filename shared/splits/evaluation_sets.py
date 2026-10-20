#!/usr/bin/env python3
"""
Shared evaluation sets for the thesis method comparison.

Spatial hold-out
----------------
Stratified selection of stations inside the Alpine Convention perimeter.
Stations are spread across the domain (coarse lon/lat grid) and across
elevation bands. Controlled by a seed so the set is reproducible.
Written once; every method loads the same list.

Temporal splits
---------------
Whole year-quarters assigned to train / tune / test.
Default ratio 70 / 15 / 15. Quarters are distributed so every season
appears in each split (as far as the available quarters allow).
Ratio and seed are configurable.

Primary scores: held-out stations × test quarters.
Secondary scores: pool stations (train + tune) on the same times.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import urllib.request
import tempfile

import numpy as np
import pandas as pd
import yaml

try:
    import geopandas as gpd
    from shapely.geometry import Point
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False


# ---------------------------------------------------------------------------
# Paths (relative to this file so the module works without the full paths.py)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
HOLDOUT_PATH = _HERE / "holdout_stations.yaml"
QUARTER_SPLITS_PATH = _HERE / "quarter_splits.yaml"

PERIMETER_URL = (
    "https://services5.arcgis.com/qjjtr5pMpt3PDiiR/ArcGIS/rest/services/"
    "Alpine_Convention_Perimeter/FeatureServer/1/query?"
    "where=1%3D1&outFields=*&f=geojson"
)
PERIMETER_CACHE = _HERE / "geometry_cache" / "alpine_convention_perimeter.geojson"


# ---------------------------------------------------------------------------
# Alpine Convention perimeter
# ---------------------------------------------------------------------------
def load_alpine_perimeter(cache: bool = True) -> "gpd.GeoDataFrame":
    """Load Alpine Convention perimeter (EPSG:4326)."""
    if not HAS_GEOPANDAS:
        raise ImportError("geopandas is required for the Alpine perimeter filter")
    PERIMETER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if cache and PERIMETER_CACHE.exists():
        return gpd.read_file(PERIMETER_CACHE)
    with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as tmp:
        urllib.request.urlretrieve(PERIMETER_URL, tmp.name)
        gdf = gpd.read_file(tmp.name)
    gdf = gdf.to_crs("EPSG:4326")
    if cache:
        gdf.to_file(PERIMETER_CACHE, driver="GeoJSON")
    return gdf


def stations_inside_perimeter(
    stations: pd.DataFrame,
    lon_col: str = "lon",
    lat_col: str = "lat",
) -> pd.DataFrame:
    """Return the subset of stations that lie inside the Alpine Convention perimeter."""
    peri = load_alpine_perimeter()
    pts = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations[lon_col], stations[lat_col]),
        crs="EPSG:4326",
    )
    inside = pts.within(peri.union_all())
    return stations.loc[inside.to_numpy()].copy()


# ---------------------------------------------------------------------------
# Stratified spatial hold-out
# ---------------------------------------------------------------------------
def _elev_bins(elev: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile bins; falls back to equal-width if too few unique values."""
    elev = np.asarray(elev, dtype=float)
    finite = elev[np.isfinite(elev)]
    if len(finite) == 0 or n_bins < 2:
        return np.zeros(len(elev), dtype=int)
    try:
        _, edges = pd.qcut(finite, q=n_bins, retbins=True, duplicates="drop")
    except ValueError:
        edges = np.linspace(finite.min(), finite.max() + 1e-6, n_bins + 1)
    # Assign every station (including non-finite) to a bin
    bins = np.digitize(elev, edges[1:-1], right=False)
    bins[~np.isfinite(elev)] = 0
    return bins.astype(int)


def select_stratified_holdout(
    stations: pd.DataFrame,
    fraction: float = 0.15,
    seed: int = 42,
    n_elev_bins: int = 4,
    spatial_step_deg: float = 0.75,
    lon_col: str = "lon",
    lat_col: str = "lat",
    elev_col: str = "elev",
    id_col: str = "station_name",
    alpine_only: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    Stratified hold-out.

    1. Optionally restrict to Alpine Convention perimeter.
    2. Bin stations by a coarse lon/lat grid and by elevation quantiles.
    3. Sample proportionally inside each non-empty (grid-cell × elev-bin)
       so the held-out set is spread in space and height.
    4. Deterministic given the seed.

    Returns (held_out_ids, pool_ids).
    """
    df = stations.dropna(subset=[lon_col, lat_col, id_col]).copy()
    if elev_col not in df.columns:
        # try common aliases
        for alias in ("elev_dem", "hoehe", "elevation", "height"):
            if alias in df.columns:
                df = df.rename(columns={alias: elev_col})
                break
        else:
            df[elev_col] = 0.0  # no elevation info → spatial only

    if alpine_only:
        df = stations_inside_perimeter(df, lon_col, lat_col)
        if df.empty:
            raise RuntimeError("No stations inside the Alpine Convention perimeter")

    df = df.reset_index(drop=True)
    n = len(df)
    n_hold = max(1, int(round(n * fraction)))

    # Spatial grid
    lon0 = np.floor(df[lon_col].min() / spatial_step_deg) * spatial_step_deg
    lat0 = np.floor(df[lat_col].min() / spatial_step_deg) * spatial_step_deg
    i_lon = ((df[lon_col] - lon0) / spatial_step_deg).astype(int)
    i_lat = ((df[lat_col] - lat0) / spatial_step_deg).astype(int)
    elev_bin = _elev_bins(df[elev_col].to_numpy(), n_elev_bins)

    df["_cell"] = [
        (int(a), int(b), int(c))
        for a, b, c in zip(i_lon, i_lat, elev_bin)
    ]

    rng = np.random.default_rng(seed)
    held_idx: List[int] = []

    # Proportional allocation per cell, then fill remaining slots
    cells = df.groupby("_cell").indices
    cell_sizes = {c: len(ix) for c, ix in cells.items()}
    # First pass: at least one from large cells if budget allows, else proportional
    alloc = {}
    remaining = n_hold
    for cell, size in sorted(cell_sizes.items(), key=lambda kv: -kv[1]):
        share = max(0, int(round(size / n * n_hold)))
        share = min(share, size, remaining)
        alloc[cell] = share
        remaining -= share
    # Distribute any leftover randomly among cells that still have capacity
    if remaining > 0:
        candidates = [c for c, size in cell_sizes.items() if alloc.get(c, 0) < size]
        rng.shuffle(candidates)
        for c in candidates:
            if remaining <= 0:
                break
            alloc[c] = alloc.get(c, 0) + 1
            remaining -= 1

    for cell, ix in cells.items():
        k = alloc.get(cell, 0)
        if k <= 0:
            continue
        chosen = rng.choice(ix, size=min(k, len(ix)), replace=False)
        held_idx.extend(chosen.tolist())

    # If we are still short (rounding), fill from the largest remaining stations
    if len(held_idx) < n_hold:
        pool_left = np.setdiff1d(np.arange(n), held_idx)
        need = n_hold - len(held_idx)
        extra = rng.choice(pool_left, size=min(need, len(pool_left)), replace=False)
        held_idx.extend(extra.tolist())

    held_idx = sorted(set(held_idx))[:n_hold]
    held_ids = df.loc[held_idx, id_col].astype(str).tolist()
    pool_ids = df.loc[~df.index.isin(held_idx), id_col].astype(str).tolist()
    return held_ids, pool_ids


# ---------------------------------------------------------------------------
# Quarter-based temporal splits
# ---------------------------------------------------------------------------
def _all_quarters(start_year: int, end_year: int) -> List[str]:
    """Return ['2020-Q1', ..., 'end_year-Q4']."""
    out = []
    for y in range(start_year, end_year + 1):
        for q in (1, 2, 3, 4):
            out.append(f"{y}-Q{q}")
    return out


def assign_quarter_splits(
    start_year: int = 2020,
    end_year: int = 2025,
    train_frac: float = 0.70,
    tune_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Assign whole year-quarters to train / tune / test.

    Quarters are grouped by season (Q1–Q4) and allocated inside each
    season so that every split receives all seasons whenever possible.
    The overall fractions are matched as closely as the discrete quarters allow.
    """
    if abs(train_frac + tune_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("fractions must sum to 1")

    quarters = _all_quarters(start_year, end_year)
    by_season: Dict[int, List[str]] = {1: [], 2: [], 3: [], 4: []}
    for q in quarters:
        season = int(q[-1])
        by_season[season].append(q)

    rng = np.random.default_rng(seed)
    assignment = {"train": [], "tune": [], "test": []}

    for season, qlist in by_season.items():
        qlist = list(qlist)
        rng.shuffle(qlist)
        n = len(qlist)
        n_train = int(round(n * train_frac))
        n_tune = int(round(n * tune_frac))
        # remainder goes to test so the counts sum to n
        n_test = n - n_train - n_tune
        if n_test < 0:  # rare rounding overflow
            n_train += n_test
            n_test = 0
        assignment["train"].extend(qlist[:n_train])
        assignment["tune"].extend(qlist[n_train:n_train + n_tune])
        assignment["test"].extend(qlist[n_train + n_tune:])

    for k in assignment:
        assignment[k] = sorted(assignment[k])
    return assignment


def quarter_of(ts) -> str:
    """Map a timestamp to 'YYYY-Q#'."""
    t = pd.Timestamp(ts)
    q = (t.month - 1) // 3 + 1
    return f"{t.year}-Q{q}"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def write_holdout(held_out: List[str], pool: List[str], meta: dict, path: Path = HOLDOUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "held_out": sorted(held_out),
        "pool": sorted(pool),
        "n_held_out": len(held_out),
        "n_pool": len(pool),
        **meta,
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def write_quarter_splits(splits: Dict[str, List[str]], meta: dict, path: Path = QUARTER_SPLITS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**meta, "splits": splits}
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def load_holdout(path: Path = HOLDOUT_PATH) -> Tuple[List[str], List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return list(raw["held_out"]), list(raw["pool"])


def load_quarter_splits(path: Path = QUARTER_SPLITS_PATH) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw["splits"]


def times_in_split(times, split: str, splits: Optional[Dict[str, List[str]]] = None) -> np.ndarray:
    """Boolean mask: which timestamps belong to the named split."""
    if splits is None:
        splits = load_quarter_splits()
    allowed = set(splits[split])
    qs = [quarter_of(t) for t in pd.to_datetime(times)]
    return np.array([q in allowed for q in qs])


def prepare_for_tuning(df: pd.DataFrame, station_col: str = "station_name", time_col: str = "time") -> pd.DataFrame:
    """
    Panel ready for free-parameter tuning.

    - Drops held-out stations.
    - Remaps split so existing code that looks for 'dev' hits the tune quarters:
        train → train, tune → dev, test rows removed.
    """
    tagged, _, _, _ = apply_evaluation_protocol(df, station_col, time_col)
    out = tagged[~tagged["is_heldout"]].copy()
    out.loc[out["split"] == "tune", "split"] = "dev"
    out = out[out["split"].isin(["train", "dev"])]
    return out


def apply_evaluation_protocol(
    df: pd.DataFrame,
    station_col: str = "station_name",
    time_col: str = "time",
) -> Tuple[pd.DataFrame, List[str], List[str], Dict[str, List[str]]]:
    """
    Tag a panel with the shared evaluation protocol.

    Adds columns:
      is_heldout  – True for the fixed evaluation stations
      quarter     – 'YYYY-Q#'
      split       – 'train' | 'tune' | 'test' | 'outside'

    Returns (df, held_out_ids, pool_ids, quarter_splits).
    Methods should exclude is_heldout from all fitting and use
    split=='test' + is_heldout for the primary scores.
    """
    held, pool = load_holdout()
    splits = load_quarter_splits()
    out = df.copy()
    out["is_heldout"] = out[station_col].astype(str).isin(held)
    out["quarter"] = out[time_col].map(quarter_of)
    out["split"] = "outside"
    for name, qs in splits.items():
        out.loc[out["quarter"].isin(qs), "split"] = name
    return out, held, pool, splits
