#!/usr/bin/env python3
"""
Shared evaluation sets for the thesis method comparison.

Spatial hold-out
----------------
Stratified selection of stations that lie strictly inside the Alpine
Convention perimeter (union of all polygons, including any exclave)
AND that measure the variable of interest.
Stations are spread across the perimeter (coarse lon/lat grid) and across
elevation bands. Controlled by a seed so the set is reproducible.
Written once per variable; every method loads the list for its variable.

Pool stations (donors) may come from the wider interpolation domain
(Alpine Convention polygon + 100 km, matching plot_maps.alpine_mask).

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

try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False


# ---------------------------------------------------------------------------
# Paths (relative to this file so the module works without the full paths.py)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
HOLDOUT_PATH = _HERE / "holdout_stations.yaml"
QUARTER_SPLITS_PATH = _HERE / "quarter_splits.yaml"
TUNE_TARGETS_PATH = _HERE / "tune_targets.yaml"

# Overview names (stations_overview.csv "variables" column)
OVERVIEW_VARS = (
    "temperature",
    "precipitation",
    "wind_speed",
    "relative_humidity",
    "snow_height",
)

# Internal code names used by the methods ↔ overview names
CODE_TO_OVERVIEW = {
    "temp_mean": "temperature",
    "temperature": "temperature",
    "precip_sum": "precipitation",
    "precipitation": "precipitation",
    "wind_mean": "wind_speed",
    "wind_speed": "wind_speed",
    "rh_mean": "relative_humidity",
    "relative_humidity": "relative_humidity",
    "snow_mean": "snow_height",
    "snow_height": "snow_height",
}

PERIMETER_URL = (
    "https://services5.arcgis.com/qjjtr5pMpt3PDiiR/ArcGIS/rest/services/"
    "Alpine_Convention_Perimeter/FeatureServer/1/query?"
    "where=1%3D1&outFields=*&f=geojson"
)
PERIMETER_CACHE = _HERE / "geometry_cache" / "alpine_convention_perimeter.geojson"

# Evaluation domain: Alpine Convention polygon + 100 km buffer.
# Grids/produce use the same 100 km buffer so the interpolation extent matches.
_GRIDS_CFG = Path(__file__).resolve().parents[1] / "grids" / "config.yaml"
DEFAULT_BBOX_EPSG31287 = (-76229, 123066, 637216, 501366)
DEFAULT_BUFFER_M = 100000.0
DEFAULT_CRS = "EPSG:31287"


def _load_domain_bbox() -> Tuple[Tuple[float, float, float, float], float, str]:
    """Return (xmin, ymin, xmax, ymax) of the Alpine bbox, buffer_m, crs.

    Buffer is fixed at 100 km for station selection (same as the grids).
    """
    bbox = DEFAULT_BBOX_EPSG31287
    buf = DEFAULT_BUFFER_M  # 100 km
    crs = DEFAULT_CRS
    if _GRIDS_CFG.exists():
        with open(_GRIDS_CFG, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        bbox = tuple(cfg.get("predefined_bboxes", {}).get("full", bbox))
        crs = cfg.get("projected_crs", crs)
    return bbox, buf, crs


def stations_inside_domain(
    stations: pd.DataFrame,
    lon_col: str = "lon",
    lat_col: str = "lat",
    buffer_km: float = 100.0,
) -> pd.DataFrame:
    """
    Keep stations inside the area that is actually interpolated.

    Matches plot_maps.alpine_mask: Alpine Convention perimeter (true polygon)
    buffered by 100 km in EPSG:31287. Stations outside this region are never
    used as donors or targets.
    """
    if not HAS_GEOPANDAS:
        raise ImportError("geopandas is required for the domain filter")
    peri = load_alpine_perimeter()
    if peri.crs is None:
        peri = peri.set_crs("EPSG:4326")
    peri = peri.to_crs("EPSG:31287")
    peri = peri.copy()
    peri["geometry"] = peri.geometry.buffer(0)  # repair invalid rings
    buffered = peri.buffer(buffer_km * 1000.0).union_all()

    df = stations.dropna(subset=[lon_col, lat_col]).copy()
    pts = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    ).to_crs("EPSG:31287")
    inside = pts.intersects(buffered)
    return df.loc[inside.to_numpy()].copy()


# ---------------------------------------------------------------------------
# Alpine Convention perimeter (kept for optional use)
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
    """Return the subset of stations that lie inside the Alpine Convention perimeter.

    Uses the union of all perimeter polygons so any exclave is included.
    Invalid rings are repaired with buffer(0).
    """
    if not HAS_GEOPANDAS:
        raise ImportError("geopandas is required for the Alpine perimeter filter")
    peri = load_alpine_perimeter()
    if peri.crs is None:
        peri = peri.set_crs("EPSG:4326")
    peri = peri.copy()
    peri["geometry"] = peri.geometry.buffer(0)  # repair invalid rings
    geom = peri.union_all()

    df = stations.dropna(subset=[lon_col, lat_col]).copy()
    pts = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    )
    inside = pts.within(geom)
    return df.loc[inside.to_numpy()].copy()


def stations_measuring(
    stations: pd.DataFrame,
    variable: str,
    variables_col: str = "variables",
) -> pd.DataFrame:
    """
    Keep stations whose 'variables' cell lists the given overview name.

    Accepts either an overview name or a code name (temp_mean → temperature).
    """
    overview_name = CODE_TO_OVERVIEW.get(variable, variable)
    if variables_col not in stations.columns:
        raise ValueError(f"stations table has no '{variables_col}' column")
    def has_var(cell) -> bool:
        if pd.isna(cell):
            return False
        parts = [p.strip() for p in str(cell).split(",")]
        return overview_name in parts
    mask = stations[variables_col].map(has_var)
    return stations.loc[mask].copy()


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
    alpine_only: bool = False,
    domain_only: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Stratified hold-out.

    1. Optionally restrict to the interpolation domain or the Alpine perimeter
       (strict; union of all polygons including any exclave).
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

    if domain_only:
        df = stations_inside_domain(df, lon_col, lat_col)
        if df.empty:
            raise RuntimeError("No stations inside the interpolation domain")
    elif alpine_only:
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
    assignment: Dict[str, List[str]] = {"train": [], "tune": [], "test": []}

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
def holdout_path(variable: Optional[str] = None) -> Path:
    """Path of the hold-out YAML. None → legacy global file."""
    if variable is None:
        return HOLDOUT_PATH
    name = CODE_TO_OVERVIEW.get(variable, variable)
    return _HERE / f"holdout_stations_{name}.yaml"


def tune_targets_path(variable: Optional[str] = None) -> Path:
    """Path of the tune-targets YAML. None → legacy global file."""
    if variable is None:
        return TUNE_TARGETS_PATH
    name = CODE_TO_OVERVIEW.get(variable, variable)
    return _HERE / f"tune_targets_{name}.yaml"


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


def select_tune_targets(
    pool_stations: pd.DataFrame,
    fraction: float = 0.15,
    seed: int = 43,
    n_elev_bins: int = 4,
    spatial_step_deg: float = 0.75,
    lon_col: str = "lon",
    lat_col: str = "lat",
    elev_col: str = "elev",
    id_col: str = "station_name",
) -> Tuple[List[str], List[str]]:
    """
    Fixed stratified 15 % of the pool as tune-targets.

    Same stratification logic as the official hold-out, different seed so the
    two sets are independent. Returns (tune_target_ids, donor_ids).
    Donor_ids = pool minus tune-targets (temporary; they re-enter later).
    """
    return select_stratified_holdout(
        pool_stations,
        fraction=fraction,
        seed=seed,
        n_elev_bins=n_elev_bins,
        spatial_step_deg=spatial_step_deg,
        lon_col=lon_col,
        lat_col=lat_col,
        elev_col=elev_col,
        id_col=id_col,
        alpine_only=False,
        domain_only=False,  # pool is already domain-filtered
    )


def write_tune_targets(
    tune_targets: List[str],
    donors: List[str],
    meta: dict,
    path: Path = TUNE_TARGETS_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tune_targets": sorted(tune_targets),
        "donors_during_tune": sorted(donors),
        "n_tune_targets": len(tune_targets),
        "n_donors_during_tune": len(donors),
        **meta,
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def load_holdout(path: Optional[Path] = None, variable: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """Return (held_out_ids, pool_ids). Pass variable for the per-variable file."""
    if path is None:
        path = holdout_path(variable)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return list(raw["held_out"]), list(raw["pool"])


def load_tune_targets(path: Optional[Path] = None, variable: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """Return (tune_target_ids, donors_during_tune_ids)."""
    if path is None:
        path = tune_targets_path(variable)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return list(raw["tune_targets"]), list(raw["donors_during_tune"])


def load_donors(path: Optional[Path] = None, variable: Optional[str] = None) -> List[str]:
    """Full donor set = pool (includes the stations that are tune-targets)."""
    _, pool = load_holdout(path, variable)
    return list(pool)


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


def prepare_for_tuning(
    df: pd.DataFrame,
    station_col: str = "station_name",
    time_col: str = "time",
    variable: Optional[str] = None,
) -> pd.DataFrame:
    """
    Panel ready for free-parameter tuning.

    - Drops held-out stations.
    - Remaps split so existing code that looks for 'dev' hits the tune quarters:
        train → train, tune → dev, test rows removed.
    """
    tagged, _, _, _ = apply_evaluation_protocol(df, station_col, time_col, variable)
    out = tagged[~tagged["is_heldout"]].copy()
    out.loc[out["split"] == "tune", "split"] = "dev"
    out = out[out["split"].isin(["train", "dev"])]
    return out


def apply_evaluation_protocol(
    df: pd.DataFrame,
    station_col: str = "station_name",
    time_col: str = "time",
    variable: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str], List[str], Dict[str, List[str]]]:
    """
    Tag a panel with the shared evaluation protocol.

    Pass variable (overview name or code name) to use the variable-specific
    station sets. If omitted, the legacy global files are used.

    Adds columns:
      is_heldout      – True for the fixed evaluation stations
      is_tune_target  – True for the fixed tune-target stations (subset of pool)
      quarter         – 'YYYY-Q#'
      split           – 'train' | 'tune' | 'test' | 'outside'

    Returns (df, held_out_ids, pool_ids, quarter_splits).
    """
    held, pool = load_holdout(variable=variable)
    try:
        tune_targets, _ = load_tune_targets(variable=variable)
    except FileNotFoundError:
        tune_targets = []
    splits = load_quarter_splits()
    out = df.copy()
    out["is_heldout"] = out[station_col].astype(str).isin(held)
    out["is_tune_target"] = out[station_col].astype(str).isin(tune_targets)
    t = pd.to_datetime(out[time_col])
    q = ((t.dt.month - 1) // 3 + 1).astype(str)
    out["quarter"] = t.dt.year.astype(str) + "-Q" + q
    out["split"] = "outside"
    for name, qs in splits.items():
        out.loc[out["quarter"].isin(qs), "split"] = name
    return out, held, pool, splits


def protocol_masks(df: pd.DataFrame) -> dict:
    """
    Convenience masks for the locked protocol.

    Keys:
      heldout, tune_target, donor (full pool),
      donor_during_tune (pool minus tune-targets)
    """
    is_held = df["is_heldout"] if "is_heldout" in df.columns else False
    is_tt = df["is_tune_target"] if "is_tune_target" in df.columns else False
    return {
        "heldout": is_held,
        "tune_target": is_tt,
        "donor": ~is_held,
        "donor_during_tune": ~is_held & ~is_tt,
    }
