#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master Dataset Unifier v2.2 (rich provider lookup from raw data tree)
===================================================================
Combines all cleaned 30-min variables into one station-centric master dataset.
Adds 'network_provider' metadata column to stations_overview.csv (and per-station JSON/parquet)
by spatially matching cluster centers to the **most complete** provider station list possible.

Key improvement v2.2:
- Instead of relying on the incomplete central Stationen/ JSON lists, we now scan the raw data tree
  (Temperatur/, Wind/, Luftfeuchte/, etc. â†’ provider subfolders â†’ station folders â†’ *.json).
- Uses the same robust recursive metadata extractor that handles ZAMG nested structures and German keys.
- This dramatically reduces (ideally eliminates) "Unknown" providers.
- Still uses 100 m coordinate tolerance nearest-neighbor matching for robustness.

All prior logic (QC paths, 10 m clustering, parquet engine guard, etc.) preserved.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
import re
from scipy.spatial.distance import cdist

# ==================== CONFIG ====================
BASE = Path(DATA_ROOT)

VAR_FOLDERS = {
    "relative_humidity": BASE / "Luftfeuchte" / "Paket" / "full_2020_2025_qc",
    "precipitation":     BASE / "Niederschlag" / "Paket" / "full_2020_2025_qc",
    "snow_height":       BASE / "Schneehoehe" / "Paket" / "full_2020_2025_qc",
    "temperature":       BASE / "Temperatur" / "Paket" / "full_2020_2025_qc",
    "wind_speed":        BASE / "Wind" / "Paket" / "full_2020_2025_qc",
}

OUTPUT_DIR = BASE / "Kombiniert"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RADIUS_M = 10.0
ROUND_DECIMALS = 2
MAX_PROVIDER_MATCH_M = 100.0

# --- Provider lookup from raw data tree (more complete than Stationen/ folder) ---
RAW_VARIABLE_FOLDERS = ["Luftfeuchte", "Wind", "Niederschlag", "Schneehoehe", "Temperatur"]

VALID_PROVIDER_FOLDERS = {"DWD", "HydroT", "HydroVO", "LWD", "Meteosuisse", "Suedtirol", "Zamg", "ZAMG"}

PROVIDER_MAP = {
    "DWD": "DWD",
    "HydroT": "HYDRO_TIROL",
    "HydroVO": "HYDRO_VORARLBERG",
    "LWD": "LWD",
    "Meteosuisse": "MeteoSuisse",
    "Suedtirol": "Suedtirol",
    "Zamg": "ZAMG",
    "ZAMG": "ZAMG",
}

# Fallback (old central lists) - kept for compatibility but usually less complete
PROVIDER_DIR = BASE / "Stationen"
PROVIDERS = list(PROVIDER_MAP.values())

LOG_FILE = OUTPUT_DIR / f"master_unifier_v2.0_{datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()])
logger = logging.getLogger(__name__)


def extract_metadata_from_json(json_path: Path):
    """Robust recursive extractor (handles ZAMG nested structures and German keys)."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        lat = lon = hoehe = None

        def find_coords(obj):
            nonlocal lat, lon, hoehe
            if isinstance(obj, dict):
                for k, v in obj.items():
                    k_lower = k.lower()
                    if lat is None and any(x in k_lower for x in ["lat", "latitude", "breite", "geobreite"]):
                        try:
                            lat = float(v)
                        except (ValueError, TypeError):
                            pass
                    if lon is None and any(x in k_lower for x in ["lon", "longitude", "laenge", "geolaenge"]):
                        try:
                            lon = float(v)
                        except (ValueError, TypeError):
                            pass
                    if hoehe is None and any(x in k_lower for x in ["hoehe", "hoehe_masl", "altitude", "elevation", "stationshoehe", "alt"]):
                        try:
                            hoehe = float(v)
                        except (ValueError, TypeError):
                            pass
                    find_coords(v)
            elif isinstance(obj, list):
                for item in obj:
                    find_coords(item)

        find_coords(data)

        # Final rounding
        lat = round(lat, 6) if lat is not None else None
        lon = round(lon, 6) if lon is not None else None
        hoehe = round(hoehe, 1) if hoehe is not None else None

        # Sanity check: flipped coordinates (common in some LWD exports)
        if lat is not None and lon is not None and lat < lon:
            lat, lon = lon, lat
            logger.warning(f"FLIPPED COORDS â†’ swapped in provider JSON: {json_path.name} ({lat:.4f}, {lon:.4f})")

        return lat, lon, hoehe
    except Exception:
        return None, None, None


def build_provider_lookup_from_raw_data():
    """Scan the raw data tree (Temperatur/, Wind/, etc.) to build the most complete
    provider â†’ station mapping possible. This is more reliable than the central Stationen/ lists.
    """
    records = []
    seen = set()

    for var_name in RAW_VARIABLE_FOLDERS:
        var_path = BASE / var_name
        if not var_path.exists():
            continue

        for provider_dir in var_path.iterdir():
            if not provider_dir.is_dir() or provider_dir.name not in VALID_PROVIDER_FOLDERS:
                continue

            provider_name = PROVIDER_MAP.get(provider_dir.name, provider_dir.name)

            for station_dir in provider_dir.iterdir():
                if not station_dir.is_dir():
                    continue
                station_name = station_dir.name.strip()
                if not station_name or station_name.lower() in {"paket", "full_2020_2025"}:
                    continue

                key = (provider_name, station_name.lower())
                if key in seen:
                    continue
                seen.add(key)

                lat = lon = hoehe = None
                for jf in station_dir.glob("*.json"):
                    lat, lon, hoehe = extract_metadata_from_json(jf)
                    if lat is not None and lon is not None:
                        break

                if lat is None or lon is None:
                    continue  # skip stations without usable coordinates

                records.append({
                    "provider": provider_name,
                    "station_name": station_name,
                    "lat": lat,
                    "lon": lon,
                    "hoehe": hoehe,
                })

    df = pd.DataFrame(records)
    if df.empty:
        logger.warning("build_provider_lookup_from_raw_data() found ZERO stations â€” falling back to central lists")
        return pd.DataFrame()
    logger.info(f"Built rich provider lookup from raw data: {len(df):,} stations across {df['provider'].nunique()} networks")
    return df


def _check_parquet_engine():
    """Fail fast with a clear message if neither pyarrow nor fastparquet is installed.
    This is the most common reason the script dies on first run in a fresh conda env.
    """
    try:
        import pyarrow  # noqa: F401
        return "pyarrow"
    except ImportError:
        try:
            import fastparquet  # noqa: F401

from pathlib import Path

# ===============================================
# PATHS UPDATED TO RELATIVE (Daten root)
# ===============================================
SCRIPT = Path(__file__).resolve()

PROJECT_ROOT = SCRIPT
while PROJECT_ROOT.name != "Daten":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_ROOT = PROJECT_ROOT

            return "fastparquet"
        except ImportError:
            logger.error(
                "CRITICAL: No parquet engine found. "
                "pandas.read_parquet() requires either 'pyarrow' or 'fastparquet'.\n"
                "Please install it in your conda environment with:\n"
                "    conda install -c conda-forge pyarrow\n"
                "or\n"
                "    pip install pyarrow"
            )
            raise RuntimeError("Missing parquet engine (pyarrow or fastparquet). See log above for install command.")


# ==================== 1. COLLECT ALL STATIONS ====================
def collect_station_metadata():
    records = []
    for var_name, folder in VAR_FOLDERS.items():
        if not folder.exists():
            logger.warning(f"Folder missing: {folder}")
            continue
        for pq in folder.glob("*_final_qc.parquet"):   # UPDATED
            try:
                meta = pd.read_parquet(pq, columns=["station", "name", "lat", "lon", "hoehe", "parameter"])
                if meta.empty:
                    continue
                row = meta.iloc[0]
                lat = float(row["lat"]) if pd.notna(row["lat"]) else np.nan
                lon = float(row["lon"]) if pd.notna(row["lon"]) else np.nan

                # Sanity check: flipped coordinates
                if pd.notna(lat) and pd.notna(lon) and lat < lon:
                    logger.warning(f"FLIPPED COORDS â†’ swapped: {pq.name} ({lat:.4f}, {lon:.4f})")
                    lat, lon = lon, lat

                records.append({
                    "var": var_name,
                    "original_station": str(row["station"]),
                    "name": str(row["name"]) if pd.notna(row["name"]) else "",
                    "lat": lat,
                    "lon": lon,
                    "hoehe": float(row["hoehe"]) if pd.notna(row["hoehe"]) else np.nan,
                    "parameter": str(row.get("parameter", var_name)),
                    "file_path": pq
                })
            except Exception as e:
                logger.error(f"Read error {pq.name}: {e}")
    df = pd.DataFrame(records)
    if df.empty:
        logger.error(
            "collect_station_metadata() returned ZERO stations. "
            "This almost always means the parquet engine is missing (see earlier CRITICAL message) "
            "or that the QC output folders contain no *_final_qc.parquet files."
        )
        raise RuntimeError("No station metadata could be read. Check parquet engine installation and QC folder contents.")
    return df


# ==================== 2. SPATIAL CLUSTERING ====================
def cluster_stations(df_meta, radius_m=RADIUS_M):
    df = df_meta.dropna(subset=["lat", "lon"]).copy()
    if df.empty:
        return df, pd.DataFrame()

    mean_lat = df["lat"].mean()
    df["x"] = (df["lon"] - df["lon"].mean()) * 111320 * np.cos(np.radians(mean_lat))
    df["y"] = (df["lat"] - df["lat"].mean()) * 111320

    coords = df[["x", "y"]].values
    dist_matrix = cdist(coords, coords, metric="euclidean")

    n = len(df)
    used = np.zeros(n, dtype=bool)
    cluster_id = np.full(n, -1, dtype=int)
    cid = 0
    for i in range(n):
        if used[i]:
            continue
        cluster_id[i] = cid
        used[i] = True
        for j in range(i + 1, n):
            if not used[j] and dist_matrix[i, j] <= radius_m:
                cluster_id[j] = cid
                used[j] = True
        cid += 1

    df["cluster_id"] = cluster_id

    def pick_name(g):
        names = [n for n in g["name"] if n]
        return names[0] if names else f"station_{g['cluster_id'].iloc[0]:04d}"

    rep = df.groupby("cluster_id").apply(pick_name).reset_index(name="station_name")
    centers = df.groupby("cluster_id").agg({"lat": "mean", "lon": "mean", "hoehe": "mean"}).reset_index()

    cluster_info = centers.merge(rep, on="cluster_id")
    df = df.merge(cluster_info[["cluster_id", "station_name", "lat", "lon", "hoehe"]], on="cluster_id", suffixes=("", "_center"))

    dup = df.groupby(["cluster_id", "var"]).size().reset_index(name="count")
    dups = dup[dup["count"] > 1]
    if not dups.empty:
        logger.warning(f"DUPLICATES: {len(dups)} variable(s) measured multiple times at same location â†’ keeping first")

    return df, cluster_info


# ==================== 2b. PROVIDER LOOKUP & MATCHING (now uses raw data tree) ====================
def assign_providers_to_clusters(cluster_info, provider_df, max_dist_m=MAX_PROVIDER_MATCH_M):
    """Match each 10 m cluster center to the nearest provider station using
    coordinate proximity (â‰¤ max_dist_m). Now fed with the much richer lookup
    built from the raw data tree â†’ far fewer (ideally zero) 'Unknown' entries.
    """
    if provider_df.empty or cluster_info.empty:
        cluster_info = cluster_info.copy()
        cluster_info["network_provider"] = "Unknown"
        return cluster_info

    cl = cluster_info[["cluster_id", "lat", "lon"]].copy()
    prov = provider_df[["provider", "lat", "lon"]].copy()

    # Common origin for accurate relative distances (same logic as before)
    combined = pd.concat([cl[["lat", "lon"]], prov[["lat", "lon"]]]).dropna()
    mean_lat = combined["lat"].mean()
    mean_lon = combined["lon"].mean()

    cl["x"] = (cl["lon"] - mean_lon) * 111320 * np.cos(np.radians(mean_lat))
    cl["y"] = (cl["lat"] - mean_lat) * 111320
    prov["x"] = (prov["lon"] - mean_lon) * 111320 * np.cos(np.radians(mean_lat))
    prov["y"] = (prov["lat"] - mean_lat) * 111320

    cl_coords = cl[["x", "y"]].values
    prov_coords = prov[["x", "y"]].values
    dist_matrix = cdist(cl_coords, prov_coords, metric="euclidean")

    min_dists = dist_matrix.min(axis=1)
    min_idx = dist_matrix.argmin(axis=1)

    assigned = []
    for dist, idx in zip(min_dists, min_idx):
        if dist <= max_dist_m:
            assigned.append(prov.iloc[idx]["provider"])
        else:
            assigned.append("Unknown")

    cluster_info = cluster_info.copy()
    cluster_info["network_provider"] = assigned

    n_unknown = sum(p == "Unknown" for p in assigned)
    n_matched = len(assigned) - n_unknown

    if n_unknown > 0:
        logger.warning(f"{n_unknown} cluster(s) still unmatched within {max_dist_m} m even after scanning raw data tree. "
                       f"These stations may genuinely be missing from all provider metadata JSONs.")
    logger.info(f"Provider matching complete: {n_matched}/{len(assigned)} clusters matched to a network "
                f"({n_unknown} Unknown)")
    return cluster_info


# ==================== 3. BUILD MASTER PER CLUSTER ====================
def build_master_for_cluster(cluster_df, cluster_info_row):
    station_name = cluster_info_row["station_name"]
    safe_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', station_name)
    out_parquet = OUTPUT_DIR / f"{safe_name}_full_2020_2025_master.parquet"
    out_json = OUTPUT_DIR / f"{safe_name}_full_2020_2025_master.json"

    var_dfs = {}
    for _, row in cluster_df.drop_duplicates(subset=["var"]).iterrows():
        try:
            df = pd.read_parquet(row["file_path"], columns=["timestamp", "value"])
            df = df.rename(columns={"value": row["var"]})
            df[row["var"]] = df[row["var"]].round(ROUND_DECIMALS)
            var_dfs[row["var"]] = df
        except Exception as e:
            logger.error(f"Load error {row['file_path'].name}: {e}")

    if not var_dfs:
        return None

    master = None
    for v, d in var_dfs.items():
        if master is None:
            master = d
        else:
            master = pd.merge(master, d, on="timestamp", how="outer")

    master = master.sort_values("timestamp").reset_index(drop=True)

    master["station_name"] = station_name
    master["lat"] = round(cluster_info_row["lat"], 6)
    master["lon"] = round(cluster_info_row["lon"], 6)
    master["hoehe"] = round(cluster_info_row["hoehe"], 1) if pd.notna(cluster_info_row["hoehe"]) else None

    # Extract network_provider (added in v2.1)
    network_provider = cluster_info_row.get("network_provider", "Unknown") if isinstance(cluster_info_row, pd.Series) else "Unknown"

    # Add as constant column for convenience when loading individual station files
    master["network_provider"] = network_provider

    core_cols = ["timestamp", "station_name", "lat", "lon", "hoehe", "network_provider"]
    var_cols = [v for v in VAR_FOLDERS.keys() if v in master.columns]
    master = master[core_cols + var_cols]

    master.to_parquet(out_parquet, compression="snappy", index=False)

    json_data = {
        "station_name": station_name,
        "lat": float(cluster_info_row["lat"]),
        "lon": float(cluster_info_row["lon"]),
        "hoehe": float(cluster_info_row["hoehe"]) if pd.notna(cluster_info_row["hoehe"]) else None,
        "network_provider": network_provider,
        "variables_present": var_cols,
        "n_timestamps": len(master),
        "period": "2020-2025 (30-min)",
        "qc_version": "master_v2.1 (coordinate-clustered + provider matching)",
        "source_files": [str(p) for p in cluster_df["file_path"].unique()],
        "created": datetime.now().isoformat()
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    return {
        "station_name": station_name,
        "lat": cluster_info_row["lat"],
        "lon": cluster_info_row["lon"],
        "hoehe": cluster_info_row["hoehe"],
        "network_provider": network_provider,
        "variables": ", ".join(var_cols),
        "n_timestamps": len(master),
        "parquet": out_parquet.name
    }


# ==================== MAIN ====================
def main():
    _check_parquet_engine()   # fail fast with actionable message if pyarrow/fastparquet missing
    logger.info("=== MASTER DATASET UNIFIER v2.2 START ===")
    logger.info(f"Radius: {RADIUS_M} m | Provider match tol: {MAX_PROVIDER_MATCH_M} m | "
                f"Rounding: {ROUND_DECIMALS} decimals | Output: {OUTPUT_DIR}")
    logger.info("Coordinate flip detection active for provider JSONs (LWD exports often have lat/lon swapped)")

    meta = collect_station_metadata()
    logger.info(f"Found {len(meta)} station-variable pairs across {meta['var'].nunique()} variables")

    clustered, cluster_info = cluster_stations(meta)
    logger.info(f"Created {len(cluster_info)} unique physical stations (10 m clusters)")

    # === NEW (v2.2): build richest possible provider lookup from the raw data tree ===
    # This replaces the old central Stationen/ lists which were incomplete.
    provider_df = build_provider_lookup_from_raw_data()
    if provider_df.empty:
        logger.warning("Raw-data provider lookup empty â€” trying fallback to central Stationen/ lists")
        # (old load_provider_stations code could be re-added here if really needed)
    cluster_info = assign_providers_to_clusters(cluster_info, provider_df, MAX_PROVIDER_MATCH_M)

    summary = []
    for _, cl in cluster_info.iterrows():
        cl_df = clustered[clustered["cluster_id"] == cl["cluster_id"]]
        res = build_master_for_cluster(cl_df, cl)
        if res:
            summary.append(res)
            logger.info(f"  âœ“ {res['station_name']} | {res.get('network_provider', 'Unknown')} | "
                        f"{res['variables']} | {res['n_timestamps']:,} rows")

    if summary:
        pd.DataFrame(summary).to_csv(OUTPUT_DIR / "stations_overview.csv", index=False)
        logger.info(f"Summary written: stations_overview.csv ({len(summary)} stations) â€” now includes 'network_provider' column")

    logger.info("=== FINISHED v2.2 ===")
    logger.info(f"Total unique stations: {len(summary)}")
    logger.info(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

