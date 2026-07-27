#!/usr/bin/env python3
"""
Spatial Analysis Script for FGBoost Multi-Variable Performance
Master Thesis - Clean version (no geographic features)
"""

import os
import re
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import itertools
from pathlib import Path

from pathlib import Path

# ===============================================
# PATHS UPDATED TO RELATIVE (Daten root)
# ===============================================
SCRIPT = Path(__file__).resolve()

PROJECT_ROOT = SCRIPT
while PROJECT_ROOT.name != "Daten":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_ROOT = PROJECT_ROOT


# ====================== CONFIG ======================
DATA_DIR = Path(DATA_ROOT / "kombiniert" / "analyse" / "xgb")
ARTIFACTS_DIR = Path(DATA_ROOT / "plots" / "kombiniert")
ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)

VARS = {
    "TEMPERATURE": "temperature",
    "WIND_SPEED": "wind_speed",
    "RELATIVE_HUMIDITY": "relative_humidity",
    "SNOW_HEIGHT": "snow_height",
}

ZOOM_REGIONS = {
    "Central_Alps": {"lon_min": 10.0, "lon_max": 12.5, "lat_min": 46.3, "lat_max": 47.6},
    "Eastern_Alps": {"lon_min": 12.0, "lon_max": 13.5, "lat_min": 46.5, "lat_max": 47.3},
    "Western_Alps_Switzerland": {"lon_min": 8.0, "lon_max": 10.0, "lat_min": 46.0, "lat_max": 47.5},
    "Pre_Alps_North": {"lon_min": 9.5, "lon_max": 11.5, "lat_min": 47.2, "lat_max": 48.0},
}

TOP_K_FOR_IMPORTANCE = 5
JACCARD_SAMPLE = 200

# ====================== HELPERS ======================
def normalize_station(name: str) -> str:
    return name.strip().replace(" / ", "/").replace(" ,", ",")

def parse_similarity_log(log_path: Path) -> dict:
    print(f"[INFO] Parsing log: {log_path.name} ({log_path.stat().st_size/1e6:.2f} MB)")
    data = defaultdict(list)
    current_var = None
    current_target = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "STARTING ANALYSIS FOR:" in line:
                current_var = line.split(":")[-1].strip().upper()
                continue

            m_target = re.search(r"Top 8 Most Similar Stations to (.+?) \([a-z]+\)", line)
            if m_target:
                current_target = normalize_station(m_target.group(1))
                continue

            m_sim = re.search(
                r"(\d+)\.\s+(.+?)\s+\|\s+([\d.]+) km\s+\|\s+([\d.]+) m\s+\|\s+corr:\s*([\d.]+)\s+\|\s+score:\s*([\d.]+)",
                line,
            )
            if m_sim and current_var and current_target:
                rank, similar, dist, hdiff, corr, score = m_sim.groups()
                similar_norm = normalize_station(similar)
                data[current_var].append({
                    "target": current_target,
                    "rank": int(rank),
                    "similar_station": similar_norm,
                    "dist_km": float(dist),
                    "height_diff_m": float(hdiff),
                    "corr": float(corr),
                    "score": float(score),
                })
    print(f"[INFO] Parsed {sum(len(v) for v in data.values())} similarity records across {len(data)} variables")
    return data

def load_performance(var_key: str) -> pd.DataFrame:
    path = DATA_DIR / f"{VARS[var_key]}_clusters.csv"
    df = pd.read_csv(path)
    df["station"] = df["station"].apply(normalize_station)
    df = df.drop_duplicates(subset=["station"])
    return df

def compute_jaccard_stability(sim_list: list, top_k: int = 5, n_samples: int = JACCARD_SAMPLE) -> float:
    target_sets = defaultdict(set)
    for rec in sim_list:
        if rec["rank"] <= top_k:
            target_sets[rec["target"]].add(rec["similar_station"])

    targets = list(target_sets.keys())
    if len(targets) < 10:
        return np.nan

    np.random.seed(42)
    all_pairs = list(itertools.combinations(np.random.choice(targets, min(len(targets), 300), replace=False), 2))
    pairs = random.sample(all_pairs, n_samples) if n_samples and len(all_pairs) > n_samples else all_pairs

    jaccards = []
    for t1, t2 in pairs:
        s1, s2 = target_sets[t1], target_sets[t2]
        inter = len(s1 & s2)
        union = len(s1 | s2)
        if union > 0:
            jaccards.append(inter / union)
    return float(np.mean(jaccards)) if jaccards else np.nan

# ====================== ANALYSIS PER VARIABLE ======================
def analyze_variable(var_key: str, sim_data: dict, perf_df: pd.DataFrame):
    var_name = VARS[var_key]
    print(f"\n{'='*60}\nANALYZING: {var_key} ({var_name})\n{'='*60}")

    sim_list = sim_data.get(var_key, [])
    n_stations = perf_df["station"].nunique()
    print(f"Stations: {n_stations} | Similarities: {len(sim_list)}")

    # 1. Performance spatial maps
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle(f"{var_key} â€” Spatial Performance & Clusters", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    sc = ax.scatter(perf_df["lon"], perf_df["lat"], c=perf_df["rmse"], cmap="viridis_r", s=35, alpha=0.75)
    plt.colorbar(sc, ax=ax, label="RMSE", shrink=0.7)
    ax.set_title("RMSE (lower = better)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    sc = ax.scatter(perf_df["lon"], perf_df["lat"], c=perf_df["r2"], cmap="viridis", s=35, alpha=0.75)
    plt.colorbar(sc, ax=ax, label="RÂ²", shrink=0.7)
    ax.set_title("RÂ² (higher = better)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    colors = {0: "#1f77b4", 1: "#d62728"}
    for cl in [0, 1]:
        sub = perf_df[perf_df["cluster"] == cl]
        ax.scatter(sub["lon"], sub["lat"], c=colors[cl], s=40, alpha=0.7, label=f"Cluster {cl} (n={len(sub)})")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("K-Means Clusters (elevation-based)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for cl in [0, 1]:
        sub = perf_df[perf_df["cluster"] == cl]
        ax.scatter(sub["hoehe"], sub["rmse"], c=colors[cl], s=25, alpha=0.6, label=f"Cl {cl}")
    ax.set_xlabel("Station Height (m a.s.l.)")
    ax.set_ylabel("RMSE")
    ax.set_title("Performance vs Elevation (mountain vs valley)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.savefig(ARTIFACTS_DIR / f"{var_name}_spatial_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {var_name}_spatial_overview.png")

    # 2. Zoomed regional maps
    for region_name, bounds in ZOOM_REGIONS.items():
        mask = ((perf_df["lon"] >= bounds["lon_min"]) & (perf_df["lon"] <= bounds["lon_max"]) &
                (perf_df["lat"] >= bounds["lat_min"]) & (perf_df["lat"] <= bounds["lat_max"]))
        sub = perf_df[mask]
        if len(sub) < 8:
            continue
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(sub["lon"], sub["lat"], c=sub["rmse"], cmap="viridis_r", s=60, alpha=0.85, edgecolors="k", linewidths=0.3)
        plt.colorbar(sc, ax=ax, label="RMSE")
        ax.set_xlim(bounds["lon_min"], bounds["lon_max"])
        ax.set_ylim(bounds["lat_min"], bounds["lat_max"])
        ax.set_title(f"{var_key} RMSE â€” {region_name.replace('_', ' ')} (zoomed)")
        ax.set_xlabel("Lon"); ax.set_ylabel("Lat")
        ax.grid(True, alpha=0.4)
        worst = sub.nlargest(2, "rmse")
        for _, row in worst.iterrows():
            ax.annotate(row["station"][:12], (row["lon"], row["lat"]), fontsize=6, alpha=0.8)
        plt.savefig(ARTIFACTS_DIR / f"{var_name}_zoom_{region_name}.png", dpi=150, bbox_inches="tight")
        plt.close()
    print(f"[SAVED] {len(ZOOM_REGIONS)} zoomed maps for {var_name}")

    # 3. Station importance
    if sim_list:
        top_similar = [rec["similar_station"] for rec in sim_list if rec["rank"] <= TOP_K_FOR_IMPORTANCE]
        importance = Counter(top_similar).most_common(25)

        stations_imp, counts = zip(*importance[:15]) if importance else ([], [])
        fig, ax = plt.subplots(figsize=(9, 5))
        y_pos = np.arange(len(stations_imp))
        ax.barh(y_pos, counts, color="#2ca02c", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(stations_imp, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(f"Times in Top-{TOP_K_FOR_IMPORTANCE} Similar Lists")
        ax.set_title(f"{var_key} â€” Most 'Important' Stations (used as predictors most often)")
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(ARTIFACTS_DIR / f"{var_name}_station_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SAVED] {var_name}_station_importance.png (top station: {importance[0][0] if importance else 'N/A'})")

        imp_stations = set([s for s, _ in importance[:20]])
        perf_df["is_important"] = perf_df["station"].isin(imp_stations)
        mean_rmse_imp = perf_df[perf_df["is_important"]]["rmse"].mean()
        mean_rmse_other = perf_df[~perf_df["is_important"]]["rmse"].mean()
        print(f"  Mean RMSE of top-20 important stations: {mean_rmse_imp:.3f} vs others: {mean_rmse_other:.3f}")

    # 4. Feature stability
    if sim_list:
        stability = compute_jaccard_stability(sim_list, top_k=5)
        print(f"  Avg. Jaccard similarity of Top-5 similar sets (random pairs): {stability:.3f}")
        if stability > 0.35:
            print("  â†’ Features relatively STABLE across stations")
        elif stability > 0.15:
            print("  â†’ Features MODERATELY variable")
        else:
            print("  â†’ Features change STRONGLY across stations (highly local patterns)")

    # 5. Quick stats
    print(f"  Overall: mean RMSE={perf_df['rmse'].mean():.3f}, mean R2={perf_df['r2'].mean():.3f}")
    print(f"  Cluster 0: {len(perf_df[perf_df['cluster']==0])} stations, Cluster 1: {len(perf_df[perf_df['cluster']==1])} stations")

# ====================== MAIN ======================
def main():
    log_path = DATA_DIR / "all_variables_analysis_20260508_165005.log"
    sim_data = parse_similarity_log(log_path)

    for var_key in VARS.keys():
        perf_df = load_performance(var_key)
        analyze_variable(var_key, sim_data, perf_df)

    print("\n[COMPLETE] All spatial analyses finished. Figures saved to /home/workdir/artifacts/")
    print("Key outputs per variable:")
    print("  - *_spatial_overview.png  (RMSE, R2, clusters, height-vs-error)")
    print("  - *_zoom_*.png            (regional details for valley/mountain)")
    print("  - *_station_importance.png (which stations are most reused as similar)")

if __name__ == "__main__":
    main()
