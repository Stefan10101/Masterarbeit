#!/usr/bin/env python3
"""
Verification Script - DST + Fixed Period Comparisons (Updated for new QC structure)
Coordinate-matched station comparisons using the new full_2020_2025_qc output.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import logging

from pathlib import Path

# ===============================================
# PATHS UPDATED TO RELATIVE (Daten root)
# ===============================================
SCRIPT = Path(__file__).resolve()

PROJECT_ROOT = SCRIPT
while PROJECT_ROOT.name != "Daten":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_ROOT = PROJECT_ROOT


# ==================== UPDATED PATHS FOR NEW QC STRUCTURE ====================
CLEANED_DIR = Path(DATA_ROOT / "temperatur" / "paket" / "full_2020_2025_qc")
STATION_CSV = Path(DATA_ROOT / "station.csv")
OUTPUT_DIR = Path(DATA_ROOT / "verification")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)
plt.style.use('seaborn-v0_8-whitegrid')


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_station_metadata():
    return pd.read_csv(STATION_CSV).set_index("station_name")


def load_cleaned_data_with_coords():
    """Load from new QC output: *_final_qc.parquet"""
    files = list(CLEANED_DIR.glob("*_final_qc.parquet"))
    data = {}
    coord_map = []

    for f in files:
        try:
            df = pd.read_parquet(f)
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            df.index = pd.to_datetime(df.index, utc=True)

            station = df["station"].iloc[0] if "station" in df.columns else f.stem.split("_")[0]
            lat = float(df["lat"].iloc[0]) if "lat" in df.columns else None
            lon = float(df["lon"].iloc[0]) if "lon" in df.columns else None

            data[station] = df[["value"]].copy()
            if lat and lon:
                coord_map.append((lat, lon, station))
        except Exception as e:
            logger.debug(f"Failed to load {f.name}: {e}")
            pass
    logger.info(f"Loaded {len(data)} temperature stations from QC output")
    return data, coord_map


def find_matching_station(target_lat, target_lon, coord_map, max_distance_m=300):
    """Increased tolerance to 300m for robustness with pipeline coordinate handling."""
    best_station = None
    best_dist = 999999
    for lat, lon, station in coord_map:
        dist = haversine(target_lat, target_lon, lat, lon)
        if dist < best_dist and dist <= max_distance_m:
            best_dist = dist
            best_station = station
    return best_station, best_dist


def _apply_6h_axis(ax):
    """Apply clean 6-hour ticks with labels only at 00:00 UTC."""
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    ax.tick_params(axis='x', which='major', labelsize=8, rotation=45, length=5)

    def format_6h(x, pos=None):
        dt = mdates.num2date(x)
        if dt.hour == 0:
            return dt.strftime('%d.%m')
        return ''
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_6h))


def plot_dst_comparison_2021(station1_csv, station2_csv, provider1, provider2, data, metadata, coord_map, suffix=""):
    """DST comparison for 2021 only (March + October)."""
    if station1_csv not in metadata.index or station2_csv not in metadata.index:
        logger.warning(f"Station not found in metadata: {station1_csv} or {station2_csv}")
        return

    row1 = metadata.loc[[station1_csv]].iloc[0]
    row2 = metadata.loc[[station2_csv]].iloc[0]

    matched1, dist1 = find_matching_station(float(row1["latitude"]), float(row1["longitude"]), coord_map)
    matched2, dist2 = find_matching_station(float(row2["latitude"]), float(row2["longitude"]), coord_map)

    if not matched1 or not matched2 or matched1 not in data or matched2 not in data:
        logger.warning(f"Could not match stations within 300m: {station1_csv}â†’{matched1}, {station2_csv}â†’{matched2}")
        return

    logger.info(f"DST 2021: {station1_csv} ({matched1}, {dist1:.0f}m) vs {station2_csv} ({matched2}, {dist2:.0f}m)")

    lat1, lon1, elev1 = float(row1["latitude"]), float(row1["longitude"]), float(row1["elevation_masl"])
    lat2, lon2, elev2 = float(row2["latitude"]), float(row2["longitude"]), float(row2["elevation_masl"])
    distance = round(haversine(lat1, lon1, lat2, lon2) / 1000, 1)
    elev_diff = round(abs(elev1 - elev2))

    title = f"{provider1} vs {provider2} | {distance} km | Î”elev = {elev_diff} m | Year 2021"

    fig, axes = plt.subplots(2, 1, figsize=(15, 9))

    for ax, month, season in zip(axes, [3, 10], ["Spring DST", "Autumn DST"]):
        if month == 3:
            transition = pd.Timestamp("2021-03-28 01:00:00", tz="UTC")
        else:
            transition = pd.Timestamp("2021-10-31 01:00:00", tz="UTC")

        start = transition - pd.Timedelta(days=7)
        end = transition + pd.Timedelta(days=7)

        s1 = data[matched1].loc[start:end]
        s2 = data[matched2].loc[start:end]

        ax.plot(s1.index, s1["value"], label=f"{station1_csv} ({provider1})", linewidth=1.6)
        ax.plot(s2.index, s2["value"], label=f"{station2_csv} ({provider2})", linewidth=1.6)

        ax.axvline(transition, color='red', linestyle='--', linewidth=2.2, alpha=0.85,
                   label="DST Transition (~02:00 local)")

        try:
            for ts in s1["value"].resample("D").apply(lambda x: x.idxmax() if len(x) > 0 else pd.NaT).dropna():
                ax.axvline(ts, color='blue', linestyle=':', alpha=0.5, linewidth=1.2)
            for ts in s2["value"].resample("D").apply(lambda x: x.idxmax() if len(x) > 0 else pd.NaT).dropna():
                ax.axvline(ts, color='orange', linestyle=':', alpha=0.5, linewidth=1.2)
        except:
            pass

        ax.set_title(f"{season} Transition â€” {title}", fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_ylabel("Temperature (Â°C)")
        ax.grid(True, alpha=0.3)

        _apply_6h_axis(ax)

    plt.tight_layout()
    filename = f"{suffix}_{station1_csv}_vs_{station2_csv}_2021.png".replace(" ", "_")
    plt.savefig(OUTPUT_DIR / filename, dpi=160, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {filename}")


def plot_fixed_period_comparison(station1_csv, station2_csv, provider1, provider2, data, metadata, coord_map,
                                 start, end, suffix=""):
    """Fixed period comparison."""
    if station1_csv not in metadata.index or station2_csv not in metadata.index:
        return

    row1 = metadata.loc[[station1_csv]].iloc[0]
    row2 = metadata.loc[[station2_csv]].iloc[0]

    matched1, dist1 = find_matching_station(float(row1["latitude"]), float(row1["longitude"]), coord_map)
    matched2, dist2 = find_matching_station(float(row2["latitude"]), float(row2["longitude"]), coord_map)

    if not matched1 or not matched2 or matched1 not in data or matched2 not in data:
        logger.warning(f"Could not match: {station1_csv}â†’{matched1}, {station2_csv}â†’{matched2}")
        return

    logger.info(f"Fixed period: {station1_csv} ({matched1}, {dist1:.0f}m) vs {station2_csv} ({matched2}, {dist2:.0f}m)")

    lat1, lon1, elev1 = float(row1["latitude"]), float(row1["longitude"]), float(row1["elevation_masl"])
    lat2, lon2, elev2 = float(row2["latitude"]), float(row2["longitude"]), float(row2["elevation_masl"])
    distance = round(haversine(lat1, lon1, lat2, lon2) / 1000, 1)
    elev_diff = round(abs(elev1 - elev2))

    title = f"{provider1} vs {provider2} | {distance} km | Î”elev = {elev_diff} m | {start.strftime('%Y-%m-%d')} â€“ {end.strftime('%Y-%m-%d')}"

    fig, ax = plt.subplots(1, 1, figsize=(15, 6))

    s1 = data[matched1].loc[start:end]
    s2 = data[matched2].loc[start:end]

    ax.plot(s1.index, s1["value"], label=f"{station1_csv} ({provider1})", linewidth=1.6)
    ax.plot(s2.index, s2["value"], label=f"{station2_csv} ({provider2})", linewidth=1.6)

    try:
        for ts in s1["value"].resample("D").apply(lambda x: x.idxmax() if len(x) > 0 else pd.NaT).dropna():
            ax.axvline(ts, color='blue', linestyle=':', alpha=0.5, linewidth=1.2)
        for ts in s2["value"].resample("D").apply(lambda x: x.idxmax() if len(x) > 0 else pd.NaT).dropna():
            ax.axvline(ts, color='orange', linestyle=':', alpha=0.5, linewidth=1.2)
    except:
        pass

    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylabel("Temperature (Â°C)")
    ax.grid(True, alpha=0.3)

    _apply_6h_axis(ax)

    plt.tight_layout()
    filename = f"{suffix}_{station1_csv}_vs_{station2_csv}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.png".replace(" ", "_")
    plt.savefig(OUTPUT_DIR / filename, dpi=160, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {filename}")


def main():
    logger.info("=== VERIFICATION SCRIPT v2 (new QC structure) START ===")
    metadata = load_station_metadata()
    data, coord_map = load_cleaned_data_with_coords()

    if len(data) == 0:
        logger.error("No temperature data loaded. Check CLEANED_DIR path and that QC has been run.")
        return

    comparisons = [
        ("Patscherkofel", "Innsbruck-Seegrube", "ZAMG", "HYDRO_TIROL"),
        ("Obergurgl", "Sonnbergalm", "ZAMG", "LWD"),
        ("Obergurgl", "Rosskar", "ZAMG", "LWD"),
        ("Seefeld", "Leutasch-Kirchplatzl", "ZAMG", "HYDRO_TIROL"),
        ("Galzig", "Zugspitze", "ZAMG", "DWD"),
        ("Galzig", "SÃ¤ntis", "ZAMG", "MeteoSuisse"),
        ("Steinach_am_Brenner", "Sterzing", "ZAMG", "Suedtirol"),
        ("Oberriet___Kriessern", "Lustenau", "MeteoSuisse", "HYDRO_VO"),
    ]

    # === 2021 DST transitions only (March + October) ===
    logger.info("\n=== Processing 2021 DST transitions ===")
    for s1, s2, p1, p2 in comparisons:
        plot_dst_comparison_2021(s1, s2, p1, p2, data, metadata, coord_map, suffix="DST_2021")

    # === Fixed 5-day period: 20â€“25 February 2021 ===
    logger.info("\n=== Processing fixed period 2021-02-20 to 2021-02-25 ===")
    feb_start = pd.Timestamp("2021-02-20 00:00:00", tz="UTC")
    feb_end = pd.Timestamp("2021-02-25 23:59:59", tz="UTC")
    for s1, s2, p1, p2 in comparisons:
        plot_fixed_period_comparison(s1, s2, p1, p2, data, metadata, coord_map,
                                     start=feb_start, end=feb_end, suffix="Fixed_Feb2021")

    logger.info(f"\nAll plots saved to: {OUTPUT_DIR}")
    logger.info("=== VERIFICATION SCRIPT FINISHED ===")


if __name__ == "__main__":
    main()

