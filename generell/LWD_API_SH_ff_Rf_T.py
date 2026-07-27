import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

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
OUTPUT_DIR = Path(DATA_ROOT / "lwd")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2020, 2026))          # 2020_2021 up to 2025_2026

# PARAMS now includes LT (air temperature / Lufttemperatur)
PARAMS = ["HS", "WG", "LF", "LT"]

BASE_URL = "https://wiski.tirol.gv.at/lawine/produkte/ogd"

TEST_MODE = False
TEST_STATIONS_LIMIT = 15

# Clean station list (updated from latest run of LWD_Stationlist.py)
# Removed obvious non-station entries like "LAWINE", "OGD", "PRODUKTE"
STATION_CODES = [
    "ABIR1", "ABRA2", "ACOM1", "AKLE1", "AKLE2", "ARAU1", "ARAU2", "AXLIZ1",
    "AXLIZ3", "BJOE1", "EERF1", "EERF2", "FHOC1", "FHOC2", "FISS1", "FISS2",
    "FSEE1", "FSEE2", "FSON1", "FSON2", "FURG2", "GADA2", "GAMA1", "GAMA2",
    "GGAL1", "GGAL2", "GJAM1", "GJAM2", "GPRE1", "GPRE2", "HHAH1", "IMOS1",
    "IMOS2", "IMUT1", "IMUT2", "INAC1", "INAC2", "IPIS1", "IPIS2", "ISEE1",
    "ISEE2", "KADL1", "KAUN1", "KAUN2", "KDIA1", "KDIA2", "KFIG1", "KFIG2",
    "KGRI2", "KGRO1", "KHOR1", "KHOR2", "KSIN1", "KSIN2", "KSPI2", "KTRI1",
    "KUET1", "KWEL2", "KZWO1", "LBRE1", "LBRE2", "LGRU1", "LGRU2", "LPUI1",
    "LPUI2", "MIOF1", "MIOG1", "MIOG2", "MIOH1", "MIOH2", "NASS1", "NASS2",
    "NELF1", "NELF2", "NFRA1", "NFRA2", "NGAM1", "NGAM2", "NGAN2", "NKRI1",
    "NKRI2", "NNOW1", "NSCH1", "NSGE2", "NSGR1", "NSGS1", "NVAL1", "NVAL2",
    "OCON1", "PESE1", "PESE2", "SAAU1", "SAAU2", "SAGA1", "SAGA2", "SARE1",
    "SARE2", "SAUL2", "SAYO1", "SBAR2", "SDAW1", "SDAW2", "SFES1", "SFES2",
    "SGIG2", "SKAP1", "SKAP2", "SKRE1", "SLAG1", "SLBU1", "SLBU2", "SLRI1",
    "SLRI2", "SLSE1", "SMAS1", "SMAS2", "SMED1", "SMED2", "SPLO1", "SPLO2",
    "SRET1", "SRET2", "SROS1", "SSLA1", "SSLA2", "SSON1", "SSON2", "SSPE2",
    "STHU1", "SVVO1", "SVVO2", "SVZI1", "SVZI2", "THOH1", "TLAE1", "TLAE2",
    "TRAU1", "TRAU2", "TTUX1", "TTUX2", "TWAN1", "TWAN2", "ZSIL1", "ZSIL2"
]

# ====================================================

def download_year_file(station: str, param: str, year: int):
    """Download one specific year file. Skips if already exists."""
    next_year = year + 1
    url = f"{BASE_URL}/{station}/{station}_{param}_{year}_{next_year}.csv"
    filepath = OUTPUT_DIR / f"{station}_{param}_{year}_{next_year}.csv"
    
    # === IMPORTANT: Check if file already exists ===
    if filepath.exists():
        return "SKIP", filepath.name
    
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            filepath.write_bytes(r.content)
            return "DONE", filepath.name
        elif r.status_code == 404:
            return "MISS", f"{station}_{param}_{year}_{next_year}"
        else:
            return "ERROR", f"{r.status_code} {station}_{param}_{year}_{next_year}"
    except requests.exceptions.Timeout:
        return "TIMEOUT", f"{station}_{param}_{year}_{next_year}"
    except Exception as e:
        return "FAIL", f"{station}_{param}_{year}_{next_year} â†’ {e}"


def main():
    print("=" * 95)
    print("LWD Year-Specific Downloader - Resume capable (with Air Temperature)")
    print("=" * 95)
    print(f"Output folder: {OUTPUT_DIR}\n")
    
    print("Parameters: HS (Snow height), WG (Wind), LF (Relative humidity), LT (Air temperature)")
    print("Note: 'TP' (dewpoint) was previously replaced by 'LF' (relative humidity)\n")
    
    stations = STATION_CODES[:TEST_STATIONS_LIMIT] if TEST_MODE else STATION_CODES
    
    if TEST_MODE:
        print(f"TEST MODE â†’ {len(stations)} stations")
    else:
        print(f"Full run â†’ {len(stations)} stations Ã— {len(YEARS)} seasons Ã— {len(PARAMS)} params")
    
    tasks = [(s, p, y) for s in stations for y in YEARS for p in PARAMS]
    print(f"Total tasks: {len(tasks)}\n")
    
    counters = {"DONE": 0, "SKIP": 0, "MISS": 0, "ERROR": 0, "FAIL": 0, "TIMEOUT": 0}
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=4) as executor:   # 4 is safe
        futures = {executor.submit(download_year_file, *task): task for task in tasks}
        
        for future in as_completed(futures):
            status, msg = future.result()
            counters[status] += 1
            if status in ("DONE", "ERROR", "FAIL", "TIMEOUT") or (status == "MISS" and len(tasks) < 500):
                print(f"{status:8} | {msg}")
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 95)
    print("DOWNLOAD SUMMARY")
    print("=" * 95)
    print(f"âœ… DONE    : {counters['DONE']} files downloaded")
    print(f"â­ï¸  SKIP    : {counters['SKIP']} files already existed (skipped)")
    print(f"â“ MISS    : {counters['MISS']} files not available")
    print(f"â³ TIMEOUT : {counters['TIMEOUT']} timeouts")
    print(f"âš ï¸  ERROR   : {counters['ERROR']} other errors")
    print(f"âŒ FAIL    : {counters['FAIL']} failures")
    print(f"\nTime taken : {elapsed/60:.1f} minutes")
    print(f"Files saved in: {OUTPUT_DIR}")
    print("=" * 95)
    
    if counters["TIMEOUT"] > 0:
        print("\nâ†’ Some timeouts occurred. Simply re-run the script â€” it will continue from where it left off.")

if __name__ == "__main__":
    main()
