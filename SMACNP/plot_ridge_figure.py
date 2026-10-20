#!/usr/bin/env python3
"""Generate the ridge + coastline overview figure for SMACNP features."""

from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from SMACNP.geometry_helpers import (
    load_alpine_perimeter,
    load_natural_earth_coastline,
    classify_coast_segments,
    derive_main_alpine_ridge,
    plot_ridge_figure,
)


def main():
    print("Loading Alpine Convention perimeter...")
    peri = load_alpine_perimeter()
    print(f"Perimeter features: {len(peri)}, bounds: {peri.total_bounds}")

    print("Loading Natural Earth coastline...")
    coast = load_natural_earth_coastline()
    med, atl = classify_coast_segments(coast, peri)
    print(f"Mediterranean segments: {len(med)}, Atlantic segments: {len(atl)}")

    ridge = derive_main_alpine_ridge()
    out = Path(__file__).resolve().parent / "figures" / "ridge_and_coasts.png"
    plot_ridge_figure(peri, ridge, out, med, atl)
    print(f"Figure written to {out}")


if __name__ == "__main__":
    main()
