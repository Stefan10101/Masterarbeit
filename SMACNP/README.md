# SMACNP

Localized Spatial Multi-Attention Conditional Neural Process, adapted from
Dumitrescu 2026 (https://github.com/alexdum/climate-gridder).

## Design (locked)
- Per-variable models
- Localized encoder only
- Daily first; monthly / weekly / hh wired
- Two-step (occurrence + amount) for precip_sum and snow_mean
- Fixed held-out stations (configurable fraction, random) for final evaluation.
- Random day split (configurable fraction, no temporal cut) inside the training-pool stations for early stopping.
- Ocean distances from Natural Earth 10m coastline (same source as station_overview_plots.py cartopy features), scaled by characteristic lengths.
- Main Alpine ridge: DEM-derived high-elevation watershed divide (see below).
- Alpine Convention perimeter loaded from the same ArcGIS FeatureServer URL used by station_overview_plots.py.
- Normalization: paper min-max / log1p+min-max
- On-the-grid training
- Fixed held-out stations for evaluation
- Emits mean + variance

## Status
Core model and data adapter stubs are in place.
DEM feature geometry (distance to coast, signed distance to main Alpine ridge)
still needs implementation before training.

## Run order
1. tune_smacnp.py
2. llocv_smacnp.py
3. produce_smacnp_maps.py
