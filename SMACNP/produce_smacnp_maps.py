#!/usr/bin/env python3
"""Produce SMACNP maps (mean + variance).

Loads a frozen checkpoint, uses all stations as context, predicts on the
master grid, and writes a netCDF with mean and variance variables.
"""

from __future__ import annotations

from pathlib import Path
import sys
import argparse

import numpy as np
import pandas as pd
import torch

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from smacnp_data import (
    load_config,
    load_panel,
    compute_dem_features,
    attach_clc_onehot,
    PaperScaler,
    FeatureScaler,
)
from smacnp_core import SMACNP
from shared.map_io import save_map_with_partials
from paths import get_smacnp_model_path, get_master_grid_path, get_method_output_dir

try:
    import xarray as xr
except ImportError:
    xr = None


def get_device(cfg: dict) -> torch.device:
    dev = cfg["smacnp"].get("device", "auto")
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def build_model_from_ckpt(ckpt: dict, device: torch.device) -> SMACNP:
    cfg_sm = ckpt["cfg"]
    n_features = len(ckpt["feature_cols"])
    model = SMACNP(
        input_dim=n_features,
        spatial_dim=2,
        attr_dim=n_features - 2,
        y_dim=1,
        two_step=ckpt["two_step"],
        r_dim=cfg_sm["r_dim"],
        w_dim=cfg_sm["w_dim"],
        v_dim=cfg_sm["v_dim"],
        hidden_size=cfg_sm["hidden_size"],
        num_layers_mlp=cfg_sm["num_layers_mlp"],
        num_heads=cfg_sm["num_heads"],
        dropout_rate=cfg_sm["dropout_rate"],
        k_localized=cfg_sm["k_localized"],
        use_positional_encoding=cfg_sm["use_positional_encoding"],
        pe_dim=cfg_sm["pe_dim"],
        use_layer_norm=cfg_sm["use_layer_norm"],
        variance_floor=cfg_sm["variance_floor"],
    )
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def grid_features(grid: xr.Dataset, feat_cols: list[str], cfg: dict,
                  resolution: int = 1000) -> tuple[np.ndarray, tuple[int, int]]:
    """Build (N, n_features) on the master grid.

    Prefer the terrain cache at the same resolution as the master grid.
    Fall back to the 1000 m cache with nearest reindex only if the matching
    file is missing.
    """
    from paths import get_grid_terrain_path
    from shared.clc import one_hot_level2

    cache_path = get_grid_terrain_path(int(resolution))
    if not cache_path.exists() and int(resolution) != 1000:
        alt = get_grid_terrain_path(1000)
        if alt.exists():
            print(f"  terrain cache {resolution}m missing, falling back to 1000m")
            cache_path = alt
    if not cache_path.exists():
        raise FileNotFoundError(f"grid terrain cache missing: {cache_path}")
    print(f"  terrain cache: {cache_path.name}")
    cache = xr.open_dataset(cache_path)

    # Align only when the cache grid differs from the master grid.
    same = (
        cache.y.size == grid.y.size
        and cache.x.size == grid.x.size
        and np.allclose(cache.y.values, grid.y.values)
        and np.allclose(cache.x.values, grid.x.values)
    )
    aligned = cache if same else cache.reindex(y=grid.y, x=grid.x, method="nearest")
    ny, nx = int(grid.y.size), int(grid.x.size)
    xx, yy = np.meshgrid(grid.x.values, grid.y.values)

    # Precompute CLC one-hot from the integer class field when requested
    clc_oh = None
    if any(c.startswith("clc_l2_") for c in feat_cols) and "clc_level2" in aligned:
        codes = np.nan_to_num(aligned["clc_level2"].values, nan=0).astype(np.int32).ravel()
        clc_oh = one_hot_level2(codes)  # (N, 16), column i = clc_l2_i

    # Precompute scaled 250 m elev-bin from grid elev when requested
    elev_bin_s = None
    if "elev_bin" in feat_cols and "elev" in aligned:
        from shared.terrain import elev_bin
        b = elev_bin(np.nan_to_num(aligned["elev"].values, nan=0.0).ravel()).astype(np.float32)
        mu = float(b.mean()) if b.size else 0.0
        sd = float(b.std()) if b.size else 1.0
        if sd < 1e-8:
            sd = 1.0
        elev_bin_s = (b - mu) / sd

    cols = []
    for c in feat_cols:
        if c == "x":
            cols.append(xx.ravel())
        elif c == "y":
            cols.append(yy.ravel())
        elif c.startswith("clc_l2_") and clc_oh is not None:
            idx = int(c.rsplit("_", 1)[-1])
            cols.append(clc_oh[:, idx] if 0 <= idx < clc_oh.shape[1] else np.zeros(ny * nx, dtype=np.float32))
        elif c == "elev_bin" and elev_bin_s is not None:
            cols.append(elev_bin_s)
        elif c in aligned and aligned[c].ndim == 2:
            cols.append(np.nan_to_num(aligned[c].values, nan=0.0).ravel())
        else:
            cols.append(np.zeros(ny * nx, dtype=np.float32))
    mat = np.column_stack(cols).astype(np.float32)
    cache.close()
    return mat, (ny, nx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variable", default=None)
    parser.add_argument("--time-resolution", default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--max-days", type=int, default=None, help="Produce N randomly sampled days")
    parser.add_argument("--months", default=None, help="comma-separated YYYY-MM (e.g. 2025-01,2025-02,...) or seasonal4")
    parser.add_argument("--zero-ridge", action="store_true", help="Test: zero signed_dist_to_ridge on the grid")
    parser.add_argument("--zero-coast", action="store_true", help="Test: zero dist_to_coast on the grid")
    parser.add_argument("--no-tile", action="store_true", help="Test: predict the full grid in one pass")
    parser.add_argument("--rows-per-tile", type=int, default=None, help="Override tile height in rows")
    args = parser.parse_args()

    if xr is None:
        raise RuntimeError("xarray is required")

    cfg = load_config()
    if args.time_resolution:
        cfg["time_resolution"] = args.time_resolution
    if args.variable is None:
        import subprocess
        import sys
        vs = list(cfg.get("variables_to_process") or ["temp_mean"])
        print(f"no --variable; running {vs}", flush=True)
        for v in vs:
            cmd = [sys.executable, str(Path(__file__).resolve()), "--variable", v]
            if args.time_resolution:
                cmd += ["--time-resolution", args.time_resolution]
            if args.resolution is not None:
                cmd += ["--resolution", str(args.resolution)]
            if args.max_days is not None:
                cmd += ["--max-days", str(args.max_days)]
            if args.months:
                cmd += ["--months", args.months]
            if args.zero_ridge:
                cmd.append("--zero-ridge")
            if args.zero_coast:
                cmd.append("--zero-coast")
            if args.no_tile:
                cmd.append("--no-tile")
            if args.rows_per_tile is not None:
                cmd += ["--rows-per-tile", str(args.rows_per_tile)]
            subprocess.check_call(cmd)
        return
    var = args.variable
    res = args.resolution if args.resolution is not None else int((cfg.get("resolutions_to_process") or [1000])[0])
    from paths import get_method_output_dir
    out_dir = get_method_output_dir("SMACNP") / "interpolated_maps" / cfg["time_resolution"]
    out_path = out_dir / f"{var}_{cfg['time_resolution']}_{res}m.nc"
    if out_path.exists() and not cfg.get("overwrite_maps", True):
        print(f"exists, skip {out_path}")
        return
    device = get_device(cfg)

    ckpt_path = get_smacnp_model_path(var, cfg["time_resolution"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model_from_ckpt(ckpt, device)
    scaler = PaperScaler(two_step=ckpt["two_step"])
    scaler.__dict__.update(ckpt["scaler"])
    feat_scaler = FeatureScaler().load_state_dict(ckpt["feature_scaler"])
    feat_cols = ckpt["feature_cols"]
    print(f"loaded {ckpt_path}")

    # Grid
    _, _, grid_method = (
        (cfg.get("paths") or {}).get("aggregated_from", "RFSI"),
        (cfg.get("paths") or {}).get("stations_from", "RFSI"),
        (cfg.get("paths") or {}).get("grids_from", "RFSI"),
    )
    grid_path = get_master_grid_path(grid_method, res)
    grid = xr.open_dataset(grid_path)
    print(f"grid {grid_path}  shape=({grid.y.size}, {grid.x.size})")

    print("Computing grid features...")
    grid_mat, (ny, nx) = grid_features(grid, feat_cols, cfg, resolution=res)

    # Station context (all stations)
    print("Loading stations...")
    df = load_panel(cfg, var)
    df = df.rename(columns={var: "value"})
    stations = df[["station_name", "x", "y"]].drop_duplicates("station_name")
    if "elev" in df.columns:
        stations = stations.merge(
            df[["station_name", "elev"]].drop_duplicates("station_name"), on="station_name", how="left"
        )
    stations = compute_dem_features(stations, cfg)
    stations = attach_clc_onehot(stations)
    from smacnp_data import attach_elev_bin
    stations = attach_elev_bin(stations)
    df = df.drop(columns=[c for c in feat_cols if c in df.columns and c not in ("x", "y")], errors="ignore")
    df = df.merge(
        stations[["station_name"] + [c for c in feat_cols if c not in ("x", "y")]],
        on="station_name", how="left",
    )
    df = feat_scaler.transform(df)

    # Standardize grid features with the same stats
    mean = feat_scaler.mean_
    std = feat_scaler.std_
    grid_mat = (grid_mat - mean) / std
    if args.zero_ridge and "signed_dist_to_ridge" in feat_cols:
        grid_mat[:, feat_cols.index("signed_dist_to_ridge")] = 0.0
        print("TEST: signed_dist_to_ridge zeroed on grid")
    if args.zero_coast and "dist_to_coast" in feat_cols:
        grid_mat[:, feat_cols.index("dist_to_coast")] = 0.0
        print("TEST: dist_to_coast zeroed on grid")

    times_u = np.array(sorted(df["time"].unique()))
    unit = {"monthly": "months", "weekly": "weeks", "daily": "days",
            "half_hourly": "slots"}.get(cfg["time_resolution"], "times")
    if args.months and str(args.months).strip().lower() not in ("", "all", "none"):
        spec = str(args.months).strip().lower()
        t = pd.to_datetime(times_u)
        if spec == "seasonal4":
            times_u = times_u[np.isin(t.month, (2, 5, 8, 11))]
        else:
            keys = {s.strip() for s in spec.split(",") if s.strip()}
            stamp = t.strftime("%Y-%m")
            times_u = times_u[np.isin(stamp, list(keys))]
    if args.max_days is not None and args.max_days < len(times_u):
        rng = np.random.default_rng(cfg["smacnp"].get("seed", 42))
        pick = rng.choice(len(times_u), args.max_days, replace=False)
        times_u = np.sort(times_u[pick])
    print(f"predicting {len(times_u)} {unit}")

    # Tile by whole rows. Variance encoder attention (v_dim=512) on a
    # ~200k-cell tile tries to allocate >12 GB and OOMs a 12 GB card.
    # ~40k cells keeps the (M, k, D) projections inside VRAM; the location
    # encoder is already chunked so the smaller tile no longer stalls on cdist.
    rows_per_tile = ny if args.no_tile else max(1, 40000 // nx)
    if args.rows_per_tile is not None:
        rows_per_tile = max(1, int(args.rows_per_tile))
    if args.no_tile:
        print("TEST: full grid in one pass")
    print(f"  rows_per_tile={rows_per_tile}  (~{rows_per_tile * nx:,} cells)")
    means = []
    variances = []
    times = []

    for i, ts in enumerate(times_u):
        print(f"  {unit[:-1] if unit.endswith('s') else unit} {i+1}/{len(times_u)}  {ts}", flush=True)
        df_day = df[df["time"] == ts].dropna(subset=feat_cols + ["value"])
        if len(df_day) < 2:
            continue
        x_ctx = torch.from_numpy(df_day[feat_cols].to_numpy(dtype=np.float32)).unsqueeze(0).to(device)
        y_ctx = torch.from_numpy(
            scaler.transform(df_day["value"].to_numpy(dtype=np.float64)).astype(np.float32)
        ).unsqueeze(0).unsqueeze(-1).to(device)

        mu_full = np.full((ny, nx), np.nan, dtype=np.float32)
        var_full = np.full((ny, nx), np.nan, dtype=np.float32)

        n_tiles = (ny + rows_per_tile - 1) // rows_per_tile
        for ti, r0 in enumerate(range(0, ny, rows_per_tile)):
            r1 = min(r0 + rows_per_tile, ny)
            sl = grid_mat.reshape(ny, nx, -1)[r0:r1].reshape(-1, grid_mat.shape[1])
            x_tgt = torch.from_numpy(sl).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(x_ctx, y_ctx, x_tgt)
            if ckpt["two_step"]:
                p_occ = torch.sigmoid(pred["occurrence_logits"]).cpu().numpy().squeeze()
                mu_n = pred["amount_mu"].cpu().numpy().squeeze()
                var_n = pred["amount_variance"].cpu().numpy().squeeze()
                amt = np.maximum(scaler.inverse_mean(mu_n, var_n), 0.0)
                p_occ = np.clip(p_occ, 0.0, 1.0)
                mu_full[r0:r1] = (p_occ * amt).reshape(r1 - r0, nx).astype(np.float32)
                var_full[r0:r1] = var_n.reshape(r1 - r0, nx).astype(np.float32)
            else:
                mu_full[r0:r1] = scaler.inverse(pred["mu"].cpu().numpy().squeeze()).reshape(r1 - r0, nx).astype(np.float32)
                var_full[r0:r1] = pred["variance"].cpu().numpy().squeeze().reshape(r1 - r0, nx).astype(np.float32)
            if (ti + 1) % 10 == 0 or ti == n_tiles - 1:
                print(f"    tile {ti+1}/{n_tiles}", flush=True)
            del x_tgt, pred
            # Do not empty_cache every tile. That forces a full device sync
            # and is what turns a normal forward into the spike-then-freeze.

        means.append(mu_full)
        variances.append(var_full)
        times.append(ts)

    if not means:
        print("no maps produced")
        return

    ds = xr.Dataset(
        {
            f"{var}_mean": (("time", "y", "x"), np.stack(means)),
            f"{var}_variance": (("time", "y", "x"), np.stack(variances)),
        },
        coords={"time": times, "y": grid.y.values, "x": grid.x.values},
    )
    if "elev" in grid:
        ds["elev"] = (("y", "x"), grid.elev.values.astype(np.float32))

    out_dir = get_method_output_dir("SMACNP") / "interpolated_maps" / cfg["time_resolution"]
    out_dir.mkdir(parents=True, exist_ok=True)
    save_map_with_partials(ds, out_path, cfg["time_resolution"])
    print(f"wrote {out_path}")
    grid.close()


if __name__ == "__main__":
    main()
