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
    PaperScaler,
    FeatureScaler,
)
from smacnp_core import SMACNP
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


def grid_features(grid: xr.Dataset, feat_cols: list[str], cfg: dict) -> tuple[np.ndarray, tuple[int, int]]:
    """Build (N, n_features) on the master grid.

    Terrain cache may have a different shape. Features are reindexed onto
    the master grid coordinates (nearest) so the output matches the master grid.
    """
    from paths import get_grid_terrain_path

    cache_path = get_grid_terrain_path(1000)
    if not cache_path.exists():
        raise FileNotFoundError(f"grid terrain cache missing: {cache_path}")
    cache = xr.open_dataset(cache_path)

    # Align cache fields onto the master grid
    aligned = cache.reindex(y=grid.y, x=grid.x, method="nearest")
    ny, nx = int(grid.y.size), int(grid.x.size)
    xx, yy = np.meshgrid(grid.x.values, grid.y.values)

    cols = []
    for c in feat_cols:
        if c == "x":
            cols.append(xx.ravel())
        elif c == "y":
            cols.append(yy.ravel())
        elif c in aligned and aligned[c].ndim == 2:
            cols.append(np.nan_to_num(aligned[c].values, nan=0.0).ravel())
        else:
            cols.append(np.zeros(ny * nx, dtype=np.float32))
    mat = np.column_stack(cols).astype(np.float32)
    cache.close()
    return mat, (ny, nx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variable", default="temp_mean")
    parser.add_argument("--time-resolution", default=None)
    parser.add_argument("--resolution", type=int, default=1000)
    parser.add_argument("--max-days", type=int, default=None, help="Produce N randomly sampled days")
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
    var = args.variable
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
        cfg["paths"].get("aggregated_from", "RFSI"),
        cfg["paths"].get("stations_from", "RFSI"),
        cfg["paths"].get("grids_from", "RFSI"),
    )
    grid_path = get_master_grid_path(grid_method, args.resolution)
    grid = xr.open_dataset(grid_path)
    print(f"grid {grid_path}  shape=({grid.y.size}, {grid.x.size})")

    print("Computing grid features...")
    grid_mat, (ny, nx) = grid_features(grid, feat_cols, cfg)

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

    days = np.array(sorted(df["time"].unique()))
    if args.max_days is not None and args.max_days < len(days):
        rng = np.random.default_rng(cfg["smacnp"].get("seed", 42))
        pick = rng.choice(len(days), args.max_days, replace=False)
        days = np.sort(days[pick])
    print(f"predicting {len(days)} days")

    # Tile by whole rows so cuts never split a row (avoids horizontal seams)
    rows_per_tile = ny if args.no_tile else max(1, 50000 // nx)
    if args.no_tile:
        print("TEST: full grid in one pass")
    means = []
    variances = []
    times = []

    for i, day in enumerate(days):
        if i % 10 == 0 or i == len(days) - 1:
            print(f"  day {i+1}/{len(days)}  {day}", flush=True)
        df_day = df[df["time"] == day].dropna(subset=feat_cols + ["value"])
        if len(df_day) < 2:
            continue
        x_ctx = torch.from_numpy(df_day[feat_cols].to_numpy(dtype=np.float32)).unsqueeze(0).to(device)
        y_ctx = torch.from_numpy(
            scaler.transform(df_day["value"].to_numpy(dtype=np.float64)).astype(np.float32)
        ).unsqueeze(0).unsqueeze(-1).to(device)

        mu_full = np.full((ny, nx), np.nan, dtype=np.float32)
        var_full = np.full((ny, nx), np.nan, dtype=np.float32)

        for r0 in range(0, ny, rows_per_tile):
            r1 = min(r0 + rows_per_tile, ny)
            sl = grid_mat.reshape(ny, nx, -1)[r0:r1].reshape(-1, grid_mat.shape[1])
            x_tgt = torch.from_numpy(sl).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(x_ctx, y_ctx, x_tgt)
            if ckpt["two_step"]:
                p_occ = torch.sigmoid(pred["occurrence_logits"]).cpu().numpy().squeeze()
                amt = scaler.inverse(pred["amount_mu"].cpu().numpy().squeeze())
                mu_full[r0:r1] = (p_occ * np.maximum(amt, 0.0)).reshape(r1 - r0, nx).astype(np.float32)
                var_full[r0:r1] = pred["amount_variance"].cpu().numpy().squeeze().reshape(r1 - r0, nx).astype(np.float32)
            else:
                mu_full[r0:r1] = scaler.inverse(pred["mu"].cpu().numpy().squeeze()).reshape(r1 - r0, nx).astype(np.float32)
                var_full[r0:r1] = pred["variance"].cpu().numpy().squeeze().reshape(r1 - r0, nx).astype(np.float32)

        means.append(mu_full)
        variances.append(var_full)
        times.append(day)

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
    out_path = out_dir / f"{var}_{cfg['time_resolution']}_{args.resolution}m.nc"
    ds.to_netcdf(out_path)
    print(f"wrote {out_path}")
    grid.close()


if __name__ == "__main__":
    main()
