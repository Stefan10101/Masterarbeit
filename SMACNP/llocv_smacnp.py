#!/usr/bin/env python3
"""Held-out station evaluation for SMACNP.

Loads a trained checkpoint, predicts on the fixed held-out stations
(using only training-pool stations as context), computes metrics on
the original scale, and writes a results parquet.
"""

from __future__ import annotations

from pathlib import Path
import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from smacnp_data import (
    load_config,
    load_panel,
    compute_dem_features,
    feature_columns,
    PaperScaler,
    FeatureScaler,
)
from smacnp_core import SMACNP
from paths import get_smacnp_model_path, get_method_output_dir
from shared.splits.evaluation_sets import load_holdout, load_quarter_splits, quarter_of


def get_device(cfg: dict) -> torch.device:
    dev = cfg["smacnp"].get("device", "auto")
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def build_model_from_ckpt(ckpt: dict, device: torch.device) -> SMACNP:
    cfg_sm = ckpt["cfg"]
    n_features = len(ckpt["feature_cols"])
    spatial_dim = 2
    attr_dim = n_features - spatial_dim
    model = SMACNP(
        input_dim=n_features,
        spatial_dim=spatial_dim,
        attr_dim=attr_dim,
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


def metrics(obs: np.ndarray, pred: np.ndarray) -> dict:
    o = np.asarray(obs, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    mask = np.isfinite(o) & np.isfinite(p)
    o, p = o[mask], p[mask]
    if len(o) == 0:
        return {k: np.nan for k in ("rmse", "mae", "bias", "r")}
    err = p - o
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    if o.std() > 0 and p.std() > 0:
        r = float(np.corrcoef(o, p)[0, 1])
    else:
        r = np.nan
    return {"rmse": rmse, "mae": mae, "bias": bias, "r": r, "n": int(len(o))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variable", default="temp_mean")
    parser.add_argument("--time-resolution", default=None)
    parser.add_argument("--jobs", type=int, default=None,
                        help="parallel test days (default: half the logical cores). "
                             "GPU forwards serialize on the single device.")
    args = parser.parse_args()

    cfg = load_config()
    if args.time_resolution:
        cfg["time_resolution"] = args.time_resolution
    var = args.variable
    device = get_device(cfg)
    ckpt_path = get_smacnp_model_path(var, cfg["time_resolution"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"loaded {ckpt_path}")

    model = build_model_from_ckpt(ckpt, device)
    scaler = PaperScaler(two_step=ckpt["two_step"])
    scaler.__dict__.update(ckpt["scaler"])
    feat_scaler = FeatureScaler().load_state_dict(ckpt["feature_scaler"])
    feat_cols = ckpt["feature_cols"]

    print("Loading panel and features...")
    df = load_panel(cfg, var)
    df = df.rename(columns={var: "value"})
    stations = df[["station_name", "x", "y"]].drop_duplicates("station_name")
    if "elev" in df.columns:
        stations = stations.merge(
            df[["station_name", "elev"]].drop_duplicates("station_name"), on="station_name", how="left"
        )
    stations = compute_dem_features(stations, cfg)
    from smacnp_data import attach_clc_onehot
    stations = attach_clc_onehot(stations)
    df = df.drop(columns=[c for c in feat_cols if c in df.columns and c not in ("x", "y")], errors="ignore")
    df = df.merge(
        stations[["station_name"] + [c for c in feat_cols if c not in ("x", "y")]],
        on="station_name", how="left",
    )
    df = feat_scaler.transform(df)

    station_names = df["station_name"].unique()
    held_out, pool = load_holdout()
    train_stations = np.array([s for s in station_names if s in set(pool)])
    held_stations = np.array([s for s in station_names if s in set(held_out)])
    print(f"evaluating on {len(held_stations)} held-out stations")

    # Restrict to test quarters for the primary scores
    qsplits = load_quarter_splits()
    df["quarter"] = df["time"].map(quarter_of)
    test_days = set(df.loc[df["quarter"].isin(qsplits["test"]), "time"].unique())
    print(f"test quarters={len(qsplits['test'])}  test days={len(test_days)}")

    # For evaluation: context = all training-pool stations present that day
    # target = held-out stations present that day
    rows = []
    diag_p, diag_amt, diag_wet = [], [], []
    days = [d for d in df["time"].unique() if d in test_days]
    jobs = int(args.jobs) if args.jobs is not None else max(1, (os.cpu_count() or 1) // 2)
    print(f"scoring {len(days)} test days  jobs={jobs}")

    def one(day):
        df_day = df[df["time"] == day].dropna(subset=feat_cols + ["value"])
        df_ctx = df_day[df_day["station_name"].isin(train_stations)]
        df_tgt = df_day[df_day["station_name"].isin(held_stations)]
        if len(df_ctx) < 1 or len(df_tgt) < 1:
            return None
        x_ctx = torch.from_numpy(df_ctx[feat_cols].to_numpy(dtype=np.float32)).unsqueeze(0).to(device)
        y_ctx = torch.from_numpy(
            scaler.transform(df_ctx["value"].to_numpy(dtype=np.float64)).astype(np.float32)
        ).unsqueeze(0).unsqueeze(-1).to(device)
        x_tgt = torch.from_numpy(df_tgt[feat_cols].to_numpy(dtype=np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x_ctx, y_ctx, x_tgt)
        if ckpt["two_step"]:
            p_occ = torch.sigmoid(pred["occurrence_logits"]).cpu().numpy().squeeze()
            mu_n = pred["amount_mu"].cpu().numpy().squeeze()
            var_n = pred["amount_variance"].cpu().numpy().squeeze()
            amt = scaler.inverse_mean(mu_n, var_n)
            y_hat = np.where(p_occ >= 0.5, np.maximum(amt, 0.0), 0.0)
            diag = (np.atleast_1d(p_occ), np.atleast_1d(amt),
                    df_tgt["value"].to_numpy(dtype=np.float64) > 1e-6)
        else:
            y_hat = scaler.inverse(pred["mu"].cpu().numpy().squeeze())
            diag = None
        y_true = df_tgt["value"].to_numpy(dtype=np.float64)
        out = [{"time": day, "station_name": name, "obs": yt, "pred": yp}
               for name, yt, yp in zip(df_tgt["station_name"], y_true, y_hat)]
        return out, diag

    if jobs == 1:
        results = [one(d) for d in tqdm(days, desc="  smacnp")]
    else:
        results = Parallel(n_jobs=jobs, backend="threading")(
            delayed(one)(d) for d in tqdm(days, desc="  smacnp")
        )
    for r in results:
        if r is None:
            continue
        out, diag = r
        rows.extend(out)
        if diag is not None:
            diag_p.append(diag[0])
            diag_amt.append(diag[1])
            diag_wet.append(diag[2])

    res = pd.DataFrame(rows)
    if res.empty:
        print("no evaluation pairs produced")
        return
    if diag_p:
        p = np.concatenate(diag_p)
        a = np.concatenate(diag_amt)
        wet = np.concatenate(diag_wet)
        print(
            f"diag  p_occ mean={p.mean():.3f}  p>=0.5={float(np.mean(p >= 0.5)):.3f}  "
            f"amt mean={float(np.nanmean(a)):.1f}  "
            f"amt|wet mean={float(np.nanmean(a[wet])) if wet.any() else float('nan'):.1f}  "
            f"obs wet frac={float(wet.mean()):.3f}"
        )
    met = metrics(res["obs"].to_numpy(), res["pred"].to_numpy())
    print(f"held-out  RMSE={met['rmse']:.3f}  MAE={met['mae']:.3f}  bias={met['bias']:.3f}  r={met['r']:.3f}  n={met['n']}")

    out_dir = get_method_output_dir("SMACNP") / "llocv" / cfg["time_resolution"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{var}_heldout.parquet"
    res.to_parquet(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
