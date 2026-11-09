#!/usr/bin/env python3
"""Train Localized SMACNP for one variable.

Protocol (pooled): fit on full pool × train quarters;
early-stop on fixed tune-targets × tune quarters
(context excludes the tune-targets).
Saves the checkpoint.
"""

from __future__ import annotations

from pathlib import Path
import sys
import argparse
import itertools
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from smacnp_data import (
    load_config,
    load_panel,
    compute_dem_features,
    feature_columns,
    usable_feature_columns,
    fill_feature_nans,
    PaperScaler,
    FeatureScaler,
    prepare_day_tensors,
)
from smacnp_core import SMACNP, gaussian_nll, two_step_loss
from paths import get_smacnp_model_path
from shared.splits.evaluation_sets import (
    load_holdout,
    load_quarter_splits,
    load_tune_targets,
    quarter_of,
)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cfg: dict) -> torch.device:
    dev = cfg["smacnp"].get("device", "auto")
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def build_model(cfg: dict, n_features: int, two_step: bool) -> SMACNP:
    s = cfg["smacnp"]
    spatial_dim = 2
    attr_dim = n_features - spatial_dim
    return SMACNP(
        input_dim=n_features,
        spatial_dim=spatial_dim,
        attr_dim=attr_dim,
        y_dim=1,
        two_step=two_step,
        r_dim=s["r_dim"],
        w_dim=s["w_dim"],
        v_dim=s["v_dim"],
        hidden_size=s["hidden_size"],
        num_layers_mlp=s["num_layers_mlp"],
        num_heads=s["num_heads"],
        dropout_rate=s["dropout_rate"],
        k_localized=s["k_localized"],
        use_positional_encoding=s["use_positional_encoding"],
        pe_dim=s["pe_dim"],
        use_layer_norm=s["use_layer_norm"],
        variance_floor=s["variance_floor"],
    )


def precompute_days(days, df, feature_cols, scaler, train_stations, cfg,
                    on_grid: bool | None = None, epoch: int = 0):
    """Build all day tensors. Resampled each epoch so the model sees
    different donor subsets; a frozen split is a train/inference mismatch
    (LLOCV uses the full pool as context).
    """
    rng = np.random.default_rng(int(cfg["smacnp"]["seed"]) + int(epoch))
    if on_grid is None:
        on_grid = bool(cfg["smacnp"]["on_grid_training"])
    ctx_frac = cfg["smacnp"]["context_fraction"]
    grouped = {t: g for t, g in df.groupby("time", sort=False)}
    out = []
    for day in days:
        g = grouped.get(day)
        if g is None:
            continue
        tensors = prepare_day_tensors(g, feature_cols, scaler, train_stations, on_grid, ctx_frac, rng)
        if tensors is not None:
            out.append(tensors)
    return out


def run_epoch(model, day_tensors, cfg, device, optimizer=None):
    """One pass over precomputed day tensors."""
    model.train(optimizer is not None)
    total_loss = 0.0
    n = 0
    two_step = model.two_step
    use_amp = bool(cfg["smacnp"].get("use_amp", False)) and device.type == "cuda"
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)

    for tensors in day_tensors:
        x_ctx = torch.from_numpy(tensors["x_context"]).unsqueeze(0).to(device, non_blocking=True)
        y_ctx = torch.from_numpy(tensors["y_context"]).unsqueeze(0).to(device, non_blocking=True)
        x_tgt = torch.from_numpy(tensors["x_target"]).unsqueeze(0).to(device, non_blocking=True)
        y_tgt = torch.from_numpy(tensors["y_target"]).unsqueeze(0).to(device, non_blocking=True)
        y_raw = None
        if "y_raw" in tensors:
            y_raw = torch.from_numpy(tensors["y_raw"]).unsqueeze(0).to(device, non_blocking=True)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(x_ctx, y_ctx, x_tgt)
            if two_step:
                loss = two_step_loss(pred, y_tgt, y_raw=y_raw)
            else:
                loss = gaussian_nll(pred["mu"], pred["variance"], y_tgt)
        if not torch.isfinite(loss):
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            continue
        if optimizer is not None:
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg["smacnp"]["clip_grad_norm"]))
            scaler_amp.step(optimizer)
            scaler_amp.update()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def val_rmse_original(model, day_tensors, scaler, device) -> float:
    """RMSE in original units on the validation tensors.

    The printed loss is beta-NLL on the [0,1] scale and is not comparable
    to the other methods. This is the number that is.
    """
    model.eval()
    se = 0.0
    n = 0
    two_step = model.two_step
    for tensors in day_tensors:
        x_ctx = torch.from_numpy(tensors["x_context"]).unsqueeze(0).to(device)
        y_ctx = torch.from_numpy(tensors["y_context"]).unsqueeze(0).to(device)
        x_tgt = torch.from_numpy(tensors["x_target"]).unsqueeze(0).to(device)
        y_raw = tensors["y_raw"].reshape(-1)
        pred = model(x_ctx, y_ctx, x_tgt)
        if two_step:
            p = torch.sigmoid(pred["occurrence_logits"]).cpu().numpy().reshape(-1)
            mu_n = pred["amount_mu"].cpu().numpy().reshape(-1)
            var_n = pred["amount_variance"].cpu().numpy().reshape(-1)
            amt = scaler.inverse_mean(mu_n, var_n)
            y_hat = np.where(p >= 0.5, np.maximum(amt, 0.0), 0.0)
        else:
            mu_n = pred["mu"].cpu().numpy().reshape(-1)
            y_hat = scaler.inverse(mu_n)
        err = y_hat - y_raw
        se += float(np.dot(err, err))
        n += err.size
    model.train()
    return float(np.sqrt(se / max(n, 1)))


def search_grid(cfg: dict, quick: bool) -> list[dict]:
    """Full factorial of the config search block. --quick keeps the first level."""
    block = cfg["smacnp"].get("search") or {}
    keys = [k for k, v in block.items() if isinstance(v, (list, tuple)) and len(v)]
    if not keys:
        return [{}]
    levels = []
    for k in keys:
        vals = list(block[k])
        levels.append(vals[:1] if quick else vals)
    return [dict(zip(keys, combo)) for combo in itertools.product(*levels)]


def train_one(cfg, feat_cols, two_step, device, y_scaler,
              train_days, val_days, df, train_stations, val_tensors) -> tuple[float, list, dict]:
    """Train one search cell. Returns (best val RMSE, history, state)."""
    model = build_model(cfg, len(feat_cols), two_step).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["smacnp"]["lr"]),
        weight_decay=float(cfg["smacnp"]["weight_decay"]),
    )
    best_rmse = float("inf")
    best_state = None
    patience = 0
    max_patience = int(cfg["smacnp"]["early_stopping_patience"])
    history = []
    t0 = time.time()
    for epoch in range(int(cfg["smacnp"]["num_epochs"])):
        train_tensors = precompute_days(
            train_days, df, feat_cols, y_scaler, train_stations, cfg,
            on_grid=False, epoch=epoch,
        )
        tr_loss = run_epoch(model, train_tensors, cfg, device, optimizer)
        with torch.no_grad():
            va_loss = run_epoch(model, val_tensors, cfg, device, None)
            rmse = val_rmse_original(model, val_tensors, y_scaler, device)
        print(f"  epoch {epoch:4d}  train={tr_loss:.4f}  val={va_loss:.4f}  val_rmse={rmse:.3f}")
        history.append({"epoch": epoch, "train": tr_loss, "val": va_loss, "val_rmse": rmse})
        if not np.isfinite(rmse):
            print("  val rmse not finite, stopping cell")
            break
        if rmse < best_rmse - 1e-4:
            best_rmse = rmse
            patience = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= max_patience:
                print(f"  early stop at epoch {epoch}")
                break
    wall = time.time() - t0
    return best_rmse, history, {"state": best_state, "epochs": len(history), "wall_s": wall}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variable", default="temp_mean")
    parser.add_argument("--time-resolution", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-times", type=int, default=400,
                        help="cap timestamps, sampled within each split")
    args = parser.parse_args()

    cfg = load_config()
    if args.time_resolution:
        cfg["time_resolution"] = args.time_resolution
    if args.quick:
        cfg["smacnp"]["num_epochs"] = min(int(cfg["smacnp"].get("num_epochs", 500)), 15)
        cfg["smacnp"]["early_stopping_patience"] = min(
            int(cfg["smacnp"].get("early_stopping_patience", 50)), 4
        )
    # Cap LR. 4e-4 diverged on plain NLL. Beta-NLL holds at 1e-4; 2e-5
    # undertrained (best epoch ~34, loss barely off init, RMSE 2.16).
    cfg["smacnp"]["lr"] = min(float(cfg["smacnp"].get("lr", 4e-4)), 1e-4)
    var = args.variable
    two_step = var in cfg["smacnp"].get("two_step_vars", [])
    set_seed(cfg["smacnp"]["seed"])
    device = get_device(cfg)
    print(f"device={device}  variable={var}  two_step={two_step}  time_res={cfg['time_resolution']}  lr={cfg['smacnp']['lr']}")

    print("Loading panel...")
    df = load_panel(cfg, var)
    if var not in df.columns:
        raise KeyError(f"column {var} not in panel")
    df = df.rename(columns={var: "value"})
    print(f"panel rows={len(df)}  stations={df['station_name'].nunique()}")

    print("Computing features...")
    # Features are station-static; compute once and merge
    df["station_name"] = df["station_name"].astype(str)
    stations = df[["station_name", "x", "y"]].drop_duplicates("station_name")
    if "elev" in df.columns:
        stations = stations.merge(df[["station_name", "elev"]].drop_duplicates("station_name"), on="station_name", how="left")
    stations = compute_dem_features(stations, cfg)
    from smacnp_data import attach_clc_onehot, attach_elev_bin
    stations = attach_clc_onehot(stations)
    stations = attach_elev_bin(stations)
    feat_cols = feature_columns(cfg)
    df = df.drop(columns=[c for c in feat_cols if c in df.columns and c not in ("x", "y")], errors="ignore")
    df = df.merge(stations[["station_name"] + [c for c in feat_cols if c not in ("x", "y") and c in stations.columns]], on="station_name", how="left")

    station_names = df["station_name"].unique()
    held_out, pool = load_holdout(variable=var)
    tune_targets, _ = load_tune_targets(variable=var)
    train_stations = np.array([s for s in station_names if s in set(pool)])
    held_stations = np.array([s for s in station_names if s in set(held_out)])
    # Temporary donors for validation: pool minus the fixed tune-targets
    val_context = np.array([s for s in train_stations if s not in set(tune_targets)])
    print(f"train stations={len(train_stations)}  held-out={len(held_stations)}  "
          f"tune-targets={len(tune_targets)}  val-context={len(val_context)}")

    # Drop feature columns that are missing or all-NaN on the training pool,
    # then impute the rest. Otherwise dropna in prepare_day_tensors yields
    # zero tensors and the model trains on nothing.
    train_mask = df["station_name"].isin(train_stations)
    feat_cols = usable_feature_columns(df, feat_cols, train_mask)
    df = fill_feature_nans(df, feat_cols, train_mask)
    print(f"  features ({len(feat_cols)}): {feat_cols}")

    # Wet-fraction gate on the training pool (profile: on if wet in (0.2, 0.8); snow always)
    if two_step:
        raw = df.loc[df["station_name"].isin(train_stations), "value"].to_numpy()
        trace = 0.5 if str(var).lower().startswith("snow") else 0.1
        wet = float(np.mean(raw > trace)) if len(raw) else 0.0
        if str(var).lower().startswith("snow"):
            print(f"  two_step on (snow always)  wet={wet:.3f}")
        elif wet >= 0.80 or wet <= 0.20:
            two_step = False
            print(f"  two_step off: wet={wet:.3f} (indicator is constant)")
        else:
            print(f"  two_step on: wet={wet:.3f}")

    # Scalers on training-pool only
    y_scaler = PaperScaler(two_step=two_step).fit(df.loc[train_mask, "value"].to_numpy())
    feat_scaler = FeatureScaler().fit(df.loc[train_mask], feat_cols)
    df = feat_scaler.transform(df)

    # Quarter-based temporal split (shared protocol)
    qsplits = load_quarter_splits()
    df["quarter"] = df["time"].map(quarter_of)
    train_days = df.loc[df["quarter"].isin(qsplits["train"]), "time"].unique()
    val_days = df.loc[df["quarter"].isin(qsplits["tune"]), "time"].unique()
    if args.max_times:
        rng = np.random.default_rng(int(cfg["smacnp"].get("seed", 42)))
        if len(train_days) > args.max_times:
            train_days = rng.choice(train_days, size=args.max_times, replace=False)
        n_val = max(1, args.max_times // 4)
        if len(val_days) > n_val:
            val_days = rng.choice(val_days, size=n_val, replace=False)
    print(f"train quarters={len(qsplits['train'])}  tune quarters={len(qsplits['tune'])}")
    print(f"train days={len(train_days)}  val days={len(val_days)}")
    # Validation split is fixed so cells are comparable. Training context
    # is resampled inside each epoch.
    val_tensors = precompute_days(
        val_days, df, feat_cols, y_scaler, val_context, cfg, on_grid=False, epoch=0,
    )
    print(f"val tensors={len(val_tensors)}")

    grid = search_grid(cfg, args.quick)
    print(f"search cells={len(grid)}  {[tuple(c.items()) for c in grid]}")
    from paths import get_smacnp_tuned_params_path
    ckpt_path = get_smacnp_model_path(var, cfg["time_resolution"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    search_path = ckpt_path.with_name(f"{var}_{cfg['time_resolution']}_search.csv")

    trials = []
    best = None
    for i, cell in enumerate(grid):
        for k, v in cell.items():
            cfg["smacnp"][k] = v
        print(f"\ncell {i+1}/{len(grid)}  {cell}")
        rmse, history, info = train_one(
            cfg, feat_cols, two_step, device, y_scaler,
            train_days, val_days, df, train_stations, val_tensors,
        )
        row = {**cell, "val_rmse": rmse, "epochs": info["epochs"], "wall_s": round(info["wall_s"], 1)}
        trials.append(row)
        pd.DataFrame(trials).to_csv(search_path, index=False)
        curve = ckpt_path.with_name(f"{var}_{cfg['time_resolution']}_curve_{i}.csv")
        pd.DataFrame(history).to_csv(curve, index=False)
        print(f"  cell val_rmse={rmse:.3f}  epochs={info['epochs']}  wall={info['wall_s']:.0f}s")
        if info["state"] is not None and (best is None or rmse < best["rmse"]):
            best = {"rmse": rmse, "cell": dict(cell), "state": info["state"], "history": history}

    if best is None:
        raise RuntimeError("no search cell produced a checkpoint")
    for k, v in best["cell"].items():
        cfg["smacnp"][k] = v
    torch.save(
        {
            "model": best["state"],
            "scaler": y_scaler.__dict__,
            "feature_scaler": feat_scaler.state_dict(),
            "feature_cols": feat_cols,
            "two_step": two_step,
            "variable": var,
            "time_resolution": cfg["time_resolution"],
            "cfg": cfg["smacnp"],
            "search_cell": best["cell"],
        },
        ckpt_path,
    )
    params_path = get_smacnp_tuned_params_path(var, cfg["time_resolution"])
    params_path.parent.mkdir(parents=True, exist_ok=True)
    import yaml
    with open(params_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"variable": var, "time_resolution": cfg["time_resolution"],
                         **best["cell"], "val_rmse": best["rmse"]}, f)
    pd.DataFrame(best["history"]).to_csv(
        ckpt_path.with_name(f"{var}_{cfg['time_resolution']}_curve.csv"), index=False,
    )
    print(f"\nbest {best['cell']}  val_rmse={best['rmse']:.3f}")
    print(f"saved {ckpt_path}")
    print(f"wrote {search_path}")


if __name__ == "__main__":
    main()
