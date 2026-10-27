#!/usr/bin/env python3
"""Shared experiment helpers for synthetic recovery tests.

Six checks, same numbers for every method:
  1. covariate ablation (caller supplies with/without preds)
  2. baseline: training-set mean and plain IDW
  3. two-step gate diagnostic
  4. multi-field aggregate
  5. neural loss curve (caller supplies history)
  6. held-out leak check
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_recovery(obs, pred, var: str, allow_constant: bool = False) -> dict:
    obs = np.asarray(obs, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if not np.isfinite(pred).all():
        raise AssertionError(f"{var}: non-finite predictions")
    if not np.isfinite(obs).all():
        raise AssertionError(f"{var}: non-finite observations")
    if (not allow_constant) and np.std(pred) < 1e-8:
        raise AssertionError(f"{var}: constant predictions (std={np.std(pred):.2e})")
    err = pred - obs
    out = {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "r": float(np.corrcoef(obs, pred)[0, 1]) if np.std(pred) > 1e-8 else 0.0,
        "n": int(len(obs)),
    }
    v = var.lower()
    if v.startswith("precip") or v.startswith("snow"):
        wet = obs > 0.1
        both = wet & (pred > 0.1)
        out["wet_n"] = int(wet.sum())
        out["wetwet_n"] = int(both.sum())
        out["wet_rmse"] = (
            float(np.sqrt(np.mean((pred[wet] - obs[wet]) ** 2))) if wet.sum() >= 5 else np.nan
        )
        out["wetwet_rmse"] = (
            float(np.sqrt(np.mean((pred[both] - obs[both]) ** 2))) if both.sum() >= 5 else np.nan
        )
    return out


def print_score(method: str, var: str, met: dict, held: int, tag: str = "") -> None:
    extra = ""
    if "wet_rmse" in met and np.isfinite(met.get("wet_rmse", np.nan)):
        extra += f"  wet_rmse={met['wet_rmse']:.3f}"
    if "wetwet_rmse" in met and np.isfinite(met.get("wetwet_rmse", np.nan)):
        extra += f"  amt_rmse={met['wetwet_rmse']:.3f}"
    label = f"{method:8s} {var:12s}"
    if tag:
        label += f" [{tag}]"
    print(f"{label}  rmse={met['rmse']:.3f}  mae={met['mae']:.3f}  "
          f"r={met['r']:.2f}{extra}  n={met['n']}  held={held}")


def assert_no_leak(donors: pd.DataFrame, held_names: set) -> None:
    """6. Held-out stations must not appear in the fit set."""
    leak = set(donors["station_name"].unique()) & set(held_names)
    if leak:
        raise AssertionError(f"held-out leak: {sorted(leak)[:5]}")


def mean_baseline(donors: pd.DataFrame, queries: pd.DataFrame, var: str) -> np.ndarray:
    mu = float(donors[var].mean())
    return np.full(len(queries), mu, dtype=np.float64)


def idw_baseline(donors: pd.DataFrame, queries: pd.DataFrame, var: str, k: int = 12) -> np.ndarray:
    """Plain inverse-distance (p=2) on x,y. No elevation."""
    dx = queries["x"].to_numpy()[:, None] - donors["x"].to_numpy()[None, :]
    dy = queries["y"].to_numpy()[:, None] - donors["y"].to_numpy()[None, :]
    d = np.hypot(dx, dy)
    d = np.maximum(d, 1.0)
    kk = min(k, d.shape[1])
    idx = np.argpartition(d, kk - 1, axis=1)[:, :kk]
    rows = np.arange(len(queries))[:, None]
    dd = d[rows, idx]
    w = 1.0 / dd ** 2
    z = donors[var].to_numpy()[idx]
    return (w * z).sum(axis=1) / w.sum(axis=1)


def gate_diagnostic(obs, pred, var: str, p_wet=None, tau: float = 0.5) -> dict:
    """3. Two-step gate: donor wet fraction, fraction gated on, amount RMSE."""
    obs = np.asarray(obs, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    wet_obs = obs > 0.1
    out = {
        "obs_wet_frac": float(wet_obs.mean()) if len(obs) else 0.0,
        "pred_pos_frac": float((pred > 0.1).mean()) if len(pred) else 0.0,
    }
    if p_wet is not None:
        p = np.asarray(p_wet, dtype=np.float64)
        out["frac_p_ge_tau"] = float((p >= tau).mean())
        out["p_wet_mean"] = float(np.nanmean(p))
    both = wet_obs & (pred > 0.1)
    if both.sum() >= 5:
        out["amt_rmse"] = float(np.sqrt(np.mean((pred[both] - obs[both]) ** 2)))
        out["amt_n"] = int(both.sum())
    else:
        out["amt_rmse"] = np.nan
        out["amt_n"] = int(both.sum())
    return out


def print_gate(method: str, var: str, g: dict) -> None:
    bits = [f"obs_wet={g['obs_wet_frac']:.2f}", f"pred_pos={g['pred_pos_frac']:.2f}"]
    if "frac_p_ge_tau" in g:
        bits.append(f"p>tau={g['frac_p_ge_tau']:.2f}")
    if np.isfinite(g.get("amt_rmse", np.nan)):
        bits.append(f"amt_rmse={g['amt_rmse']:.2f}(n={g['amt_n']})")
    print(f"  gate {method:8s} {var:12s}  " + "  ".join(bits))


def aggregate(mets: list[dict]) -> dict:
    """4. Mean of per-field scores."""
    if not mets:
        return {}
    keys = [k for k in mets[0] if k != "n"]
    out = {k: float(np.nanmean([m[k] for m in mets if k in m])) for k in keys}
    out["n"] = int(sum(m["n"] for m in mets))
    out["n_fields"] = len(mets)
    return out


def print_baselines(var: str, obs, mean_hat, idw_hat, method_hat) -> None:
    """2. Method must beat the mean; report IDW too."""
    def _r(a, b):
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])
    def _rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))
    print(f"  base {var:12s}  mean_rmse={_rmse(obs, mean_hat):.3f}  "
          f"idw_rmse={_rmse(obs, idw_hat):.3f}  method_rmse={_rmse(obs, method_hat):.3f}  "
          f"idw_r={_r(obs, idw_hat):.2f}")


def print_ablation(var: str, full: dict, dropped: dict, covar: str) -> None:
    """1. Recovery should drop when the carrying covariate is removed."""
    print(f"  ablate {var:12s}  full_r={full['r']:.2f}  no_{covar}_r={dropped['r']:.2f}  "
          f"delta={full['r'] - dropped['r']:+.2f}")


def print_loss_curve(method: str, history: list[float]) -> None:
    """5. Train loss over epochs."""
    if not history:
        return
    head = " ".join(f"{v:.3f}" for v in history[:8])
    tail = f"{history[-1]:.3f}"
    fell = history[-1] < history[0] - 1e-6
    print(f"  curve {method:8s}  n={len(history)}  first8=[{head}]  last={tail}  fell={fell}")
