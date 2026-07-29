#!/usr/bin/env python3
"""
compare_regimes.py
Compare regime clustering results (kmeans / gmm / som) and write diagnostic plots.

Reads artefacts produced by identify_regimes.py and writes figures to
Plots/Clusters/{resolution}/ via paths.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    silhouette_score,
)

THIS_DIR = Path(__file__).resolve().parent
CODE_DIR = THIS_DIR.parents[1]
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(THIS_DIR))

from paths import get_clusters_dir, get_clusters_plot_dir, ensure_dir

# clustering helpers (works from shared/clustering or shared/plotting)
_CLUSTERING_DIR = THIS_DIR if (THIS_DIR / "clustering_core.py").exists() else THIS_DIR.parent / "clustering"
sys.path.insert(0, str(_CLUSTERING_DIR))
try:
    from clustering_core import parse_time_label
except ImportError:
    parse_time_label = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Compare regime clustering results")
    p.add_argument("--method", default="IDW",
                   help="Method folder to read cluster artefacts from "
                        "(results are identical across methods)")
    p.add_argument("--resolution", default="half_hourly")
    p.add_argument("--cluster-methods", nargs="+",
                   default=["gmm", "kmeans", "som"],
                   help="Clustering algorithms to compare")
    p.add_argument("--variables", nargs="+", default=None,
                   help="Subset of variables (default = all found)")
    p.add_argument("--max-timeline-days", type=int, default=60,
                   help="For timeline plot: show at most this many days "
                        "(centred on a dense period). 0 = full series (slow).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _cluster_dir(method: str, resolution: str, cluster_method: str) -> Path:
    return get_clusters_dir(method) / resolution / cluster_method


def discover_variables(method: str, resolution: str,
                       cluster_methods: List[str]) -> List[str]:
    found = set()
    skip = {"all_regime"}  # combined file, not a real variable
    for cm in cluster_methods:
        d = _cluster_dir(method, resolution, cm)
        if not d.exists():
            continue
        for f in d.glob("*_assignments.parquet"):
            name = f.name.replace("_assignments.parquet", "")
            if name not in skip:
                found.add(name)
    return sorted(found)


def load_assignments(method: str, resolution: str, cluster_method: str,
                     var: str) -> Optional[pd.DataFrame]:
    path = _cluster_dir(method, resolution, cluster_method) / f"{var}_assignments.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    # rename generic time column to timestamp for downstream plots
    if col != "timestamp":
        df = df.rename(columns={col: "timestamp"})

    raw = df["timestamp"]
    if pd.api.types.is_datetime64_any_dtype(raw):
        ts = raw
        if getattr(ts.dt, "tz", None) is not None:
            ts = ts.dt.tz_convert(None)
    elif parse_time_label is not None:
        ts = raw.map(lambda x: parse_time_label(x, resolution))
    else:
        ts = pd.to_datetime(raw, errors="coerce")

    df["timestamp"] = ts
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def load_features(method: str, resolution: str, cluster_method: str,
                  var: str) -> Optional[pd.DataFrame]:
    path = _cluster_dir(method, resolution, cluster_method) / f"{var}_features.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df


def load_summary(method: str, resolution: str, cluster_method: str,
                 var: str) -> Optional[dict]:
    path = _cluster_dir(method, resolution, cluster_method) / f"{var}_summary.yaml"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def pairwise_agreement(labels_a: np.ndarray, labels_b: np.ndarray) -> Tuple[float, float]:
    ari = float(adjusted_rand_score(labels_a, labels_b))
    nmi = float(normalized_mutual_info_score(labels_a, labels_b))
    return ari, nmi


def run_lengths(labels: np.ndarray) -> np.ndarray:
    """Lengths of consecutive identical-label runs."""
    if len(labels) == 0:
        return np.array([])
    lengths = []
    run = 1
    for i in range(1, len(labels)):
        if labels[i] == labels[i - 1]:
            run += 1
        else:
            lengths.append(run)
            run = 1
    lengths.append(run)
    return np.asarray(lengths, dtype=int)


def transition_matrix(labels: np.ndarray) -> np.ndarray:
    ids = np.unique(labels)
    idx = {c: i for i, c in enumerate(ids)}
    n = len(ids)
    T = np.zeros((n, n), dtype=float)
    for a, b in zip(labels[:-1], labels[1:]):
        T[idx[a], idx[b]] += 1
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return T / row_sums


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _save(fig, path: Path):
    ensure_dir(path.parent)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


# ---- 1. Agreement ---------------------------------------------------------

def plot_ari_nmi(assignments: Dict[str, pd.DataFrame], var: str, out_dir: Path):
    methods = sorted(assignments.keys())
    n = len(methods)
    if n < 2:
        return
    ari = np.zeros((n, n))
    nmi = np.zeros((n, n))
    # align on common timestamps
    base = assignments[methods[0]][["timestamp"]].copy()
    aligned = {}
    for m in methods:
        merged = base.merge(
            assignments[m][["timestamp", "cluster_id"]],
            on="timestamp", how="inner",
        )
        aligned[m] = merged["cluster_id"].to_numpy()

    for i, mi in enumerate(methods):
        for j, mj in enumerate(methods):
            if len(aligned[mi]) == 0:
                continue
            a, b = pairwise_agreement(aligned[mi], aligned[mj])
            ari[i, j] = a
            nmi[i, j] = b

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, mat, title in zip(axes, [ari, nmi], ["Adjusted Rand Index", "NMI"]):
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.set_yticklabels(methods)
        ax.set_title(title)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="white" if mat[i, j] < 0.5 else "black", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Cross-method agreement — {var}", fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir / "agreement" / f"ari_nmi_{var}.png")


def plot_contingency(assignments: Dict[str, pd.DataFrame], var: str, out_dir: Path):
    methods = sorted(assignments.keys())
    if len(methods) < 2:
        return
    base_ts = assignments[methods[0]][["timestamp"]]
    for i, mi in enumerate(methods):
        for mj in methods[i + 1:]:
            a = base_ts.merge(assignments[mi][["timestamp", "cluster_id"]], on="timestamp")
            b = base_ts.merge(assignments[mj][["timestamp", "cluster_id"]], on="timestamp")
            merged = a.merge(b, on="timestamp", suffixes=("_a", "_b"))
            if merged.empty:
                continue
            ct = pd.crosstab(merged["cluster_id_a"], merged["cluster_id_b"])
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(ct.values, cmap="Blues", aspect="auto")
            ax.set_xticks(range(ct.shape[1]))
            ax.set_yticks(range(ct.shape[0]))
            ax.set_xticklabels(ct.columns)
            ax.set_yticklabels(ct.index)
            ax.set_xlabel(mj)
            ax.set_ylabel(mi)
            ax.set_title(f"Contingency — {var}: {mi} vs {mj}")
            for r in range(ct.shape[0]):
                for c in range(ct.shape[1]):
                    ax.text(c, r, str(ct.values[r, c]), ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout()
            _save(fig, out_dir / "agreement" / f"contingency_{var}_{mi}_vs_{mj}.png")


def plot_cluster_sizes(summaries: Dict[str, dict], var: str, out_dir: Path):
    methods = sorted(summaries.keys())
    if not methods:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / max(len(methods), 1)
    for i, m in enumerate(methods):
        stats = summaries[m].get("cluster_stats", {})
        ids = sorted(int(k) for k in stats.keys())
        sizes = [stats[str(c)]["size"] if str(c) in stats else stats[c]["size"]
                 for c in ids]
        # yaml may load keys as int
        sizes = []
        for c in ids:
            s = stats.get(c, stats.get(str(c), {}))
            sizes.append(s.get("size", 0))
        x = np.arange(len(ids)) + i * width
        ax.bar(x, sizes, width=width, label=m)
    ax.set_xlabel("cluster_id")
    ax.set_ylabel("n timestamps")
    ax.set_title(f"Cluster sizes — {var}")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir / "agreement" / f"cluster_sizes_{var}.png")


# ---- 2. Internal quality --------------------------------------------------

def plot_quality_metrics(features: Dict[str, pd.DataFrame],
                         assignments: Dict[str, pd.DataFrame],
                         var: str, out_dir: Path):
    rows = []
    for m, feat in features.items():
        if "cluster_id" not in feat.columns:
            continue
        feature_cols = [c for c in feat.columns
                        if c not in ("cluster_id", "dist_to_centroid", "n_stations")]
        X = np.array(feat[feature_cols].to_numpy(dtype=float), copy=True)
        # simple impute
        for j in range(X.shape[1]):
            col = X[:, j]
            med = np.nanmedian(col)
            if not np.isfinite(med):
                med = 0.0
            nan_mask = ~np.isfinite(col)
            if nan_mask.any():
                col[nan_mask] = med
                X[:, j] = col
        labels = feat["cluster_id"].to_numpy()
        if len(np.unique(labels)) < 2:
            continue
        try:
            db = float(davies_bouldin_score(X, labels))
            ch = float(calinski_harabasz_score(X, labels))
            sil = float(silhouette_score(X, labels, sample_size=min(5000, len(labels))))
        except Exception:
            continue
        # persistence
        asg = assignments.get(m)
        mean_life = float(np.mean(run_lengths(asg["cluster_id"].to_numpy()))) if asg is not None else np.nan
        rows.append({"method": m, "davies_bouldin": db, "calinski_harabasz": ch,
                     "silhouette": sil, "mean_lifetime_steps": mean_life})
    if not rows:
        return
    df = pd.DataFrame(rows)
    metrics = ["silhouette", "davies_bouldin", "calinski_harabasz", "mean_lifetime_steps"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 3.5))
    for ax, met in zip(axes, metrics):
        ax.bar(df["method"], df[met], color="steelblue")
        ax.set_title(met)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle(f"Internal quality — {var}", fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir / "quality" / f"quality_metrics_{var}.png")


def plot_persistence_hist(assignments: Dict[str, pd.DataFrame], var: str, out_dir: Path):
    for m, asg in assignments.items():
        lengths = run_lengths(asg["cluster_id"].to_numpy())
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(lengths, bins=min(50, max(10, lengths.max())), color="steelblue", edgecolor="white")
        ax.set_xlabel("run length (half-hour steps)")
        ax.set_ylabel("count")
        ax.set_title(f"Persistence — {var} / {m}  (mean={lengths.mean():.1f})")
        fig.tight_layout()
        _save(fig, out_dir / "quality" / f"persistence_hist_{var}_{m}.png")


def plot_transition_matrices(assignments: Dict[str, pd.DataFrame], var: str, out_dir: Path):
    for m, asg in assignments.items():
        labels = asg["cluster_id"].to_numpy()
        ids = np.unique(labels)
        T = transition_matrix(labels)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(T, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(len(ids)))
        ax.set_yticks(range(len(ids)))
        ax.set_xticklabels(ids)
        ax.set_yticklabels(ids)
        ax.set_xlabel("to cluster")
        ax.set_ylabel("from cluster")
        ax.set_title(f"Transition matrix — {var} / {m}")
        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                ax.text(j, i, f"{T[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        _save(fig, out_dir / "quality" / f"transition_matrix_{var}_{m}.png")


# ---- 3. Meteorological feature content ------------------------------------

def plot_feature_radar(summaries: Dict[str, dict], var: str, out_dir: Path):
    for m, summary in summaries.items():
        stats = summary.get("cluster_stats", {})
        if not stats:
            continue
        # collect feature names from first cluster
        first = next(iter(stats.values()))
        feat_names = list(first.get("feature_means", {}).keys())
        if not feat_names:
            continue
        # limit to at most 10 features for readability
        feat_names = feat_names[:10]
        n_feat = len(feat_names)
        angles = np.linspace(0, 2 * np.pi, n_feat, endpoint=False).tolist()
        angles += angles[:1]

        # standardise means across clusters for comparable radar
        means_mat = []
        cids = sorted(stats.keys(), key=lambda x: int(x))
        for c in cids:
            fm = stats[c].get("feature_means", {})
            means_mat.append([fm.get(f, np.nan) for f in feat_names])
        means_mat = np.array(means_mat, dtype=float)
        col_mean = np.nanmean(means_mat, axis=0)
        col_std = np.nanstd(means_mat, axis=0)
        col_std[col_std < 1e-12] = 1.0
        means_z = (means_mat - col_mean) / col_std

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        for i, c in enumerate(cids):
            vals = means_z[i].tolist()
            vals += vals[:1]
            ax.plot(angles, vals, label=f"c{c}")
            ax.fill(angles, vals, alpha=0.1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feat_names, fontsize=8)
        ax.set_title(f"Feature means (z-scored) — {var} / {m}", y=1.08)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
        fig.tight_layout()
        _save(fig, out_dir / "features" / f"radar_{var}_{m}.png")


def plot_feature_boxplots(features: Dict[str, pd.DataFrame], var: str, out_dir: Path):
    key_candidates = [
        "elev_corr", "elev_corr_all", "elev_corr_wet", "elev_gradient",
        "wet_fraction", "spatial_autocorr", "mean", "std",
        "high_low_ratio", "temp_corr", "zero_fraction",
    ]
    for m, feat in features.items():
        if "cluster_id" not in feat.columns:
            continue
        cols = [c for c in key_candidates if c in feat.columns]
        if not cols:
            cols = [c for c in feat.columns
                    if c not in ("cluster_id", "dist_to_centroid", "n_stations")][:4]
        n = len(cols)
        fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.5), squeeze=False)
        for ax, col in zip(axes[0], cols):
            data_by_c = [
                feat.loc[feat["cluster_id"] == c, col].dropna().values
                for c in sorted(feat["cluster_id"].unique())
            ]
            ax.boxplot(data_by_c, labels=sorted(feat["cluster_id"].unique()), showfliers=False)
            ax.set_title(col, fontsize=9)
            ax.set_xlabel("cluster")
        fig.suptitle(f"Feature boxplots — {var} / {m}", fontsize=11)
        fig.tight_layout()
        _save(fig, out_dir / "features" / f"boxplots_{var}_{m}.png")


# ---- 4. Temporal structure ------------------------------------------------

def plot_timeline(assignments: Dict[str, pd.DataFrame], var: str, out_dir: Path,
                  max_days: int = 60):
    methods = sorted(assignments.keys())
    if not methods:
        return

    # choose a window with high activity if max_days > 0
    ref = assignments[methods[0]].copy()
    if max_days > 0 and len(ref) > max_days * 48:
        # pick the window with most cluster switches (interesting period)
        switches = (ref["cluster_id"].values[1:] != ref["cluster_id"].values[:-1]).astype(int)
        window = max_days * 48
        cum = np.cumsum(switches)
        if len(cum) > window:
            scores = cum[window:] - np.concatenate([[0], cum[:-window - 1]])
            start_idx = int(np.argmax(scores))
            t0 = ref["timestamp"].iloc[start_idx]
            t1 = t0 + pd.Timedelta(days=max_days)
        else:
            t0, t1 = ref["timestamp"].iloc[0], ref["timestamp"].iloc[-1]
    else:
        t0, t1 = ref["timestamp"].iloc[0], ref["timestamp"].iloc[-1]

    fig, axes = plt.subplots(len(methods), 1, figsize=(14, 2.2 * len(methods)),
                             sharex=True, squeeze=False)
    for ax, m in zip(axes[:, 0], methods):
        asg = assignments[m]
        mask = (asg["timestamp"] >= t0) & (asg["timestamp"] <= t1)
        sub = asg.loc[mask]
        if sub.empty:
            continue
        labels = sub["cluster_id"].to_numpy()
        n_c = int(labels.max()) + 1 if len(labels) else 1
        cmap = plt.colormaps["tab10"].resampled(max(n_c, 3))
        ax.scatter(sub["timestamp"], np.zeros(len(sub)), c=labels,
                   cmap=cmap, marker="|", s=80, vmin=-0.5, vmax=max(n_c - 0.5, 2.5))
        ax.set_yticks([])
        ax.set_ylabel(m, rotation=0, labelpad=30, va="center")
        ax.set_xlim(t0, t1)
    axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.suptitle(f"Regime timeline — {var}  ({t0.date()} → {t1.date()})", fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir / "temporal" / f"timeline_{var}.png")


def plot_seasonal_occupancy(assignments: Dict[str, pd.DataFrame], var: str, out_dir: Path):
    for m, asg in assignments.items():
        df = asg.copy()
        df["month"] = df["timestamp"].dt.month
        pivot = df.groupby(["month", "cluster_id"]).size().unstack(fill_value=0)
        # normalise to fraction per month
        pivot = pivot.div(pivot.sum(axis=1), axis=0)
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(pivot.T.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
        ax.set_xticks(range(len(pivot.index)))
        ax.set_xticklabels(pivot.index)
        ax.set_yticks(range(len(pivot.columns)))
        ax.set_yticklabels(pivot.columns)
        ax.set_xlabel("month")
        ax.set_ylabel("cluster_id")
        ax.set_title(f"Seasonal occupancy — {var} / {m}")
        fig.colorbar(im, ax=ax, fraction=0.046, label="fraction")
        fig.tight_layout()
        _save(fig, out_dir / "temporal" / f"seasonal_occupancy_{var}_{m}.png")


def plot_diurnal_occupancy(assignments: Dict[str, pd.DataFrame], var: str, out_dir: Path):
    for m, asg in assignments.items():
        df = asg.copy()
        df["hour"] = df["timestamp"].dt.hour
        pivot = df.groupby(["hour", "cluster_id"]).size().unstack(fill_value=0)
        pivot = pivot.div(pivot.sum(axis=1), axis=0)
        fig, ax = plt.subplots(figsize=(9, 4))
        im = ax.imshow(pivot.T.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
        ax.set_xticks(range(len(pivot.index)))
        ax.set_xticklabels(pivot.index)
        ax.set_yticks(range(len(pivot.columns)))
        ax.set_yticklabels(pivot.columns)
        ax.set_xlabel("hour (UTC)")
        ax.set_ylabel("cluster_id")
        ax.set_title(f"Diurnal occupancy — {var} / {m}")
        fig.colorbar(im, ax=ax, fraction=0.046, label="fraction")
        fig.tight_layout()
        _save(fig, out_dir / "temporal" / f"diurnal_occupancy_{var}_{m}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    method = args.method.upper()
    resolution = args.resolution
    cluster_methods = args.cluster_methods

    print("=" * 72)
    print(f"Compare regimes | source method={method} | resolution={resolution}")
    print(f"cluster methods: {cluster_methods}")
    print("=" * 72)

    variables = args.variables or discover_variables(method, resolution, cluster_methods)
    if not variables:
        raise SystemExit(
            f"No assignment files found under "
            f"{get_clusters_dir(method) / resolution} for {cluster_methods}"
        )
    print(f"Variables: {variables}")

    out_dir = get_clusters_plot_dir(resolution)
    print(f"Plot output: {out_dir}")

    for var in variables:
        print(f"\n----- {var} -----")
        assignments = {}
        features = {}
        summaries = {}
        for cm in cluster_methods:
            asg = load_assignments(method, resolution, cm, var)
            feat = load_features(method, resolution, cm, var)
            summary = load_summary(method, resolution, cm, var)
            if asg is not None:
                assignments[cm] = asg
            if feat is not None:
                features[cm] = feat
            if summary is not None:
                summaries[cm] = summary
            status = "OK" if asg is not None else "missing"
            print(f"  {cm}: {status}")

        if not assignments:
            print(f"  [SKIP] no assignments for {var}")
            continue

        # 1. Agreement
        plot_ari_nmi(assignments, var, out_dir)
        plot_contingency(assignments, var, out_dir)
        plot_cluster_sizes(summaries, var, out_dir)

        # 2. Quality
        plot_quality_metrics(features, assignments, var, out_dir)
        plot_persistence_hist(assignments, var, out_dir)
        plot_transition_matrices(assignments, var, out_dir)

        # 3. Features
        plot_feature_radar(summaries, var, out_dir)
        plot_feature_boxplots(features, var, out_dir)

        # 4. Temporal
        plot_timeline(assignments, var, out_dir, max_days=args.max_timeline_days)
        plot_seasonal_occupancy(assignments, var, out_dir)
        plot_diurnal_occupancy(assignments, var, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
