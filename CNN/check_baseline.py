"""One-day check. Computes the point IDW from the pool stations itself.

Does not trust the cache. Prints the old cache too, so a stale file is obvious.
"""
import sys
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from cnn_data import load_cnn_config, load_panel
from cnn_core import idw_fz_points
from shared.splits.evaluation_sets import apply_evaluation_protocol
from train_cnn import _idw_baseline


def main():
    var = "precip_sum"
    res = "daily"
    cfg = load_cnn_config()
    panel = load_panel(cfg, var)
    tagged, _, _, _ = apply_evaluation_protocol(panel)
    dev = tagged[(tagged["is_heldout"]) & (tagged["split"] == "tune")]
    pool = tagged[~tagged["is_heldout"]]
    p, fz, k = _idw_baseline(var, res)
    print(f"baseline IDW p={p} Fz={fz} k={k}")

    ts = dev.groupby("time").size().sort_values(ascending=False).index[0]
    part = dev[dev["time"] == ts]
    don = pool[pool["time"] == ts]
    print(f"day {ts}  held-out={len(part)}  pool={len(don)}")
    print(f"  held-out mean={part[var].mean():.2f}  pool mean={don[var].mean():.2f}")

    obs = part[var].to_numpy()
    pts = idw_fz_points(
        don["x"].to_numpy(), don["y"].to_numpy(), don["elev"].to_numpy(), don[var].to_numpy(),
        part["x"].to_numpy(), part["y"].to_numpy(), part["elev"].to_numpy(),
        p=p, Fz=fz, k=k, exclude_self=False,
    )
    err = obs - pts
    print(f"  point IDW from pool  mean={float(pts.mean()):.2f}  "
          f"RMSE={float(np.sqrt(np.mean(err**2))):.2f}")
    print(f"  predict zero RMSE={float(np.sqrt(np.mean(obs**2))):.2f}")
    print("  a good baseline is close to the held-out mean, not to 3 mm")


if __name__ == "__main__":
    main()
