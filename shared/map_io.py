#!/usr/bin/env python3
"""Durable map writing: per-unit partials then merge.

Map-production scripts compute fields, then call save_map_with_partials
instead of a single ds.to_netcdf. Partials are written under
<out_file.parent>/.partials_<stem>/ so a hang or kill during the final
merge does not lose the computed fields.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr
from tqdm import tqdm


def _period_label(ts: pd.Timestamp, time_res: str) -> str:
    ts = pd.Timestamp(ts)
    if time_res == "monthly":
        return ts.strftime("%Y-%m")
    if time_res == "weekly":
        # ISO week
        y, w, _ = ts.isocalendar()
        return f"{y}-W{w:02d}"
    if time_res == "daily":
        return ts.strftime("%Y-%m-%d")
    # half_hourly and fallback: one file per timestamp
    return ts.strftime("%Y-%m-%dT%H%M%S")


def save_map_with_partials(
    ds: xr.Dataset,
    out_file: Path,
    time_res: str,
    *,
    show_progress: bool = True,
) -> None:
    """Write ds in time-resolution units, then merge into out_file.

    Partial files stay on disk after a successful merge so a later run
    can inspect them. The final file is what plotting expects.
    """
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    partials_dir = out_file.parent / f".partials_{out_file.stem}"
    partials_dir.mkdir(parents=True, exist_ok=True)

    if "time" not in ds.dims:
        # nothing to split
        ds.to_netcdf(out_file, engine="netcdf4")
        return

    times = pd.to_datetime(ds["time"].values)
    labels = [_period_label(t, time_res) for t in times]
    groups: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        groups.setdefault(lab, []).append(i)

    ordered = sorted(groups)
    iterator = tqdm(ordered, desc=f"  save partials ({time_res})", unit="file") if show_progress else ordered

    written = []
    for lab in iterator:
        idx = groups[lab]
        sub = ds.isel(time=idx)
        part_path = partials_dir / f"{lab}.nc"
        sub.to_netcdf(part_path, engine="netcdf4")
        written.append(part_path)

    if not written:
        return

    # merge without dask (open_mfdataset requires it)
    print(f"  merging {len(written)} partials → {out_file.name}")
    opened = [xr.open_dataset(p) for p in written]
    try:
        merged = xr.concat(opened, dim="time").sortby("time")
        merged.to_netcdf(out_file, engine="netcdf4")
    finally:
        for d in opened:
            d.close()
    print(f"  saved {out_file}")