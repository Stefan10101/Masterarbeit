#!/usr/bin/env python3
"""Load the shared train / dev / test time windows."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import yaml

sys.path.append(str(Path(__file__).resolve().parents[2]))
from paths import get_time_splits_path


SPLITS = ("train", "dev", "test")


def load_time_splits(path: Path | None = None) -> dict:
    path = Path(path) if path else get_time_splits_path()
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    purge = pd.Timedelta(days=int(raw.get("purge_days", 0) or 0))
    out = {"purge_days": int(purge.days), "windows": {}}
    prev_end = None
    for name in SPLITS:
        start = pd.Timestamp(raw[name]["start"])
        end = pd.Timestamp(raw[name]["end"])
        if prev_end is not None and purge.days > 0:
            start = max(start, prev_end + pd.Timedelta(days=1) + purge)
        if start > end:
            raise ValueError(f"{name} window empty after purge={purge.days}d")
        out["windows"][name] = (start, end)
        prev_end = end
    return out


def label_times(times) -> pd.Series:
    """Map timestamps to train/dev/test/outside."""
    spec = load_time_splits()
    t = pd.DatetimeIndex(pd.to_datetime(times)).tz_localize(None)
    labels = pd.Series("outside", index=range(len(t)))
    for name in SPLITS:
        start, end = spec["windows"][name]
        labels[(t >= start) & (t <= end)] = name
    return labels


def in_splits(times, names) -> pd.Series:
    names = {names} if isinstance(names, str) else set(names)
    return label_times(times).isin(names)


def filter_to_split(timestamps, split: str = "dev") -> pd.DatetimeIndex:
    """Keep timestamps that fall in train/dev/test."""
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True)).tz_convert("UTC").tz_localize(None)
    labels = label_times(ts)
    return ts[labels.to_numpy() == split]
