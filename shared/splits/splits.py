#!/usr/bin/env python3
"""Load the shared train / dev / test time windows.

For the thesis evaluation protocol (stratified station hold-out +
quarter-based 70/15/15 splits) see evaluation_sets.py and
make_evaluation_sets.py in this directory.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import yaml

sys.path.append(str(Path(__file__).resolve().parents[2]))
from paths import get_time_splits_path


SPLITS = ("train", "dev", "test")


def load_time_splits(path: Path | None = None) -> dict:
    """Deprecated. Continuous windows were replaced by quarter_splits.yaml.

    Kept so old callers fail with a clear message instead of a missing file.
    Use label_times() or evaluation_sets.load_quarter_splits().
    """
    raise FileNotFoundError(
        "time_splits.yaml was removed. Use quarter_splits.yaml via "
        "label_times() or shared.splits.evaluation_sets.load_quarter_splits()."
    )
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
    """Map timestamps to train/dev/test/outside using quarter_splits.yaml.

    The tune quarters are labelled 'dev' so existing code that looks for
    the old DEV window hits the designated tune subset.
    """
    from shared.splits.evaluation_sets import load_quarter_splits, quarter_of
    splits = load_quarter_splits()
    q_to_split = {}
    for name, qs in splits.items():
        label = "dev" if name == "tune" else name
        for q in qs:
            q_to_split[q] = label
    t = pd.to_datetime(times)
    labels = [q_to_split.get(quarter_of(x), "outside") for x in t]
    return pd.Series(labels, index=range(len(labels)))


def in_splits(times, names) -> pd.Series:
    names = {names} if isinstance(names, str) else set(names)
    return label_times(times).isin(names)


def filter_to_split(timestamps, split: str = "dev") -> pd.DatetimeIndex:
    """Keep timestamps that fall in train/dev/test."""
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True)).tz_convert("UTC").tz_localize(None)
    labels = label_times(ts)
    return ts[labels.to_numpy() == split]
