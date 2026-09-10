#!/usr/bin/env python3
"""Shared time-resolution switch and compute policy.

Configs stay on time_resolution: monthly. Override with --time-resolution.
Do not retune spatial grid in the same step. Do not share hyperparameters
across timescales.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

TIME_RESOLUTIONS = ("monthly", "weekly", "daily", "half_hourly")
_WEEK_RE = re.compile(r"^(\d{4})-W(\d{1,2})$")
_MONTH_RE = re.compile(r"^(\d{4})-(\d{2})$")

# Science vs operations (locked 2026-09-10)
# weekly   – seasonal4 tune, full nested LLOCV, full 2020-2025 maps
# daily    – seasonal4 tune, TEST + seasonal4 DEV LLOCV, TEST maps first
# hh       – subsampled hours for tune/LLOCV; no 6-year 1 km cubes
POLICY = {
    "monthly": {
        "purge_days": 0,
        "tune_months": "all",
        "llocv_months": "all",
        "maps": "full",
        "subsample_hours": None,
    },
    "weekly": {
        "purge_days": 0,
        "tune_months": "seasonal4",
        "llocv_months": "all",
        "maps": "full",
        "subsample_hours": None,
    },
    "daily": {
        "purge_days": 0,
        "tune_months": "seasonal4",
        "llocv_months": "seasonal4",
        "maps": "test",
        "subsample_hours": None,
    },
    "half_hourly": {
        "purge_days": 3,
        "tune_months": "seasonal4",
        "llocv_months": "seasonal4",
        "maps": "test_seasonal4",
        "subsample_hours": (0, 6, 12, 18),
    },
}


def add_hours_arg(parser, dest: str = "hours"):
    parser.add_argument(
        "--hours",
        dest=dest,
        default=None,
        help="UTC hours as 0,6,12,18 or 'all'. half_hourly default is 0,6,12,18.",
    )
    return parser


def parse_hours(spec):
    if spec is None or spec == "":
        return None
    if spec == "all":
        return "all"
    if isinstance(spec, (list, tuple)):
        return tuple(int(h) for h in spec)
    return tuple(int(x.strip()) for x in str(spec).split(",") if x.strip())


def default_hours(time_res: str, explicit=None):
    if explicit is not None:
        return parse_hours(explicit)
    if time_res != "half_hourly":
        return None
    return POLICY["half_hourly"]["subsample_hours"]


def subset_hours(panel: pd.DataFrame, hours, time_col: str = "time") -> pd.DataFrame:
    if hours is None or hours == "" or hours == "all":
        return panel
    if time_col not in panel.columns:
        return panel
    want = {int(h) for h in hours}
    if not want:
        return panel
    return panel.loc[pd.to_datetime(panel[time_col]).dt.hour.isin(want)].copy()


def apply_hour_cut(panel: pd.DataFrame, time_res: str, hours_arg=None, time_col: str = "time"):
    """Apply locked hh 00/06/12/18 cut unless --hours all."""
    hours = default_hours(time_res, hours_arg)
    return subset_hours(panel, hours, time_col=time_col), hours


def cap_jobs(jobs: int, n_rows: int, limit: int = 2, row_cut: int = 2_000_000) -> int:
    jobs = max(1, int(jobs))
    if n_rows > row_cut and jobs > limit:
        return limit
    return jobs


def add_time_res_arg(parser, dest: str = "time_resolution"):
    parser.add_argument(
        "--time-resolution",
        dest=dest,
        default=None,
        choices=list(TIME_RESOLUTIONS),
        help="Override config.yaml time_resolution (yaml default stays monthly).",
    )
    return parser


def apply_time_res(cfg: dict, args, attr: str = "time_resolution") -> str:
    val = getattr(args, attr, None)
    if val:
        cfg["time_resolution"] = val
    res = str(cfg.get("time_resolution") or "monthly")
    if res not in TIME_RESOLUTIONS and res != "seasonal":
        raise ValueError(f"unknown time_resolution={res!r}")
    return res


def policy_for(time_res: str) -> dict:
    return dict(POLICY.get(time_res, POLICY["monthly"]))


def infer_time_res(sample: pd.Series) -> str | None:
    text = sample.astype(str).str.strip()
    text = text[text.notna() & (text != "") & (text != "NaT")]
    if text.empty:
        return None
    head = text.head(32)
    if head.str.match(_WEEK_RE).mean() >= 0.5:
        return "weekly"
    if head.str.match(_MONTH_RE).mean() >= 0.5:
        return "monthly"
    return None


def parse_time_index(values, time_res: str | None = None) -> pd.DatetimeIndex:
    """Parse aggregated time keys to tz-naive UTC.

    weekly 'YYYY-Www' including W00 (clamped to week 1), monthly 'YYYY-MM',
    else pandas datetime. Matches clustering_core.parse_time_label.
    """
    s = pd.Series(values)
    if s.empty:
        return pd.DatetimeIndex([], dtype="datetime64[ns]")
    if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_datetime64tz_dtype(s):
        idx = pd.DatetimeIndex(pd.to_datetime(s, utc=True))
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        return idx

    text = s.astype(str).str.strip()
    res = time_res or infer_time_res(text)
    if res == "weekly":
        year = text.str.extract(r"^(\d{4})-W", expand=False)
        week = text.str.extract(r"-W(\d+)", expand=False)
        year_i = pd.to_numeric(year, errors="coerce")
        week_i = pd.to_numeric(week, errors="coerce").clip(lower=1, upper=53)
        stamp = (
            year_i.astype("Int64").astype(str)
            + "-W"
            + week_i.astype("Int64").astype(str).str.zfill(2)
            + "-1"
        )
        bad = year_i.isna() | week_i.isna()
        stamp = stamp.mask(bad, pd.NA)
        idx = pd.to_datetime(stamp, format="%Y-W%W-%w", utc=True, errors="coerce")
        idx = pd.DatetimeIndex(idx)
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        return idx
    if res == "monthly":
        idx = pd.to_datetime(text + "-01", utc=True, errors="coerce")
        idx = pd.DatetimeIndex(idx)
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        return idx
    idx = pd.DatetimeIndex(pd.to_datetime(text, utc=True, errors="coerce"))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx


def to_naive_utc(values, time_res: str | None = None) -> np.ndarray:
    return parse_time_index(values, time_res).to_numpy(dtype="datetime64[ns]")
