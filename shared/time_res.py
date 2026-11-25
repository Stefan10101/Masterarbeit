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

TIME_RESOLUTIONS = ("monthly", "weekly", "daily", "six_hourly", "half_hourly")
DEFAULT_VARIABLES = (
    "temp_mean", "precip_sum", "wind_mean", "rh_mean", "snow_mean",
)


def variables_from_cfg(cfg: dict, override=None) -> list[str]:
    """CLI list wins. Otherwise top-level variables_to_process."""
    if override:
        return [str(v) for v in override]
    raw = (cfg or {}).get("variables_to_process") or list(DEFAULT_VARIABLES)
    return [str(v) for v in raw]

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
    "six_hourly": {
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


def cap_timestamps(panel: pd.DataFrame, max_times, time_res: str,
                   seed: int = 22, time_col: str = "time") -> pd.DataFrame:
    """Cap unique timestamps. max_times <= 0 or None means no cap.

    half_hourly: sample whole days so the 4 hour slots stay balanced and the
    total stays <= max_times (365 // 4 = 91 days = 364 slots).
    other resolutions: uniform random sample of timestamps.
    Call after the hour cut so half_hourly only sees the kept hours.
    """
    if panel is None or len(panel) == 0:
        return panel
    if not max_times or int(max_times) <= 0:
        return panel
    max_times = int(max_times)
    times = panel[time_col].drop_duplicates()
    if len(times) <= max_times:
        return panel
    rng = np.random.default_rng(int(seed))
    if str(time_res).lower() == "half_hourly":
        dt = pd.to_datetime(times)
        day = dt.dt.floor("D")
        counts = day.value_counts()
        n_per = int(counts.mode().iloc[0]) if len(counts) else 1
        n_days = max(1, max_times // max(n_per, 1))
        n_days = min(n_days, int(day.nunique()))
        chosen = set(rng.choice(day.drop_duplicates().to_numpy(), size=n_days, replace=False))
        keep = times.loc[day.isin(chosen)]
        if len(keep) > max_times:
            # rare: uneven slots/day. trim with equal hour counts.
            hours = pd.to_datetime(keep).dt.hour
            per_h = max(1, max_times // max(int(hours.nunique()), 1))
            picked = []
            for _, grp in keep.groupby(hours):
                take = min(per_h, len(grp))
                picked.extend(rng.choice(grp.to_numpy(), size=take, replace=False).tolist())
            if len(picked) > max_times:
                picked = rng.choice(np.asarray(picked), size=max_times, replace=False).tolist()
            keep = np.asarray(picked)
        else:
            keep = keep.to_numpy()
    else:
        keep = rng.choice(times.to_numpy(), size=max_times, replace=False)
    return panel.loc[panel[time_col].isin(keep)].copy()


def cap_score_split(panel: pd.DataFrame, max_times, time_res: str,
                     score_splits=("test",), seed: int = 22,
                     time_col: str = "time", split_col: str = "split") -> pd.DataFrame:
    """Cap only the score split (default test). Train/tune rows stay intact
    so pooled methods still fit on the full pool. half_hourly uses the
    day-balanced sampler inside cap_timestamps.
    """
    if panel is None or len(panel) == 0 or not max_times or int(max_times) <= 0:
        return panel
    if split_col not in panel.columns:
        return cap_timestamps(panel, max_times, time_res, seed, time_col)
    want = set(score_splits)
    score = panel[panel[split_col].isin(want)]
    rest = panel[~panel[split_col].isin(want)]
    score = cap_timestamps(score, max_times, time_res, seed, time_col)
    if rest.empty:
        return score
    return pd.concat([rest, score], ignore_index=True)


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
