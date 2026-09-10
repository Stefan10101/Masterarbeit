#!/usr/bin/env python3
"""Shared time-resolution switch and compute policy.

Configs stay on time_resolution: monthly. Override with --time-resolution.
Do not retune spatial grid in the same step. Do not share hyperparameters
across timescales.
"""

from __future__ import annotations

TIME_RESOLUTIONS = ("monthly", "weekly", "daily", "half_hourly")

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
