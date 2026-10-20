#!/usr/bin/env python3
"""
Shared CLC level-2 utilities.

Level 2 is the first two digits of the 3-digit CLC2018 code (15 classes).
All methods that use land cover should call these helpers so the grouping
is identical.

Representation guidance (applied in the method code):
  RFSI, Kriging, GAM, CNN, SMACNP  → one-hot / dummies
  RGI                              → learned embedding of a compact index
"""

from __future__ import annotations

import numpy as np

# 3-digit CLC → level-2 (15 classes)
CLC_LEVEL2 = {
    111: 11, 112: 11,
    121: 12, 122: 12, 123: 12, 124: 12,
    131: 13, 132: 13, 133: 13,
    141: 14, 142: 14,
    211: 21, 212: 21, 213: 21,
    221: 22, 222: 22, 223: 22,
    231: 23,
    241: 24, 242: 24, 243: 24, 244: 24,
    311: 31, 312: 31, 313: 31,
    321: 32, 322: 32, 323: 32, 324: 32,
    331: 33, 332: 33, 333: 33, 334: 33, 335: 33,
    411: 41, 412: 41,
    421: 42, 422: 42, 423: 42,
    511: 51, 512: 51,
    521: 52, 522: 52, 523: 52,
}

# Compact index: 0 = missing/other, 1..15 = the level-2 classes
LEVEL2_CLASSES = [11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 41, 42, 51, 52]
_LEVEL2_TO_INDEX = {c: i + 1 for i, c in enumerate(LEVEL2_CLASSES)}
N_LEVEL2 = 16  # index 0..15


def to_level2(codes) -> np.ndarray:
    """Map raw CLC codes to level-2 integers (0 if unknown)."""
    c = np.asarray(codes, dtype=np.int32).reshape(-1)
    out = np.zeros(c.shape, dtype=np.int32)
    for raw, lv in CLC_LEVEL2.items():
        out[c == raw] = lv
    # already level-2
    known = np.isin(c, LEVEL2_CLASSES)
    out[known] = c[known]
    return out


def level2_index(codes) -> np.ndarray:
    """Map raw or level-2 codes to compact index 0..15 (0 = missing)."""
    lv = to_level2(codes)
    idx = np.zeros(lv.shape, dtype=np.int32)
    for code, i in _LEVEL2_TO_INDEX.items():
        idx[lv == code] = i
    return idx


def one_hot_level2(codes, dtype=np.float32) -> np.ndarray:
    """One-hot of shape (n, 16). Column 0 is the missing/other class."""
    idx = level2_index(codes)
    return np.eye(N_LEVEL2, dtype=dtype)[idx]
