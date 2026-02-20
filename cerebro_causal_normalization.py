#!/usr/bin/env python3
"""
CEREBRO CAUSAL NORMALIZATION — No future leakage
================================================
All transforms are causal: at year t, use stats computed from data ≤ t only.
Replaces full-series min/max and mean/std that contaminate with future info.
"""

import pandas as pd
import numpy as np
from typing import Union


def expanding_minmax_to_10pt(
    series: Union[pd.Series, dict],
    min_periods: int = 10,
    target_lo: float = -5.0,
    target_hi: float = 5.0,
) -> pd.Series:
    """
    Causal min-max normalization. At index t, use min/max of series[:t] only.
    Returns values in [target_lo, target_hi] scale. NaN until min_periods.
    """
    if isinstance(series, dict):
        s = pd.Series(series).sort_index()
    else:
        s = series.dropna().sort_index()
    if len(s) < min_periods:
        return pd.Series(index=s.index, dtype=float)
    out = []
    for i, (idx, val) in enumerate(s.items()):
        if i < min_periods - 1:
            out.append((idx, np.nan))
            continue
        hist = s.iloc[: i + 1].dropna()
        if len(hist) < min_periods:
            out.append((idx, np.nan))
            continue
        lo, hi = hist.min(), hist.max()
        if hi == lo:
            out.append((idx, 0.0))
        else:
            n = target_lo + (val - lo) / (hi - lo) * (target_hi - target_lo)
            out.append((idx, float(n)))
    return pd.Series({k: v for k, v in out}).reindex(s.index)


def expanding_zscore_to_10pt(
    series: Union[pd.Series, dict],
    min_periods: int = 10,
    clip: float = 3.0,
    target_lo: float = -5.0,
    target_hi: float = 5.0,
) -> pd.Series:
    """
    Causal z-score normalization. At index t, use mean/std of series[:t] only.
    z = (x - mean) / std, clipped to [-clip, clip], mapped to [target_lo, target_hi].
    NaN until min_periods.
    """
    if isinstance(series, dict):
        s = pd.Series(series).sort_index()
    else:
        s = series.dropna().sort_index()
    if len(s) < min_periods:
        return pd.Series(index=s.index, dtype=float)
    out = []
    for i, (idx, val) in enumerate(s.items()):
        if i < min_periods - 1:
            out.append((idx, np.nan))
            continue
        hist = s.iloc[: i + 1].dropna()
        if len(hist) < min_periods:
            out.append((idx, np.nan))
            continue
        mu, sig = hist.mean(), hist.std()
        if pd.isna(sig) or sig < 1e-9:
            out.append((idx, 0.0))
            continue
        z = (val - mu) / sig
        z = np.clip(z, -clip, clip)
        n = target_lo + (z + clip) / (2 * clip) * (target_hi - target_lo)
        out.append((idx, float(n)))
    return pd.Series({k: v for k, v in out}).reindex(s.index)


def causal_velocity(series: pd.Series, window: int = 1) -> pd.Series:
    """Causal velocity: pos[t] - pos[t-window]. No lookahead."""
    return series.diff(window)


def causal_acceleration(velocity: pd.Series, window: int = 1) -> pd.Series:
    """Causal acceleration: vel[t] - vel[t-window]. No lookahead."""
    return velocity.diff(window)


def norm_causal(
    series_dict: dict,
    invert: bool = False,
    min_periods: int = 10,
) -> pd.Series:
    """
    Causal normalization to [-1, +1]. Same interface as legacy norm() but uses
    expanding min/max (no future leakage). invert=True flips direction.
    """
    s = expanding_minmax_to_10pt(
        series_dict,
        min_periods=min_periods,
        target_lo=-1.0,
        target_hi=1.0,
    )
    if invert:
        s = -s
    return s
