"""
sync/av_sync.py
===============

Utilities for estimating and correcting audio/video offsets. When dealing
with recorded videos, there can be a constant offset between the
timestamps used for pose detection and the audio beat times. This module
exposes a ``estimate_offset`` function that should compute the offset
between two time series by minimising the error.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def estimate_offset(step_times: List[float], beat_times: List[float], search_range: Tuple[float, float] = (-0.2, 0.2), step: float = 0.01) -> float:
    """Estimate a global time offset between steps and beats.

    Parameters
    ----------
    step_times : List[float]
        Times of detected foot strikes (in seconds).
    beat_times : List[float]
        Times of detected beats (in seconds).
    search_range : Tuple[float, float], optional
        Lower and upper bound (in seconds) of offsets to search over.
    step : float, optional
        Step size (in seconds) when searching the offset space.

    Returns
    -------
    float
        The offset (in seconds) that minimises the mean absolute error between
        the shifted steps and the nearest beats.

    Notes
    -----
    This is currently a placeholder. An implementation might iterate over
    candidate offsets, compute the absolute differences between shifted
    ``step_times`` and the nearest ``beat_times``, and return the offset with
    the lowest error.
    """

    if not step_times or not beat_times:
        return 0.0

    steps = np.asarray(step_times, dtype=float)
    beats = np.asarray(beat_times, dtype=float)
    beats.sort()

    def mean_abs_error(off: float) -> float:
        shifted = steps + off
        # For each shifted step, compute distance to nearest beat efficiently
        idx = np.searchsorted(beats, shifted)
        # candidates: previous beat and next beat
        prev_idx = np.clip(idx - 1, 0, len(beats) - 1)
        next_idx = np.clip(idx, 0, len(beats) - 1)
        prev_dist = np.abs(shifted - beats[prev_idx])
        next_dist = np.abs(shifted - beats[next_idx])
        d = np.minimum(prev_dist, next_dist)
        return float(np.mean(d)) if d.size > 0 else float("inf")

    lo, hi = float(search_range[0]), float(search_range[1])
    step_sz = float(step)
    offsets = np.arange(lo, hi + 1e-9, step_sz)
    errors = np.array([mean_abs_error(float(o)) for o in offsets])
    best_idx = int(np.argmin(errors))
    return float(offsets[best_idx])