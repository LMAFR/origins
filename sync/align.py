"""
sync/align.py
==============

Functions for aligning detected foot strikes to beat times and computing
desynchronisation metrics. This module defines ``nearest_beat_delta`` and
``evaluate_steps``, which respectively compute the time difference between a
step and its nearest beat and summarise the timing accuracy over a set of
steps.
"""

from __future__ import annotations

from bisect import bisect_left
from typing import List, Tuple


def nearest_beat_delta(t: float, beats: List[float]) -> Tuple[float, float]:
    """Find the nearest beat to a given timestamp and return the difference.

    Parameters
    ----------
    t : float
        Timestamp of a step in seconds.
    beats : List[float]
        Sorted list of beat timestamps in seconds.

    Returns
    -------
    Tuple[float, float]
        A tuple ``(delta, beat_time)`` where ``delta`` is the signed time
        difference between ``t`` and the nearest beat, and ``beat_time`` is
        the timestamp of that beat.

    Notes
    -----
    This function is currently a placeholder and does not perform any
    computation. Replace the body with code that finds the nearest beat in
    ``beats`` to ``t`` and returns the signed difference.
    """

    if not beats:
        return 0.0, t
    i = bisect_left(beats, t)
    if i == 0:
        beat_time = beats[0]
    elif i == len(beats):
        beat_time = beats[-1]
    else:
        # Choose closer of beats[i-1] and beats[i]
        before = beats[i - 1]
        after = beats[i]
        beat_time = before if (t - before) <= (after - t) else after
    delta = t - beat_time
    return float(delta), float(beat_time)


def evaluate_steps(step_times: List[float], beat_times: List[float], tau_ms: float = 60.0) -> Tuple[List[dict], float]:
    """Evaluate timing accuracy of steps relative to beats.

    Parameters
    ----------
    step_times : List[float]
        Times of detected foot strikes (in seconds).
    beat_times : List[float]
        Times of detected beats (in seconds).
    tau_ms : float, optional
        Standard deviation (in milliseconds) of the Gaussian used to compute
        per-step scores. Defaults to 60 ms.

    Returns
    -------
    Tuple[List[dict], float]
        A list of dictionaries with per-step metrics (time of step, nearest
        beat time, delta in milliseconds and score) and the overall mean
        score.

    Notes
    -----
    This function is currently a placeholder and raises
    ``NotImplementedError``. Replace the body with code that iterates over
    ``step_times``, computes the nearest beat and delta using
    ``nearest_beat_delta``, converts the delta to milliseconds, computes a
    Gaussian score and aggregates the results.
    """

    from scoring.metrics import score_delta_ms

    per_step: List[dict] = []
    if not step_times or not beat_times:
        return per_step, 0.0

    for st in step_times:
        d, bt = nearest_beat_delta(st, beat_times)
        delta_ms = d * 1000.0
        score = score_delta_ms(delta_ms, tau_ms=tau_ms)
        per_step.append({
            "step_time": float(st),
            "nearest_beat_time": float(bt),
            "delta_ms": float(delta_ms),
            "score": float(score),
        })

    mean_score = float(sum(p["score"] for p in per_step) / len(per_step)) if per_step else 0.0
    return per_step, mean_score