"""
scoring/metrics.py
==================

This module defines placeholder functions for computing timing metrics
based on the difference between foot strikes and beat times. Use these
functions to summarise the timing accuracy of a dancer over a window
of time or across the entire song.
"""

from __future__ import annotations

from typing import List, Dict
import math
import numpy as np


def score_delta_ms(delta_ms: float, tau_ms: float = 60.0) -> float:
    """Compute a Gaussian score for a timing error in milliseconds.

    Parameters
    ----------
    delta_ms : float
        Signed difference between a step and its nearest beat, in
        milliseconds.
    tau_ms : float, optional
        Standard deviation of the Gaussian scoring function. Smaller values
        penalise deviations more strongly. Defaults to 60 ms.

    Returns
    -------
    float
        A score between 0 and 1, where 1 indicates perfect synchrony.

    Notes
    -----
    Implements ``exp(-(abs(delta_ms) / tau_ms)**2)``. The score is in [0, 1].
    """
    if tau_ms <= 0:
        return 0.0
    return float(math.exp(-((abs(delta_ms) / float(tau_ms)) ** 2)))


def summarise_scores(per_step_scores: List[float]) -> Dict[str, float]:
    """Summarise a list of per-step scores.

    Parameters
    ----------
    per_step_scores : List[float]
        Individual timing scores for each step.

    Returns
    -------
    Dict[str, float]
        A dictionary with summary statistics such as the mean score,
        median score and standard deviation.

    Notes
    -----
    Returns mean, median, std and count.
    """
    if not per_step_scores:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "count": 0.0}
    arr = np.asarray(per_step_scores, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "count": float(arr.size),
    }