"""
video/footstrike_detector.py
============================

This module contains placeholder code for detecting heel strikes (foot
contacts) from pose keypoint sequences. The primary function,
``detect_heel_strikes``, should take a sequence of 2D ankle positions and
detect local minima corresponding to foot contacts with the ground.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

try:
    from scipy.signal import find_peaks
except Exception:  # pragma: no cover
    find_peaks = None  # type: ignore

from utils.smoothing import smooth_signal


def detect_heel_strikes(times: List[float], ankle_x: List[float], ankle_y: List[float], fps: float) -> List[float]:
    """Detect times of heel strikes from ankle trajectories.

    Parameters
    ----------
    times : List[float]
        Timestamps for each frame in seconds.
    ankle_x : List[float]
        Horizontal positions of the ankle in pixel coordinates.
    ankle_y : List[float]
        Vertical positions of the ankle in pixel coordinates.
    fps : float
        Frames per second of the video.

    Returns
    -------
    List[float]
        A list of timestamps (in seconds) at which heel strikes occur.

    Notes
    -----
    Heuristic method:
      - Smooth ankle trajectories with Savitzky–Golay.
      - Compute velocities and accelerations via numerical gradients.
      - Detect local minima in vertical position (y) using ``find_peaks`` on
        the inverted signal.
      - Filter candidates by requiring large acceleration magnitude and low
        horizontal velocity.
      - Apply a refractory period of at least 250 ms.
    """

    if len(times) == 0:
        return []

    # Convert to arrays
    t = np.asarray(times, dtype=float)
    x = np.asarray(ankle_x, dtype=float)
    y = np.asarray(ankle_y, dtype=float)

    # Smoothing: choose window roughly ~0.15s long, odd number of samples
    if fps <= 0:
        fps = 30.0
    wl = int(max(5, round(0.15 * fps)))
    if wl % 2 == 0:
        wl += 1
    x_s = smooth_signal(x, window_length=wl, polyorder=2)
    y_s = smooth_signal(y, window_length=wl, polyorder=2)

    # Velocities and accelerations (finite differences)
    dt = np.gradient(t)
    vx = np.gradient(x_s, t, edge_order=2)
    vy = np.gradient(y_s, t, edge_order=2)
    ax = np.gradient(vx, t, edge_order=2)
    ay = np.gradient(vy, t, edge_order=2)
    a_mag = np.sqrt(ax**2 + ay**2)

    # Candidate local minima in vertical position (foot closest to ground)
    # In image coordinates y increases downward; heel strike tends to be a local
    # minimum in y when the foot is planted and body rises. We still use minima.
    if find_peaks is None:
        # Simple fallback: minima by comparing neighbors
        inv_y = -y_s
        candidates = [i for i in range(1, len(inv_y) - 1) if inv_y[i] > inv_y[i - 1] and inv_y[i] > inv_y[i + 1]]
    else:
        inv_y = -y_s
        distance_samples = max(1, int(round(0.25 * fps)))  # ~250 ms refractory as min distance
        peaks, _ = find_peaks(inv_y, distance=distance_samples)
        candidates = peaks.tolist()

    if len(candidates) == 0:
        return []

    # Heuristic thresholds based on percentiles
    a_thr = float(np.percentile(a_mag[candidates], 60)) if len(candidates) > 0 else float(np.median(a_mag))
    vx_abs = np.abs(vx)
    vx_thr = float(np.percentile(vx_abs[candidates], 40)) if len(candidates) > 0 else float(np.median(vx_abs))

    # Filter candidates
    accepted_idx: List[int] = []
    last_accept_time = -1e9
    refractory = 0.25  # seconds
    for i in candidates:
        if a_mag[i] >= a_thr and vx_abs[i] <= vx_thr:
            if t[i] - last_accept_time >= refractory:
                accepted_idx.append(i)
                last_accept_time = t[i]

    return [float(t[i]) for i in accepted_idx]