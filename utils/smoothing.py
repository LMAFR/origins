"""
utils/smoothing.py
===================

Signal smoothing utilities.

Many of the algorithms in this project operate on noisy time-series
data, such as ankle positions extracted from pose estimation. This
module defines helper functions for smoothing such data using
techniques like the Savitzky–Golay filter or low-pass Butterworth
filters.

The functions provided here are placeholders. Replace them with
implementations that call into ``scipy.signal`` or ``numpy`` as
appropriate.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

try:
    # Savitzky–Golay filter for robust smoothing
    from scipy.signal import savgol_filter
except Exception:  # pragma: no cover - fallback if SciPy import fails at runtime
    savgol_filter = None  # type: ignore


def smooth_signal(data: Iterable[float], window_length: int = 11, polyorder: int = 2) -> np.ndarray:
    """Smooth a 1D signal using a Savitzky–Golay filter.

    Parameters
    ----------
    data : Iterable[float]
        The input signal to smooth.
    window_length : int, optional
        Length of the filter window (must be odd and greater than polyorder).
        Defaults to 11.
    polyorder : int, optional
        Order of the polynomial used to fit the samples. Must be less
        than ``window_length``. Defaults to 2.

    Returns
    -------
    np.ndarray
        The smoothed signal as a NumPy array of the same length as
        ``data``.

    Notes
    -----
    Uses ``scipy.signal.savgol_filter`` when available; if SciPy is not
    available or inputs are too short for the requested window, this
    function returns the input as a NumPy array (no smoothing).
    """

    x = np.asarray(list(data), dtype=float)
    n = x.shape[0]
    if n == 0:
        return x

    # Ensure window_length is a valid odd integer and not longer than the data
    wl = int(window_length)
    if wl % 2 == 0:
        wl += 1  # make odd
    wl = max(3, wl)
    wl = min(wl, n if n % 2 == 1 else n - 1)  # must be odd and <= n
    po = int(polyorder)
    po = max(0, min(po, wl - 1))

    if savgol_filter is None or wl < 3 or n < 3:
        # Fallback: return original signal if smoothing not possible
        return x

    try:
        return savgol_filter(x, window_length=wl, polyorder=po, mode="interp")
    except Exception:
        # If anything goes wrong (e.g., numerical issues), return original
        return x