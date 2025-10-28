"""
audio/beat_tracking.py
=======================

This module contains placeholder code for beat and downbeat tracking.
It defines a ``compute_beats`` function that should load an audio file,
optionally isolate drums using source separation, compute an onset strength
envelope and track beat and downbeat positions. It currently raises
``NotImplementedError``. Refer to the project documentation for
guidance on implementing this function.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import tempfile
import pathlib
import os

try:
    import librosa
except Exception:  # pragma: no cover
    librosa = None  # type: ignore

# Optional source separation libraries
try:  # pragma: no cover - heavy dependency
    from spleeter.separator import Separator  # type: ignore
except Exception:
    Separator = None  # type: ignore

try:  # pragma: no cover - heavy dependency
    import demucs.separate as demucs_separate  # type: ignore
except Exception:
    demucs_separate = None  # type: ignore

# Optional downbeat tracking with madmom
try:
    from madmom.features.downbeats import (
        DBNDownBeatTrackingProcessor,
        RNNDownBeatProcessor,
    )
except Exception:  # pragma: no cover
    DBNDownBeatTrackingProcessor = None  # type: ignore
    RNNDownBeatProcessor = None  # type: ignore


def compute_beats(audio_path: str, use_drums: bool = False) -> Dict[str, object]:
    """Compute beat and downbeat times from an audio file.

    Parameters
    ----------
    audio_path : str
        Path to an audio file on disk.
    use_drums : bool, optional
        If True, isolate the drums track before tracking beats. This can
        improve onset detection in music with dense instrumentation.

    Returns
    -------
    Dict[str, object]
        A dictionary containing the sample rate (``sr``), the onset strength
        envelope (``oenv``) and lists of beat times (``beat_times``) and
        downbeat times (``downbeat_times``). When implemented, all times
        should be given in seconds.

    Notes
    -----
    The current implementation is a placeholder and simply raises
    ``NotImplementedError``. Replace the body of this function with code
    that calls functions from ``librosa`` and ``madmom`` to extract beat and
    downbeat positions.
    """

    if librosa is None:
        raise RuntimeError("librosa is required for audio processing")

    # Optionally isolate drums/percussion to boost onset clarity
    source_audio_path = audio_path
    tmp_dir: Optional[tempfile.TemporaryDirectory[str]] = None
    try:
        if use_drums:
            # Try Spleeter first
            if Separator is not None:
                tmp_dir = tempfile.TemporaryDirectory()
                out_dir = pathlib.Path(tmp_dir.name)
                # 5stems: vocals, drums, bass, piano, other
                separator = Separator("spleeter:5stems")
                separator.separate_to_file(audio_path, out_dir.as_posix())
                # Locate drums file
                # Spleeter writes to out_dir/<basename>/drums.wav
                base = pathlib.Path(audio_path).stem
                candidate = out_dir / base / "drums.wav"
                if candidate.exists():
                    source_audio_path = candidate.as_posix()
            # If Spleeter not available, try demucs CLI entry
            elif demucs_separate is not None:
                tmp_dir = tempfile.TemporaryDirectory()
                out_parent = pathlib.Path(tmp_dir.name)
                # Run demucs separation programmatically
                # demucs writes into out_parent/separated/<model>/<basename>/drums.wav
                try:
                    demucs_separate.main([
                        "--two-stems",
                        "drums",
                        "-o",
                        out_parent.as_posix(),
                        audio_path,
                    ])
                    # Find drums file
                    sep_root = next((out_parent / "separated").glob("**/drums.wav"), None)
                    if sep_root is not None and sep_root.exists():
                        source_audio_path = sep_root.as_posix()
                except SystemExit:
                    # demucs may call sys.exit; ignore
                    pass

        # Load audio
        y, sr = librosa.load(source_audio_path, sr=None, mono=True)

        # Onset strength envelope
        hop_length = 512
        oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

        beat_times: List[float] = []
        downbeat_times: List[float] = []

        # Prefer madmom for downbeat tracking
        if RNNDownBeatProcessor is not None and DBNDownBeatTrackingProcessor is not None:
            try:
                act = RNNDownBeatProcessor()(source_audio_path)
                dbn = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
                # Each row: [time, beat_index] where beat_index 1=downbeat, else 0
                # madmom returns array [[t, beat_idx], ...]
                # Note: Some versions return (time, beat) pairs directly; handle both
                proc = dbn(act)
                # proc shape: (N, 2), first column: time, second: beat index (1=downbeat)
                for t, b in proc:
                    beat_times.append(float(t))
                    if int(round(b)) == 1:
                        downbeat_times.append(float(t))
            except Exception:
                beat_times, downbeat_times = _fallback_librosa_beats(y, sr, hop_length)
        else:
            beat_times, downbeat_times = _fallback_librosa_beats(y, sr, hop_length)

        return {
            "sr": int(sr),
            "oenv": oenv,
            "beat_times": beat_times,
            "downbeat_times": downbeat_times,
        }
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()


def _fallback_librosa_beats(y: np.ndarray, sr: int, hop_length: int) -> tuple[List[float], List[float]]:
    """Fallback beat tracking using librosa when madmom is unavailable.

    Returns beat times, and an empty downbeat list.
    """
    # Tempo and beat frames
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length).tolist()
    return list(map(float, beat_times)), []