"""
utils/io.py
============

Helper functions for reading and writing audio and video files.

This module provides convenience wrappers around common I/O tasks used
throughout the project, such as extracting the audio track from a
video, loading raw audio samples and iterating over video frames.

The implementations provided here are placeholders and raise
``NotImplementedError``. Replace them with code that uses libraries
such as ``ffmpeg``, ``moviepy``, ``librosa`` and ``opencv-python`` to
perform the actual I/O.
"""

from __future__ import annotations

from typing import Iterable, Tuple, Generator

import numpy as np

import pathlib
from contextlib import contextmanager

try:
    import librosa
except Exception:  # pragma: no cover
    librosa = None  # type: ignore

try:
    # moviepy relies on ffmpeg being available in PATH
    from moviepy.editor import VideoFileClip
except Exception:  # pragma: no cover
    VideoFileClip = None  # type: ignore

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


def extract_audio_from_video(video_path: str, output_path: str) -> None:
    """Extract the audio track from a video file.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    output_path : str
        Path to the output audio file (e.g. ``.wav``).

    Notes
    -----
    This function is a placeholder. Implement it using ``ffmpeg`` or
    ``moviepy`` to demux the audio stream from ``video_path`` and
    save it to ``output_path``.
    """

    # Prefer moviepy for portability; falls back to raising if unavailable
    vpath = str(video_path)
    out = str(output_path)
    path_obj = pathlib.Path(out)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    if VideoFileClip is not None:
        # Write a WAV file to avoid codec surprises downstream
        with VideoFileClip(vpath) as clip:
            if clip.audio is None:
                raise RuntimeError("No audio stream found in the video")
            # default fps=44100; write as PCM WAV
            clip.audio.write_audiofile(out, fps=44100, codec="pcm_s16le", verbose=False, logger=None)
            return

    # Fallback to ffmpeg CLI if available
    import shutil
    import subprocess
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("Neither moviepy nor ffmpeg are available to extract audio")
    cmd = [ffmpeg, "-y", "-i", vpath, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", out]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to extract audio: {proc.stderr.decode(errors='ignore')[:300]}")


def load_audio(audio_path: str) -> Tuple[np.ndarray, int]:
    """Load an audio file and return samples and sample rate.

    Parameters
    ----------
    audio_path : str
        Path to an audio file.

    Returns
    -------
    Tuple[np.ndarray, int]
        A tuple ``(y, sr)`` where ``y`` is a 1D array of audio samples
        and ``sr`` is the sample rate in Hz.

    Notes
    -----
    This is a placeholder implementation. Replace with code that uses
    ``librosa.load`` or a similar function to read audio from
    ``audio_path``.
    """

    if librosa is None:
        raise RuntimeError("librosa is not installed; cannot load audio")
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    return y, int(sr)


def iter_video_frames(video_path: str) -> Iterable[np.ndarray]:
    """Iterate over frames in a video file.

    Parameters
    ----------
    video_path : str
        Path to a video file.

    Yields
    ------
    np.ndarray
        Video frames as NumPy arrays (e.g. RGB images).

    Notes
    -----
    This is a placeholder generator. Replace with code that uses
    ``opencv-python`` or ``moviepy`` to iterate over frames in
    ``video_path``.
    """

    if cv2 is None:
        raise RuntimeError("opencv-python is not installed; cannot read video frames")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            # Convert BGR (OpenCV) to RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            yield frame_rgb
    finally:
        cap.release()


def get_video_fps(video_path: str) -> float:
    """Return frames-per-second (FPS) for the given video file.

    Parameters
    ----------
    video_path : str
        Path to a video file on disk.

    Returns
    -------
    float
        Frames per second of the video stream.
    """
    if cv2 is None:
        raise RuntimeError("opencv-python is not installed; cannot read video FPS")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    finally:
        cap.release()
    return float(fps)