"""
video/pose_extractor.py
=======================

This module defines utilities for extracting human pose keypoints from
video frames. It currently exposes a single placeholder function,
``extract_poses``, which should be replaced with an implementation
that uses a pose estimation model such as MediaPipe Pose or MoveNet.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple, Generator

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import mediapipe as mp
except Exception:  # pragma: no cover
    mp = None  # type: ignore


def extract_poses(video_path: str) -> Iterable[np.ndarray]:
    """Yield pose keypoints for each frame in a video.

    Parameters
    ----------
    video_path : str
        Path to a video file on disk.

    Yields
    ------
    np.ndarray
        An array of shape (N, 3) per frame, where ``N`` is the number of
        keypoints and each keypoint has (x, y, confidence) values. When
        implemented, this generator will produce an array for each frame
        containing the 2D coordinates of detected keypoints.

    Notes
    -----
    The current implementation is a placeholder and raises
    ``NotImplementedError`` on first use. Replace the body of this function
    with code that loads frames from ``video_path`` and runs a pose
    estimation model.
    """

    if cv2 is None or mp is None:
        raise RuntimeError("opencv-python and mediapipe are required for pose extraction")

    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            h, w = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = pose.process(frame_rgb)
            if not res.pose_landmarks:
                # Yield an empty array to keep timing consistent
                yield np.zeros((0, 3), dtype=float)
                continue
            lm = res.pose_landmarks.landmark
            # MediaPipe Pose has 33 landmarks
            arr = np.zeros((len(lm), 3), dtype=float)
            for i, p in enumerate(lm):
                # Convert normalized coords to pixel space for downstream use
                arr[i, 0] = p.x * w
                arr[i, 1] = p.y * h
                arr[i, 2] = p.visibility
            yield arr
    finally:
        cap.release()
        pose.close()


def extract_ankle_trajectories(video_path: str) -> Tuple[List[float], List[float], List[float], float]:
    """Extract a single-dancer ankle trajectory over time from a video.

    Chooses per-frame the ankle (left or right) with higher visibility to
    build a robust ankle track.

    Parameters
    ----------
    video_path : str
        Path to the input video file.

    Returns
    -------
    times : List[float]
        Timestamp for each frame in seconds.
    ankle_x : List[float]
        Selected ankle x-coordinate (pixels) per frame.
    ankle_y : List[float]
        Selected ankle y-coordinate (pixels) per frame.
    fps : float
        Frames-per-second of the video.
    """
    if cv2 is None or mp is None:
        raise RuntimeError("opencv-python and mediapipe are required for pose extraction")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    # Prepare mediapipe
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    times: List[float] = []
    ax: List[float] = []
    ay: List[float] = []
    idx_left = mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value
    idx_right = mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value
    frame_idx = 0
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            h, w = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = pose.process(frame_rgb)
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                l = lm[idx_left]
                r = lm[idx_right]
                # Choose ankle with higher visibility for this frame
                use_left = (l.visibility or 0.0) >= (r.visibility or 0.0)
                if use_left:
                    ax.append(l.x * w)
                    ay.append(l.y * h)
                else:
                    ax.append(r.x * w)
                    ay.append(r.y * h)
            else:
                # No detection; repeat last value if available, else 0
                if ax:
                    ax.append(ax[-1])
                    ay.append(ay[-1])
                else:
                    ax.append(0.0)
                    ay.append(0.0)
            times.append(frame_idx / fps if fps > 0 else float(frame_idx))
            frame_idx += 1
    finally:
        cap.release()
        pose.close()
    return times, ax, ay, float(fps)