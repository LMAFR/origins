"""
Origins Streamlit UI
====================

Interactive viewer that plays a selected video with audio while showing
live audio visualisations (like a DJ tool) and a dynamic table with the
latest detected steps and beats.

Notes
-----
- This app relies on the implemented pipeline modules in this repo.
- For best results, ensure `requirements.txt` are installed, including
  streamlit, mediapipe, opencv-python, librosa, and scipy.
"""

from __future__ import annotations

import time
import pathlib
from typing import Dict, List, Tuple, Sequence, cast

import numpy as np
import pandas as pd
import streamlit as st

from utils.io import extract_audio_from_video, load_audio
from audio.beat_tracking import compute_beats
from video.pose_extractor import extract_ankle_trajectories
from video.footstrike_detector import detect_heel_strikes

try:
    import librosa
    import librosa.display  # noqa: F401
except Exception:
    st.error("librosa is required for this app. Please install dependencies.")
    raise


@st.cache_data(show_spinner=True)
def prepare_media(video_path: str, output_dir: str) -> Dict[str, object]:
    """Extract audio, compute features, beats, and steps for a video.

    Returns a dict with:
      - audio_path, sr, y
      - times_oenv, oenv
      - times_rms, rms_full, rms_perc, rms_harm
      - band_times, band_low, band_mid, band_high
      - beat_times, step_times
    """
    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_path = out_dir / f"{pathlib.Path(video_path).stem}_audio.wav"
    if not audio_path.exists():
        extract_audio_from_video(video_path, str(audio_path))

    # Load audio
    y, sr = load_audio(str(audio_path))

    # Onset strength envelope
    hop_length = 512
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    times_oenv = librosa.times_like(oenv, sr=sr, hop_length=hop_length)

    # HPSS for percussive/harmonic RMS
    y_harm, y_perc = librosa.effects.hpss(y)
    frame_length = 2048
    hop_rms = 512
    rms_full = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_rms)[0]
    rms_perc = librosa.feature.rms(y=y_perc, frame_length=frame_length, hop_length=hop_rms)[0]
    rms_harm = librosa.feature.rms(y=y_harm, frame_length=frame_length, hop_length=hop_rms)[0]
    times_rms = librosa.times_like(rms_full, sr=sr, hop_length=hop_rms)

    # Band energies (low/mid/high) from mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=hop_rms, n_mels=64, fmin=20, fmax=8000)
    mel_freqs = librosa.mel_frequencies(n_mels=S.shape[0], fmin=20, fmax=8000)
    # Define bands
    low_idx = mel_freqs <= 150
    mid_idx = (mel_freqs > 150) & (mel_freqs <= 2000)
    high_idx = mel_freqs > 2000
    band_low = S[low_idx].mean(axis=0)
    band_mid = S[mid_idx].mean(axis=0)
    band_high = S[high_idx].mean(axis=0)
    band_times = librosa.times_like(band_low, sr=sr, hop_length=hop_rms)

    # Beat tracking (prefer madmom if available via compute_beats)
    beats_info = compute_beats(str(audio_path), use_drums=False)
    beat_times_seq = beats_info.get("beat_times", [])  # type: ignore[assignment]
    try:
        beat_times = [float(b) for b in cast(Sequence[float], beat_times_seq)]
    except Exception:
        beat_times = [float(b) for b in list(beat_times_seq) if b is not None]  # type: ignore[arg-type]

    # Steps via pose + foot-strike detection
    times_v, ax, ay, fps = extract_ankle_trajectories(video_path)
    step_times = detect_heel_strikes(times_v, ax, ay, fps)

    return {
        "audio_path": str(audio_path),
        "sr": int(sr),
        "y": y,
        "times_oenv": times_oenv,
        "oenv": oenv,
        "times_rms": times_rms,
        "rms_full": rms_full,
        "rms_perc": rms_perc,
        "rms_harm": rms_harm,
        "band_times": band_times,
        "band_low": band_low,
        "band_mid": band_mid,
        "band_high": band_high,
        "beat_times": beat_times,
        "step_times": step_times,
    }


def slice_last_window(times: np.ndarray, values: np.ndarray, t_now: float, window: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """Slice a time series to the last `window` seconds before t_now."""
    if times.size == 0:
        return times, values
    start_t = max(0.0, t_now - window)
    i0 = int(np.searchsorted(times, start_t))
    i1 = int(np.searchsorted(times, t_now))
    return times[i0:i1], values[i0:i1]


def last_n_events_before(t_now: float, events: List[float], n: int = 3) -> List[float]:
    arr = np.asarray(events, dtype=float)
    idx = np.searchsorted(arr, t_now) - 1
    idx = int(max(-1, idx))
    if idx < 0:
        return []
    i0 = max(0, idx - (n - 1))
    return arr[i0 : idx + 1].tolist()


def main() -> None:
    st.set_page_config(page_title="Origins Live Viewer", layout="wide")
    st.title("Origins Live Viewer")
    st.caption("Play a video and view live audio graphs + last steps/beats table.")

    # Sidebar: media selection
    test_dir = pathlib.Path("test-files")
    videos = sorted([p for p in test_dir.glob("*.mp4")] + [p for p in test_dir.glob("*.mov")] + [p for p in test_dir.glob("*.MOV")])
    if not videos:
        st.warning("Place some videos in 'test-files/' to begin.")
        return
    video_path = st.sidebar.selectbox("Select video", videos, format_func=lambda p: p.name)
    output_dir = st.sidebar.text_input("Output directory", "output")
    viz_window = st.sidebar.slider("Visualization window (s)", 5, 30, 12)
    st.sidebar.divider()

    # Prepare data
    with st.spinner("Preparing media (audio features, beats, steps)..."):
        data = prepare_media(str(video_path), output_dir)

    st.success("Media ready")

    # Playback + controls
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Video")
        st.video(str(video_path))
        st.caption("Use the player controls to play/pause. The visualisation advances using an app clock.")

    with c2:
        st.subheader("Controls")
        if "running" not in st.session_state:
            st.session_state["running"] = False
        if "t0" not in st.session_state:
            st.session_state["t0"] = 0.0
        if "offset" not in st.session_state:
            st.session_state["offset"] = 0.0

        st.session_state["offset"] = st.number_input("Visualization time offset (s)", value=float(st.session_state["offset"]))
        start_btn = st.button("Start")
        stop_btn = st.button("Stop")
        reset_btn = st.button("Reset")

        if start_btn and not st.session_state["running"]:
            st.session_state["running"] = True
            st.session_state["t0"] = time.perf_counter()
        if stop_btn and st.session_state["running"]:
            st.session_state["running"] = False
        if reset_btn:
            st.session_state["t0"] = time.perf_counter()

    # Live plots and table
    lc1, lc2 = st.columns([2, 1])
    with lc1:
        st.subheader("Audio graphs")
        placeholder_oenv = st.empty()
        placeholder_rms = st.empty()
        placeholder_bands = st.empty()
    with lc2:
        st.subheader("Latest events (last 3)")
        placeholder_table = st.empty()

    # Data for plotting
    times_oenv = np.asarray(data["times_oenv"], dtype=float)
    oenv = np.asarray(data["oenv"], dtype=float)
    times_rms = np.asarray(data["times_rms"], dtype=float)
    rms_full = np.asarray(data["rms_full"], dtype=float)
    rms_perc = np.asarray(data["rms_perc"], dtype=float)
    rms_harm = np.asarray(data["rms_harm"], dtype=float)
    band_times = np.asarray(data["band_times"], dtype=float)
    band_low = np.asarray(data["band_low"], dtype=float)
    band_mid = np.asarray(data["band_mid"], dtype=float)
    band_high = np.asarray(data["band_high"], dtype=float)
    beat_times = [float(b) for b in cast(Sequence[float], data["beat_times"])]  # type: ignore[arg-type]
    step_times = [float(s) for s in cast(Sequence[float], data["step_times"])]  # type: ignore[arg-type]

    # Main update loop
    while True:
        if not st.session_state.get("running", False):
            time.sleep(0.1)
            # Still render current snapshot so UI doesn't look empty
            t_now = (time.perf_counter() - st.session_state.get("t0", 0.0)) + st.session_state.get("offset", 0.0)
        else:
            t_now = (time.perf_counter() - st.session_state.get("t0", 0.0)) + st.session_state.get("offset", 0.0)

        # Slice windows
        toenv, voenv = slice_last_window(times_oenv, oenv, t_now, viz_window)
        trms, vf, = slice_last_window(times_rms, rms_full, t_now, viz_window)
        _, vp = slice_last_window(times_rms, rms_perc, t_now, viz_window)
        _, vh = slice_last_window(times_rms, rms_harm, t_now, viz_window)
        tb, bl = slice_last_window(band_times, band_low, t_now, viz_window)
        _, bm = slice_last_window(band_times, band_mid, t_now, viz_window)
        _, bh = slice_last_window(band_times, band_high, t_now, viz_window)

        # Prepare DataFrames for plotting
        df_oenv = pd.DataFrame({"time": toenv, "onset_env": voenv})
        df_rms = pd.DataFrame({"time": trms, "full": vf, "percussive": vp, "harmonic": vh})
        df_bands = pd.DataFrame({"time": tb, "low": bl, "mid": bm, "high": bh})

        with lc1:
            placeholder_oenv.line_chart(df_oenv.set_index("time"), height=150, use_container_width=True)
            placeholder_rms.line_chart(df_rms.set_index("time"), height=150, use_container_width=True)
            placeholder_bands.line_chart(df_bands.set_index("time"), height=150, use_container_width=True)

        # Dynamic table with last 3 beats and steps
        last_beats = last_n_events_before(t_now, beat_times, n=3)
        last_steps = last_n_events_before(t_now, step_times, n=3)
        df_events = pd.DataFrame({
            "type": ["beat"] * len(last_beats) + ["step"] * len(last_steps),
            "time (s)": last_beats + last_steps,
        }).sort_values("time (s)", ascending=False, ignore_index=True)
        with lc2:
            placeholder_table.dataframe(df_events, hide_index=True, use_container_width=True)

        # End iteration
        time.sleep(0.1)


if __name__ == "__main__":
    main()