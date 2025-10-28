"""
main.py
========

Entry point for the Origins timing analysis toolkit.

This script wires together the audio analysis, pose extraction, foot
strike detection, synchronisation and scoring components to provide
a command-line interface for evaluating dance timing.

This script implements the Phase 1 MVP CLI: it accepts a video file,
extracts audio and beats, estimates the pose to get ankle trajectories,
detects heel strikes, estimates a global A/V offset, and evaluates the
timing accuracy relative to musical beats. Optionally writes a CSV of
per-step metrics.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from typing import List

from audio.beat_tracking import compute_beats
from utils.io import extract_audio_from_video
from video.pose_extractor import extract_ankle_trajectories
from video.footstrike_detector import detect_heel_strikes
from sync.av_sync import estimate_offset
from sync.align import evaluate_steps


def main(argv: list[str] | None = None) -> int:
    """Run the Origins analysis pipeline from the command line.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Defaults to ``None``, in which case
        ``sys.argv[1:]`` is used.

    Returns
    -------
    int
        Exit status code (0 for success).
    """

    parser = argparse.ArgumentParser(description="Analyse dance timing relative to music.")
    parser.add_argument("video", type=pathlib.Path, help="Path to the input video file")
    parser.add_argument("--use-drums", action="store_true", help="Isolate drums for beat tracking")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("output"), help="Output directory for results")
    parser.add_argument("--tau-ms", type=float, default=60.0, help="Tau (ms) for Gaussian scoring")
    parser.add_argument("--write-csv", action="store_true", help="Write per-step metrics to CSV in output directory")
    args = parser.parse_args(argv)

    video_path = args.video
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Extract audio from video
    audio_path = out_dir / f"{video_path.stem}_audio.wav"
    extract_audio_from_video(str(video_path), str(audio_path))

    # 2) Compute beats and (optional) downbeats
    beats_info = compute_beats(str(audio_path), use_drums=bool(args.use_drums))
    beat_times: List[float] = [float(b) for b in beats_info.get("beat_times", [])]  # type: ignore[arg-type]

    if not beat_times:
        print("No beats detected; aborting.")
        return 2

    # 3) Extract ankle trajectories
    times, ankle_x, ankle_y, fps = extract_ankle_trajectories(str(video_path))

    # 4) Heel-strike detection
    step_times = detect_heel_strikes(times, ankle_x, ankle_y, fps)
    if not step_times:
        print("No heel strikes detected; aborting.")
        return 3

    # 5) Estimate global offset (steps -> beats)
    best_offset = estimate_offset(step_times, beat_times, search_range=(-0.2, 0.2), step=0.005)
    step_times_aligned = [s + best_offset for s in step_times]

    # 6) Evaluate timing
    per_step, mean_score = evaluate_steps(step_times_aligned, beat_times, tau_ms=float(args.tau_ms))

    # Extra summary: percentage within ±tau
    tau_ms = float(args.tau_ms)
    within_tau = sum(1 for p in per_step if abs(p["delta_ms"]) <= tau_ms)
    pct_within_tau = 100.0 * within_tau / len(per_step)

    # Print summary
    print("--- Origins Timing Analysis ---")
    print(f"Video:          {video_path}")
    print(f"Audio (WAV):    {audio_path}")
    print(f"Detected beats: {len(beat_times)}")
    print(f"Detected steps: {len(step_times)}")
    print(f"Best offset:    {best_offset*1000:.1f} ms (applied to steps)")
    print(f"Mean score:     {mean_score:.3f}")
    print(f"Within ±{tau_ms:.0f} ms: {within_tau}/{len(per_step)} ({pct_within_tau:.1f}%)")

    # Optional CSV export
    if args.write_csv:
        csv_path = out_dir / f"{video_path.stem}_per_step.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["step_time", "nearest_beat_time", "delta_ms", "score"]) 
            writer.writeheader()
            for row in per_step:
                writer.writerow(row)
        print(f"Per-step metrics written to: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())