# Origins

Origins is a research project aiming to analyse the temporal synchrony of
dancers relative to the music in Brazilian Zouk. It provides a set of
Python modules and a Streamlit interface to extract beats and downbeats
from audio, detect foot strikes from video, align the two and compute
metrics describing how "in time" a dancer is with the music.

This repository contains only the **code and configuration files**. It
does not include any sample media; instead, you should place your own
videos and audio files into a local directory named `test-files`. This
directory is excluded from version control via `.gitignore` to avoid
committing large files.

## Project layout

```
Origins/
├── audio/                # Audio analysis modules (beat detection)
│   ├── __init__.py
│   └── beat_tracking.py
├── video/                # Video and pose analysis modules
│   ├── __init__.py
│   ├── pose_extractor.py
│   └── footstrike_detector.py
├── sync/                 # Synchronisation utilities
│   ├── __init__.py
│   ├── av_sync.py        # Audio/video offset estimation
│   └── align.py          # Align steps to beats and compute deltas
├── scoring/              # Scoring and evaluation utilities
│   ├── __init__.py
│   └── metrics.py
├── utils/                # Helper functions for I/O and smoothing
│   ├── __init__.py
│   ├── io.py
│   └── smoothing.py
├── ui/
│   ├── __init__.py
│   └── app_streamlit.py  # Streamlit live viewer (video + audio graphs)
├── main.py               # Command-line entry point (MVP pipeline)
├── requirements.txt      # List of Python dependencies
├── .gitignore            # Files and directories to ignore in Git
└── README.md             # This file
```

The repository now contains a working Phase 1 MVP:

- Audio beat tracking (librosa fallback; optional madmom if installed)
- Pose extraction with MediaPipe Pose and ankle trajectory construction
- Heel-strike detection from ankle trajectories
- A/V offset estimation and alignment
- Timing evaluation with Gaussian scoring
- CLI to run end to end and optional CSV export
- Streamlit live viewer with DJ-style audio graphs and a rolling table of last events

## Getting started

1. **Create a virtual environment** (recommended):

   ```bash
   # Windows (Git Bash)
   python -m venv .venv
   source .venv/Scripts/activate

   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare media files**:

   Create a directory called `test-files` at the root of this project and
   place your video files there. These files will not be tracked by Git.

4. **Run the command-line script**:

   ```bash
   python main.py test-files/your_video.mp4 --write-csv --output output
   ```

   This runs the full pipeline:
   - extracts audio from the video (moviepy; falls back to ffmpeg if needed)
   - tracks beats
   - extracts ankle trajectories and detects heel strikes
   - estimates a global offset and evaluates timing vs beats
   - prints a summary and optionally writes per-step CSV in `output/`

5. **Launch the Streamlit app**:

   ```bash
   # If `streamlit` is not on PATH, use the venv Python module invocation:
   python -m streamlit run ui/app_streamlit.py
   ```

   A web page will open showing:
   - the selected video
   - live audio graphs (onset envelope, RMS percussive/harmonic, band energies)
   - a dynamic table with the last 3 beats and last 3 steps

   Tip: Place your test videos in `test-files/` so the app can list them.

## Contributing

If you'd like to contribute to Origins, feel free to submit pull
requests or open issues. Refer to the docstrings in each module for
guidance on the expected functionality.

## License

This project is provided for educational purposes and does not yet have
a licence. You are free to use and adapt the code, but please
acknowledge the original authors when doing so.
