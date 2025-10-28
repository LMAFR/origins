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
│   └── app_streamlit.py  # Placeholder Streamlit interface
├── main.py               # Command-line entry point (placeholder)
├── requirements.txt      # List of Python dependencies
├── .gitignore            # Files and directories to ignore in Git
└── README.md             # This file
```

Most modules currently contain **placeholder implementations**. They
include docstrings describing their intended functionality and raise
`NotImplementedError` when called. This structure allows you to
incrementally fill in the implementations while maintaining a coherent
package layout.

## Getting started

1. **Create a virtual environment** (recommended):

   ```bash
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

4. **Run the command-line script** (placeholder):

   ```bash
   python main.py path/to/your/video.mp4
   ```

   This will currently print a message that the analysis is not yet
   implemented. Future versions will analyse the video and output
   statistics.

5. **Launch the Streamlit app** (placeholder):

   ```bash
   streamlit run Origins/ui/app_streamlit.py
   ```

   A web page will open with a placeholder message. Eventually, this
   interface will allow you to upload videos, run the analysis and
   visualise the results.

## Contributing

If you'd like to contribute to Origins, feel free to submit pull
requests or open issues. Refer to the docstrings in each module for
guidance on the expected functionality.

## License

This project is provided for educational purposes and does not yet have
a licence. You are free to use and adapt the code, but please
acknowledge the original authors when doing so.