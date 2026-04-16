# OpenLecture

OpenLecture is an open-source Python CLI for transcribing local audio files into Markdown.

The project is intentionally narrow in scope: local transcription, a thin terminal UX, and a reusable internal pipeline. By default it uses `faster-whisper`. It also includes an optional `transformers` backend for environments where PyTorch is the better fit, including AMD setups through ROCm or DirectML.

## What The Project Is

OpenLecture is:

- a local CLI tool
- a Python package with a reusable transcription pipeline
- a Markdown transcript generator
- a dual-backend Whisper wrapper
- a codebase prepared for future output formats

OpenLecture is not:

- a web app
- an HTTP API
- a database-backed service
- a diarization tool
- a summarizer
- an exporter for JSON, SRT, or VTT yet

## Features

- Transcribes a local audio file into Markdown
- Writes plain Markdown transcript lines by default
- Keeps timestamps disabled by default and enables them with `--timestamps`
- Supports both `faster-whisper` and `transformers`
- Supports AMD-oriented device aliases through the `transformers` backend
- Skips chunking for files shorter than 5 minutes
- Splits longer files into overlapping chunks
- Shows a smooth `tqdm` progress bar for both direct and chunked transcription
- Falls back to callback-driven progress when duration estimation is unavailable
- Rejects obvious non-audio inputs such as `.md`, `.txt`, `.json`, `.srt`, and `.vtt`
- Drops duplicate segments that fall entirely inside already-processed overlap
- Exposes model, device, decoding, and language controls through the CLI
- Returns structured `Segment` objects from the internal Python API

## Requirements

- Python 3.10+
- `ffmpeg` available on the system

The audio pipeline uses `pydub` for chunking and PyAV as a decoding fallback. In practice, having `ffmpeg` installed remains the safest setup.

## Installation

Install the package normally:

```bash
python -m pip install .
```

Install it in editable mode during development:

```bash
python -m pip install -e .
```

Install test dependencies:

```bash
python -m pip install -e .[test]
```

Install optional Transformers support:

```bash
python -m pip install -e .[transformers]
```

Important: the `transformers` extra installs `transformers`, but you still need a compatible PyTorch runtime for your environment.

## Quick Start

Transcribe a file and write a Markdown transcript next to it:

```bash
openlecture lecture.mp3
```

This creates:

```text
lecture.md
```

Write to a custom output file:

```bash
openlecture lecture.mp3 --output notes/lecture.md
```

Enable timestamps in the Markdown output:

```bash
openlecture lecture.mp3 --timestamps
```

Use a specific model size:

```bash
openlecture lecture.mp3 --model small
```

Force the input language:

```bash
openlecture lecture.mp3 --language it
```

Use the Transformers backend on AMD:

```bash
openlecture lecture.mp3 --backend transformers --device amd --compute-type auto
```

## CLI Usage

The supported public CLI style is:

```bash
openlecture path/to/lecture.mp3
```

The command style below is not the intended public interface:

```bash
openlecture transcribe lecture.mp3
```

## CLI Options

```text
--output PATH
--progress / --no-progress
--timestamps / --no-timestamps
--chunk-length-ms INTEGER
--backend TEXT
--model TEXT
--beam-size INTEGER
--device TEXT
--compute-type TEXT
--language TEXT
--verbose, -v
```

## CLI Behavior

With progress enabled, the CLI behaves differently depending on the file:

- Small files use direct transcription and show a smooth estimated progress bar.
- Long files use chunked transcription and show a smooth chunk-aware progress bar.
- If duration estimation fails up front, the CLI falls back to progress updates driven only by chunk completion callbacks.
- `--no-progress` disables the progress bar but still allows plain status messages.

The progress bar lives only in `openlecture/cli.py`. The core pipeline stays callback-based and does not print directly.

## Input Validation

OpenLecture expects an audio or video container with an audio stream, such as:

- `.mp3`
- `.wav`
- `.m4a`
- `.mp4`

If you accidentally pass an obvious transcript or text file like `.md`, `.txt`, `.json`, `.srt`, or `.vtt`, the CLI fails early with a clear error instead of attempting to decode it as audio.

## Output Format

OpenLecture writes Markdown like this when timestamps are enabled:

```md
# Lecture Transcript

[00:00:01] Today we talk about Fourier transform.

[00:00:04] The Fourier transform converts signals.
```

Or like this by default:

```md
# Lecture Transcript

Today we talk about Fourier transform.

The Fourier transform converts signals.
```

The formatter:

- always starts with `# Lecture Transcript`
- skips empty transcript lines
- uses `HH:MM:SS` timestamps
- can render output with or without timestamps

The CLI defaults to output without timestamps. Use `--timestamps` to enable them.

## Backend Notes

### Default backend: `faster-whisper`

This is the default path and requires no backend flag:

```bash
openlecture lecture.mp3
```

Useful when you want the simplest setup and default CLI behavior.

### Optional backend: `transformers`

Use this when you explicitly want Hugging Face Whisper through PyTorch:

```bash
openlecture lecture.mp3 --backend transformers
```

The code resolves short model names like `medium` into Hugging Face model IDs such as `openai/whisper-medium`.

### AMD GPU setups

For AMD GPUs, use the `transformers` backend.

Linux with ROCm:

1. Install a ROCm-enabled PyTorch build for your system.
2. Install OpenLecture with the `transformers` extra.
3. Run OpenLecture with `--backend transformers --device rocm`.

Windows with DirectML:

1. Install `torch-directml`.
2. Install OpenLecture with the `transformers` extra.
3. Run OpenLecture with `--backend transformers --device dml`.

Examples:

```bash
openlecture lecture.mp3 --backend transformers --device rocm
openlecture lecture.mp3 --backend transformers --device dml
openlecture lecture.mp3 --backend transformers --device amd
```

`--device amd` is a convenience alias. It prefers ROCm when PyTorch exposes the GPU as `cuda`, otherwise it falls back to DirectML when available.

## How The Repository Works

At a high level, the repository contains:

- the CLI entrypoint
- the internal transcription pipeline
- audio chunking and decoding helpers
- a Markdown formatter
- tests
- a benchmark script
- CI configuration
- an AI-oriented context document in `docs/CONTESTO_PROGETTO_AI.md`

### Repository Layout

```text
OpenLecture/
|-- .github/workflows/ci.yml
|-- docs/
|   `-- CONTESTO_PROGETTO_AI.md
|-- openlecture/
|   |-- __init__.py
|   |-- audio_utils.py
|   |-- cli.py
|   |-- models.py
|   |-- output_formatter.py
|   `-- transcribe.py
|-- scripts/
|   `-- benchmark.py
|-- tests/
|   |-- conftest.py
|   |-- test_audio_utils.py
|   |-- test_backends.py
|   |-- test_cli.py
|   |-- test_output_formatter.py
|   `-- test_progress.py
|-- Ossessione.mp3
|-- Ossessione.md
|-- pyproject.toml
|-- requirements.txt
`-- README.md
```

### What Each Part Does

`openlecture/cli.py`

- defines the Typer CLI
- validates and prepares the output path
- creates the `tqdm` progress bar wrapper
- forwards CLI options into the transcription layer
- writes the final Markdown file
- shows clean errors unless `--verbose` is enabled

`openlecture/transcribe.py`

- implements the core transcription pipeline
- loads and caches models
- normalizes backend, device, compute type, and language options
- decides whether to transcribe directly or chunk the audio
- merges chunk results into absolute `Segment` objects
- adapts the `transformers` backend to the same shape expected by the pipeline

`openlecture/audio_utils.py`

- validates audio paths
- rejects obvious non-audio transcript-like files
- finds audio duration
- loads audio through `pydub`, with PyAV fallback
- splits long audio into overlapping chunks
- exports chunk audio to temporary WAV files when needed

`openlecture/output_formatter.py`

- converts `list[Segment]` into Markdown

`openlecture/models.py`

- defines the core `Segment` dataclass used throughout the project

`tests/`

- covers the CLI, chunking behavior, backend wiring, progress reporting, and output formatting

`scripts/benchmark.py`

- measures transcription time relative to audio duration

`docs/CONTESTO_PROGETTO_AI.md`

- describes the repository for another AI that needs to understand the project quickly and safely

## Internal Flow

The end-to-end flow is:

1. The CLI receives an input audio file and optional runtime flags.
2. The output path is resolved and parent directories are created if needed.
3. Audio path validation runs before decoding.
4. A Whisper model is loaded and cached for the selected backend.
5. Audio duration is measured.
6. Files shorter than 5 minutes are transcribed directly.
7. Longer files are split into overlapping chunks.
8. Each chunk is exported to a temporary WAV file and transcribed.
9. Chunk-local timestamps are converted into absolute timestamps.
10. Segments fully contained inside already-processed overlap are discarded.
11. The final list of `Segment` objects is rendered to Markdown.
12. The transcript is written to disk.

## Chunking Strategy

Chunking exists to handle long audio more safely.

Current defaults:

- chunk length: `60000 ms`
- overlap: `2000 ms`
- small-file direct transcription threshold: `300 seconds`

This means:

- files shorter than 5 minutes skip chunk splitting entirely
- long files are split into 60-second chunks
- adjacent chunks overlap by 2 seconds
- the overlap helps reduce word cuts at chunk boundaries
- duplicate overlap-only segments are removed during merge

## Python API

The internal pipeline is reusable from Python.

```python
from openlecture.transcribe import transcribe_audio

segments = transcribe_audio(
    "lecture.mp3",
    backend="transformers",
    model_size="medium",
    beam_size=5,
    device="amd",
    compute_type="auto",
    language=None,
)

for segment in segments:
    print(segment.start, segment.end, segment.text)
```

`transcribe_audio()` returns a list of `Segment` objects:

```python
from openlecture.models import Segment

Segment(start=0.0, end=2.4, text="Hello and welcome")
```

The core API currently revolves around this single data model.

## Development

Install development dependencies:

```bash
python -m pip install -e .[test]
```

Run the test suite:

```bash
pytest
```

Run the benchmark script:

```bash
python scripts/benchmark.py path/to/audio.mp3
```

Show progress during benchmarking:

```bash
python scripts/benchmark.py path/to/audio.mp3 --progress
```

## CI

The repository includes a GitHub Actions workflow at `.github/workflows/ci.yml`.

It currently:

- runs on push and pull request
- tests on Ubuntu and Windows
- uses Python 3.12
- installs `-e .[test]`
- runs `pytest`

## Current Scope And Limits

The project currently focuses on:

- local transcription through the CLI
- Markdown transcript generation
- a reusable internal pipeline

The project does not currently implement:

- JSON export
- SRT export
- VTT export
- diarization
- speaker labeling
- batch folder processing
- a web interface
- an API server
- transcript summarization

## Design Intent

The current architecture is deliberately simple:

- `Segment` is the common data shape
- transcription and formatting are separated
- the CLI is thin and forwards into reusable Python functions
- backend-specific behavior is isolated inside the transcription layer
- progress UI remains in the CLI, not in the pipeline

That structure makes it easier to add future output formats such as JSON, SRT, or VTT without rewriting the entire pipeline.
