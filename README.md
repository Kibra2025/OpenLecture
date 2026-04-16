# OpenLecture

OpenLecture is an open-source CLI for transcribing lectures and long-form audio.

By default it uses `faster-whisper`, and it also ships with an optional `transformers` backend for environments where PyTorch is a better fit, including AMD GPU setups.

It is designed to be simple to run from the terminal, while keeping the transcription pipeline reusable as a Python module.

## What It Does

- Transcribes audio files into Markdown
- Produces timestamped transcript lines
- Can optionally omit timestamps in Markdown output
- Supports both `faster-whisper` and `transformers` backends
- Can use AMD GPUs through the `transformers` backend with ROCm or DirectML
- Skips chunking for short files
- Splits long files into overlapping chunks to reduce word cuts at boundaries
- Exposes model and decoding controls from the CLI
- Returns structured `Segment` objects from the internal Python API

## Requirements

- Python 3.10+
- `ffmpeg` available on your system

OpenLecture uses `pydub` for chunking and `PyAV` as a decoding fallback. In practice, having `ffmpeg` installed is still the safest setup.

## Installation

Install the default CLI in editable mode during development:

```bash
python -m pip install -e .
```

Or install it normally:

```bash
python -m pip install .
```

Install test dependencies:

```bash
python -m pip install .[test]
```

Install the optional Transformers backend support:

```bash
python -m pip install .[transformers]
```

### AMD GPU Setup

For AMD GPUs, use `--backend transformers`.

Linux with ROCm:

1. Install a ROCm-enabled PyTorch build from the official PyTorch instructions for your system.
2. Install OpenLecture with the optional Transformers support:

```bash
python -m pip install .[transformers]
```

Windows with DirectML:

1. Install `torch-directml`.
2. Install OpenLecture with the optional Transformers support:

```bash
python -m pip install .[transformers]
```

Examples:

```bash
openlecture lecture.mp3 --backend transformers --device rocm
openlecture lecture.mp3 --backend transformers --device dml
openlecture lecture.mp3 --backend transformers --device amd
```

`--device amd` is a convenience alias that prefers ROCm when PyTorch exposes the GPU as `cuda`, and otherwise falls back to DirectML when available.

## CLI Usage

The supported CLI style is:

```bash
openlecture path/to/lecture.mp3
```

By default, OpenLecture writes a Markdown file next to the input audio using the same base name.

Example:

```bash
openlecture lecture.mp3
```

This creates:

```text
lecture.md
```

Choose a custom output file:

```bash
openlecture lecture.mp3 --output notes/lecture.md
```

Disable progress output:

```bash
openlecture lecture.mp3 --no-progress
```

Disable timestamps in the generated Markdown:

```bash
openlecture lecture.mp3 --no-timestamps
```

Select the backend explicitly:

```bash
openlecture lecture.mp3 --backend transformers
```

Show full traceback on errors:

```bash
openlecture lecture.mp3 --verbose
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

### Examples

Use a smaller model with the default backend:

```bash
openlecture lecture.mp3 --model small
```

Tune decoding parameters for `faster-whisper`:

```bash
openlecture lecture.mp3 --beam-size 3 --device cpu --compute-type int8
```

Use the Transformers backend on AMD:

```bash
openlecture lecture.mp3 --backend transformers --device amd --compute-type auto
```

Force the input language instead of auto-detection:

```bash
openlecture lecture.mp3 --language it
```

Use larger chunks for long recordings:

```bash
openlecture lecture.mp3 --chunk-length-ms 120000
```

## Output Format

OpenLecture writes Markdown like this:

```md
# Lecture Transcript

[00:00:01] Today we talk about Fourier transform.

[00:00:04] The Fourier transform converts signals.
```

Or without timestamps:

```md
# Lecture Transcript

Today we talk about Fourier transform.

The Fourier transform converts signals.
```

## How It Works

- Files shorter than 5 minutes are transcribed directly without chunking.
- Longer files are split into overlapping chunks.
- Each chunk is exported to a temporary WAV file, transcribed, and deleted immediately.
- Chunks use a small default overlap to reduce boundary errors.
- Segments that fall fully inside already-processed overlap are discarded during merge.
- The final output is rendered from structured `Segment` objects.
- The `transformers` backend runs Whisper through `processor(...)` plus `model.generate(...)` and normalizes the returned segments back into the same `Segment` model.

## Python API

The internal transcription pipeline is reusable from Python.

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
Segment(start=0.0, end=2.4, text="Hello and welcome")
```

## Development

Run the test suite with:

```bash
pytest
```

## Current Scope

OpenLecture currently focuses on:

- local transcription through the CLI
- Markdown transcript generation
- a clean internal pipeline for future outputs such as JSON, SRT, or VTT
