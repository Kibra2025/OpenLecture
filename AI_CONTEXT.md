# AI Context for OpenLecture

## TL;DR

OpenLecture is a small Python CLI project that transcribes an audio file into a Markdown document using `faster-whisper`.

The real pipeline is:

1. read an input audio file from the CLI
2. split it into fixed-duration chunks with `pydub`
3. export chunks as temporary `.wav` files
4. transcribe each chunk sequentially with Whisper
5. concatenate all text into one flat transcript string
6. convert that string into a very simple Markdown document

This is not a web app, not a service, and not a library with many public APIs. It is primarily a local command-line transcription tool.

## What The Project Is For

The goal of the project is to turn a lecture or spoken-audio file into a Markdown transcript.

Typical use case:

- user has an `.mp3`, `.m4a`, `.mp4`, or `.wav`
- user runs `openlecture <audio-file>`
- tool generates `<audio-file>.md`

The output is intentionally simple:

- one top-level heading: `# Lecture Transcript`
- transcript text split into paragraphs by periods

There is no timestamping, no speaker diarization, no structured JSON output, and no semantic post-processing.

## Repository Map

### Product code

- `openlecture/cli.py`
  CLI entrypoint built with Typer.
- `openlecture/transcribe.py`
  Core transcription pipeline, Whisper model loading, chunk orchestration, progress handling, and temporary directory management.
- `openlecture/audio_utils.py`
  Audio loading/splitting/exporting utilities. Handles `ffmpeg` discovery and a `PyAV` fallback decoder.
- `openlecture/output_formatter.py`
  Converts the flat transcript string into Markdown.
- `openlecture/__init__.py`
  Empty package marker.

### Metadata and docs

- `pyproject.toml`
  Project metadata and console script registration.
- `requirements.txt`
  Pip-style dependency list.
- `README.md`
  Minimal project overview and usage example.

### Non-product / local artifacts

- `venv/`
  Local virtual environment, not part of the application design.
- `__pycache__/`
  Python cache artifacts.
- `Ossessione.mp3`
  Sample/local audio file in the repo root, useful as a manual test input but not part of the package API.

## Current Snapshot Observed During Analysis

Snapshot date: 2026-03-18.

At analysis time, the git worktree was not clean:

- `openlecture/transcribe.py` had local uncommitted modifications
- `Ossessione.md` was deleted from the working tree

An AI working on this repository should check `git status` before assuming the current workspace matches the last commit.

## High-Level Architecture

The architecture is small but modular:

- `cli.py` handles user input and file output
- `transcribe.py` owns the end-to-end audio-to-text workflow
- `audio_utils.py` owns audio decoding, chunk splitting, and chunk export
- `output_formatter.py` owns final Markdown rendering

Mental model:

`CLI -> validate args -> transcribe_audio() -> split/export audio -> Whisper transcription -> markdown formatting -> write .md file`

## End-to-End Flow

### 1. CLI entrypoint

The console script is registered in `pyproject.toml` as:

`openlecture = "openlecture.cli:app"`

Important detail:

- the Typer app contains only one command, `transcribe`
- Typer promotes that single command to the root command
- in practice, the UX is `openlecture audio.mp3`, not `openlecture transcribe audio.mp3`

`cli.py` also defines a `main()` function that strips the literal `transcribe` argument if the module is run directly and the user types it explicitly. That is a compatibility convenience for `python -m openlecture.cli transcribe ...`.

### 2. CLI options

The command accepts:

- positional `audio_file`
- optional `--output`
- optional `--progress/--no-progress`
- optional `--chunk-length-ms` with default `60000`

Default output path:

- if `--output` is omitted, the tool writes next to the input file using the same basename and a `.md` extension

### 3. Transcription pipeline

`cli.py` calls:

`transcribe_audio(str(audio_file), show_progress=progress, chunk_length_ms=chunk_length_ms)`

Then it converts the returned transcript with:

`transcript_to_markdown(transcript)`

Then it writes the Markdown file to disk.

### 4. Validation and model loading

`transcribe_audio()` first validates the input path with `_validate_audio_file()`.

Then it loads the Whisper model through `_get_model()`.

Important implementation details:

- model size is hardcoded as `MODEL_SIZE = "medium"`
- model loading is cached with `@lru_cache(maxsize=1)`
- device selection is automatic: `device="auto"`
- compute type selection is automatic: `compute_type="auto"`

Consequence:

- within one Python process the model is loaded only once
- model size is currently not user-configurable from the CLI

### 5. Audio splitting

`transcribe_audio()` calls:

`split_audio(audio_path, chunk_length_ms=...)`

`split_audio()`:

- validates the input path and chunk size
- imports `pydub.AudioSegment` lazily
- tries to decode the file with `pydub`
- if `pydub` decoding fails, tries a fallback decoder implemented with `PyAV`
- returns a list of in-memory `AudioSegment` chunks

Chunking behavior:

- chunk length is fixed in milliseconds
- default is 60 seconds
- chunks are created sequentially from the start of the audio

### 6. Temporary WAV export

The chunks are not transcribed directly from memory.

Instead, `transcribe_audio()`:

- creates a temporary directory
- exports each chunk as `chunk_000.wav`, `chunk_001.wav`, etc.
- transcribes those temporary files one by one
- deletes the temporary directory afterwards

Temporary directory strategy:

- first try `<cwd>/.openlecture_tmp`
- then try `<audio_file_parent>/.openlecture_tmp`
- finally fall back to `<cwd>/openlecture_chunks_<uuid>`

This matters because the code assumes it can create temporary directories in a writable location near the current working directory or near the audio file.

### 7. Chunk transcription

Each exported chunk file is transcribed by `_transcribe_file()`.

Whisper parameters currently used:

- `beam_size=5`
- `vad_filter=True`
- `log_progress=show_progress`

Text handling:

- empty or whitespace-only segment texts are ignored
- non-empty segment texts are stripped and appended
- at the end, all collected text pieces are joined with `" "`

Consequence:

- final output is a single flat transcript string
- original chunk boundaries are discarded
- segment timing information is discarded

### 8. Progress bar behavior

Progress handling is more complex than the rest of the project and is the most non-obvious part of the codebase.

`transcribe.py` monkey-patches `faster_whisper.transcribe.tqdm` at runtime so that Whisper progress output uses a custom progress bar implementation.

The custom progress system uses:

- `ContextVar` to store chunk-local progress state
- a shared session object so multiple chunk transcriptions reuse one bar
- computed fields such as total audio processed, current chunk progress, speed, and ETA

Important implication:

- progress reporting is not just a wrapper around logging
- it depends on runtime monkey-patching of the `faster_whisper` internals
- if `faster-whisper` changes its internal `tqdm` integration, this code may require adaptation

### 9. Markdown formatting

After transcription, `transcript_to_markdown()` formats the text.

Formatting rules are intentionally simple:

- split transcript on `.`
- trim whitespace
- discard empty parts
- add back a trailing `.`
- join paragraphs with blank lines
- prepend `# Lecture Transcript`

This produces readable output quickly, but it is intentionally naive.

## Module-by-Module Notes

### `openlecture/cli.py`

Responsibilities:

- define the Typer command
- parse CLI options
- call the transcription pipeline
- format transcript to Markdown
- write final output file
- exit with code `1` on error

Non-obvious behavior:

- Typer exposes the single command as the app root
- `main()` only exists to gracefully accept an explicit `transcribe` token when running as a module

### `openlecture/audio_utils.py`

Responsibilities:

- discover `ffmpeg` and `ffprobe`
- configure `pydub` dynamically
- decode audio files
- split audio into chunks
- export chunks to `.wav`

Important details:

- supported extension-to-format inference is hardcoded in `SUPPORTED_FORMATS`
- on Windows, the code explicitly searches common WinGet FFmpeg install directories
- `pydub` import is deferred until needed
- warnings about missing `ffmpeg`/`ffprobe` are suppressed during import/load attempts

Decoder strategy:

1. try `pydub` + `ffmpeg`/`ffprobe`
2. if that fails, try `PyAV`

This makes the project more resilient than the README suggests.

### `openlecture/transcribe.py`

Responsibilities:

- validate input file
- load/cache the Whisper model
- split audio and export temporary chunk files
- run transcription chunk by chunk
- manage progress display
- merge chunk transcripts into one final string

Most important constants:

- `MODEL_SIZE = "medium"`
- `DEFAULT_CHUNK_LENGTH_MS = 60_000`

Most important internal contracts:

- input is a local path string
- output is a plain transcript string, not a structured object
- processing is sequential, not parallel

### `openlecture/output_formatter.py`

Responsibilities:

- produce Markdown from a plain transcript string

Important limitation:

- sentence splitting is based only on periods
- abbreviations, decimals, ellipses, or language-specific punctuation are not handled robustly

## Dependencies and Runtime Requirements

### Python dependencies actually used by the code

- `faster-whisper`
- `typer`
- `pydub`
- `av` / `PyAV`

### Declared dependency files

`pyproject.toml` declares:

- `faster-whisper`
- `typer`
- `pydub`
- `av`

`requirements.txt` declares:

- `faster-whisper`
- `ffmpeg-python`
- `typer`
- `pydub`
- `av`

Important inconsistency:

- `ffmpeg-python` is present in `requirements.txt`
- it is not declared in `pyproject.toml`
- the current code does not import `ffmpeg-python`

So `ffmpeg-python` appears to be unused by the current implementation.

### External system requirement

For common audio decoding paths, `ffmpeg` and `ffprobe` are important runtime dependencies.

However, this project also includes a fallback decoder based on `PyAV`, so the real runtime story is:

- `ffmpeg`/`ffprobe` are preferred and often necessary for `pydub`-based decoding
- `PyAV` provides a fallback path if direct `pydub` decoding fails

An AI modifying installation docs should reflect this more accurately than the current README does.

## Actual CLI Usage

Observed help output shows the effective command shape is:

```bash
python -m openlecture.cli [OPTIONS] AUDIO_FILE
```

Equivalent installed-console usage should be:

```bash
openlecture [OPTIONS] AUDIO_FILE
```

Example:

```bash
openlecture .\lecture.mp3 --chunk-length-ms 120000
```

## What The Output Looks Like

Generated Markdown is roughly:

```md
# Lecture Transcript

First sentence.

Second sentence.

Third sentence.
```

The formatter does not preserve:

- timestamps
- chunk boundaries
- speaker information
- original segment metadata

## Known Limitations and Risks

### Product limitations

- no speaker diarization
- no timestamps
- no subtitle formats such as `.srt` or `.vtt`
- no JSON or structured transcript output
- no language-selection option exposed in the CLI
- no model-size option exposed in the CLI
- no batching or parallel chunk transcription

### Code / maintenance limitations

- no automated tests were present in the repository during analysis
- no CI configuration was present in the repository during analysis
- README is minimal and does not explain the real runtime behavior in depth
- dependency declarations are inconsistent between `pyproject.toml` and `requirements.txt`
- progress integration depends on monkey-patching an internal `faster-whisper` module attribute
- many exceptions are rewrapped into generic `RuntimeError`, which simplifies UX but can hide precise root causes

### Output-quality limitations

- final transcript is flattened into a single text stream
- Markdown paragraph splitting is simplistic
- formatting assumes period-based sentence boundaries

## Extension Guidance For Another AI

If you need to extend this project, these are the safest mental anchors:

### If you want to change CLI behavior

Start from `openlecture/cli.py`.

Check:

- Typer single-command behavior
- whether `pyproject.toml` entrypoints stay valid
- whether direct-module execution through `main()` still behaves correctly

### If you want to change supported audio formats or decoding

Start from `openlecture/audio_utils.py`.

Check:

- `SUPPORTED_FORMATS`
- `_find_ffmpeg_binaries()`
- `_load_with_pydub()`
- `_load_with_pyav()`

### If you want to change transcription quality, speed, or model settings

Start from `openlecture/transcribe.py`.

Check:

- `MODEL_SIZE`
- `_get_model()`
- `_transcribe_file()`
- `transcribe_audio()`

### If you want richer output

The current bottleneck is architectural:

- `_transcribe_file()` discards segment structure and keeps only text
- `transcribe_audio()` returns a single string
- `output_formatter.py` only knows how to format a string

If you want timestamps, sections, or speaker-aware output, the likely refactor is:

1. preserve structured segment data from Whisper
2. return a richer data structure from `transcribe_audio()`
3. teach the formatter to render that structured data

### If you want to touch progress reporting

Be careful.

The current progress system is coupled to:

- `ContextVar` state
- a shared session object
- runtime monkey-patching of `faster_whisper.transcribe.tqdm`

This is the most delicate part of the codebase.

## Bottom Line

OpenLecture is a focused local transcription CLI:

- input: one local audio file
- engine: `faster-whisper`
- preprocessing: chunking via `pydub`
- decoding fallback: `PyAV`
- output: one simple Markdown transcript

The codebase is small and understandable in one sitting. The most important non-obvious pieces are:

- Typer single-command root behavior
- `ffmpeg` discovery and `PyAV` fallback
- temporary chunk export to `.wav`
- custom progress-bar monkey-patching
- the fact that transcript structure is flattened before formatting
