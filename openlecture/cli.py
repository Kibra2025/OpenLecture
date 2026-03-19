"""CLI entrypoint for OpenLecture."""

import errno
from pathlib import Path
from typing import Callable

import typer

from .output_formatter import transcript_to_markdown
from .transcribe import (
    DEFAULT_BEAM_SIZE,
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DEVICE,
    DEFAULT_MODEL_SIZE,
    transcribe_audio,
)

app = typer.Typer()


def _resolve_output_path(output_path: Path) -> Path:
    """Validate the target output path and create parent directories as needed."""
    if not str(output_path).strip():
        raise ValueError("Output path must not be empty")

    resolved_output_path = output_path.expanduser()

    if resolved_output_path.exists() and resolved_output_path.is_dir():
        raise ValueError(f"Output path points to a directory: {resolved_output_path}")

    parent_dir = resolved_output_path.parent

    if parent_dir.exists() and not parent_dir.is_dir():
        raise ValueError(f"Output parent path is not a directory: {parent_dir}")

    try:
        parent_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise PermissionError(
            f"Permission denied while creating output directory: {parent_dir}"
        ) from exc
    except OSError as exc:
        if exc.errno in {errno.EINVAL, errno.ENAMETOOLONG}:
            raise ValueError(f"Invalid output path: {resolved_output_path}") from exc
        raise RuntimeError(
            f"Failed to prepare output directory: {parent_dir}"
        ) from exc

    return resolved_output_path


def _write_output(output_path: Path, content: str) -> None:
    """Write the transcript to disk with clear filesystem errors."""
    try:
        output_path.write_text(content, encoding="utf-8")
    except PermissionError as exc:
        raise PermissionError(
            f"Permission denied while writing output file: {output_path}"
        ) from exc
    except OSError as exc:
        if exc.errno in {errno.EINVAL, errno.ENAMETOOLONG}:
            raise ValueError(f"Invalid output path: {output_path}") from exc
        raise RuntimeError(f"Failed to write output file: {output_path}") from exc


def _build_progress_callback() -> Callable[[int, int], None]:
    """Create a simple CLI progress renderer based on chunk counts."""
    last_reported = {"current": 0}

    def progress_callback(current: int, total: int) -> None:
        if current == last_reported["current"]:
            return

        last_reported["current"] = current
        typer.secho(f"Processing chunk {current}/{total}", fg=typer.colors.YELLOW)

    return progress_callback


@app.command()
def transcribe(
    audio_file: Path,
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Markdown output path. Defaults to the input filename with a .md extension.",
    ),
    progress: bool = typer.Option(
        True,
        "--progress/--no-progress",
        help="Show transcription progress in the terminal.",
    ),
    chunk_length_ms: int = typer.Option(
        60000,
        "--chunk-length-ms",
        min=1,
        help="Split audio into chunks of this length before transcription.",
    ),
    model: str = typer.Option(
        DEFAULT_MODEL_SIZE,
        "--model",
        help="Whisper model size to load.",
    ),
    beam_size: int = typer.Option(
        DEFAULT_BEAM_SIZE,
        "--beam-size",
        min=1,
        help="Beam size used during decoding.",
    ),
    device: str = typer.Option(
        DEFAULT_DEVICE,
        "--device",
        help="Execution device passed to faster-whisper.",
    ),
    compute_type: str = typer.Option(
        DEFAULT_COMPUTE_TYPE,
        "--compute-type",
        help="Computation type passed to faster-whisper.",
    ),
    language: str | None = typer.Option(
        None,
        "--language",
        help="Optional source language code. If omitted, Whisper auto-detects it.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show the full traceback when an error occurs.",
    ),
) -> None:
    """Transcribe an audio file using Whisper."""
    typer.echo("Starting OpenLecture...")

    try:
        output_path = _resolve_output_path(output or audio_file.with_suffix(".md"))
        transcribe_kwargs = {
            "show_progress": progress,
            "chunk_length_ms": chunk_length_ms,
            "status_callback": typer.echo,
        }

        if progress:
            transcribe_kwargs["progress_callback"] = _build_progress_callback()

        if model != DEFAULT_MODEL_SIZE:
            transcribe_kwargs["model_size"] = model

        if beam_size != DEFAULT_BEAM_SIZE:
            transcribe_kwargs["beam_size"] = beam_size

        if device != DEFAULT_DEVICE:
            transcribe_kwargs["device"] = device

        if compute_type != DEFAULT_COMPUTE_TYPE:
            transcribe_kwargs["compute_type"] = compute_type

        if language is not None:
            transcribe_kwargs["language"] = language

        segments = transcribe_audio(str(audio_file), **transcribe_kwargs)
        markdown_transcript = transcript_to_markdown(segments)
        _write_output(output_path, markdown_transcript)
    except Exception as exc:
        if verbose:
            raise
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.secho(f"Transcript saved to {output_path}", fg=typer.colors.GREEN)


def main() -> None:
    """Run the OpenLecture CLI."""
    app()


if __name__ == "__main__":
    main()
