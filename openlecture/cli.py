"""CLI entrypoint for OpenLecture."""

from contextlib import suppress
import errno
import os
from pathlib import Path

import typer

from .output_formatter import transcript_to_markdown
from .transcribe import transcribe_audio

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
        transcript = transcribe_audio(
            str(audio_file),
            show_progress=progress,
            chunk_length_ms=chunk_length_ms,
        )
        markdown_transcript = transcript_to_markdown(transcript)
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
