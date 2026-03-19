"""CLI entrypoint for OpenLecture."""

from pathlib import Path

import typer

from .output_formatter import transcript_to_markdown
from .transcribe import transcribe_audio

app = typer.Typer()


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
) -> None:
    """Transcribe an audio file using Whisper."""
    typer.echo("Starting OpenLecture...")

    try:
        output_path = output or audio_file.with_suffix(".md")
        transcript = transcribe_audio(
            str(audio_file),
            show_progress=progress,
            chunk_length_ms=chunk_length_ms,
        )
        markdown_transcript = transcript_to_markdown(transcript)
        output_path.write_text(markdown_transcript, encoding="utf-8")
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Transcript saved to {output_path}")


def main() -> None:
    """Run the OpenLecture CLI."""
    app()


if __name__ == "__main__":
    main()
