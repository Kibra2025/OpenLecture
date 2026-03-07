import typer
import sys
from pathlib import Path
from .transcribe import transcribe_audio

app = typer.Typer()


@app.command()
def transcribe(
    audio_file: Path,
    output: str = "transcript.txt",
    progress: bool = typer.Option(
        True,
        "--progress/--no-progress",
        help="Show transcription progress in the terminal.",
    ),
):
    """
    Transcribe an audio file using Whisper.
    """

    typer.echo("Starting OpenLecture...")

    try:
        transcript = transcribe_audio(str(audio_file), show_progress=progress)

        with open(output, "w", encoding="utf-8") as f:
            f.write(transcript)
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Transcript saved to {output}")


def main() -> None:
    # Typer promotes a single command to the app root, so accept the
    # explicit subcommand form users naturally try as well.
    if len(sys.argv) > 1 and sys.argv[1] == "transcribe":
        sys.argv.pop(1)

    app()


if __name__ == "__main__":
    main()
