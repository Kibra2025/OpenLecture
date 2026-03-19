"""CLI tests for OpenLecture."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from openlecture import cli


runner = CliRunner()


def test_help_works() -> None:
    """The CLI should expose help text successfully."""
    result = runner.invoke(cli.app, ["--help"])

    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "Transcribe an audio file using Whisper." in result.output


def test_default_output_file_is_created(
    workspace_tmp_path: Path,
    monkeypatch,
) -> None:
    """The CLI should write Markdown next to the input file by default."""
    audio_file = workspace_tmp_path / "lecture.mp3"
    audio_file.write_bytes(b"fake audio")

    def fake_transcribe_audio(
        audio_path: str,
        *,
        show_progress: bool,
        chunk_length_ms: int,
    ) -> str:
        assert audio_path == str(audio_file)
        assert show_progress is False
        assert chunk_length_ms == 120000
        return "First sentence. Second sentence."

    monkeypatch.setattr(cli, "transcribe_audio", fake_transcribe_audio)

    result = runner.invoke(
        cli.app,
        [str(audio_file), "--no-progress", "--chunk-length-ms", "120000"],
    )

    output_file = audio_file.with_suffix(".md")
    assert result.exit_code == 0
    assert output_file.exists()
    assert output_file.read_text(encoding="utf-8") == (
        "# Lecture Transcript\n\nFirst sentence.\n\nSecond sentence."
    )
    assert f"Transcript saved to {output_file}" in result.output


def test_custom_output_file_is_created(
    workspace_tmp_path: Path,
    monkeypatch,
) -> None:
    """The CLI should respect an explicit output path."""
    audio_file = workspace_tmp_path / "lecture.mp3"
    audio_file.write_bytes(b"fake audio")
    output_file = workspace_tmp_path / "custom.md"

    def fake_transcribe_audio(*args, **kwargs) -> str:
        return "Only sentence."

    monkeypatch.setattr(cli, "transcribe_audio", fake_transcribe_audio)

    result = runner.invoke(
        cli.app,
        [str(audio_file), "--output", str(output_file)],
    )

    assert result.exit_code == 0
    assert output_file.exists()
    assert output_file.read_text(encoding="utf-8") == (
        "# Lecture Transcript\n\nOnly sentence."
    )
    assert not audio_file.with_suffix(".md").exists()


def test_output_parent_directories_are_created(
    workspace_tmp_path: Path,
    monkeypatch,
) -> None:
    """The CLI should create missing parent directories for the output file."""
    audio_file = workspace_tmp_path / "lecture.mp3"
    audio_file.write_bytes(b"fake audio")
    output_file = workspace_tmp_path / "nested" / "dir" / "custom.md"

    def fake_transcribe_audio(*args, **kwargs) -> str:
        return "Only sentence."

    monkeypatch.setattr(cli, "transcribe_audio", fake_transcribe_audio)

    result = runner.invoke(
        cli.app,
        [str(audio_file), "--output", str(output_file)],
    )

    assert result.exit_code == 0
    assert output_file.exists()
    assert output_file.read_text(encoding="utf-8") == (
        "# Lecture Transcript\n\nOnly sentence."
    )


def test_missing_input_argument_returns_error() -> None:
    """The CLI should reject invocations without an audio file."""
    result = runner.invoke(cli.app, [])

    assert result.exit_code == 2
    assert "Missing argument" in result.output


def test_cli_uses_root_command_style() -> None:
    """The supported public CLI style is ``openlecture file.mp3``."""
    result = runner.invoke(cli.app, ["transcribe", "lecture.mp3"])

    assert result.exit_code == 2


def test_cli_shows_clean_error_message_without_verbose(
    workspace_tmp_path: Path,
    monkeypatch,
) -> None:
    """The CLI should show a concise error in normal mode."""
    audio_file = workspace_tmp_path / "lecture.mp3"
    audio_file.write_bytes(b"fake audio")

    def fail_transcription(*args, **kwargs) -> str:
        raise RuntimeError("Failed to load Whisper model")

    monkeypatch.setattr(cli, "transcribe_audio", fail_transcription)

    result = runner.invoke(cli.app, [str(audio_file)])

    assert result.exit_code == 1
    assert "Error: Failed to load Whisper model" in result.output
    assert "Traceback" not in result.output


def test_cli_reraises_errors_in_verbose_mode(
    workspace_tmp_path: Path,
    monkeypatch,
) -> None:
    """The CLI should re-raise exceptions when verbose mode is enabled."""
    audio_file = workspace_tmp_path / "lecture.mp3"
    audio_file.write_bytes(b"fake audio")

    def fail_transcription(*args, **kwargs) -> str:
        raise RuntimeError("Failed to load Whisper model")

    monkeypatch.setattr(cli, "transcribe_audio", fail_transcription)

    with pytest.raises(RuntimeError, match="Failed to load Whisper model"):
        runner.invoke(cli.app, [str(audio_file), "--verbose"], catch_exceptions=False)
