"""CLI tests for OpenLecture."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from openlecture import cli
from openlecture.models import Segment


runner = CliRunner()


class FakeProgressBar:
    """Minimal tqdm-like object for CLI progress tests."""

    def __init__(self, total: int = 0) -> None:
        self.total = total
        self.updated_by: list[int] = []
        self.clear_calls = 0
        self.refresh_calls = 0

    def update(self, amount: int) -> None:
        self.updated_by.append(amount)

    def clear(self) -> None:
        self.clear_calls += 1

    def refresh(self) -> None:
        self.refresh_calls += 1


class FakeSmoothProgressBar:
    """Minimal smooth progress bar double for CLI integration tests."""

    def __init__(self, total: int, *, estimated_chunk_duration: float | None = None) -> None:
        self.total = total
        self.estimated_chunk_duration = estimated_chunk_duration
        self.advance_calls: list[tuple[int, int]] = []
        self.clear_calls = 0
        self.refresh_calls = 0
        self.closed = False

    def advance_to(self, current: int, total: int) -> None:
        self.advance_calls.append((current, total))

    def clear(self) -> None:
        self.clear_calls += 1

    def refresh(self) -> None:
        self.refresh_calls += 1

    def close(self) -> None:
        self.closed = True


def test_progress_callback_updates_tqdm_incrementally() -> None:
    """The CLI progress callback should increment tqdm without resetting it."""
    progress_bar = FakeProgressBar(total=8)
    progress_callback = cli._build_progress_callback(progress_bar)

    progress_callback(1, 8)
    progress_callback(1, 8)
    progress_callback(3, 8)

    assert progress_bar.total == 8
    assert progress_bar.updated_by == [1, 2]


def test_status_callback_replaces_direct_transcribe_message(monkeypatch) -> None:
    """Small-file CLI status should use a simple transcribing message."""
    emitted_messages: list[str] = []

    monkeypatch.setattr(cli.typer, "echo", lambda message: emitted_messages.append(message))
    status_callback = cli._build_status_callback(replace_direct_transcribe_status=True)

    status_callback("Loading Whisper model...")
    status_callback("Transcribing audio directly (02:00 total audio)...")
    status_callback("Transcribing audio directly (02:00 total audio)...")

    assert emitted_messages == [
        "Loading Whisper model...",
        "Transcribing...",
    ]


def test_cli_uses_estimated_progress_bar_for_small_files(
    workspace_tmp_path: Path,
    monkeypatch,
) -> None:
    """Small files with known duration should use a smooth estimated progress bar."""
    audio_file = workspace_tmp_path / "lecture.mp3"
    audio_file.write_bytes(b"fake audio")
    output_file = workspace_tmp_path / "custom.md"
    created_bars: list[FakeSmoothProgressBar] = []

    def fake_smooth_progress_bar(
        total: int,
        *,
        estimated_chunk_duration: float | None = None,
    ) -> FakeSmoothProgressBar:
        progress_bar = FakeSmoothProgressBar(
            total,
            estimated_chunk_duration=estimated_chunk_duration,
        )
        created_bars.append(progress_bar)
        return progress_bar

    def fake_transcribe_audio(audio_path: str, **kwargs) -> list[Segment]:
        assert audio_path == str(audio_file)
        assert kwargs["show_progress"] is True
        assert "progress_callback" in kwargs
        kwargs["status_callback"]("Loading Whisper model...")
        kwargs["status_callback"]("Transcribing audio directly (00:10 total audio)...")
        kwargs["progress_callback"](1, 1)
        return [Segment(start=0.0, end=1.0, text="Only sentence.")]

    monkeypatch.setattr(cli, "_SmoothTqdmProgressBar", fake_smooth_progress_bar)
    monkeypatch.setattr(cli, "get_audio_duration_seconds", lambda _audio_path: 10.0)
    monkeypatch.setattr(cli, "transcribe_audio", fake_transcribe_audio)

    result = runner.invoke(
        cli.app,
        [str(audio_file), "--output", str(output_file)],
    )

    assert result.exit_code == 0
    assert len(created_bars) == 1
    assert created_bars[0].total == 1
    assert created_bars[0].estimated_chunk_duration == pytest.approx(12.0)
    assert created_bars[0].advance_calls == [(1, 1)]
    assert created_bars[0].clear_calls == 2
    assert created_bars[0].refresh_calls == 2
    assert created_bars[0].closed is True
    assert output_file.read_text(encoding="utf-8") == "# Lecture Transcript\n\nOnly sentence."


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
        status_callback,
    ) -> list[Segment]:
        assert audio_path == str(audio_file)
        assert show_progress is False
        assert chunk_length_ms == 120000
        assert status_callback is not None
        return [
            Segment(start=1.0, end=2.0, text="First sentence."),
            Segment(start=3.0, end=4.0, text="Second sentence."),
        ]

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


def test_cli_forwards_custom_transcription_options(
    workspace_tmp_path: Path,
    monkeypatch,
) -> None:
    """The CLI should forward custom Whisper options to the transcription layer."""
    audio_file = workspace_tmp_path / "lecture.mp3"
    audio_file.write_bytes(b"fake audio")
    captured_kwargs: dict[str, object] = {}

    def fake_transcribe_audio(audio_path: str, **kwargs) -> list[Segment]:
        assert audio_path == str(audio_file)
        captured_kwargs.update(kwargs)
        return [Segment(start=0.0, end=1.0, text="Only sentence.")]

    monkeypatch.setattr(cli, "transcribe_audio", fake_transcribe_audio)

    result = runner.invoke(
        cli.app,
        [
            str(audio_file),
            "--no-progress",
            "--model",
            "small",
            "--beam-size",
            "3",
            "--device",
            "cpu",
            "--compute-type",
            "int8",
            "--language",
            "it",
        ],
    )

    assert result.exit_code == 0
    assert captured_kwargs == {
        "show_progress": False,
        "chunk_length_ms": 60000,
        "status_callback": cli.typer.echo,
        "model_size": "small",
        "beam_size": 3,
        "device": "cpu",
        "compute_type": "int8",
        "language": "it",
    }


def test_custom_output_file_is_created(
    workspace_tmp_path: Path,
    monkeypatch,
) -> None:
    """The CLI should respect an explicit output path."""
    audio_file = workspace_tmp_path / "lecture.mp3"
    audio_file.write_bytes(b"fake audio")
    output_file = workspace_tmp_path / "custom.md"

    def fake_transcribe_audio(*args, **kwargs) -> list[Segment]:
        return [Segment(start=0.0, end=1.0, text="Only sentence.")]

    monkeypatch.setattr(cli, "transcribe_audio", fake_transcribe_audio)

    result = runner.invoke(
        cli.app,
        [str(audio_file), "--output", str(output_file)],
    )

    assert result.exit_code == 0
    assert output_file.exists()
    assert output_file.read_text(encoding="utf-8") == "# Lecture Transcript\n\nOnly sentence."
    assert not audio_file.with_suffix(".md").exists()


def test_cli_can_enable_timestamps(
    workspace_tmp_path: Path,
    monkeypatch,
) -> None:
    """The CLI should include Markdown timestamps only when explicitly requested."""
    audio_file = workspace_tmp_path / "lecture.mp3"
    audio_file.write_bytes(b"fake audio")
    output_file = workspace_tmp_path / "custom.md"

    def fake_transcribe_audio(*args, **kwargs) -> list[Segment]:
        return [Segment(start=0.0, end=1.0, text="Only sentence.")]

    monkeypatch.setattr(cli, "transcribe_audio", fake_transcribe_audio)

    result = runner.invoke(
        cli.app,
        [str(audio_file), "--output", str(output_file), "--timestamps"],
    )

    assert result.exit_code == 0
    assert output_file.exists()
    assert output_file.read_text(encoding="utf-8") == (
        "# Lecture Transcript\n\n[00:00:00] Only sentence."
    )


def test_output_parent_directories_are_created(
    workspace_tmp_path: Path,
    monkeypatch,
) -> None:
    """The CLI should create missing parent directories for the output file."""
    audio_file = workspace_tmp_path / "lecture.mp3"
    audio_file.write_bytes(b"fake audio")
    output_file = workspace_tmp_path / "nested" / "dir" / "custom.md"

    def fake_transcribe_audio(*args, **kwargs) -> list[Segment]:
        return [Segment(start=0.0, end=1.0, text="Only sentence.")]

    monkeypatch.setattr(cli, "transcribe_audio", fake_transcribe_audio)

    result = runner.invoke(
        cli.app,
        [str(audio_file), "--output", str(output_file)],
    )

    assert result.exit_code == 0
    assert output_file.exists()
    assert output_file.read_text(encoding="utf-8") == "# Lecture Transcript\n\nOnly sentence."


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

    def fail_transcription(*args, **kwargs) -> list[Segment]:
        raise RuntimeError("Failed to load Whisper model")

    monkeypatch.setattr(cli, "transcribe_audio", fail_transcription)

    result = runner.invoke(cli.app, [str(audio_file)])

    assert result.exit_code == 1
    assert "Error: Failed to load Whisper model" in result.output
    assert "Traceback" not in result.output


def test_cli_shows_explicit_error_for_markdown_input(workspace_tmp_path: Path) -> None:
    """The CLI should clearly reject transcript files passed as input."""
    transcript_file = workspace_tmp_path / "lecture.md"
    transcript_file.write_text("# Lecture Transcript\n\nHello", encoding="utf-8")

    result = runner.invoke(cli.app, [str(transcript_file)])

    assert result.exit_code == 1
    assert "Expected an audio file, but got a transcript or text file" in result.output
    assert "Traceback" not in result.output


def test_cli_reraises_errors_in_verbose_mode(
    workspace_tmp_path: Path,
    monkeypatch,
) -> None:
    """The CLI should re-raise exceptions when verbose mode is enabled."""
    audio_file = workspace_tmp_path / "lecture.mp3"
    audio_file.write_bytes(b"fake audio")

    def fail_transcription(*args, **kwargs) -> list[Segment]:
        raise RuntimeError("Failed to load Whisper model")

    monkeypatch.setattr(cli, "transcribe_audio", fail_transcription)

    with pytest.raises(RuntimeError, match="Failed to load Whisper model"):
        runner.invoke(cli.app, [str(audio_file), "--verbose"], catch_exceptions=False)
