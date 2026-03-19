"""Tests for explicit progress reporting in the transcription pipeline."""

from pathlib import Path

from openlecture import transcribe
from openlecture.models import Segment
from openlecture.transcribe import transcribe_audio


class FakeChunk:
    """Minimal chunk object exposing only a duration."""

    def __init__(self, length_ms: int) -> None:
        self.length_ms = length_ms

    def __len__(self) -> int:
        return self.length_ms


def test_progress_callback_is_called_for_each_chunk(
    workspace_tmp_path: Path,
    monkeypatch,
) -> None:
    """Chunk-level progress should be reported through the explicit callback."""
    audio_file = workspace_tmp_path / "lecture.mp3"
    audio_file.write_bytes(b"fake audio")
    chunks = [FakeChunk(60000), FakeChunk(60000)]

    monkeypatch.setattr(transcribe, "_get_model", lambda: object())
    monkeypatch.setattr(transcribe, "get_audio_duration_seconds", lambda _audio_path: 600.0)
    monkeypatch.setattr(transcribe, "iter_audio_chunks", lambda *args, **kwargs: iter(chunks))
    monkeypatch.setattr(
        transcribe,
        "_transcribe_chunk",
        lambda model, chunk, **kwargs: [
            Segment(
                start=kwargs["time_offset_seconds"],
                end=kwargs["time_offset_seconds"] + 1.0,
                text=str(len(chunk)),
            )
        ],
    )

    progress_events: list[tuple[int, int]] = []
    status_messages: list[str] = []

    result = transcribe_audio(
        str(audio_file),
        status_callback=status_messages.append,
        progress_callback=lambda current, total: progress_events.append((current, total)),
        overlap_ms=0,
    )

    assert result == [
        Segment(start=0.0, end=1.0, text="60000"),
        Segment(start=60.0, end=61.0, text="60000"),
    ]
    assert progress_events == [(1, 10), (2, 10)]
    assert status_messages == [
        "Loading Whisper model...",
        "Splitting audio into chunks...",
        "Transcribing 10 chunk(s) (10:00 total audio)...",
    ]


def test_progress_falls_back_to_status_messages_when_callback_fails(
    workspace_tmp_path: Path,
    monkeypatch,
) -> None:
    """Progress callback failures should not stop transcription."""
    audio_file = workspace_tmp_path / "lecture.mp3"
    audio_file.write_bytes(b"fake audio")
    chunks = [FakeChunk(60000), FakeChunk(60000)]

    monkeypatch.setattr(transcribe, "_get_model", lambda: object())
    monkeypatch.setattr(transcribe, "get_audio_duration_seconds", lambda _audio_path: 600.0)
    monkeypatch.setattr(transcribe, "iter_audio_chunks", lambda *args, **kwargs: iter(chunks))
    monkeypatch.setattr(
        transcribe,
        "_transcribe_chunk",
        lambda model, chunk, **kwargs: [
            Segment(
                start=kwargs["time_offset_seconds"],
                end=kwargs["time_offset_seconds"] + 1.0,
                text=str(len(chunk)),
            )
        ],
    )

    status_messages: list[str] = []

    def broken_progress_callback(current: int, total: int) -> None:
        raise RuntimeError("progress unavailable")

    result = transcribe_audio(
        str(audio_file),
        status_callback=status_messages.append,
        progress_callback=broken_progress_callback,
        overlap_ms=0,
    )

    assert result == [
        Segment(start=0.0, end=1.0, text="60000"),
        Segment(start=60.0, end=61.0, text="60000"),
    ]
    assert status_messages == [
        "Loading Whisper model...",
        "Splitting audio into chunks...",
        "Transcribing 10 chunk(s) (10:00 total audio)...",
        "Processing chunk 1/10",
        "Processing chunk 2/10",
    ]


def test_transcription_still_works_without_progress(
    workspace_tmp_path: Path,
    monkeypatch,
) -> None:
    """The pipeline should still work when progress reporting is disabled."""
    audio_file = workspace_tmp_path / "lecture.mp3"
    audio_file.write_bytes(b"fake audio")

    monkeypatch.setattr(transcribe, "_get_model", lambda: object())
    monkeypatch.setattr(transcribe, "get_audio_duration_seconds", lambda _audio_path: 120.0)
    monkeypatch.setattr(
        transcribe,
        "_transcribe_file",
        lambda model, audio_file, **kwargs: [
            Segment(start=0.0, end=1.0, text="transcript")
        ],
    )

    result = transcribe_audio(str(audio_file), show_progress=False)

    assert result == [Segment(start=0.0, end=1.0, text="transcript")]
