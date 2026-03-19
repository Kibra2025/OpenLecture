"""Validation tests for audio and transcription entrypoints."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from openlecture import audio_utils, transcribe
from openlecture.audio_utils import DEFAULT_OVERLAP_MS, iter_audio_chunks, split_audio
from openlecture.models import Segment
from openlecture.transcribe import transcribe_audio


class FakeAudio:
    """Minimal audio object for chunk iteration tests."""

    def __init__(self, length_ms: int) -> None:
        self.length_ms = length_ms

    def __len__(self) -> int:
        return self.length_ms

    def __getitem__(self, item):
        start = 0 if item.start is None else item.start
        stop = self.length_ms if item.stop is None else min(item.stop, self.length_ms)
        return FakeAudio(max(0, stop - start))


class FakeExportableChunk:
    """Minimal chunk object with export support."""

    def __init__(self, length_ms: int, payload: bytes = b"chunk") -> None:
        self.length_ms = length_ms
        self.payload = payload

    def __len__(self) -> int:
        return self.length_ms

    def export(self, output_path: str, format: str) -> None:
        assert format == "wav"
        Path(output_path).write_bytes(self.payload)


def test_split_audio_rejects_empty_path() -> None:
    """split_audio should reject empty input paths."""
    with pytest.raises(ValueError, match="audio_path must not be empty"):
        split_audio("")


def test_split_audio_rejects_missing_file(workspace_tmp_path: Path) -> None:
    """split_audio should reject files that do not exist."""
    missing_file = workspace_tmp_path / "missing.mp3"

    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        split_audio(str(missing_file))


def test_split_audio_rejects_directory_path(workspace_tmp_path: Path) -> None:
    """split_audio should reject directory paths."""
    with pytest.raises(ValueError, match="Audio path is not a file"):
        split_audio(str(workspace_tmp_path))


@pytest.mark.parametrize("chunk_length_ms", [0, -1])
def test_split_audio_rejects_non_positive_chunk_length(
    workspace_tmp_path: Path,
    chunk_length_ms: int,
) -> None:
    """split_audio should reject non-positive chunk lengths."""
    audio_file = workspace_tmp_path / "lecture.mp3"
    audio_file.write_bytes(b"fake audio")

    with pytest.raises(ValueError, match="chunk_length_ms must be greater than 0"):
        split_audio(str(audio_file), chunk_length_ms=chunk_length_ms)


def test_iter_audio_chunks_yields_one_chunk_at_a_time(
    workspace_tmp_path: Path,
    monkeypatch,
) -> None:
    """iter_audio_chunks should stream chunks without building a full list first."""
    audio_file = workspace_tmp_path / "lecture.mp3"
    audio_file.write_bytes(b"fake audio")

    monkeypatch.setattr(
        audio_utils,
        "_load_audio_segment",
        lambda _audio_file: FakeAudio(150000),
    )

    chunks = list(
        iter_audio_chunks(
            str(audio_file),
            chunk_length_ms=60000,
            overlap_ms=0,
        )
    )

    assert [len(chunk) for chunk in chunks] == [60000, 60000, 30000]


def test_iter_audio_chunks_applies_overlap_by_default(
    workspace_tmp_path: Path,
    monkeypatch,
) -> None:
    """iter_audio_chunks should keep a small overlap between adjacent chunks."""
    audio_file = workspace_tmp_path / "lecture.mp3"
    audio_file.write_bytes(b"fake audio")

    monkeypatch.setattr(
        audio_utils,
        "_load_audio_segment",
        lambda _audio_file: FakeAudio(150000),
    )

    chunks = list(iter_audio_chunks(str(audio_file), chunk_length_ms=60000))

    assert [len(chunk) for chunk in chunks] == [60000, 60000, 34000]


def test_split_audio_raises_specific_decode_error(
    workspace_tmp_path: Path,
    monkeypatch,
) -> None:
    """split_audio should expose a distinct decode failure error."""
    audio_file = workspace_tmp_path / "lecture.mp3"
    audio_file.write_bytes(b"fake audio")

    monkeypatch.setattr(
        audio_utils,
        "_load_audio_segment",
        lambda _audio_file: (_ for _ in ()).throw(RuntimeError("decode failed")),
    )

    with pytest.raises(RuntimeError, match="decode failed"):
        list(iter_audio_chunks(str(audio_file)))


def test_get_audio_duration_seconds_uses_pyav_microseconds(
    workspace_tmp_path: Path,
    monkeypatch,
) -> None:
    """PyAV container durations should be converted from microseconds to seconds."""
    audio_file = workspace_tmp_path / "lecture.mp3"
    audio_file.write_bytes(b"fake audio")

    class FakeContainer:
        duration = 188_290_612
        streams = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_av = SimpleNamespace(
        time_base=1_000_000,
        open=lambda _path: FakeContainer(),
    )
    monkeypatch.setitem(sys.modules, "av", fake_av)

    duration = audio_utils.get_audio_duration_seconds(str(audio_file))

    assert duration == pytest.approx(188.290612)


def test_transcribe_audio_rejects_empty_path() -> None:
    """transcribe_audio should reject empty input paths before model loading."""
    with pytest.raises(ValueError, match="audio_path must not be empty"):
        transcribe_audio("")


def test_transcribe_audio_rejects_missing_file(workspace_tmp_path: Path) -> None:
    """transcribe_audio should reject files that do not exist."""
    missing_file = workspace_tmp_path / "missing.mp3"

    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        transcribe_audio(str(missing_file))


def test_transcribe_audio_rejects_directory_path(workspace_tmp_path: Path) -> None:
    """transcribe_audio should reject directory paths."""
    with pytest.raises(ValueError, match="Audio path is not a file"):
        transcribe_audio(str(workspace_tmp_path))


@pytest.mark.parametrize("chunk_length_ms", [0, -1])
def test_transcribe_audio_rejects_non_positive_chunk_length(
    chunk_length_ms: int,
) -> None:
    """transcribe_audio should reject non-positive chunk lengths."""
    with pytest.raises(ValueError, match="chunk_length_ms must be greater than 0"):
        transcribe_audio("lecture.mp3", chunk_length_ms=chunk_length_ms)


def test_transcribe_audio_raises_specific_model_load_error(
    workspace_tmp_path: Path,
    monkeypatch,
) -> None:
    """transcribe_audio should expose a distinct model load failure error."""
    audio_file = workspace_tmp_path / "lecture.mp3"
    audio_file.write_bytes(b"fake audio")

    monkeypatch.setattr(transcribe, "get_audio_duration_seconds", lambda _audio_path: 120.0)

    def fail_model_load():
        raise RuntimeError("whisper init failed")

    monkeypatch.setattr(transcribe, "_get_model", fail_model_load)

    with pytest.raises(RuntimeError, match="Failed to load Whisper model"):
        transcribe_audio(str(audio_file))


def test_transcribe_file_returns_structured_segments_with_absolute_offsets() -> None:
    """_transcribe_file should preserve timestamps, trim text, and pass options."""
    captured_kwargs: dict[str, object] = {}

    class FakeModel:
        def transcribe(self, audio_path: str, **kwargs):
            assert audio_path.endswith("lecture.wav")
            captured_kwargs.update(kwargs)
            return (
                [
                    SimpleNamespace(start=0.2, end=1.4, text=" First line. "),
                    SimpleNamespace(start=1.5, end=2.0, text="   "),
                    SimpleNamespace(start=2.1, end=3.6, text="Second line"),
                ],
                None,
            )

    result = transcribe._transcribe_file(
        FakeModel(),
        Path("lecture.wav"),
        show_progress=False,
        beam_size=7,
        language="it",
        time_offset_seconds=60.0,
    )

    assert captured_kwargs == {"beam_size": 7, "vad_filter": True, "language": "it"}
    assert result == [
        Segment(start=60.2, end=61.4, text="First line."),
        Segment(start=62.1, end=63.6, text="Second line"),
    ]


def test_transcribe_audio_forwards_model_and_decode_options(
    workspace_tmp_path: Path,
    monkeypatch,
) -> None:
    """transcribe_audio should forward runtime configuration to faster-whisper."""
    audio_file = workspace_tmp_path / "lecture.mp3"
    audio_file.write_bytes(b"fake audio")
    model_load_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class FakeModel:
        def transcribe(self, audio_path: str, **kwargs):
            assert audio_path == str(audio_file)
            assert kwargs == {"beam_size": 3, "vad_filter": True, "language": "it"}
            return ([SimpleNamespace(start=0.0, end=1.0, text="ciao")], None)

    def fake_get_model(*args, **kwargs):
        model_load_calls.append((args, kwargs))
        return FakeModel()

    monkeypatch.setattr(transcribe, "_get_model", fake_get_model)
    monkeypatch.setattr(transcribe, "get_audio_duration_seconds", lambda _audio_path: 120.0)

    result = transcribe_audio(
        str(audio_file),
        show_progress=False,
        model_size="small",
        beam_size=3,
        device="cpu",
        compute_type="int8",
        language="it",
    )

    assert model_load_calls == [
        ((), {"model_size": "small", "device": "cpu", "compute_type": "int8"})
    ]
    assert result == [Segment(start=0.0, end=1.0, text="ciao")]


def test_transcribe_audio_skips_chunking_for_small_files(
    workspace_tmp_path: Path,
    monkeypatch,
) -> None:
    """Small files should be transcribed directly without chunk iteration."""
    audio_file = workspace_tmp_path / "lecture.mp3"
    audio_file.write_bytes(b"fake audio")

    monkeypatch.setattr(transcribe, "_get_model", lambda: object())
    monkeypatch.setattr(transcribe, "get_audio_duration_seconds", lambda _audio_path: 120.0)
    monkeypatch.setattr(
        transcribe,
        "_transcribe_file",
        lambda model, audio_file, **kwargs: [
            Segment(start=0.0, end=1.0, text=audio_file.name)
        ],
    )
    monkeypatch.setattr(
        transcribe,
        "iter_audio_chunks",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not chunk")),
    )

    messages: list[str] = []
    result = transcribe_audio(
        str(audio_file),
        show_progress=False,
        status_callback=messages.append,
    )

    assert result == [Segment(start=0.0, end=1.0, text="lecture.mp3")]
    assert messages == [
        "Loading Whisper model...",
        "Transcribing audio directly (02:00 total audio)...",
    ]


def test_transcribe_chunk_deletes_temp_file_after_use(monkeypatch) -> None:
    """Each exported chunk file should be deleted immediately after transcription."""
    observed_path: Path | None = None

    def fake_transcribe_file(model, audio_file: Path, **kwargs):
        nonlocal observed_path
        observed_path = audio_file
        assert audio_file.exists()
        return [Segment(start=0.0, end=1.0, text="ok")]

    monkeypatch.setattr(transcribe, "_transcribe_file", fake_transcribe_file)

    result = transcribe._transcribe_chunk(
        object(),
        FakeExportableChunk(1000),
        show_progress=False,
    )

    assert result == [Segment(start=0.0, end=1.0, text="ok")]
    assert observed_path is not None
    assert not observed_path.exists()


def test_transcribe_audio_discards_segments_fully_inside_overlap(
    workspace_tmp_path: Path,
    monkeypatch,
) -> None:
    """Overlapped chunks should drop segments fully covered by earlier audio."""
    audio_file = workspace_tmp_path / "lecture.mp3"
    audio_file.write_bytes(b"fake audio")

    chunks = [FakeExportableChunk(60000), FakeExportableChunk(60000)]

    monkeypatch.setattr(transcribe, "_get_model", lambda: object())
    monkeypatch.setattr(transcribe, "get_audio_duration_seconds", lambda _audio_path: 600.0)
    monkeypatch.setattr(transcribe, "iter_audio_chunks", lambda *args, **kwargs: iter(chunks))

    def fake_transcribe_chunk(model, chunk, **kwargs):
        start = kwargs["time_offset_seconds"]
        if start == 0.0:
            return [Segment(start=58.0, end=60.5, text="tail")]
        return [
            Segment(start=start, end=start + 1.5, text="duplicate overlap"),
            Segment(start=start + 1.5, end=start + 2.5, text="carry over"),
            Segment(start=start + 3.0, end=start + 4.0, text="new text"),
        ]

    monkeypatch.setattr(transcribe, "_transcribe_chunk", fake_transcribe_chunk)

    result = transcribe_audio(
        str(audio_file),
        show_progress=False,
        chunk_length_ms=60000,
        overlap_ms=DEFAULT_OVERLAP_MS,
    )

    assert result == [
        Segment(start=58.0, end=60.5, text="tail"),
        Segment(start=59.5, end=60.5, text="carry over"),
        Segment(start=61.0, end=62.0, text="new text"),
    ]


def test_transcribe_audio_emits_status_only_through_callback(
    workspace_tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """transcribe_audio should stay silent unless the caller provides a callback."""
    audio_file = workspace_tmp_path / "lecture.mp3"
    audio_file.write_bytes(b"fake audio")

    chunks = [FakeExportableChunk(120000), FakeExportableChunk(180000)]

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

    messages: list[str] = []
    result = transcribe_audio(
        str(audio_file),
        show_progress=False,
        chunk_length_ms=300000,
        overlap_ms=0,
        status_callback=messages.append,
    )

    assert result == [
        Segment(start=0.0, end=1.0, text="120000"),
        Segment(start=120.0, end=121.0, text="180000"),
    ]
    assert messages == [
        "Loading Whisper model...",
        "Splitting audio into chunks...",
        "Transcribing 2 chunk(s) (10:00 total audio)...",
        "Transcribing chunk 1/2...",
        "Transcribing chunk 2/2...",
    ]

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_transcribe_audio_stays_silent_without_status_callback(
    workspace_tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """transcribe_audio should not write CLI output when no callback is provided."""
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
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
