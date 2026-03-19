"""Validation tests for audio and transcription entrypoints."""

from pathlib import Path

import pytest

from openlecture import audio_utils, transcribe
from openlecture.audio_utils import split_audio
from openlecture.transcribe import transcribe_audio


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


def test_split_audio_raises_specific_decode_error(
    workspace_tmp_path: Path,
    monkeypatch,
) -> None:
    """split_audio should expose a distinct decode failure error."""
    audio_file = workspace_tmp_path / "lecture.mp3"
    audio_file.write_bytes(b"fake audio")

    monkeypatch.setattr(audio_utils, "_get_audio_segment_class", lambda: object())

    def fail_pydub(*args, **kwargs):
        raise RuntimeError("pydub failed")

    def fail_pyav(*args, **kwargs):
        raise RuntimeError("pyav failed")

    monkeypatch.setattr(audio_utils, "_load_with_pydub", fail_pydub)
    monkeypatch.setattr(audio_utils, "_load_with_pyav", fail_pyav)

    with pytest.raises(RuntimeError, match="Failed to decode audio file"):
        split_audio(str(audio_file))


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

    def fail_model_load():
        raise RuntimeError("whisper init failed")

    monkeypatch.setattr(transcribe, "_get_model", fail_model_load)

    with pytest.raises(RuntimeError, match="Failed to load Whisper model"):
        transcribe_audio(str(audio_file))
