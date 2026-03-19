"""Validation tests for audio and transcription entrypoints."""

from pathlib import Path

import pytest

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
