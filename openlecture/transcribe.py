"""Core transcription pipeline for OpenLecture."""

from __future__ import annotations

from contextlib import contextmanager, suppress
from functools import lru_cache
from math import ceil
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Callable, Iterator

from .audio_utils import DEFAULT_OVERLAP_MS, get_audio_duration_seconds, iter_audio_chunks
from .models import Segment

if TYPE_CHECKING:
    from faster_whisper import WhisperModel
    from pydub import AudioSegment


StatusCallback = Callable[[str], None]
ProgressCallback = Callable[[int, int], None]

DEFAULT_MODEL_SIZE = "medium"
DEFAULT_BEAM_SIZE = 5
DEFAULT_DEVICE = "auto"
DEFAULT_COMPUTE_TYPE = "auto"
DEFAULT_CHUNK_LENGTH_MS = 60_000
SMALL_FILE_CHUNKING_THRESHOLD_SECONDS = 300


def _format_clock(seconds: float | None, *, round_up: bool = False) -> str:
    """Format seconds as MM:SS or HH:MM:SS for terminal output."""
    if seconds is None:
        return "--:--"

    whole_seconds = max(0, ceil(seconds) if round_up else int(seconds))
    minutes, secs = divmod(whole_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    return f"{minutes:02d}:{secs:02d}"


def _emit_status(status_callback: StatusCallback | None, message: str) -> None:
    """Send a status message to the caller when a callback is provided."""
    if status_callback is not None:
        status_callback(message)


def _report_progress(
    *,
    show_progress: bool,
    progress_callback: ProgressCallback | None,
    status_callback: StatusCallback | None,
    current: int,
    total: int,
) -> None:
    """Report chunk-level progress with a safe fallback to status messages."""
    if not show_progress:
        return

    if progress_callback is not None:
        try:
            progress_callback(current, total)
            return
        except Exception:
            pass

    _emit_status(status_callback, f"Processing chunk {current}/{total}")


def _normalize_required_text_option(name: str, value: str) -> str:
    """Normalize required text configuration values."""
    if not value or not value.strip():
        raise ValueError(f"{name} must not be empty.")

    return value.strip()


def _normalize_language(language: str | None) -> str | None:
    """Normalize the optional Whisper language parameter."""
    if language is None:
        return None

    normalized = language.strip()
    return normalized or None


def _validate_transcription_options(
    *,
    chunk_length_ms: int,
    beam_size: int,
    overlap_ms: int,
) -> None:
    """Validate transcription options exposed to callers."""
    if chunk_length_ms <= 0:
        raise ValueError("chunk_length_ms must be greater than 0.")

    if beam_size <= 0:
        raise ValueError("beam_size must be greater than 0.")

    if overlap_ms < 0:
        raise ValueError("overlap_ms must be greater than or equal to 0.")

    if overlap_ms >= chunk_length_ms:
        raise ValueError("overlap_ms must be smaller than chunk_length_ms.")


@contextmanager
def _temporary_chunk_file() -> Iterator[Path]:
    """Create a temporary WAV file for a single exported chunk."""
    temp_path: Path | None = None
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        yield temp_path
    finally:
        if temp_path is not None:
            with suppress(OSError):
                temp_path.unlink()


def _validate_audio_file(audio_path: str) -> Path:
    """Validate the input path and return it as a ``Path`` object."""
    if not audio_path or not audio_path.strip():
        raise ValueError("audio_path must not be empty.")

    audio_file = Path(audio_path)

    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    if not audio_file.is_file():
        raise ValueError(f"Audio path is not a file: {audio_file}")

    return audio_file


@lru_cache(maxsize=8)
def _get_model(
    model_size: str = DEFAULT_MODEL_SIZE,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
) -> "WhisperModel":
    """Load and cache a Whisper model for the given runtime configuration."""
    from faster_whisper import WhisperModel

    return WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
    )


def _load_model(
    *,
    model_size: str,
    device: str,
    compute_type: str,
) -> "WhisperModel":
    """Load a Whisper model while preserving no-argument compatibility for defaults."""
    if (
        model_size == DEFAULT_MODEL_SIZE
        and device == DEFAULT_DEVICE
        and compute_type == DEFAULT_COMPUTE_TYPE
    ):
        return _get_model()

    return _get_model(
        model_size=model_size,
        device=device,
        compute_type=compute_type,
    )


def _transcribe_file(
    model: "WhisperModel",
    audio_file: Path,
    *,
    show_progress: bool,
    beam_size: int = DEFAULT_BEAM_SIZE,
    language: str | None = None,
    time_offset_seconds: float = 0.0,
) -> list[Segment]:
    """Transcribe a single audio file and return structured segments."""
    transcript_segments: list[Segment] = []
    transcribe_kwargs = {
        "beam_size": beam_size,
        "vad_filter": True,
    }

    if language is not None:
        transcribe_kwargs["language"] = language

    segments, _ = model.transcribe(str(audio_file), **transcribe_kwargs)

    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue

        transcript_segments.append(
            Segment(
                start=time_offset_seconds + float(segment.start),
                end=time_offset_seconds + float(segment.end),
                text=text,
            )
        )

    return transcript_segments


def _estimate_total_chunks(
    total_audio_seconds: float,
    chunk_length_ms: int,
    overlap_ms: int,
) -> int:
    """Estimate how many overlapped chunks an audio file will produce."""
    if total_audio_seconds <= 0:
        return 1

    chunk_length_seconds = chunk_length_ms / 1000
    if total_audio_seconds <= chunk_length_seconds:
        return 1

    step_seconds = (chunk_length_ms - overlap_ms) / 1000
    remaining_seconds = total_audio_seconds - chunk_length_seconds
    return max(1, 1 + ceil(remaining_seconds / step_seconds))


def _discard_processed_overlap_segments(
    segments: list[Segment],
    *,
    chunk_start_seconds: float,
    overlap_ms: int,
) -> list[Segment]:
    """Discard segments that fall fully inside the already-processed overlap."""
    if overlap_ms <= 0:
        return segments

    overlap_cutoff_seconds = chunk_start_seconds + (overlap_ms / 1000)
    return [segment for segment in segments if segment.end > overlap_cutoff_seconds]


def _transcribe_chunk(
    model: "WhisperModel",
    chunk: "AudioSegment",
    *,
    show_progress: bool,
    beam_size: int = DEFAULT_BEAM_SIZE,
    language: str | None = None,
    time_offset_seconds: float = 0.0,
) -> list[Segment]:
    """Export one chunk, transcribe it, and delete the temporary file."""
    with _temporary_chunk_file() as chunk_path:
        chunk.export(str(chunk_path), format="wav")
        return _transcribe_file(
            model,
            chunk_path,
            show_progress=show_progress,
            beam_size=beam_size,
            language=language,
            time_offset_seconds=time_offset_seconds,
        )


def transcribe_audio(
    audio_path: str,
    show_progress: bool = True,
    chunk_length_ms: int = DEFAULT_CHUNK_LENGTH_MS,
    status_callback: StatusCallback | None = None,
    progress_callback: ProgressCallback | None = None,
    model_size: str = DEFAULT_MODEL_SIZE,
    beam_size: int = DEFAULT_BEAM_SIZE,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
    language: str | None = None,
    overlap_ms: int = DEFAULT_OVERLAP_MS,
) -> list[Segment]:
    """Transcribe an audio file into structured segments.

    The pipeline chunks only large files, preserves absolute segment timestamps,
    and reports chunk-level progress through optional callbacks.
    """
    _validate_transcription_options(
        chunk_length_ms=chunk_length_ms,
        beam_size=beam_size,
        overlap_ms=overlap_ms,
    )

    normalized_model_size = _normalize_required_text_option("model_size", model_size)
    normalized_device = _normalize_required_text_option("device", device)
    normalized_compute_type = _normalize_required_text_option(
        "compute_type",
        compute_type,
    )
    normalized_language = _normalize_language(language)

    audio_file = _validate_audio_file(audio_path)

    _emit_status(status_callback, "Loading Whisper model...")

    try:
        model = _load_model(
            model_size=normalized_model_size,
            device=normalized_device,
            compute_type=normalized_compute_type,
        )
    except Exception as exc:
        raise RuntimeError("Failed to load Whisper model") from exc

    try:
        total_audio_seconds = get_audio_duration_seconds(str(audio_file))
    except (FileNotFoundError, ValueError):
        raise
    except Exception as exc:
        raise RuntimeError("Failed to decode audio file") from exc

    if total_audio_seconds < SMALL_FILE_CHUNKING_THRESHOLD_SECONDS:
        _emit_status(
            status_callback,
            f"Transcribing audio directly ({_format_clock(total_audio_seconds)} total audio)...",
        )

        transcript_segments = _transcribe_file(
            model,
            audio_file,
            show_progress=show_progress,
            beam_size=beam_size,
            language=normalized_language,
            time_offset_seconds=0.0,
        )
        _report_progress(
            show_progress=show_progress,
            progress_callback=progress_callback,
            status_callback=status_callback,
            current=1,
            total=1,
        )
        return transcript_segments

    _emit_status(status_callback, "Splitting audio into chunks...")

    total_chunks = _estimate_total_chunks(total_audio_seconds, chunk_length_ms, overlap_ms)
    _emit_status(
        status_callback,
        f"Transcribing {total_chunks} chunk(s) ({_format_clock(total_audio_seconds)} total audio)...",
    )

    transcript_segments: list[Segment] = []
    current_chunk_start_seconds = 0.0
    overlap_seconds = overlap_ms / 1000

    try:
        for index, chunk in enumerate(
            iter_audio_chunks(
                str(audio_file),
                chunk_length_ms=chunk_length_ms,
                overlap_ms=overlap_ms,
            ),
            start=1,
        ):
            chunk_duration_seconds = len(chunk) / 1000

            if not show_progress:
                _emit_status(status_callback, f"Transcribing chunk {index}/{total_chunks}...")

            chunk_segments = _transcribe_chunk(
                model,
                chunk,
                show_progress=show_progress,
                beam_size=beam_size,
                language=normalized_language,
                time_offset_seconds=current_chunk_start_seconds,
            )

            if index > 1:
                chunk_segments = _discard_processed_overlap_segments(
                    chunk_segments,
                    chunk_start_seconds=current_chunk_start_seconds,
                    overlap_ms=overlap_ms,
                )

            transcript_segments.extend(chunk_segments)
            _report_progress(
                show_progress=show_progress,
                progress_callback=progress_callback,
                status_callback=status_callback,
                current=index,
                total=total_chunks,
            )
            current_chunk_start_seconds += max(0.0, chunk_duration_seconds - overlap_seconds)
    except Exception as exc:
        raise RuntimeError(f"Failed to transcribe audio file: {audio_file}") from exc

    return transcript_segments
