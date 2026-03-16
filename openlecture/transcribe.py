from contextlib import contextmanager, suppress
from functools import lru_cache
from importlib import import_module
from math import ceil
from pathlib import Path
import shutil
from typing import Iterator
from uuid import uuid4

from faster_whisper import WhisperModel

from .audio_utils import export_chunks, split_audio

# Define which version of the AI model to use
MODEL_SIZE = "medium"
DEFAULT_CHUNK_LENGTH_MS = 60_000
PROGRESS_BAR_FORMAT = (
    "{desc}: {percentage:3.0f}%|{bar}| {audio_done}/{audio_total} | ETA {eta_human}"
)


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


def _build_whisper_progress_bar(base_tqdm):
    class WhisperProgressBar(base_tqdm):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("desc", "Transcribing")
            kwargs.setdefault("dynamic_ncols", True)
            kwargs.setdefault("bar_format", PROGRESS_BAR_FORMAT)
            super().__init__(*args, **kwargs)

        @property
        def format_dict(self):
            data = super().format_dict
            total = data.get("total")
            completed = data.get("n", 0.0)
            elapsed = data.get("elapsed", 0.0)

            if total is not None and completed >= total:
                remaining = 0.0
            elif total and completed > 0 and elapsed > 0:
                remaining = max(0.0, (total - completed) * (elapsed / completed))
            else:
                remaining = None

            data.update(
                audio_done=_format_clock(completed),
                audio_total=_format_clock(total),
                eta_human=_format_clock(remaining, round_up=True),
            )
            return data

    return WhisperProgressBar


@contextmanager
def _patched_whisper_progress_bar(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return

    whisper_transcribe = import_module("faster_whisper.transcribe")
    original_tqdm = whisper_transcribe.tqdm
    whisper_transcribe.tqdm = _build_whisper_progress_bar(original_tqdm)

    try:
        yield
    finally:
        whisper_transcribe.tqdm = original_tqdm


@contextmanager
def _temporary_chunk_directory(audio_file: Path) -> Iterator[Path]:
    """Create a writable temporary directory for exported chunks."""
    candidate_roots = (
        Path.cwd() / ".openlecture_tmp",
        audio_file.parent / ".openlecture_tmp",
    )

    for root in candidate_roots:
        try:
            root.mkdir(parents=True, exist_ok=True)
            temp_dir = root / f"chunks_{uuid4().hex}"
            temp_dir.mkdir()
            try:
                yield temp_dir
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
            with suppress(OSError):
                root.rmdir()
            return
        except OSError:
            continue

    temp_dir = Path.cwd() / f"openlecture_chunks_{uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


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


# This "decorator" saves the model in memory so we don't reload it every time
@lru_cache(maxsize=1)
def _get_model() -> WhisperModel:
    """Loads the Whisper model and keeps it cached."""
    
    print("Loading Whisper model...")
    
    return WhisperModel(
        MODEL_SIZE,
        device="auto",
        compute_type="auto"
    )

def _transcribe_file(
    model: WhisperModel,
    audio_file: Path,
    *,
    show_progress: bool,
) -> list[str]:
    """Transcribe a single audio file and return its text segments."""
    transcript_parts: list[str] = []

    with _patched_whisper_progress_bar(show_progress):
        segments, _ = model.transcribe(
            str(audio_file),
            beam_size=5,
            vad_filter=True,
            log_progress=show_progress,
        )

        for segment in segments:
            if segment.text.strip():
                transcript_parts.append(segment.text.strip())

    return transcript_parts


def transcribe_audio(
    audio_path: str,
    show_progress: bool = True,
    chunk_length_ms: int = DEFAULT_CHUNK_LENGTH_MS,
) -> str:
    """Transcribe an audio file by splitting it into smaller chunks first."""
    if chunk_length_ms <= 0:
        raise ValueError("chunk_length_ms must be greater than 0.")

    audio_file = _validate_audio_file(audio_path)

    try:
        model = _get_model()
    except Exception as exc:
        raise RuntimeError("Failed to load Whisper model.") from exc

    print("Splitting audio into chunks...")

    try:
        chunks = split_audio(str(audio_file), chunk_length_ms=chunk_length_ms)
    except Exception as exc:
        raise RuntimeError(f"Failed to split audio file: {audio_file}") from exc

    print(f"Transcribing {len(chunks)} chunk(s)...")

    transcript_parts: list[str] = []

    try:
        with _temporary_chunk_directory(audio_file) as temp_dir:
            chunk_paths = export_chunks(chunks, temp_dir)

            for index, chunk_path in enumerate(chunk_paths, start=1):
                print(f"Transcribing chunk {index}/{len(chunk_paths)}...")
                transcript_parts.extend(
                    _transcribe_file(
                        model,
                        chunk_path,
                        show_progress=show_progress,
                    )
                )
    except Exception as exc:
        raise RuntimeError(f"Failed to transcribe audio file: {audio_file}") from exc

    return " ".join(transcript_parts)
