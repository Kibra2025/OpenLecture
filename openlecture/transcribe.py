from contextlib import contextmanager
from functools import lru_cache
from importlib import import_module
from math import ceil
from pathlib import Path
from typing import Iterator

from faster_whisper import WhisperModel

# Define which version of the AI model to use
MODEL_SIZE = "medium"
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

def transcribe_audio(audio_path: str, show_progress: bool = True) -> str:
    """Main function to turn an audio file into text."""
    
    # 1. PRE-FLIGHT CHECKS
    # Check if the path provided is empty or just spaces
    if not audio_path or not audio_path.strip():
        raise ValueError("audio_path must not be empty.")

    # Convert the string path into a special 'Path' object for easier handling
    audio_file = Path(audio_path)

    # Check if the file actually exists on the disk
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    # Check if the path points to a file and not a folder
    if not audio_file.is_file():
        raise ValueError(f"Audio path is not a file: {audio_file}")

    # 2. LOADING THE AI MODEL
    try:
        model = _get_model()
    except Exception as exc:
        # If something goes wrong during loading, tell the user why
        raise RuntimeError("Failed to load Whisper model.") from exc

    print("Transcribing audio...")

    # 3. TRANSCRIPTION PROCESS
    transcript_parts = []

    try:
        with _patched_whisper_progress_bar(show_progress):
            # Ask the model to listen. It returns segments of text.
            segments, _ = model.transcribe(
                str(audio_file),
                beam_size=5,
                vad_filter=True,
                log_progress=show_progress,
            )

            for segment in segments:
                if segment.text.strip():
                    transcript_parts.append(segment.text.strip())
    except Exception as exc:
        raise RuntimeError(f"Failed to transcribe audio file: {audio_file}") from exc

    return " ".join(transcript_parts)
