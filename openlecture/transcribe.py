from contextvars import ContextVar
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from math import ceil
from pathlib import Path
import shutil
from time import perf_counter
from typing import TYPE_CHECKING, Any, Iterator
from uuid import uuid4

from .audio_utils import export_chunks, split_audio
import typer

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

# Define which version of the AI model to use
MODEL_SIZE = "medium"
DEFAULT_CHUNK_LENGTH_MS = 60_000
PROGRESS_BAR_FORMAT = (
    "{desc}: {percentage:3.0f}%|{bar}| total {audio_done}/{audio_total} | "
    "chunk {chunk_done}/{chunk_total} | {speed_label} | ETA {eta_human}"
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


def _estimate_remaining(
    completed: float,
    total: float | None,
    elapsed: float | None,
) -> float | None:
    """Estimate the remaining wall-clock time from audio progress."""
    if total is not None and completed >= total:
        return 0.0

    if total and completed > 0 and elapsed and elapsed > 0:
        return max(0.0, (total - completed) * (elapsed / completed))

    return None


def _format_speed(completed: float, elapsed: float | None) -> str:
    """Format transcription throughput as audio seconds processed per second."""
    if elapsed and elapsed > 0 and completed > 0:
        return f"{completed / elapsed:.2f}x"

    return "--.-x"


@dataclass(frozen=True)
class _ProgressState:
    chunk_index: int
    total_chunks: int
    completed_before_seconds: float
    total_audio_seconds: float
    chunk_audio_seconds: float
    started_at: float


_PROGRESS_STATE: ContextVar[_ProgressState | None] = ContextVar(
    "openlecture_progress_state",
    default=None,
)


@dataclass
class _ProgressSession:
    bar: Any = None
    active_state: _ProgressState | None = None


_PROGRESS_SESSION: ContextVar[_ProgressSession | None] = ContextVar(
    "openlecture_progress_session",
    default=None,
)


@contextmanager
def _progress_state_scope(state: _ProgressState | None) -> Iterator[None]:
    token = _PROGRESS_STATE.set(state)
    try:
        yield
    finally:
        _PROGRESS_STATE.reset(token)


@contextmanager
def _progress_bar_session(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return

    session = _ProgressSession()
    token = _PROGRESS_SESSION.set(session)
    try:
        yield
    finally:
        if session.bar is not None:
            session.bar.close()
        _PROGRESS_SESSION.reset(token)


def _format_chunk_label(state: _ProgressState) -> str:
    """Render the current chunk label for the shared progress bar."""
    return f"Chunk {state.chunk_index}/{state.total_chunks}"


def _configure_progress_bar_kwargs(kwargs: dict[str, Any]) -> None:
    """Apply consistent CLI-friendly progress bar defaults."""
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("bar_format", PROGRESS_BAR_FORMAT)
    kwargs.setdefault("leave", True)
    kwargs.setdefault("mininterval", 0.0)
    kwargs.setdefault("miniters", 1)
    kwargs.setdefault("smoothing", 0.0)


def _build_whisper_progress_bar(base_tqdm):
    class _ManagedWhisperProgressBar(base_tqdm):
        def __init__(self, *args, session: _ProgressSession | None = None, **kwargs):
            self._progress_session = session
            super().__init__(*args, **kwargs)

        @property
        def format_dict(self):
            data = super().format_dict
            total = data.get("total")
            completed = float(data.get("n", 0.0))
            progress_state = None

            if self._progress_session is not None:
                progress_state = self._progress_session.active_state

            if progress_state is None:
                elapsed = data.get("elapsed", 0.0)
                chunk_completed = completed
                chunk_total = total
            else:
                elapsed = perf_counter() - progress_state.started_at
                chunk_completed = max(
                    0.0,
                    completed - progress_state.completed_before_seconds,
                )
                chunk_total = progress_state.chunk_audio_seconds

            remaining = _estimate_remaining(completed, total, elapsed)

            data.update(
                audio_done=_format_clock(completed),
                audio_total=_format_clock(total),
                chunk_done=_format_clock(chunk_completed),
                chunk_total=_format_clock(chunk_total),
                eta_human=_format_clock(remaining, round_up=True),
                speed_label=_format_speed(completed, elapsed),
            )
            return data

    class WhisperProgressBar:
        def __init__(self, *args, **kwargs):
            self._progress_state = _PROGRESS_STATE.get()
            self._progress_session = _PROGRESS_SESSION.get()
            self._is_shared_bar = (
                self._progress_state is not None and self._progress_session is not None
            )

            if self._is_shared_bar:
                self._progress_session.active_state = self._progress_state
                self._bar = self._prepare_shared_bar(*args, **kwargs)
            else:
                self._bar = self._create_standalone_bar(*args, **kwargs)

        def _prepare_shared_bar(self, *args, **kwargs):
            progress_state = self._progress_state
            session = self._progress_session

            if session.bar is None:
                kwargs["total"] = progress_state.total_audio_seconds
                kwargs["initial"] = progress_state.completed_before_seconds
                kwargs.setdefault("desc", _format_chunk_label(progress_state))
                _configure_progress_bar_kwargs(kwargs)
                session.bar = _ManagedWhisperProgressBar(
                    *args,
                    session=session,
                    **kwargs,
                )
                return session.bar

            session.bar.n = max(
                float(session.bar.n),
                progress_state.completed_before_seconds,
            )
            session.bar.last_print_n = session.bar.n
            session.bar.set_description_str(
                _format_chunk_label(progress_state),
                refresh=False,
            )
            session.bar.refresh()
            return session.bar

        def _create_standalone_bar(self, *args, **kwargs):
            if self._progress_state is None:
                kwargs.setdefault("desc", "Transcribing")
            else:
                kwargs["total"] = self._progress_state.total_audio_seconds
                kwargs["initial"] = self._progress_state.completed_before_seconds
                kwargs.setdefault("desc", _format_chunk_label(self._progress_state))

            _configure_progress_bar_kwargs(kwargs)
            return _ManagedWhisperProgressBar(*args, **kwargs)

        def update(self, n=1):
            self._bar.update(n)

        def close(self):
            if not self._is_shared_bar:
                self._bar.close()
                return

            if self._progress_state.chunk_index >= self._progress_state.total_chunks:
                self._bar.refresh()
                self._bar.close()
                self._progress_session.bar = None
            else:
                self._bar.refresh()

        def __getattr__(self, name):
            return getattr(self._bar, name)

    return WhisperProgressBar


@contextmanager
def _patched_whisper_progress_bar(
    enabled: bool,
    *,
    progress_state: _ProgressState | None = None,
) -> Iterator[None]:
    if not enabled:
        yield
        return

    whisper_transcribe = import_module("faster_whisper.transcribe")
    original_tqdm = whisper_transcribe.tqdm
    whisper_transcribe.tqdm = _build_whisper_progress_bar(original_tqdm)

    try:
        with _progress_state_scope(progress_state):
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
def _get_model() -> "WhisperModel":
    """Loads the Whisper model and keeps it cached."""

    from faster_whisper import WhisperModel

    typer.echo("Loading Whisper model...")

    return WhisperModel(
        MODEL_SIZE,
        device="auto",
        compute_type="auto",
    )

def _transcribe_file(
    model: "WhisperModel",
    audio_file: Path,
    *,
    show_progress: bool,
    progress_state: _ProgressState | None = None,
) -> list[str]:
    """Transcribe a single audio file and return its text segments."""
    transcript_parts: list[str] = []

    with _patched_whisper_progress_bar(
        show_progress,
        progress_state=progress_state,
    ):
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
        raise RuntimeError("Failed to load Whisper model") from exc

    typer.echo("Splitting audio into chunks...")

    try:
        chunks = split_audio(str(audio_file), chunk_length_ms=chunk_length_ms)
    except (FileNotFoundError, ValueError):
        raise
    except Exception as exc:
        raise RuntimeError("Failed to decode audio file") from exc

    chunk_durations = [len(chunk) / 1000 for chunk in chunks]
    total_audio_seconds = sum(chunk_durations)

    typer.echo(
        f"Transcribing {len(chunks)} chunk(s) "
        f"({_format_clock(total_audio_seconds)} total audio)..."
    )

    transcript_parts: list[str] = []
    started_at = perf_counter()
    completed_audio_seconds = 0.0

    try:
        with _progress_bar_session(show_progress):
            with _temporary_chunk_directory(audio_file) as temp_dir:
                chunk_paths = export_chunks(chunks, temp_dir)

                for index, (chunk_path, chunk_duration) in enumerate(
                    zip(chunk_paths, chunk_durations, strict=True),
                    start=1,
                ):
                    progress_state = None
                    if show_progress:
                        progress_state = _ProgressState(
                            chunk_index=index,
                            total_chunks=len(chunk_paths),
                            completed_before_seconds=completed_audio_seconds,
                            total_audio_seconds=total_audio_seconds,
                            chunk_audio_seconds=chunk_duration,
                            started_at=started_at,
                        )
                    else:
                        typer.echo(f"Transcribing chunk {index}/{len(chunk_paths)}...")

                    transcript_parts.extend(
                        _transcribe_file(
                            model,
                            chunk_path,
                            show_progress=show_progress,
                            progress_state=progress_state,
                        )
                    )
                    completed_audio_seconds += chunk_duration
    except Exception as exc:
        raise RuntimeError(f"Failed to transcribe audio file: {audio_file}") from exc

    return " ".join(transcript_parts)
