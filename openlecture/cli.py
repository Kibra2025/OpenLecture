"""CLI entrypoint for OpenLecture."""

import errno
from math import ceil
from pathlib import Path
import sys
from threading import Event, Lock, Thread
from time import monotonic
from typing import Any, Callable

import typer
from tqdm import tqdm

from .audio_utils import DEFAULT_OVERLAP_MS, get_audio_duration_seconds
from .output_formatter import transcript_to_markdown
from .transcribe import (
    DEFAULT_BACKEND,
    DEFAULT_BEAM_SIZE,
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DEVICE,
    DEFAULT_MODEL_SIZE,
    SMALL_FILE_CHUNKING_THRESHOLD_SECONDS,
    transcribe_audio,
)

app = typer.Typer()


def _estimate_total_chunks(
    total_audio_seconds: float,
    *,
    chunk_length_ms: int,
    overlap_ms: int = DEFAULT_OVERLAP_MS,
) -> int:
    """Estimate the chunk count used by the transcription pipeline."""
    if total_audio_seconds <= 0:
        return 1

    chunk_length_seconds = chunk_length_ms / 1000
    if total_audio_seconds <= chunk_length_seconds:
        return 1

    step_seconds = (chunk_length_ms - overlap_ms) / 1000
    remaining_seconds = total_audio_seconds - chunk_length_seconds
    return max(1, 1 + ceil(remaining_seconds / step_seconds))


def _estimate_small_file_processing_seconds(total_audio_seconds: float) -> float:
    """Estimate a smooth progress duration for direct small-file transcription."""
    safe_duration = max(0.0, total_audio_seconds)
    return max(2.0, min(20.0, safe_duration + 2.0))


class _SmoothTqdmProgressBar:
    """Wrap tqdm with optional smoothing between real chunk completions."""

    _MAX_SIMULATED_FRACTION = 0.97
    _REFRESH_INTERVAL_SECONDS = 0.1
    _ESTIMATE_BLEND = 0.35

    def __init__(
        self,
        total: int,
        *,
        estimated_chunk_duration: float | None = None,
    ) -> None:
        self._lock = Lock()
        self._stop_event = Event()
        self._real_completed = 0
        self._displayed_progress = 0.0
        self._estimated_chunk_duration = (
            estimated_chunk_duration if estimated_chunk_duration and estimated_chunk_duration > 0 else None
        )
        self._last_completion_time = monotonic()
        self._progress_bar = tqdm(
            total=max(1, total),
            desc="Transcribing",
            file=sys.stdout,
            dynamic_ncols=True,
            leave=True,
            bar_format="{desc}: [{bar}] {percentage:3.0f}% | {postfix}",
        )
        self._progress_bar.set_postfix_str(
            f"{self._real_completed}/{self._progress_bar.total} chunks",
            refresh=False,
        )
        self._thread = Thread(target=self._run_smoothing_loop, daemon=True)
        self._thread.start()

    @property
    def total(self) -> int:
        return int(self._progress_bar.total or 0)

    @total.setter
    def total(self, value: int) -> None:
        with self._lock:
            self._progress_bar.total = max(1, value)
            self._progress_bar.set_postfix_str(
                f"{self._real_completed}/{self._progress_bar.total} chunks",
                refresh=False,
            )
            self._progress_bar.refresh()

    def advance_to(self, current: int, total: int) -> None:
        """Sync the displayed bar to a real chunk completion."""
        now = monotonic()
        with self._lock:
            if total > 0 and self._progress_bar.total != total:
                self._progress_bar.total = total

            completed_delta = max(0, current - self._real_completed)
            if completed_delta > 0:
                observed_chunk_duration = (now - self._last_completion_time) / completed_delta
                if observed_chunk_duration > 0:
                    if self._estimated_chunk_duration is None:
                        self._estimated_chunk_duration = observed_chunk_duration
                    else:
                        self._estimated_chunk_duration = (
                            (1 - self._ESTIMATE_BLEND) * self._estimated_chunk_duration
                            + self._ESTIMATE_BLEND * observed_chunk_duration
                        )

                self._real_completed = current
                self._last_completion_time = now
                self._sync_progress_locked(float(current))

            self._progress_bar.set_postfix_str(
                f"{self._real_completed}/{self._progress_bar.total} chunks",
                refresh=False,
            )
            self._progress_bar.refresh()

            if self._real_completed >= self._progress_bar.total:
                self._stop_event.set()

    def clear(self) -> None:
        with self._lock:
            self._progress_bar.clear()

    def refresh(self) -> None:
        with self._lock:
            self._progress_bar.refresh()

    def close(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        with self._lock:
            self._progress_bar.close()

    def _run_smoothing_loop(self) -> None:
        while not self._stop_event.wait(self._REFRESH_INTERVAL_SECONDS):
            with self._lock:
                if self._estimated_chunk_duration is None:
                    continue

                if self._real_completed >= self._progress_bar.total:
                    continue

                elapsed_since_completion = monotonic() - self._last_completion_time
                simulated_fraction = min(
                    self._MAX_SIMULATED_FRACTION,
                    elapsed_since_completion / self._estimated_chunk_duration,
                )
                simulated_target = min(
                    self._progress_bar.total,
                    self._real_completed + simulated_fraction,
                )
                self._sync_progress_locked(simulated_target)
                self._progress_bar.refresh()

    def _sync_progress_locked(self, target: float) -> None:
        clamped_target = min(float(self._progress_bar.total), max(0.0, target))
        delta = clamped_target - self._displayed_progress
        if delta <= 0:
            return

        self._progress_bar.update(delta)
        self._displayed_progress = clamped_target


class _DeferredTqdmProgressBar:
    """Create a chunk progress bar lazily only when chunked progress is confirmed."""

    def __init__(self) -> None:
        self._delegate: _SmoothTqdmProgressBar | None = None
        self._lock = Lock()

    def advance_to(self, current: int, total: int) -> None:
        with self._lock:
            if self._delegate is None:
                if total <= 1:
                    return
                self._delegate = _SmoothTqdmProgressBar(
                    total,
                    estimated_chunk_duration=None,
                )
            delegate = self._delegate

        delegate.advance_to(current, total)

    def clear(self) -> None:
        with self._lock:
            if self._delegate is not None:
                self._delegate.clear()

    def refresh(self) -> None:
        with self._lock:
            if self._delegate is not None:
                self._delegate.refresh()

    def close(self) -> None:
        with self._lock:
            delegate = self._delegate
        if delegate is not None:
            delegate.close()


def _resolve_output_path(output_path: Path) -> Path:
    """Validate the target output path and create parent directories as needed."""
    if not str(output_path).strip():
        raise ValueError("Output path must not be empty")

    resolved_output_path = output_path.expanduser()

    if resolved_output_path.exists() and resolved_output_path.is_dir():
        raise ValueError(f"Output path points to a directory: {resolved_output_path}")

    parent_dir = resolved_output_path.parent

    if parent_dir.exists() and not parent_dir.is_dir():
        raise ValueError(f"Output parent path is not a directory: {parent_dir}")

    try:
        parent_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise PermissionError(
            f"Permission denied while creating output directory: {parent_dir}"
        ) from exc
    except OSError as exc:
        if exc.errno in {errno.EINVAL, errno.ENAMETOOLONG}:
            raise ValueError(f"Invalid output path: {resolved_output_path}") from exc
        raise RuntimeError(
            f"Failed to prepare output directory: {parent_dir}"
        ) from exc

    return resolved_output_path


def _write_output(output_path: Path, content: str) -> None:
    """Write the transcript to disk with clear filesystem errors."""
    try:
        output_path.write_text(content, encoding="utf-8")
    except PermissionError as exc:
        raise PermissionError(
            f"Permission denied while writing output file: {output_path}"
        ) from exc
    except OSError as exc:
        if exc.errno in {errno.EINVAL, errno.ENAMETOOLONG}:
            raise ValueError(f"Invalid output path: {output_path}") from exc
        raise RuntimeError(f"Failed to write output file: {output_path}") from exc


def _build_progress_callback(progress_bar: Any) -> Callable[[int, int], None]:
    """Create a tqdm-backed progress callback for chunked transcription."""
    if hasattr(progress_bar, "advance_to"):
        def progress_callback(current: int, total: int) -> None:
            progress_bar.advance_to(current, total)

        return progress_callback

    last_reported = {"current": 0}

    def progress_callback(current: int, total: int) -> None:
        if total > 0 and progress_bar.total != total:
            progress_bar.total = total
            progress_bar.refresh()

        increment = max(0, current - last_reported["current"])
        if increment <= 0:
            return

        progress_bar.update(increment)
        last_reported["current"] = current

    return progress_callback


def _build_status_callback(
    *,
    progress_bar: Any | None = None,
    replace_direct_transcribe_status: bool = False,
) -> Callable[[str], None]:
    """Create a CLI status callback that coexists cleanly with tqdm."""
    direct_transcribe_announced = {"value": False}

    def status_callback(message: str) -> None:
        rendered_message = message
        if (
            replace_direct_transcribe_status
            and message.startswith("Transcribing audio directly")
        ):
            if direct_transcribe_announced["value"]:
                return
            direct_transcribe_announced["value"] = True
            rendered_message = "Transcribing..."

        if progress_bar is not None:
            progress_bar.clear()
            typer.echo(rendered_message)
            progress_bar.refresh()
            return

        typer.echo(rendered_message)

    return status_callback


@app.command()
def transcribe(
    audio_file: Path,
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Markdown output path. Defaults to the input filename with a .md extension.",
    ),
    progress: bool = typer.Option(
        True,
        "--progress/--no-progress",
        help="Show transcription progress in the terminal.",
    ),
    timestamps: bool = typer.Option(
        False,
        "--timestamps/--no-timestamps",
        help="Include segment timestamps in the Markdown output. Disabled by default.",
    ),
    chunk_length_ms: int = typer.Option(
        60000,
        "--chunk-length-ms",
        min=1,
        help="Split audio into chunks of this length before transcription.",
    ),
    backend: str = typer.Option(
        DEFAULT_BACKEND,
        "--backend",
        help="Transcription backend: faster-whisper or transformers.",
    ),
    model: str = typer.Option(
        DEFAULT_MODEL_SIZE,
        "--model",
        help="Whisper model size or model ID to load.",
    ),
    beam_size: int = typer.Option(
        DEFAULT_BEAM_SIZE,
        "--beam-size",
        min=1,
        help="Beam size used during decoding.",
    ),
    device: str = typer.Option(
        DEFAULT_DEVICE,
        "--device",
        help="Execution device for the selected backend. transformers also supports amd/rocm/dml aliases.",
    ),
    compute_type: str = typer.Option(
        DEFAULT_COMPUTE_TYPE,
        "--compute-type",
        help="Computation type for the selected backend.",
    ),
    language: str | None = typer.Option(
        None,
        "--language",
        help="Optional source language code. If omitted, Whisper auto-detects it.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show the full traceback when an error occurs.",
    ),
) -> None:
    """Transcribe an audio file using Whisper."""
    typer.echo("Starting OpenLecture...")
    progress_bar = None

    try:
        output_path = _resolve_output_path(output or audio_file.with_suffix(".md"))
        transcribe_kwargs = {
            "show_progress": progress,
            "chunk_length_ms": chunk_length_ms,
        }

        if progress:
            total_audio_seconds = None
            try:
                total_audio_seconds = get_audio_duration_seconds(str(audio_file))
            except Exception:
                total_audio_seconds = None

            if total_audio_seconds is None:
                progress_bar = _DeferredTqdmProgressBar()
                transcribe_kwargs["status_callback"] = typer.echo
                transcribe_kwargs["progress_callback"] = _build_progress_callback(
                    progress_bar
                )
            else:
                use_chunk_progress = (
                    total_audio_seconds >= SMALL_FILE_CHUNKING_THRESHOLD_SECONDS
                )

                if use_chunk_progress:
                    total_chunks = _estimate_total_chunks(
                        total_audio_seconds,
                        chunk_length_ms=chunk_length_ms,
                    )
                    estimated_chunk_duration = (
                        total_audio_seconds / total_chunks if total_chunks > 0 else None
                    )
                    progress_bar = _SmoothTqdmProgressBar(
                        total_chunks,
                        estimated_chunk_duration=estimated_chunk_duration,
                    )
                    transcribe_kwargs["status_callback"] = _build_status_callback(
                        progress_bar=progress_bar
                    )
                    transcribe_kwargs["progress_callback"] = _build_progress_callback(
                        progress_bar
                    )
                else:
                    progress_bar = _SmoothTqdmProgressBar(
                        1,
                        estimated_chunk_duration=_estimate_small_file_processing_seconds(
                            total_audio_seconds
                        ),
                    )
                    transcribe_kwargs["status_callback"] = _build_status_callback(
                        progress_bar=progress_bar,
                        replace_direct_transcribe_status=True,
                    )
                    transcribe_kwargs["progress_callback"] = _build_progress_callback(
                        progress_bar
                    )
        else:
            transcribe_kwargs["status_callback"] = typer.echo

        if model != DEFAULT_MODEL_SIZE:
            transcribe_kwargs["model_size"] = model

        if beam_size != DEFAULT_BEAM_SIZE:
            transcribe_kwargs["beam_size"] = beam_size

        if backend != DEFAULT_BACKEND:
            transcribe_kwargs["backend"] = backend

        if device != DEFAULT_DEVICE:
            transcribe_kwargs["device"] = device

        if compute_type != DEFAULT_COMPUTE_TYPE:
            transcribe_kwargs["compute_type"] = compute_type

        if language is not None:
            transcribe_kwargs["language"] = language

        segments = transcribe_audio(str(audio_file), **transcribe_kwargs)
        markdown_transcript = transcript_to_markdown(
            segments,
            include_timestamps=timestamps,
        )
        _write_output(output_path, markdown_transcript)
    except Exception as exc:
        if verbose:
            raise
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    finally:
        if progress_bar is not None:
            progress_bar.close()

    typer.secho(f"Transcript saved to {output_path}", fg=typer.colors.GREEN)


def main() -> None:
    """Run the OpenLecture CLI."""
    app()


if __name__ == "__main__":
    main()
