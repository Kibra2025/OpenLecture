"""Utilities for inspecting, splitting, and exporting audio."""

from __future__ import annotations

import os
import shutil
import warnings
from collections.abc import Iterable, Iterator
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydub import AudioSegment

SUPPORTED_FORMATS = {
    ".m4a": "mp4",
    ".mp3": "mp3",
    ".mp4": "mp4",
    ".wav": "wav",
}

WINDOWS_WINGET_FFMPEG_PACKAGES = (
    "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe",
    "Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe",
)

DEFAULT_OVERLAP_MS = 2_000


@lru_cache(maxsize=1)
def _find_ffmpeg_binaries() -> tuple[str | None, str | None]:
    """Locate ffmpeg and ffprobe, including common WinGet install paths."""
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")

    if ffmpeg_path and ffprobe_path:
        return ffmpeg_path, ffprobe_path

    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        return ffmpeg_path, ffprobe_path

    winget_packages_dir = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"

    for package_name in WINDOWS_WINGET_FFMPEG_PACKAGES:
        package_dir = winget_packages_dir / package_name
        if not package_dir.exists():
            continue

        for bin_dir in package_dir.glob("*/bin"):
            ffmpeg_candidate = bin_dir / "ffmpeg.exe"
            ffprobe_candidate = bin_dir / "ffprobe.exe"

            if ffmpeg_candidate.is_file() and ffprobe_candidate.is_file():
                return str(ffmpeg_candidate), str(ffprobe_candidate)

    return ffmpeg_path, ffprobe_path


def _configure_pydub_binaries(audio_segment_class) -> None:
    """Point pydub at the installed ffmpeg/ffprobe binaries when available."""
    ffmpeg_path, ffprobe_path = _find_ffmpeg_binaries()

    if ffmpeg_path:
        ffmpeg_bin_dir = str(Path(ffmpeg_path).parent)
        current_path = os.environ.get("PATH", "")
        path_entries = current_path.split(os.pathsep) if current_path else []

        if ffmpeg_bin_dir not in path_entries:
            os.environ["PATH"] = os.pathsep.join(
                [ffmpeg_bin_dir, current_path] if current_path else [ffmpeg_bin_dir]
            )

        audio_segment_class.converter = ffmpeg_path

    if ffprobe_path:
        import pydub.utils as pydub_utils

        pydub_utils.get_prober_name = lambda: ffprobe_path


def _get_audio_segment_class():
    """Import and return ``pydub.AudioSegment`` on demand."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Couldn't find ffmpeg or avconv.*",
                category=RuntimeWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Couldn't find ffprobe or avprobe.*",
                category=RuntimeWarning,
            )
            from pydub import AudioSegment
    except ImportError as exc:
        raise ImportError(
            "pydub is required for audio chunking. Install it and ensure ffmpeg "
            "is available on your system PATH."
        ) from exc

    _configure_pydub_binaries(AudioSegment)

    return AudioSegment


def _infer_audio_format(audio_file: Path) -> str | None:
    """Infer the decoder format from the file extension when possible."""
    return SUPPORTED_FORMATS.get(audio_file.suffix.lower())


def _validate_audio_path(audio_path: str) -> Path:
    """Validate the input path and return it as a ``Path`` object."""
    if not audio_path or not audio_path.strip():
        raise ValueError("audio_path must not be empty.")

    audio_file = Path(audio_path).expanduser()

    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    if not audio_file.is_file():
        raise ValueError(f"Audio path is not a file: {audio_file}")

    return audio_file


def _validate_chunking_options(chunk_length_ms: int, overlap_ms: int) -> None:
    """Validate chunk size and overlap settings."""
    if chunk_length_ms <= 0:
        raise ValueError("chunk_length_ms must be greater than 0.")

    if overlap_ms < 0:
        raise ValueError("overlap_ms must be greater than or equal to 0.")

    if overlap_ms >= chunk_length_ms:
        raise ValueError("overlap_ms must be smaller than chunk_length_ms.")


def _load_with_pydub(
    audio_file: Path,
    audio_format: str | None,
    audio_segment_class,
) -> "AudioSegment":
    """Load audio with pydub using ffmpeg/ffprobe when available."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Couldn't find ffprobe or avprobe.*",
                category=RuntimeWarning,
            )
            return audio_segment_class.from_file(str(audio_file), format=audio_format)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg/ffprobe is not installed or is not available on PATH."
        ) from exc
    except Exception as exc:
        raise RuntimeError("Failed to decode audio file") from exc


def _load_with_pyav(audio_file: Path, audio_segment_class) -> "AudioSegment":
    """Load audio using PyAV and convert it into a pydub ``AudioSegment``."""
    try:
        import av
    except ImportError as exc:
        raise RuntimeError(
            "PyAV is not installed, so no fallback decoder is available."
        ) from exc

    try:
        with av.open(str(audio_file)) as container:
            audio_stream = next(
                (stream for stream in container.streams if stream.type == "audio"),
                None,
            )
            if audio_stream is None:
                raise RuntimeError(f"No audio stream found in file: {audio_file}")

            sample_rate = audio_stream.rate or audio_stream.codec_context.sample_rate
            channels = audio_stream.channels or audio_stream.codec_context.channels
            layout_name = getattr(audio_stream.layout, "name", None)

            if sample_rate is None or channels is None:
                raise RuntimeError(
                    f"Could not determine audio stream properties for: {audio_file}"
                )

            resampler = av.audio.resampler.AudioResampler(
                format="s16",
                layout=layout_name or channels,
                rate=sample_rate,
            )

            pcm_parts: list[bytes] = []

            for frame in container.decode(audio_stream):
                resampled_frames = resampler.resample(frame)
                if resampled_frames is None:
                    continue

                if not isinstance(resampled_frames, list):
                    resampled_frames = [resampled_frames]

                for resampled_frame in resampled_frames:
                    pcm_parts.append(resampled_frame.to_ndarray().tobytes())

            flushed_frames = resampler.resample(None)
            if flushed_frames is not None:
                if not isinstance(flushed_frames, list):
                    flushed_frames = [flushed_frames]

                for flushed_frame in flushed_frames:
                    pcm_parts.append(flushed_frame.to_ndarray().tobytes())

        if not pcm_parts:
            raise RuntimeError(f"No audio frames were decoded from file: {audio_file}")

        return audio_segment_class(
            data=b"".join(pcm_parts),
            sample_width=2,
            frame_rate=sample_rate,
            channels=channels,
        )
    except Exception as exc:
        raise RuntimeError("Failed to decode audio file") from exc


def _load_audio_segment(audio_file: Path) -> "AudioSegment":
    """Load an audio file into a single ``AudioSegment`` instance."""
    AudioSegment = _get_audio_segment_class()
    audio_format = _infer_audio_format(audio_file)

    try:
        return _load_with_pydub(audio_file, audio_format, AudioSegment)
    except RuntimeError:
        try:
            return _load_with_pyav(audio_file, AudioSegment)
        except RuntimeError as fallback_exc:
            raise RuntimeError("Failed to decode audio file") from fallback_exc


def get_audio_duration_seconds(audio_path: str) -> float:
    """Return the audio duration in seconds."""
    audio_file = _validate_audio_path(audio_path)

    try:
        import av

        with av.open(str(audio_file)) as container:
            if container.duration is not None:
                return float(container.duration / av.time_base)

            audio_stream = next(
                (stream for stream in container.streams if stream.type == "audio"),
                None,
            )
            if (
                audio_stream is not None
                and audio_stream.duration is not None
                and audio_stream.time_base is not None
            ):
                return float(audio_stream.duration * audio_stream.time_base)
    except Exception:
        pass

    audio = _load_audio_segment(audio_file)
    return len(audio) / 1000


def iter_audio_chunks(
    audio_path: str,
    chunk_length_ms: int = 60000,
    overlap_ms: int = DEFAULT_OVERLAP_MS,
) -> Iterator["AudioSegment"]:
    """Yield one audio chunk at a time from the input file.

    Each chunk overlaps the next one by ``overlap_ms`` to reduce word cuts at
    chunk boundaries.
    """
    _validate_chunking_options(chunk_length_ms, overlap_ms)

    audio_file = _validate_audio_path(audio_path)
    audio = _load_audio_segment(audio_file)
    step_ms = chunk_length_ms - overlap_ms

    for start_index in range(0, len(audio), step_ms):
        end_index = min(start_index + chunk_length_ms, len(audio))
        yield audio[start_index:end_index]
        if end_index >= len(audio):
            break


def split_audio(
    audio_path: str,
    chunk_length_ms: int = 60000,
    overlap_ms: int = DEFAULT_OVERLAP_MS,
) -> list["AudioSegment"]:
    """Load an audio file and split it into fixed-length chunks.

    Args:
        audio_path: Path to the input audio file.
        chunk_length_ms: Length of each chunk in milliseconds.
        overlap_ms: Overlap between adjacent chunks in milliseconds.

    Returns:
        A list of ``pydub.AudioSegment`` chunks.

    Raises:
        ValueError: If ``audio_path`` is empty, not a file, or chunk settings are
            invalid.
        FileNotFoundError: If the input file does not exist.
        ImportError: If ``pydub`` is not installed.
        RuntimeError: If the file cannot be decoded.
    """
    return list(
        iter_audio_chunks(
            audio_path,
            chunk_length_ms=chunk_length_ms,
            overlap_ms=overlap_ms,
        )
    )


def export_chunks(
    chunks: Iterable["AudioSegment"],
    output_dir: str | Path,
) -> list[Path]:
    """Export audio chunks as sequentially numbered WAV files.

    Args:
        chunks: Iterable of ``pydub.AudioSegment`` instances.
        output_dir: Directory where the chunk files will be written.

    Returns:
        A list of exported file paths.

    Raises:
        ValueError: If ``output_dir`` is empty or points to a file.
        ImportError: If ``pydub`` is not installed.
        TypeError: If any item in ``chunks`` is not an ``AudioSegment``.
        RuntimeError: If a chunk cannot be exported.
    """
    if output_dir is None or not str(output_dir).strip():
        raise ValueError("output_dir must not be empty.")

    output_path = Path(output_dir).expanduser()

    if output_path.exists() and not output_path.is_dir():
        raise ValueError(f"Output path is not a directory: {output_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    AudioSegment = _get_audio_segment_class()
    exported_files: list[Path] = []

    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, AudioSegment):
            raise TypeError("Each chunk must be a pydub.AudioSegment instance.")

        chunk_path = output_path / f"chunk_{index:03d}.wav"

        try:
            chunk.export(str(chunk_path), format="wav")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to export chunk {index} to {chunk_path}."
            ) from exc

        exported_files.append(chunk_path)

    return exported_files


__all__ = [
    "DEFAULT_OVERLAP_MS",
    "split_audio",
    "iter_audio_chunks",
    "get_audio_duration_seconds",
    "export_chunks",
]
