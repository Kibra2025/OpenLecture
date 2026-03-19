"""Benchmark the OpenLecture transcription pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openlecture.audio_utils import get_audio_duration_seconds
from openlecture.transcribe import transcribe_audio


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the benchmark script."""
    parser = argparse.ArgumentParser(description="Benchmark OpenLecture transcription speed.")
    parser.add_argument("audio_file", type=Path, help="Path to the audio file to transcribe.")
    parser.add_argument(
        "--chunk-length-ms",
        type=int,
        default=60000,
        help="Chunk length in milliseconds for large files.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show transcription progress output during the benchmark.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the benchmark and print timing metrics."""
    args = parse_args()
    audio_file = args.audio_file.expanduser()

    audio_duration_seconds = get_audio_duration_seconds(str(audio_file))

    started_at = perf_counter()
    transcribe_audio(
        str(audio_file),
        show_progress=args.progress,
        chunk_length_ms=args.chunk_length_ms,
    )
    elapsed_seconds = perf_counter() - started_at

    speed = audio_duration_seconds / elapsed_seconds if elapsed_seconds > 0 else float("inf")

    print(f"Audio duration: {audio_duration_seconds:.2f} sec")
    print(f"Processing time: {elapsed_seconds:.2f} sec")
    print(f"Speed: {speed:.2f}x realtime")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
