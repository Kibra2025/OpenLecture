"""Domain models used across the OpenLecture transcription pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    """A single transcribed segment with absolute timestamps."""

    start: float
    end: float
    text: str
