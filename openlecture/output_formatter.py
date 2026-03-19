"""Helpers for converting transcripts into readable Markdown."""

from __future__ import annotations

from .models import Segment

MARKDOWN_TITLE = "# Lecture Transcript"


def _format_timestamp(seconds: float) -> str:
    """Format a timestamp in seconds as ``HH:MM:SS``."""
    whole_seconds = max(0, int(seconds))
    minutes, secs = divmod(whole_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def transcript_to_markdown(segments: list[Segment]) -> str:
    """Convert structured transcript segments into timestamped Markdown."""
    if not isinstance(segments, list):
        raise TypeError("segments must be a list of Segment instances.")

    lines: list[str] = []
    for segment in segments:
        if not isinstance(segment, Segment):
            raise TypeError("segments must be a list of Segment instances.")

        text = segment.text.strip()
        if not text:
            continue

        lines.append(f"[{_format_timestamp(segment.start)}] {text}")

    if not lines:
        return MARKDOWN_TITLE

    body = "\n\n".join(lines)
    return f"{MARKDOWN_TITLE}\n\n{body}"


__all__ = ["transcript_to_markdown"]
