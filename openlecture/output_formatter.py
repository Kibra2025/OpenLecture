"""Helpers for converting transcripts into readable Markdown."""

from __future__ import annotations

MARKDOWN_TITLE = "# Lecture Transcript"


def transcript_to_markdown(transcript: str) -> str:
    """Convert a transcript string into simple Markdown paragraphs.

    The transcript is split on periods, empty items are discarded, and each
    sentence is rendered as its own paragraph under a single top-level heading.

    Args:
        transcript: Raw transcript text.

    Returns:
        A Markdown-formatted transcript.

    Raises:
        TypeError: If ``transcript`` is not a string.
    """
    if not isinstance(transcript, str):
        raise TypeError("transcript must be a string.")

    sentences = [
        f"{sentence}."
        for sentence in (part.strip() for part in transcript.split("."))
        if sentence
    ]

    if not sentences:
        return MARKDOWN_TITLE

    body = "\n\n".join(sentences)
    return f"{MARKDOWN_TITLE}\n\n{body}"


__all__ = ["transcript_to_markdown"]
