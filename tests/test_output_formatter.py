"""Tests for Markdown transcript formatting."""

import pytest

from openlecture.models import Segment
from openlecture.output_formatter import transcript_to_markdown


def test_transcript_to_markdown_returns_title_for_empty_transcript() -> None:
    """An empty transcript should still produce a Markdown title."""
    assert transcript_to_markdown([]) == "# Lecture Transcript"


def test_transcript_to_markdown_formats_segments_with_timestamps() -> None:
    """A normal transcript should be rendered as timestamped Markdown lines."""
    transcript = [
        Segment(start=0.0, end=0.5, text="   "),
        Segment(start=1.2, end=3.9, text="Today we talk about Fourier transform."),
        Segment(start=4.0, end=6.1, text="The Fourier transform converts signals."),
    ]

    assert transcript_to_markdown(transcript) == (
        "# Lecture Transcript\n\n"
        "[00:00:01] Today we talk about Fourier transform.\n\n"
        "[00:00:04] The Fourier transform converts signals."
    )


def test_transcript_to_markdown_can_skip_timestamps() -> None:
    """The formatter should support Markdown output without timestamps."""
    transcript = [
        Segment(start=1.2, end=3.9, text="Today we talk about Fourier transform."),
        Segment(start=4.0, end=6.1, text="The Fourier transform converts signals."),
    ]

    assert transcript_to_markdown(transcript, include_timestamps=False) == (
        "# Lecture Transcript\n\n"
        "Today we talk about Fourier transform.\n\n"
        "The Fourier transform converts signals."
    )


@pytest.mark.parametrize(
    "value",
    [None, 123, "not a segment list", [Segment(0.0, 1.0, "ok"), "bad"]],
)
def test_transcript_to_markdown_rejects_invalid_input(value: object) -> None:
    """Invalid inputs should raise a TypeError."""
    with pytest.raises(TypeError, match="segments must be a list of Segment instances"):
        transcript_to_markdown(value)  # type: ignore[arg-type]
