"""Tests for Markdown transcript formatting."""

import pytest

from openlecture.output_formatter import transcript_to_markdown


def test_transcript_to_markdown_returns_title_for_empty_transcript() -> None:
    """An empty transcript should still produce a Markdown title."""
    assert transcript_to_markdown("") == "# Lecture Transcript"


def test_transcript_to_markdown_formats_sentences_as_paragraphs() -> None:
    """A normal transcript should be rendered as Markdown paragraphs."""
    transcript = "First sentence. Second sentence."

    assert transcript_to_markdown(transcript) == (
        "# Lecture Transcript\n\nFirst sentence.\n\nSecond sentence."
    )


@pytest.mark.parametrize("value", [None, 123, ["not", "a", "string"]])
def test_transcript_to_markdown_rejects_non_string_input(value: object) -> None:
    """Non-string inputs should raise a TypeError."""
    with pytest.raises(TypeError, match="transcript must be a string"):
        transcript_to_markdown(value)  # type: ignore[arg-type]
