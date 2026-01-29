"""Tests for soft_trim and content_preview utility functions."""

import pytest

from memora.storage import content_preview, soft_trim


class TestSoftTrim:
    """Tests for the soft_trim function."""

    def test_short_content_unchanged(self):
        """Content shorter than max_length should be returned unchanged."""
        content = "This is short content."
        result = soft_trim(content, max_length=100)
        assert result == content

    def test_exact_length_unchanged(self):
        """Content exactly at max_length should be returned unchanged."""
        content = "A" * 100
        result = soft_trim(content, max_length=100)
        assert result == content

    def test_long_content_trimmed(self):
        """Long content should be trimmed with ellipsis in the middle."""
        content = "A" * 1000
        result = soft_trim(content, max_length=100)

        # Should contain head, ellipsis message, and tail
        assert "...[" in result
        assert "chars truncated]..." in result
        assert result.startswith("A" * 50)  # Head (60% of 100 = 60, minus some)

    def test_truncated_char_count_accurate(self):
        """The truncated char count in the message should be accurate."""
        content = "X" * 1000
        result = soft_trim(content, max_length=100, head_ratio=0.6, tail_ratio=0.3)

        # Extract the truncated count from the message
        # Format: "...[N chars truncated]..."
        import re

        match = re.search(r"\[(\d+) chars truncated\]", result)
        assert match is not None
        truncated_count = int(match.group(1))

        # Head = 60 chars, Tail = 30 chars, Total = 1000
        # Truncated should be 1000 - 60 - 30 = 910
        assert truncated_count == 910

    def test_custom_ratios(self):
        """Custom head/tail ratios should work correctly."""
        content = "H" * 200 + "M" * 600 + "T" * 200  # 1000 chars total
        result = soft_trim(content, max_length=100, head_ratio=0.8, tail_ratio=0.1)

        # Head should be 80% = 80 chars of H's
        # Tail should be 10% = 10 chars of T's
        assert result.startswith("H" * 70)  # Some H's at start
        assert result.endswith("T" * 8)  # Some T's at end (after .lstrip())

    def test_zero_tail_ratio(self):
        """Zero tail ratio should work (head only)."""
        content = "A" * 1000
        result = soft_trim(content, max_length=100, head_ratio=0.9, tail_ratio=0.0)

        # Should end with truncation message, no tail
        assert result.endswith("chars truncated]...")
        assert not result.endswith("A")

    def test_preserves_word_boundaries_approximately(self):
        """Content should be trimmed (rstrip/lstrip applied to clean edges)."""
        content = "Hello world this is a test. " * 50  # Long content with spaces
        result = soft_trim(content, max_length=100)

        # The result should not have trailing spaces on head or leading on tail
        assert not result.split("\n")[0].endswith(" ")


class TestContentPreview:
    """Tests for the content_preview function."""

    def test_short_content_unchanged(self):
        """Content shorter than max_length should be returned unchanged."""
        content = "This is short."
        result = content_preview(content, max_length=100)
        assert result == content

    def test_exact_length_unchanged(self):
        """Content exactly at max_length should be returned unchanged."""
        content = "A" * 200
        result = content_preview(content, max_length=200)
        assert result == content

    def test_long_content_truncated(self):
        """Long content should be truncated with ellipsis."""
        content = "A" * 300
        result = content_preview(content, max_length=200)

        assert len(result) == 203  # 200 chars + "..."
        assert result.endswith("...")
        assert result.startswith("A" * 200)

    def test_default_length_200(self):
        """Default max_length should be 200."""
        content = "B" * 500
        result = content_preview(content)

        assert result == "B" * 200 + "..."

    def test_trailing_spaces_stripped(self):
        """Trailing spaces before ellipsis should be stripped."""
        content = "Hello world     " + "X" * 200
        result = content_preview(content, max_length=15)

        # "Hello world     " is 16 chars, truncated to 15
        # After rstrip, should not end with space before ...
        assert not result[:-3].endswith(" ")

    def test_empty_content(self):
        """Empty content should return empty string."""
        result = content_preview("", max_length=100)
        assert result == ""

    def test_whitespace_only(self):
        """Whitespace-only content within limit should be returned as-is."""
        content = "   "
        result = content_preview(content, max_length=100)
        assert result == "   "


class TestIntegration:
    """Integration tests for soft_trim in compact memory contexts."""

    def test_preview_vs_soft_trim(self):
        """content_preview and soft_trim should serve different purposes."""
        long_content = "START: " + "X" * 500 + " :END"

        preview = content_preview(long_content, max_length=100)
        trimmed = soft_trim(long_content, max_length=100)

        # Preview only shows beginning
        assert preview.startswith("START:")
        assert ":END" not in preview

        # Soft trim shows both beginning and end
        assert trimmed.startswith("START:")
        assert ":END" in trimmed

    def test_soft_trim_for_verbose_memories(self):
        """Soft trim should work well for typical verbose memory content."""
        verbose_memory = """# Meeting Notes - January 2026

## Attendees
- Alice
- Bob
- Charlie

## Discussion Points

1. Project timeline review
   - Current status: On track
   - Key milestones achieved
   - Remaining work items

2. Technical decisions
   - Architecture choices finalized
   - API design approved
   - Database schema locked

3. Action items
   - Alice: Complete documentation
   - Bob: Set up CI/CD
   - Charlie: Code review

## Next Steps

Schedule follow-up meeting for next week.
Review progress on action items.
"""
        result = soft_trim(verbose_memory, max_length=300)

        # Should preserve the header (important context)
        assert "Meeting Notes" in result

        # Should preserve the ending (action items/next steps)
        assert "Next Steps" in result or "follow-up" in result

        # Should indicate truncation
        assert "chars truncated" in result
