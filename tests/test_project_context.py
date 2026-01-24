"""Tests for project context discovery and ingestion."""

import os
import tempfile
from pathlib import Path

import pytest

from memora.project_context import (
    CORE_INSTRUCTION_FILES,
    FileFormat,
    InstructionFileType,
    ParsedSection,
    ProjectContextConfig,
    compute_hash,
    discover_instruction_files,
    parse_instruction_file,
    parse_markdown_sections,
    parse_plain_text_sections,
    parse_yaml_sections,
    scan_project_context,
    should_ignore_path,
    slugify,
)


class TestComputeHash:
    """Tests for content hashing."""

    def test_consistent_hash(self):
        """Same content produces same hash."""
        content = "test content"
        assert compute_hash(content) == compute_hash(content)

    def test_different_hash(self):
        """Different content produces different hash."""
        assert compute_hash("content a") != compute_hash("content b")

    def test_hash_length(self):
        """Hash is 16 characters (truncated SHA-256)."""
        assert len(compute_hash("test")) == 16


class TestSlugify:
    """Tests for URL-friendly slug generation."""

    def test_basic_slug(self):
        """Basic text becomes lowercase with hyphens."""
        assert slugify("Hello World") == "hello-world"

    def test_special_characters(self):
        """Special characters are removed."""
        assert slugify("Test (with) special!") == "test-with-special"

    def test_multiple_spaces(self):
        """Multiple spaces become single hyphen."""
        assert slugify("too   many   spaces") == "too-many-spaces"

    def test_leading_trailing(self):
        """Leading/trailing hyphens are removed."""
        assert slugify("  test  ") == "test"


class TestParseMarkdownSections:
    """Tests for markdown section parsing."""

    def test_no_headings(self):
        """Content without headings becomes single section."""
        content = "Just some plain text content."
        sections = parse_markdown_sections(content)

        assert len(sections) == 1
        assert sections[0].title == "(Document Root)"
        assert sections[0].content == content

    def test_single_heading(self):
        """Single heading creates one section."""
        content = """# Main Title

This is the content under the main heading.
"""
        sections = parse_markdown_sections(content)

        assert len(sections) == 1
        assert sections[0].title == "Main Title"
        assert "This is the content" in sections[0].content
        assert sections[0].heading_level == 1

    def test_multiple_headings(self):
        """Multiple headings create multiple sections."""
        content = """# First Section

Content for first.

## Second Section

Content for second.

# Third Section

Content for third.
"""
        sections = parse_markdown_sections(content)

        assert len(sections) == 3
        assert sections[0].title == "First Section"
        assert sections[1].title == "Second Section"
        assert sections[2].title == "Third Section"

    def test_nested_headings(self):
        """Nested headings build section path."""
        content = """# Parent

Intro.

## Child

Child content.

### Grandchild

Grandchild content.
"""
        sections = parse_markdown_sections(content)

        # Find the grandchild section
        grandchild = next(s for s in sections if s.title == "Grandchild")
        assert "Parent > Child > Grandchild" in grandchild.section_path

    def test_empty_sections_skipped(self):
        """Sections with no content are skipped."""
        content = """# Has Content

Some text here.

# Empty Section

# Another With Content

More text.
"""
        sections = parse_markdown_sections(content)

        # Empty section should be skipped
        titles = [s.title for s in sections]
        assert "Empty Section" not in titles
        assert len(sections) == 2

    def test_heading_anchor(self):
        """Heading anchor is slugified."""
        content = """# My Fancy Title

Content here.
"""
        sections = parse_markdown_sections(content)

        assert sections[0].heading_anchor == "my-fancy-title"


class TestParseYamlSections:
    """Tests for YAML section parsing."""

    def test_basic_yaml(self):
        """Basic YAML dict creates sections per key."""
        content = """
name: Test
version: 1.0
description: A test file
"""
        sections = parse_yaml_sections(content)

        assert len(sections) == 3
        titles = [s.title for s in sections]
        assert "name" in titles
        assert "version" in titles
        assert "description" in titles

    def test_nested_yaml(self):
        """Nested YAML is serialized as content."""
        content = """
settings:
  timeout: 30
  retry: true
  paths:
    - /tmp
    - /var
"""
        sections = parse_yaml_sections(content)

        settings_section = next(s for s in sections if s.title == "settings")
        assert "timeout: 30" in settings_section.content
        assert "retry: true" in settings_section.content

    def test_invalid_yaml(self):
        """Invalid YAML creates error section."""
        content = """
invalid: yaml: syntax:
  - missing quote
"""
        sections = parse_yaml_sections(content)

        assert len(sections) == 1
        assert "Parse Error" in sections[0].title
        assert "YAML parse error" in sections[0].content


class TestParsePlainTextSections:
    """Tests for plain text section parsing."""

    def test_basic_text(self):
        """Plain text creates single section."""
        content = "Just plain text without structure."
        sections = parse_plain_text_sections(content)

        assert len(sections) == 1
        assert sections[0].title == "(Document)"
        assert sections[0].content == content

    def test_empty_text(self):
        """Empty text creates no sections."""
        sections = parse_plain_text_sections("")
        assert len(sections) == 0

        sections = parse_plain_text_sections("   \n  \n  ")
        assert len(sections) == 0


class TestParseInstructionFile:
    """Tests for full instruction file parsing."""

    def test_markdown_file(self):
        """Markdown file is parsed with sections."""
        content = """# CLAUDE.md

Project instructions.

## Guidelines

Follow these rules.
"""
        result = parse_instruction_file(
            content,
            InstructionFileType.CLAUDE_MD,
            FileFormat.MARKDOWN,
        )

        assert result.file_type == InstructionFileType.CLAUDE_MD
        assert result.file_format == FileFormat.MARKDOWN
        assert len(result.sections) >= 1
        assert result.file_hash == compute_hash(content)

    def test_yaml_file(self):
        """YAML file is parsed with sections per key."""
        content = """
model: gpt-4
temperature: 0.7
"""
        result = parse_instruction_file(
            content,
            InstructionFileType.AIDER_CONF,
            FileFormat.YAML,
        )

        assert result.file_type == InstructionFileType.AIDER_CONF
        assert result.file_format == FileFormat.YAML
        assert len(result.sections) == 2


class TestShouldIgnorePath:
    """Tests for path ignore logic."""

    def test_ignore_git_dir(self):
        """Paths under .git are ignored."""
        config = ProjectContextConfig()
        path = Path("/project/.git/config")
        assert should_ignore_path(path, config)

    def test_ignore_node_modules(self):
        """Paths under node_modules are ignored."""
        config = ProjectContextConfig()
        path = Path("/project/node_modules/package/file.md")
        assert should_ignore_path(path, config)

    def test_ignore_env_files(self):
        """Files matching .env* are ignored."""
        config = ProjectContextConfig()
        path = Path("/project/.env.local")
        assert should_ignore_path(path, config)

    def test_allow_normal_files(self):
        """Normal files are not ignored."""
        config = ProjectContextConfig()
        path = Path("/project/src/CLAUDE.md")
        assert not should_ignore_path(path, config)


class TestDiscoverInstructionFiles:
    """Tests for file discovery."""

    def test_discover_claude_md(self):
        """Discovers CLAUDE.md in directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            claude_md = Path(tmpdir) / "CLAUDE.md"
            claude_md.write_text("# Test")

            files = discover_instruction_files(tmpdir)

            assert len(files) == 1
            # Use resolve() to handle macOS /private/var symlink
            assert files[0][0].resolve() == claude_md.resolve()
            assert files[0][1] == InstructionFileType.CLAUDE_MD
            assert files[0][2] == FileFormat.MARKDOWN

    def test_discover_multiple_files(self):
        """Discovers multiple instruction files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "CLAUDE.md").write_text("# Claude")
            (Path(tmpdir) / ".cursorrules").write_text("rules")
            (Path(tmpdir) / "AGENTS.md").write_text("# Agents")

            files = discover_instruction_files(tmpdir)

            assert len(files) == 3

    def test_skip_large_files(self):
        """Skips files larger than max_file_size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            large_file = Path(tmpdir) / "CLAUDE.md"
            large_file.write_text("x" * 2_000_000)  # 2MB

            config = ProjectContextConfig(max_file_size=1_000_000)
            files = discover_instruction_files(tmpdir, config)

            assert len(files) == 0

    def test_copilot_instructions(self):
        """Discovers .github/copilot-instructions.md."""
        with tempfile.TemporaryDirectory() as tmpdir:
            github_dir = Path(tmpdir) / ".github"
            github_dir.mkdir()
            copilot = github_dir / "copilot-instructions.md"
            copilot.write_text("# Copilot Instructions")

            files = discover_instruction_files(tmpdir)

            assert len(files) == 1
            assert files[0][1] == InstructionFileType.COPILOT_INSTRUCTIONS


class TestScanProjectContext:
    """Tests for full project scanning."""

    def test_scan_empty_directory(self):
        """Empty directory returns no memories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memories = scan_project_context(tmpdir)
            assert len(memories) == 0

    def test_scan_creates_parent_memory(self):
        """Scanning creates parent memory for file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            claude_md = Path(tmpdir) / "CLAUDE.md"
            claude_md.write_text("# Test Project\n\nInstructions here.")

            memories = scan_project_context(tmpdir)

            # Should have parent + section memory
            assert len(memories) >= 1

            # Find parent memory
            parent = next(m for m in memories if m.get("metadata", {}).get("is_parent"))
            assert "project-context" in parent["tags"]
            assert "claude_md" in parent["tags"]

    def test_scan_creates_section_memories(self):
        """Scanning creates section memories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            claude_md = Path(tmpdir) / "CLAUDE.md"
            claude_md.write_text("""# Project

Intro text.

## Guidelines

Follow these rules.

## Architecture

System design info.
""")

            config = ProjectContextConfig(extract_sections=True)
            memories = scan_project_context(tmpdir, config)

            # Should have parent + multiple section memories
            sections = [m for m in memories if m.get("metadata", {}).get("is_section")]
            assert len(sections) >= 2

    def test_scan_without_sections(self):
        """Can disable section extraction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            claude_md = Path(tmpdir) / "CLAUDE.md"
            claude_md.write_text("""# Project

## Section One

## Section Two
""")

            config = ProjectContextConfig(extract_sections=False)
            memories = scan_project_context(tmpdir, config)

            # Should only have parent memory
            assert len(memories) == 1
            assert memories[0]["metadata"]["is_parent"]


class TestCoreInstructionFiles:
    """Tests for the instruction file registry."""

    def test_claude_md_registered(self):
        """CLAUDE.md is in the registry."""
        assert "CLAUDE.md" in CORE_INSTRUCTION_FILES

    def test_cursorrules_registered(self):
        """.cursorrules is in the registry."""
        assert ".cursorrules" in CORE_INSTRUCTION_FILES

    def test_copilot_instructions_registered(self):
        """Copilot instructions are registered."""
        assert ".github/copilot-instructions.md" in CORE_INSTRUCTION_FILES

    def test_aider_conf_registered(self):
        """Aider config is registered."""
        assert ".aider.conf.yml" in CORE_INSTRUCTION_FILES
