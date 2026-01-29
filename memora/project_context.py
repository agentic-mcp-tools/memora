"""Project context discovery and ingestion for AI instruction files.

This module provides functionality to discover and ingest AI instruction files
(CLAUDE.md, .cursorrules, AGENTS.md, etc.) from project directories and convert
them into searchable memories.

Supported file patterns:
- CLAUDE.md - Claude Code instructions
- AGENTS.md - Multi-agent system instructions
- .cursorrules - Cursor IDE rules
- .github/copilot-instructions.md - GitHub Copilot instructions
- GEMINI.md - Gemini tools instructions
- .aider.conf.yml - Aider configuration
- CONVENTIONS.md - General coding conventions
- .windsurfrules - Windsurf IDE rules
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


class FileFormat(Enum):
    """Supported file formats for instruction files."""

    MARKDOWN = "markdown"
    YAML = "yaml"
    PLAIN_TEXT = "plain_text"


class InstructionFileType(Enum):
    """Types of instruction files."""

    CLAUDE_MD = "claude_md"
    AGENTS_MD = "agents_md"
    CURSOR_RULES = "cursorrules"
    COPILOT_INSTRUCTIONS = "copilot_instructions"
    GEMINI_MD = "gemini_md"
    AIDER_CONF = "aider_conf"
    CONVENTIONS_MD = "conventions_md"
    WINDSURF_RULES = "windsurf_rules"
    CUSTOM = "custom"


@dataclass
class ParsedSection:
    """A parsed section from an instruction file."""

    title: str
    content: str
    section_path: str  # "Guidelines > Testing > Unit"
    section_index: int
    heading_level: int
    heading_anchor: str  # "unit-testing"
    content_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "title": self.title,
            "content": self.content,
            "section_path": self.section_path,
            "section_index": self.section_index,
            "heading_level": self.heading_level,
            "heading_anchor": self.heading_anchor,
            "content_hash": self.content_hash,
        }


@dataclass
class ParsedInstructions:
    """Result of parsing an instruction file."""

    sections: List[ParsedSection]
    raw_content: str
    file_hash: str
    file_type: InstructionFileType
    file_format: FileFormat

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sections": [s.to_dict() for s in self.sections],
            "raw_content": self.raw_content,
            "file_hash": self.file_hash,
            "file_type": self.file_type.value,
            "file_format": self.file_format.value,
        }


@dataclass
class ProjectContextConfig:
    """Configuration for project context discovery."""

    enabled: bool = True
    max_file_size: int = 1024 * 1024  # 1MB
    extract_sections: bool = True
    scan_parents: bool = False  # Security: don't scan parent dirs by default
    ignore_dirs: List[str] = field(
        default_factory=lambda: [
            ".git",
            "target",
            "node_modules",
            "vendor",
            ".venv",
            "__pycache__",
            "dist",
            "build",
            ".tox",
        ]
    )
    ignore_files: List[str] = field(
        default_factory=lambda: [".env*", "*.key", "*.pem", "*.p12", "secrets/*"]
    )
    default_visibility: str = "private"


# Core instruction files to scan (Phase 1)
CORE_INSTRUCTION_FILES: Dict[str, Tuple[InstructionFileType, FileFormat]] = {
    "CLAUDE.md": (InstructionFileType.CLAUDE_MD, FileFormat.MARKDOWN),
    "AGENTS.md": (InstructionFileType.AGENTS_MD, FileFormat.MARKDOWN),
    ".cursorrules": (InstructionFileType.CURSOR_RULES, FileFormat.PLAIN_TEXT),
    ".github/copilot-instructions.md": (
        InstructionFileType.COPILOT_INSTRUCTIONS,
        FileFormat.MARKDOWN,
    ),
    ".aider.conf.yml": (InstructionFileType.AIDER_CONF, FileFormat.YAML),
    "GEMINI.md": (InstructionFileType.GEMINI_MD, FileFormat.MARKDOWN),
    ".windsurfrules": (InstructionFileType.WINDSURF_RULES, FileFormat.PLAIN_TEXT),
    "CONVENTIONS.md": (InstructionFileType.CONVENTIONS_MD, FileFormat.MARKDOWN),
    "CODING_GUIDELINES.md": (InstructionFileType.CONVENTIONS_MD, FileFormat.MARKDOWN),
}


def compute_hash(content: str) -> str:
    """Compute SHA-256 hash of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def slugify(text: str) -> str:
    """Convert text to URL-friendly slug."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text).strip("-")
    return text


def parse_markdown_sections(content: str) -> List[ParsedSection]:
    """Parse markdown content into sections based on headings.

    Args:
        content: Raw markdown content

    Returns:
        List of ParsedSection objects representing each section
    """
    sections: List[ParsedSection] = []

    # Regex to match markdown headings (# to ######)
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    # Find all headings with their positions
    matches = list(heading_pattern.finditer(content))

    if not matches:
        # No headings found - treat entire content as one section
        if content.strip():
            sections.append(
                ParsedSection(
                    title="(Document Root)",
                    content=content.strip(),
                    section_path="",
                    section_index=0,
                    heading_level=0,
                    heading_anchor="root",
                    content_hash=compute_hash(content.strip()),
                )
            )
        return sections

    # Track heading hierarchy for section_path
    heading_stack: List[Tuple[int, str]] = []  # (level, title)

    for i, match in enumerate(matches):
        level = len(match.group(1))  # Number of # characters
        title = match.group(2).strip()

        # Get content between this heading and the next (or end of file)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        section_content = content[start:end].strip()

        # Skip empty sections
        if not section_content:
            # Still update the heading stack for hierarchy
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            continue

        # Build section path from heading hierarchy
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, title))

        section_path = " > ".join(t for _, t in heading_stack)

        sections.append(
            ParsedSection(
                title=title,
                content=section_content,
                section_path=section_path,
                section_index=len(sections),
                heading_level=level,
                heading_anchor=slugify(title),
                content_hash=compute_hash(section_content),
            )
        )

    return sections


def parse_yaml_sections(content: str) -> List[ParsedSection]:
    """Parse YAML content into sections based on top-level keys.

    Args:
        content: Raw YAML content

    Returns:
        List of ParsedSection objects representing each top-level key
    """
    sections: List[ParsedSection] = []

    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        # Return single section with error indicator
        sections.append(
            ParsedSection(
                title="(Parse Error)",
                content=f"YAML parse error: {e}\n\n---\n\n{content}",
                section_path="",
                section_index=0,
                heading_level=0,
                heading_anchor="parse-error",
                content_hash=compute_hash(content),
            )
        )
        return sections

    if not isinstance(data, dict):
        # Non-dict YAML - treat as single section
        sections.append(
            ParsedSection(
                title="(Document Root)",
                content=content.strip(),
                section_path="",
                section_index=0,
                heading_level=0,
                heading_anchor="root",
                content_hash=compute_hash(content),
            )
        )
        return sections

    # Create section for each top-level key
    for i, (key, value) in enumerate(data.items()):
        # Convert value to readable string
        if isinstance(value, (dict, list)):
            value_str = yaml.dump(value, default_flow_style=False)
        else:
            value_str = str(value)

        sections.append(
            ParsedSection(
                title=str(key),
                content=value_str.strip(),
                section_path=str(key),
                section_index=i,
                heading_level=1,
                heading_anchor=slugify(str(key)),
                content_hash=compute_hash(value_str),
            )
        )

    return sections


def parse_plain_text_sections(content: str) -> List[ParsedSection]:
    """Parse plain text content as a single section.

    Args:
        content: Raw text content

    Returns:
        List with single ParsedSection for the entire content
    """
    if not content.strip():
        return []

    return [
        ParsedSection(
            title="(Document)",
            content=content.strip(),
            section_path="",
            section_index=0,
            heading_level=0,
            heading_anchor="document",
            content_hash=compute_hash(content),
        )
    ]


def parse_instruction_file(
    content: str,
    file_type: InstructionFileType,
    file_format: FileFormat,
) -> ParsedInstructions:
    """Parse an instruction file into structured sections.

    Args:
        content: Raw file content
        file_type: Type of instruction file
        file_format: Format of the file (markdown, yaml, plain text)

    Returns:
        ParsedInstructions with sections and metadata
    """
    # Parse based on format
    if file_format == FileFormat.MARKDOWN:
        sections = parse_markdown_sections(content)
    elif file_format == FileFormat.YAML:
        sections = parse_yaml_sections(content)
    else:
        sections = parse_plain_text_sections(content)

    return ParsedInstructions(
        sections=sections,
        raw_content=content,
        file_hash=compute_hash(content),
        file_type=file_type,
        file_format=file_format,
    )


def should_ignore_path(path: Path, config: ProjectContextConfig) -> bool:
    """Check if a path should be ignored based on config.

    Args:
        path: Path to check
        config: Configuration with ignore patterns

    Returns:
        True if path should be ignored
    """
    path_str = str(path)

    # Check ignore directories
    for part in path.parts:
        if part in config.ignore_dirs:
            return True

    # Check ignore file patterns
    for pattern in config.ignore_files:
        if fnmatch(path.name, pattern):
            return True
        if fnmatch(path_str, pattern):
            return True

    return False


def discover_instruction_files(
    root_path: str | Path,
    config: Optional[ProjectContextConfig] = None,
) -> List[Tuple[Path, InstructionFileType, FileFormat]]:
    """Discover AI instruction files in a directory.

    Args:
        root_path: Directory to scan
        config: Optional configuration (uses defaults if not provided)

    Returns:
        List of (path, file_type, file_format) tuples for discovered files
    """
    if config is None:
        config = ProjectContextConfig()

    root = Path(root_path).resolve()
    discovered: List[Tuple[Path, InstructionFileType, FileFormat]] = []

    # Check each known instruction file pattern
    for pattern, (file_type, file_format) in CORE_INSTRUCTION_FILES.items():
        file_path = root / pattern

        if file_path.exists() and file_path.is_file():
            # Check file size
            if file_path.stat().st_size > config.max_file_size:
                continue

            # Check ignore patterns
            if should_ignore_path(file_path, config):
                continue

            discovered.append((file_path, file_type, file_format))

    return discovered


def scan_project_context(
    root_path: str | Path,
    config: Optional[ProjectContextConfig] = None,
) -> List[Dict[str, Any]]:
    """Scan a project directory for instruction files and parse them.

    Args:
        root_path: Directory to scan
        config: Optional configuration

    Returns:
        List of memory entries ready for storage
    """
    if config is None:
        config = ProjectContextConfig()

    root = Path(root_path).resolve()
    memories: List[Dict[str, Any]] = []

    # Discover instruction files
    files = discover_instruction_files(root, config)

    for file_path, file_type, file_format in files:
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception:
            continue

        # Parse the file
        parsed = parse_instruction_file(content, file_type, file_format)

        # Create parent memory for the file
        parent_metadata = {
            "source_file": str(file_path),
            "file_type": file_type.value,
            "file_format": file_format.value,
            "project_path": str(root),
            "file_hash": parsed.file_hash,
            "file_mtime": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            "section_count": len(parsed.sections),
            "is_parent": True,
        }

        parent_tags = ["project-context", file_type.value]

        # Add parent memory (summary)
        parent_content = f"# {file_path.name}\n\n"
        parent_content += f"Project context file from: {root}\n\n"
        parent_content += f"Sections: {len(parsed.sections)}\n\n"
        if parsed.sections:
            parent_content += "## Table of Contents\n\n"
            for section in parsed.sections:
                parent_content += f"- {section.title}\n"

        memories.append(
            {
                "content": parent_content,
                "metadata": parent_metadata,
                "tags": parent_tags,
            }
        )

        # Create child memories for each section (if enabled)
        if config.extract_sections and parsed.sections:
            for section in parsed.sections:
                section_metadata = {
                    "source_file": str(file_path),
                    "file_type": file_type.value,
                    "project_path": str(root),
                    "section_path": section.section_path,
                    "section_index": section.section_index,
                    "heading_level": section.heading_level,
                    "heading_anchor": section.heading_anchor,
                    "content_hash": section.content_hash,
                    "file_hash": parsed.file_hash,
                    "is_section": True,
                }

                section_tags = [
                    "project-context",
                    file_type.value,
                    f"section:{section.heading_anchor}",
                ]

                # Build section content with context
                section_content = f"# {section.title}\n\n"
                if section.section_path and section.section_path != section.title:
                    section_content += (
                        f"*From: {file_path.name} > {section.section_path}*\n\n"
                    )
                else:
                    section_content += f"*From: {file_path.name}*\n\n"
                section_content += section.content

                memories.append(
                    {
                        "content": section_content,
                        "metadata": section_metadata,
                        "tags": section_tags,
                    }
                )

    return memories


def get_file_type_from_path(
    file_path: str | Path,
) -> Optional[Tuple[InstructionFileType, FileFormat]]:
    """Get the instruction file type from a file path.

    Args:
        file_path: Path to the file

    Returns:
        Tuple of (file_type, file_format) or None if not recognized
    """
    path = Path(file_path)

    # Check exact matches first
    for pattern, (file_type, file_format) in CORE_INSTRUCTION_FILES.items():
        if path.name == pattern or str(path).endswith(pattern):
            return (file_type, file_format)

    # Check if it's a markdown file (generic)
    if path.suffix.lower() in (".md", ".markdown"):
        return (InstructionFileType.CUSTOM, FileFormat.MARKDOWN)

    # Check if it's a YAML file (generic)
    if path.suffix.lower() in (".yml", ".yaml"):
        return (InstructionFileType.CUSTOM, FileFormat.YAML)

    return None


def find_existing_context_memories(
    conn: Any,
    project_path: str,
    source_file: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Find existing project context memories for a project.

    Args:
        conn: Database connection
        project_path: Path to the project
        source_file: Optional specific source file to filter by

    Returns:
        List of existing memories matching the criteria
    """
    import json

    query = """
        SELECT id, content, metadata, tags
        FROM memories
        WHERE json_extract(metadata, '$.project_path') = ?
    """
    params = [project_path]

    if source_file:
        query += " AND json_extract(metadata, '$.source_file') = ?"
        params.append(source_file)

    rows = conn.execute(query, params).fetchall()

    results = []
    for row in rows:
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        tags = json.loads(row["tags"]) if row["tags"] else []
        results.append(
            {
                "id": row["id"],
                "content": row["content"],
                "metadata": metadata,
                "tags": tags,
            }
        )

    return results


def update_or_create_context_memory(
    conn: Any,
    memory_data: Dict[str, Any],
    existing_memories: List[Dict[str, Any]],
) -> Tuple[str, Optional[int]]:
    """Update an existing context memory or create a new one.

    Uses idempotent strategy:
    1. If section_path matches existing → update
    2. Else if content_hash matches → move/rename
    3. Else → create new

    Args:
        conn: Database connection
        memory_data: New memory data
        existing_memories: List of existing memories to check against

    Returns:
        Tuple of (action, memory_id) where action is "created", "updated", or "unchanged"
    """
    import json

    from .storage import add_memory, update_memory

    metadata = memory_data.get("metadata", {})
    source_file = metadata.get("source_file")
    section_path = metadata.get("section_path", "")
    content_hash = metadata.get("content_hash")
    is_section = metadata.get("is_section", False)

    # Find existing memory by section_path (primary key for sections)
    for existing in existing_memories:
        ex_metadata = existing.get("metadata", {})
        ex_source = ex_metadata.get("source_file")
        ex_section = ex_metadata.get("section_path", "")
        ex_hash = ex_metadata.get("content_hash")
        ex_is_section = ex_metadata.get("is_section", False)

        # Match by source file and section path
        if (
            ex_source == source_file
            and ex_section == section_path
            and ex_is_section == is_section
        ):
            # Check if content changed
            if ex_hash == content_hash:
                return ("unchanged", existing["id"])

            # Update existing memory
            update_memory(
                conn,
                existing["id"],
                content=memory_data["content"],
                metadata=memory_data["metadata"],
                tags=memory_data["tags"],
            )
            return ("updated", existing["id"])

    # Check for content hash match (section was renamed/moved)
    if is_section and content_hash:
        for existing in existing_memories:
            ex_metadata = existing.get("metadata", {})
            ex_hash = ex_metadata.get("content_hash")
            ex_source = ex_metadata.get("source_file")

            if ex_source == source_file and ex_hash == content_hash:
                # Same content, different section path - update section_path
                update_memory(
                    conn,
                    existing["id"],
                    content=memory_data["content"],
                    metadata=memory_data["metadata"],
                    tags=memory_data["tags"],
                )
                return ("moved", existing["id"])

    # Create new memory
    result = add_memory(
        conn,
        content=memory_data["content"],
        metadata=memory_data["metadata"],
        tags=memory_data["tags"],
    )

    return ("created", result.get("id") if result else None)
