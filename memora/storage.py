"""SQLite storage helpers shared by memory servers."""

from __future__ import annotations

import base64
import io
import json
import math
import mimetypes
import os
import re
import sqlite3
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from typing import Sequence as TypingSequence

from PIL import Image

from .backends import D1Connection, StorageBackend, parse_backend_uri

ROOT = Path(__file__).resolve().parent

# Storage backend configuration
# Priority: MEMORA_STORAGE_URI > MEMORA_DB_PATH (legacy) > default
_storage_uri = os.getenv("MEMORA_STORAGE_URI")
if _storage_uri:
    # New URI-based configuration (supports s3://, file://, etc.)
    STORAGE_BACKEND = parse_backend_uri(_storage_uri)
else:
    # Legacy: Use MEMORA_DB_PATH or default local path
    _db_path_env = os.getenv("MEMORA_DB_PATH")
    if _db_path_env:
        DB_PATH = Path(os.path.expanduser(os.path.expandvars(_db_path_env)))
    else:
        DB_PATH = Path.home() / ".local" / "share" / "memora" / "memories.db"
    from .backends import LocalSQLiteBackend

    STORAGE_BACKEND = LocalSQLiteBackend(DB_PATH)

# Embedding backend configuration
EMBEDDING_MODEL = os.getenv(
    "MEMORA_EMBEDDING_MODEL", "openai"
)  # openai, sentence-transformers, tfidf

# LLM configuration for deduplication comparison
LLM_ENABLED = os.getenv("MEMORA_LLM_ENABLED", "true").lower() in ("true", "1", "yes")
LLM_MODEL = os.getenv("MEMORA_LLM_MODEL", "gpt-4o-mini")

# Event notification configuration
EVENT_TRIGGER_TAG = "shared-cache"

# Content validation limits
MIN_CONTENT_LENGTH = 3
MAX_CONTENT_LENGTH = 50000  # ~50KB text

# Memory tier configuration
VALID_TIERS = {"daily", "permanent"}
DEFAULT_TIER = "permanent"

# Embedding cache configuration
EMBEDDING_CACHE_ENABLED = os.getenv("MEMORA_EMBEDDING_CACHE", "true").lower() in (
    "true",
    "1",
    "yes",
)
EMBEDDING_CACHE_MAX_ENTRIES = int(os.getenv("MEMORA_EMBEDDING_CACHE_SIZE", "50000"))

# Secret/PII detection patterns (warn only, don't block)
SECRET_PATTERNS: List[tuple[str, str]] = [
    (r"sk-(?:proj-)?[a-zA-Z0-9]{20,}", "OpenAI API key"),
    (r"sk-or-[a-zA-Z0-9-]{20,}", "OpenRouter API key"),
    (r"sk-ant-[a-zA-Z0-9-]{20,}", "Anthropic API key"),
    (r"AKIA[0-9A-Z]{16}", "AWS Access Key"),
    (r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----", "Private key"),
    (r"Bearer [a-zA-Z0-9_-]{20,}", "Bearer token"),
    (r"ghp_[a-zA-Z0-9]{36}", "GitHub PAT"),
    (r"gho_[a-zA-Z0-9]{36}", "GitHub OAuth token"),
    (r"github_pat_[a-zA-Z0-9_]{22,}", "GitHub fine-grained PAT"),
    (r"xox[baprs]-[a-zA-Z0-9-]{10,}", "Slack token"),
    (r"(?i)password\s*[:=]\s*[^\s]{4,}", "Password in plaintext"),
    (r"(?i)secret\s*[:=]\s*[^\s]{4,}", "Secret in plaintext"),
    (r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "Credit card number"),
]


def _detect_secrets(content: str) -> List[str]:
    """Detect potential secrets/PII in content. Returns list of warnings."""
    warnings = []
    for pattern, description in SECRET_PATTERNS:
        if re.search(pattern, content):
            warnings.append(description)
    return warnings


def _redact_secrets(content: str) -> tuple[str, List[str]]:
    """Redact secrets/PII from content. Returns (redacted_content, list of redacted types)."""
    redacted = []
    result = content
    for pattern, description in SECRET_PATTERNS:
        if re.search(pattern, result):
            result = re.sub(pattern, "[REDACTED]", result)
            redacted.append(description)
    return result, redacted


def soft_trim(
    content: str,
    max_length: int = 500,
    head_ratio: float = 0.6,
    tail_ratio: float = 0.3,
) -> str:
    """
    Soft-trim content, preserving head and tail with ellipsis in the middle.

    This is useful for displaying long content while keeping both the beginning
    (which often contains key context) and the end (which may have conclusions).

    Args:
        content: The content to trim
        max_length: Maximum output length (default: 500 chars)
        head_ratio: Proportion of max_length for head (default: 0.6 = 60%)
        tail_ratio: Proportion of max_length for tail (default: 0.3 = 30%)
                   The remaining 10% is reserved for the ellipsis message.

    Returns:
        Original content if within max_length, or trimmed with ellipsis.

    Example:
        >>> soft_trim("A" * 1000, max_length=100)
        'AAAAAA...\\n...[840 chars truncated]...\\nAAAAA'
    """
    if len(content) <= max_length:
        return content

    # Calculate sizes
    head_size = int(max_length * head_ratio)
    tail_size = int(max_length * tail_ratio)
    truncated_chars = len(content) - head_size - tail_size

    # Build the trimmed result
    head = content[:head_size].rstrip()
    tail = content[-tail_size:].lstrip() if tail_size > 0 else ""

    if tail:
        return f"{head}\n...[{truncated_chars} chars truncated]...\n{tail}"
    else:
        return f"{head}\n...[{truncated_chars} chars truncated]..."


def content_preview(content: str, max_length: int = 200) -> str:
    """
    Generate a preview of content for compact listings.

    Unlike soft_trim, this just takes the beginning and adds ellipsis.

    Args:
        content: The content to preview
        max_length: Maximum preview length (default: 200 chars)

    Returns:
        Content truncated to max_length with "..." if needed.
    """
    if len(content) <= max_length:
        return content
    return content[:max_length].rstrip() + "..."


# ---------------------------------------------------------------------------
# Conversation Chunking for Session Transcript Indexing
# ---------------------------------------------------------------------------


def chunk_conversation(
    messages: List[Dict[str, Any]],
    chunk_size: int = 10,
    overlap: int = 2,
    include_metadata: bool = True,
) -> List[Dict[str, Any]]:
    """
    Split conversation messages into overlapping chunks for indexing.

    Creates chunks of messages with overlap to maintain context across chunk
    boundaries. Each chunk can be indexed as a separate memory for semantic search.

    Args:
        messages: List of message dicts with 'role', 'content', and optional 'timestamp'
        chunk_size: Number of messages per chunk (default: 10)
        overlap: Number of messages to overlap between chunks (default: 2)
        include_metadata: Include chunk metadata (index, total_chunks, message_range)

    Returns:
        List of chunk dicts with:
        - content: Formatted text of the chunk
        - messages: Original message dicts in this chunk
        - chunk_index: 0-based chunk number
        - total_chunks: Total number of chunks
        - message_range: (start_index, end_index) of messages in this chunk

    Example:
        >>> messages = [{"role": "user", "content": "Hello"}, ...]
        >>> chunks = chunk_conversation(messages, chunk_size=5, overlap=1)
        >>> len(chunks)  # Number of chunks
        3
    """
    if not messages:
        return []

    # Validate overlap
    if overlap >= chunk_size:
        overlap = chunk_size - 1

    chunks = []
    step = chunk_size - overlap

    # Calculate total chunks
    if len(messages) <= chunk_size:
        total_chunks = 1
    else:
        total_chunks = max(1, (len(messages) - overlap + step - 1) // step)

    chunk_index = 0
    for i in range(0, len(messages), step):
        chunk_messages = messages[i : i + chunk_size]
        if not chunk_messages:
            break

        # Format messages as text
        formatted_lines = []
        for msg in chunk_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")

            if timestamp:
                formatted_lines.append(f"[{timestamp}] {role}: {content}")
            else:
                formatted_lines.append(f"{role}: {content}")

        chunk_content = "\n".join(formatted_lines)

        chunk_data = {
            "content": chunk_content,
            "messages": chunk_messages,
        }

        if include_metadata:
            chunk_data.update(
                {
                    "chunk_index": chunk_index,
                    "total_chunks": total_chunks,
                    "message_range": (i, i + len(chunk_messages) - 1),
                }
            )

        chunks.append(chunk_data)
        chunk_index += 1

        # Stop if we've covered all messages
        if i + chunk_size >= len(messages):
            break

    return chunks


def format_conversation_chunk(
    messages: List[Dict[str, Any]],
    include_timestamps: bool = True,
    separator: str = "\n",
) -> str:
    """
    Format a list of messages into a single string for embedding.

    Args:
        messages: List of message dicts with 'role', 'content', and optional 'timestamp'
        include_timestamps: Include timestamps in the formatted output
        separator: Separator between messages (default: newline)

    Returns:
        Formatted string representation of the messages.
    """
    formatted_lines = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", "")

        if include_timestamps and timestamp:
            formatted_lines.append(f"[{timestamp}] {role}: {content}")
        else:
            formatted_lines.append(f"{role}: {content}")

    return separator.join(formatted_lines)


def _validate_content(content: str) -> str:
    """Validate and normalize content. Raises ValueError if invalid."""
    if not isinstance(content, str):
        content = str(content)

    # Trim whitespace
    content = content.strip()

    # Normalize excessive newlines (max 2 consecutive)
    content = re.sub(r"\n{3,}", "\n\n", content)

    # Length validation
    if len(content) < MIN_CONTENT_LENGTH:
        raise ValueError(f"Content too short (min {MIN_CONTENT_LENGTH} characters)")
    if len(content) > MAX_CONTENT_LENGTH:
        raise ValueError(f"Content too long (max {MAX_CONTENT_LENGTH} characters)")

    return content


# ---------------------------------------------------------------------------
# Auto-detection of memory types (issue, todo) from content
# ---------------------------------------------------------------------------

# Keywords that suggest content is about a bug/issue
_ISSUE_KEYWORDS = [
    "bug",
    "fix",
    "fixed",
    "error",
    "crash",
    "broken",
    "resolve",
    "resolved",
    "problem",
    "issue",
    "fault",
    "defect",
    "patch",
    "hotfix",
    "regression",
]

# Keywords that suggest content is a TODO/task
_TODO_KEYWORDS = [
    "todo",
    "task",
    "implement",
    "add feature",
    "need to",
    "should add",
    "plan to",
    "will add",
    "must add",
    "want to add",
    "roadmap",
]

# Patterns that strongly suggest closed/resolved issues
_RESOLVED_PATTERNS = [
    r"\*\*fix\*\*",  # **Fix** or **fix**
    r"fix(?:ed)?:",  # Fix: or Fixed:
    r"resolved?:",  # Resolve: or Resolved:
    r"problem:.*(?:fix|solution)",  # Problem: ... fix/solution
    r"root cause:",  # Root cause analysis
]


def _detect_memory_type(
    content: str,
    metadata: Optional[Dict[str, Any]],
    tags: Optional[List[str]],
) -> Optional[Dict[str, Any]]:
    """Auto-detect if content should be an issue or TODO.

    Returns metadata dict to merge if type detected, None otherwise.
    Only detects if no explicit type is already set.
    """
    # Don't override if type is already explicitly set
    if metadata and metadata.get("type"):
        return None

    # Don't detect if already tagged as issue or todo
    if tags:
        if "memora/issues" in tags or "memora/todos" in tags:
            return None

    content_lower = content.lower()

    # Count keyword matches
    issue_matches = sum(1 for kw in _ISSUE_KEYWORDS if kw in content_lower)
    todo_matches = sum(1 for kw in _TODO_KEYWORDS if kw in content_lower)

    # Check for resolved patterns (stronger signal for closed issues)
    has_resolved_pattern = any(
        re.search(pattern, content_lower) for pattern in _RESOLVED_PATTERNS
    )

    # Require at least 2 keyword matches to avoid false positives
    # Exception: resolved patterns are a strong enough signal alone
    if issue_matches >= 2 or (issue_matches >= 1 and has_resolved_pattern):
        # Detect if it's a closed/resolved issue or open
        is_closed = has_resolved_pattern or any(
            word in content_lower for word in ["fixed", "resolved", "patched"]
        )

        return {
            "_detected_type": "issue",
            "_auto_metadata": {
                "type": "issue",
                "status": "closed" if is_closed else "open",
                "closed_reason": "complete" if is_closed else None,
                "severity": "minor",
                "category": "bug",
            },
            "_auto_tags": ["memora/issues"],
        }

    if todo_matches >= 2:
        return {
            "_detected_type": "todo",
            "_auto_metadata": {
                "type": "todo",
                "status": "open",
                "priority": "medium",
            },
            "_auto_tags": ["memora/todos"],
        }

    return None


def _apply_auto_detection(
    content: str,
    metadata: Optional[Dict[str, Any]],
    tags: Optional[List[str]],
) -> tuple[Optional[Dict[str, Any]], Optional[List[str]]]:
    """Apply auto-detection and return updated metadata and tags.

    Returns (updated_metadata, updated_tags) tuple.
    """
    detection = _detect_memory_type(content, metadata, tags)
    if not detection:
        return metadata, tags

    # Merge detected metadata with provided metadata
    updated_metadata = dict(metadata) if metadata else {}
    updated_metadata.update(detection["_auto_metadata"])

    # Add detected tags
    updated_tags = list(tags) if tags else []
    for tag in detection["_auto_tags"]:
        if tag not in updated_tags:
            updated_tags.append(tag)

    return updated_metadata, updated_tags


def _emit_event(conn: sqlite3.Connection, memory_id: int, tags: List[str]) -> None:
    """Emit an event notification if memory has the trigger tag."""
    if EVENT_TRIGGER_TAG in tags:
        tags_json = json.dumps(tags, ensure_ascii=False)
        try:
            conn.execute(
                "INSERT INTO memories_events (memory_id, tags) VALUES (?, ?)",
                (memory_id, tags_json),
            )
            conn.commit()
        except Exception:
            # Don't fail memory operations if event emission fails
            pass


def connect(*, check_same_thread: bool = True) -> sqlite3.Connection:
    """Create a database connection using the configured storage backend.

    For cloud backends, this will automatically sync from cloud before use.

    Args:
        check_same_thread: SQLite connection parameter

    Returns:
        sqlite3.Connection ready for use
    """
    conn = STORAGE_BACKEND.connect(check_same_thread=check_same_thread)
    ensure_schema(conn)
    return conn


def sync_to_cloud() -> None:
    """Sync database to cloud storage if using a cloud backend.

    This should be called after write operations to ensure changes are
    persisted to cloud storage.
    """
    STORAGE_BACKEND.sync_after_write()


def get_backend_info() -> dict:
    """Get information about the current storage backend.

    Returns:
        Dictionary with backend type, configuration, and status
    """
    return STORAGE_BACKEND.get_info()


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            metadata TEXT,
            tags TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT
        )
        """
    )
    conn.commit()
    _ensure_fts(conn)
    _ensure_embeddings_table(conn)
    _ensure_crossrefs_table(conn)
    _ensure_events_table(conn)
    _ensure_importance_columns(conn)
    _ensure_updated_at_column(conn)
    _ensure_tier_columns(conn)
    _ensure_workspace_column(conn)
    _ensure_embedding_cache_table(conn)
    _ensure_sync_tables(conn)
    _ensure_share_events_table(conn)
    _ensure_session_tables(conn)
    _ensure_identity_tables(conn)


def _ensure_fts(conn: sqlite3.Connection) -> None:
    # D1 doesn't support FTS5 virtual tables
    if isinstance(conn, D1Connection):
        return
    table_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='memories_fts'"
    ).fetchone()
    if not table_exists:
        conn.execute(
            """
            CREATE VIRTUAL TABLE memories_fts
            USING fts5(content, metadata, tags)
            """
        )
        conn.commit()


def _ensure_embeddings_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memories_embeddings (
            memory_id INTEGER PRIMARY KEY,
            embedding TEXT,
            FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
        )
        """
    )
    # Metadata table for storing embedding model info
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memories_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    conn.commit()


def _ensure_crossrefs_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memories_crossrefs (
            memory_id INTEGER PRIMARY KEY,
            related TEXT,
            FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
        )
        """
    )
    conn.commit()


def _ensure_events_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memories_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER NOT NULL,
            tags TEXT NOT NULL,
            timestamp TEXT NOT NULL DEFAULT (datetime('now')),
            consumed INTEGER DEFAULT 0,
            FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
        )
        """
    )
    conn.commit()


def _ensure_importance_columns(conn: sqlite3.Connection) -> None:
    """Add importance scoring columns to memories table if they don't exist."""
    # Check if columns already exist
    cursor = conn.execute("PRAGMA table_info(memories)")
    columns = {row[1] for row in cursor.fetchall()}

    if "importance" not in columns:
        conn.execute("ALTER TABLE memories ADD COLUMN importance REAL DEFAULT 1.0")

    if "last_accessed" not in columns:
        conn.execute("ALTER TABLE memories ADD COLUMN last_accessed TEXT")

    if "access_count" not in columns:
        conn.execute("ALTER TABLE memories ADD COLUMN access_count INTEGER DEFAULT 0")

    conn.commit()


def _ensure_updated_at_column(conn: sqlite3.Connection) -> None:
    """Add updated_at column to memories table if it doesn't exist."""
    cursor = conn.execute("PRAGMA table_info(memories)")
    columns = {row[1] for row in cursor.fetchall()}

    if "updated_at" not in columns:
        conn.execute("ALTER TABLE memories ADD COLUMN updated_at TEXT")
        conn.commit()


def _ensure_tier_columns(conn: sqlite3.Connection) -> None:
    """Add tier and expires_at columns to memories table if they don't exist."""
    cursor = conn.execute("PRAGMA table_info(memories)")
    columns = {row[1] for row in cursor.fetchall()}

    if "tier" not in columns:
        conn.execute(
            f"ALTER TABLE memories ADD COLUMN tier TEXT DEFAULT '{DEFAULT_TIER}'"
        )

    if "expires_at" not in columns:
        conn.execute("ALTER TABLE memories ADD COLUMN expires_at TEXT")

    conn.commit()


def _ensure_workspace_column(conn: sqlite3.Connection) -> None:
    """Add workspace column to memories table for multi-workspace support."""
    cursor = conn.execute("PRAGMA table_info(memories)")
    columns = {row[1] for row in cursor.fetchall()}

    if "workspace" not in columns:
        conn.execute("ALTER TABLE memories ADD COLUMN workspace TEXT DEFAULT 'default'")
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memories_workspace
            ON memories(workspace)
            """
        )
        conn.commit()


# Default workspace name
DEFAULT_WORKSPACE = "default"


def _ensure_embedding_cache_table(conn: sqlite3.Connection) -> None:
    """Create embedding cache table for LRU caching of embeddings."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embedding_cache (
            content_hash TEXT PRIMARY KEY,
            embedding TEXT NOT NULL,
            model TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            last_accessed TEXT NOT NULL DEFAULT (datetime('now')),
            access_count INTEGER DEFAULT 1
        )
        """
    )
    # Index for LRU eviction (oldest entries first)
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_embedding_cache_lru
        ON embedding_cache(last_accessed)
        """
    )
    conn.commit()


def _ensure_share_events_table(conn: sqlite3.Connection) -> None:
    """Create table for cross-session memory sharing."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS share_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER NOT NULL,
            source_agent TEXT,
            target_agents TEXT,
            message TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            acknowledged_by TEXT,
            FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_share_events_memory
        ON share_events(memory_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_share_events_created
        ON share_events(created_at)
        """
    )
    conn.commit()


def _ensure_sync_tables(conn: sqlite3.Connection) -> None:
    """Create tables for sync version tracking and delta sync."""
    cursor = conn.execute("PRAGMA table_info(memories)")
    columns = {row[1] for row in cursor.fetchall()}

    # Add sync_version column to memories if not exists
    if "sync_version" not in columns:
        conn.execute("ALTER TABLE memories ADD COLUMN sync_version INTEGER DEFAULT 0")

    # Create global sync version counter
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sync_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    # Initialize global version if not exists
    conn.execute(
        """
        INSERT OR IGNORE INTO sync_metadata (key, value) VALUES ('global_version', '0')
        """
    )

    # Create sync state table for tracking agent sync positions
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sync_state (
            agent_id TEXT PRIMARY KEY,
            last_sync_version INTEGER NOT NULL DEFAULT 0,
            last_sync_at TEXT DEFAULT (datetime('now'))
        )
        """
    )

    # Create deleted memories tracking table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS deleted_memories (
            memory_id INTEGER PRIMARY KEY,
            content_preview TEXT,
            deleted_at TEXT NOT NULL DEFAULT (datetime('now')),
            sync_version INTEGER NOT NULL
        )
        """
    )

    # Index for efficient delta queries
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_memories_sync_version
        ON memories(sync_version)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_deleted_memories_sync_version
        ON deleted_memories(sync_version)
        """
    )

    conn.commit()


def _ensure_identity_tables(conn: sqlite3.Connection) -> None:
    """Create tables for identity linking (entity unification)."""
    # Create identities table for canonical entities
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS identities (
            canonical_id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            entity_type TEXT DEFAULT 'person',
            metadata TEXT DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )

    # Create identity_aliases table for alternative names/IDs
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS identity_aliases (
            alias TEXT PRIMARY KEY,
            canonical_id TEXT NOT NULL,
            source TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY(canonical_id) REFERENCES identities(canonical_id) ON DELETE CASCADE
        )
        """
    )

    # Create memory_identity_links table to link memories to identities
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_identity_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER NOT NULL,
            identity_id TEXT NOT NULL,
            mention_text TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE,
            FOREIGN KEY(identity_id) REFERENCES identities(canonical_id) ON DELETE CASCADE,
            UNIQUE(memory_id, identity_id)
        )
        """
    )

    # Create indexes for efficient lookups
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_identity_aliases_canonical
        ON identity_aliases(canonical_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_memory_identity_links_memory
        ON memory_identity_links(memory_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_memory_identity_links_identity
        ON memory_identity_links(identity_id)
        """
    )

    conn.commit()


def _ensure_session_tables(conn: sqlite3.Connection) -> None:
    """Create tables for session transcript indexing."""
    # Create sessions table to track indexed sessions
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            title TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            last_indexed_at TEXT,
            message_count INTEGER DEFAULT 0,
            chunk_count INTEGER DEFAULT 0,
            metadata TEXT DEFAULT '{}'
        )
        """
    )

    # Create session_chunks table for indexed conversation chunks
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS session_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            message_start INTEGER NOT NULL,
            message_end INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            memory_id INTEGER,
            FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
            FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE SET NULL,
            UNIQUE(session_id, chunk_index)
        )
        """
    )

    # Create index for efficient session lookups
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_session_chunks_session_id
        ON session_chunks(session_id)
        """
    )

    # Create index for memory lookups
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_session_chunks_memory_id
        ON session_chunks(memory_id)
        """
    )

    conn.commit()


def _get_next_sync_version(conn: sqlite3.Connection) -> int:
    """Get and increment the global sync version counter."""
    row = conn.execute(
        "SELECT value FROM sync_metadata WHERE key = 'global_version'"
    ).fetchone()
    current = int(row[0]) if row else 0
    new_version = current + 1
    conn.execute(
        "UPDATE sync_metadata SET value = ? WHERE key = 'global_version'",
        (str(new_version),),
    )
    return new_version


def _validate_tier(tier: Optional[str]) -> str:
    """Validate and return tier value. Returns DEFAULT_TIER if None."""
    if tier is None:
        return DEFAULT_TIER
    if tier not in VALID_TIERS:
        raise ValueError(
            f"Invalid tier '{tier}'. Must be one of: {', '.join(sorted(VALID_TIERS))}"
        )
    return tier


def _normalize_expires_at(expires_at: Optional[str]) -> Optional[str]:
    """Normalize expires_at to canonical ISO format (YYYY-MM-DDTHH:MM:SS).

    Handles common variations like space-separated timestamps.
    Returns None if input is None or empty.
    """
    if not expires_at:
        return None

    # Already in correct format
    if "T" in expires_at:
        return expires_at

    # Convert space-separated to T-separated (e.g., "2026-01-28 12:00:00" -> "2026-01-28T12:00:00")
    if " " in expires_at:
        parts = expires_at.split(" ", 1)
        if len(parts) == 2:
            return f"{parts[0]}T{parts[1]}"

    # Return as-is if we can't parse it (will be compared as string)
    return expires_at


def _build_metadata_dict(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    """Return metadata in a canonical form with optional hierarchy path."""

    normalised: Dict[str, Any] = {}

    for key in metadata.keys():
        if not isinstance(key, str):
            raise ValueError("Metadata keys must be strings")

    tasks_value = metadata.get("tasks")
    done_present = "done" in metadata
    done_value = metadata.get("done")

    for key, value in metadata.items():
        if key in {"tasks", "done", "hierarchy", "section", "subsection"}:
            continue
        normalised[key] = value

    path: List[str] = []

    if "hierarchy" in metadata:
        hierarchy = metadata["hierarchy"]
        path_source: Optional[Sequence[Any]] = None

        if isinstance(hierarchy, Mapping):
            if "path" in hierarchy and hierarchy["path"] is not None:
                path_source = hierarchy["path"]
            else:
                collected: List[Any] = []
                for key in ("section", "subsection"):
                    if key in hierarchy and hierarchy[key] is not None:
                        collected.append(hierarchy[key])
                if collected:
                    path_source = collected
        elif isinstance(hierarchy, Sequence) and not isinstance(
            hierarchy, (str, bytes)
        ):
            path_source = hierarchy
        else:
            raise ValueError("metadata['hierarchy'] must be a mapping or sequence")

        if path_source is None:
            raise ValueError("metadata['hierarchy'] must define a path")

        try:
            path = [str(part) for part in path_source if part is not None]
        except TypeError as exc:
            raise ValueError("metadata['hierarchy'] path must be iterable") from exc

    else:
        if "section" in metadata and metadata["section"] is not None:
            path.append(str(metadata["section"]))
        if "subsection" in metadata and metadata["subsection"] is not None:
            path.append(str(metadata["subsection"]))

    # Always rewrite hierarchy to the canonical form
    normalised.pop("hierarchy", None)

    if tasks_value is not None:
        normalised["tasks"] = _normalise_tasks(tasks_value)

    if done_present:
        normalised["done"] = (
            _coerce_bool(done_value) if done_value is not None else False
        )

    if path:
        normalised["hierarchy"] = {"path": path}
        normalised["section"] = path[0]
        if len(path) > 1:
            normalised["subsection"] = path[1]
        else:
            normalised.pop("subsection", None)
    else:
        normalised.pop("section", None)
        normalised.pop("subsection", None)

    return normalised


TRUE_STRINGS = {"true", "1", "yes", "y", "on"}
FALSE_STRINGS = {"false", "0", "no", "n", "off"}


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in TRUE_STRINGS:
            return True
        if lowered in FALSE_STRINGS:
            return False
        raise ValueError("Boolean strings must be true/false, yes/no, on/off, or 1/0")
    raise ValueError("Boolean fields must be bool-like values")


def _normalise_tasks(tasks: Any) -> List[Dict[str, Any]]:
    if isinstance(tasks, (str, bytes)) or not isinstance(tasks, TypingSequence):
        raise ValueError("metadata['tasks'] must be a sequence of task entries")

    normalised: List[Dict[str, Any]] = []

    for index, item in enumerate(tasks):
        if isinstance(item, Mapping):
            if "title" not in item:
                raise ValueError(f"Task at index {index} must include a 'title'")
            title = str(item["title"]).strip()
            if not title:
                raise ValueError(
                    f"Task at index {index} must provide a non-empty title"
                )
            task_entry: Dict[str, Any] = {"title": title}
            if "done" in item and item["done"] is not None:
                try:
                    task_entry["done"] = _coerce_bool(item["done"])
                except ValueError as exc:
                    raise ValueError(
                        f"Task at index {index} has an invalid 'done' flag"
                    ) from exc
            else:
                task_entry["done"] = False
            for key, value in item.items():
                if key in {"title", "done"}:
                    continue
                task_entry[key] = value
        elif isinstance(item, str):
            title = item.strip()
            if not title:
                raise ValueError(
                    f"Task at index {index} must provide a non-empty title"
                )
            task_entry = {"title": title, "done": False}
        else:
            raise ValueError(
                "metadata['tasks'] entries must be mappings with 'title' or plain strings"
            )
        normalised.append(task_entry)

    return normalised


def _process_image_for_storage(
    src: str,
    memory_id: Optional[int] = None,
    image_index: int = 0,
    max_size: int = 1200,
    quality: int = 85,
) -> str:
    """Process image: resize, compress, and upload to R2 or encode as data URI.

    Args:
        src: Image source (file path, file:// URI, data URI, or existing URL)
        memory_id: ID of the memory (required for R2 upload)
        image_index: Index of the image within the memory
        max_size: Maximum dimension (width or height) in pixels. Default 1200 (R2 storage).
        quality: JPEG quality (1-100). Default 85.

    Returns:
        R2 URL if cloud storage configured, otherwise base64 data URI
    """
    from .image_storage import get_image_storage_instance, parse_data_uri

    image_storage = get_image_storage_instance()

    # Already an R2 reference or HTTP(S) URL - return as-is
    if (
        src.startswith("r2://")
        or src.startswith("http://")
        or src.startswith("https://")
    ):
        return src

    # Handle existing data URI - upload to R2 if configured
    if src.startswith("data:"):
        if image_storage and memory_id is not None:
            try:
                image_bytes, content_type = parse_data_uri(src)
                return image_storage.upload_image(
                    image_data=image_bytes,
                    content_type=content_type,
                    memory_id=memory_id,
                    image_index=image_index,
                )
            except Exception as e:
                # If R2 upload fails, keep the data URI
                import logging

                logging.getLogger(__name__).warning(
                    f"Failed to upload data URI to R2: {e}"
                )
                return src
        return src

    # Handle file:// URIs
    if src.startswith("file://"):
        file_path = src[7:]  # Remove file:// prefix
    else:
        file_path = src

    # Check if file exists
    path = Path(file_path).expanduser()
    if not path.exists():
        return src  # Return original if file doesn't exist

    try:
        # Open image with Pillow
        img = Image.open(path)

        # Convert RGBA to RGB if saving as JPEG (no alpha support)
        has_alpha = img.mode in ("RGBA", "LA", "P")

        # Resize if larger than max_size
        width, height = img.size
        if width > max_size or height > max_size:
            # Calculate new size maintaining aspect ratio
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Encode to bytes
        buffer = io.BytesIO()
        if has_alpha:
            # Keep PNG for images with transparency
            img.save(buffer, format="PNG", optimize=True)
            mime_type = "image/png"
        else:
            # Convert to RGB and save as JPEG for smaller size
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(buffer, format="JPEG", quality=quality, optimize=True)
            mime_type = "image/jpeg"

        image_bytes = buffer.getvalue()

        # Upload to R2 if configured
        if image_storage and memory_id is not None:
            try:
                return image_storage.upload_image(
                    image_data=image_bytes,
                    content_type=mime_type,
                    memory_id=memory_id,
                    image_index=image_index,
                )
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(
                    f"Failed to upload image to R2: {e}"
                )
                # Fall through to base64 encoding

        # Fallback: encode as base64 data URI
        b64 = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime_type};base64,{b64}"

    except Exception:
        # Fallback: read raw file if Pillow fails
        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type is None or not mime_type.startswith("image/"):
            mime_type = "image/png"
        with open(path, "rb") as f:
            raw_bytes = f.read()

        # Try R2 upload for raw file
        if image_storage and memory_id is not None:
            try:
                return image_storage.upload_image(
                    image_data=raw_bytes,
                    content_type=mime_type,
                    memory_id=memory_id,
                    image_index=image_index,
                )
            except Exception:
                pass  # Fall through to base64

        b64 = base64.b64encode(raw_bytes).decode("ascii")
        return f"data:{mime_type};base64,{b64}"


def _process_metadata_images(
    metadata: Dict[str, Any],
    memory_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Process images in metadata, uploading to R2 or encoding as data URIs.

    Args:
        metadata: Memory metadata dict potentially containing 'images' list
        memory_id: ID of the memory (required for R2 upload)

    Returns:
        Metadata dict with processed image sources
    """
    if "images" not in metadata:
        return metadata

    images = metadata.get("images")
    if not isinstance(images, list):
        return metadata

    processed_images = []
    for idx, img in enumerate(images):
        if isinstance(img, dict) and "src" in img:
            processed_img = dict(img)
            processed_img["src"] = _process_image_for_storage(
                img["src"],
                memory_id=memory_id,
                image_index=idx,
            )
            processed_images.append(processed_img)
        else:
            processed_images.append(img)

    result = dict(metadata)
    result["images"] = processed_images
    return result


def _prepare_metadata(
    metadata: Optional[Dict[str, Any]],
    memory_id: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Prepare metadata for storage, processing images if present.

    Args:
        metadata: Raw metadata dict
        memory_id: ID of the memory (required for R2 image upload)

    Returns:
        Prepared metadata dict
    """
    if metadata is None:
        return None
    if not isinstance(metadata, Mapping):
        raise ValueError("Metadata must be a mapping")
    processed = _process_metadata_images(dict(metadata), memory_id=memory_id)
    return _build_metadata_dict(processed)


def _expand_image_urls(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Expand r2:// image references to full URLs."""
    if "images" not in metadata:
        return metadata

    images = metadata.get("images")
    if not isinstance(images, list):
        return metadata

    from .image_storage import expand_r2_url

    expanded_images = []
    for img in images:
        if isinstance(img, dict) and "src" in img:
            expanded_img = dict(img)
            expanded_img["src"] = expand_r2_url(img["src"])
            expanded_images.append(expanded_img)
        else:
            expanded_images.append(img)

    result = dict(metadata)
    result["images"] = expanded_images
    return result


def _present_metadata(metadata: Optional[Any]) -> Optional[Any]:
    if metadata is None:
        return None
    if isinstance(metadata, Mapping):
        try:
            result = _build_metadata_dict(metadata)
            # Expand r2:// image URLs to full URLs
            if result and "images" in result:
                result = _expand_image_urls(result)
            return result
        except ValueError:
            # Surface legacy/invalid metadata without breaking callers
            return dict(metadata)
    return metadata


def _metadata_matches_filters(
    metadata: Optional[Any], filters: Mapping[str, Any]
) -> bool:
    if not filters:
        return True

    canonical: Dict[str, Any] = {}
    if isinstance(metadata, Mapping):
        canonical = _present_metadata(metadata) or {}
    elif metadata is None:
        canonical = {}
    else:
        canonical = {"value": metadata}

    hierarchy_entry = canonical.get("hierarchy")
    hierarchy_path: List[str] = []
    if isinstance(hierarchy_entry, Mapping):
        path_value = hierarchy_entry.get("path")
        if isinstance(path_value, Sequence) and not isinstance(
            path_value, (str, bytes)
        ):
            hierarchy_path = [str(part) for part in path_value]

    for key, expected in filters.items():
        if key == "section":
            if canonical.get("section") != expected:
                return False
        elif key == "subsection":
            if canonical.get("subsection") != expected:
                return False
        elif key in {"hierarchy", "hierarchy_path"}:
            if isinstance(expected, str):
                if expected not in hierarchy_path:
                    return False
            elif isinstance(expected, Sequence) and not isinstance(
                expected, (str, bytes)
            ):
                expected_list = [str(part) for part in expected]
                if hierarchy_path[: len(expected_list)] != expected_list:
                    return False
            else:
                return False
        else:
            if canonical.get(key) != expected:
                return False

    return True


def _validate_metadata_filters(
    metadata_filters: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if metadata_filters is None:
        return {}
    if not isinstance(metadata_filters, Mapping):
        raise ValueError("metadata_filters must be a mapping")
    validated: Dict[str, Any] = {}
    for key, value in metadata_filters.items():
        if not isinstance(key, str):
            raise ValueError("metadata_filters keys must be strings")
        validated[key] = value
    return validated


def _fts_enabled(conn: sqlite3.Connection) -> bool:
    # D1 doesn't support FTS5 virtual tables
    if isinstance(conn, D1Connection):
        return False
    return bool(
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memories_fts'"
        ).fetchone()
    )


def _fts_upsert(
    conn: sqlite3.Connection,
    memory_id: int,
    content: str,
    metadata_json: Optional[str],
    tags_json: Optional[str],
) -> None:
    if not _fts_enabled(conn):
        return
    conn.execute(
        "INSERT OR REPLACE INTO memories_fts(rowid, content, metadata, tags) VALUES (?, ?, ?, ?)",
        (
            memory_id,
            content,
            metadata_json or "",
            tags_json or "",
        ),
    )


def _fts_delete(conn: sqlite3.Connection, memory_id: int) -> None:
    if not _fts_enabled(conn):
        return
    conn.execute("DELETE FROM memories_fts WHERE rowid = ?", (memory_id,))


def _serialise_row(row: sqlite3.Row) -> Dict[str, Any]:
    metadata = row["metadata"]
    tags = row["tags"]
    row_keys = row.keys() if hasattr(row, "keys") else []
    result = {
        "id": row["id"],
        "content": row["content"],
        "metadata": _present_metadata(json.loads(metadata)) if metadata else None,
        "tags": json.loads(tags) if tags else [],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"] if "updated_at" in row_keys else None,
    }

    # Add tier fields if available (may not exist in older schemas during migration)
    if "tier" in row_keys:
        result["tier"] = row["tier"] if row["tier"] is not None else DEFAULT_TIER
    else:
        result["tier"] = DEFAULT_TIER

    if "expires_at" in row_keys:
        result["expires_at"] = row["expires_at"]

    # Add workspace field if available
    if "workspace" in row_keys:
        result["workspace"] = (
            row["workspace"] if row["workspace"] is not None else DEFAULT_WORKSPACE
        )
    else:
        result["workspace"] = DEFAULT_WORKSPACE

    # Add importance fields if available (may not exist in older schemas during migration)
    if "importance" in row_keys:
        base_importance = row["importance"] if row["importance"] is not None else 1.0
        access_count = (
            row["access_count"]
            if "access_count" in row_keys and row["access_count"] is not None
            else 0
        )
        result["importance"] = base_importance
        result["access_count"] = access_count
        result["last_accessed"] = (
            row["last_accessed"] if "last_accessed" in row_keys else None
        )
        # Calculate current importance score with decay
        result["importance_score"] = calculate_importance(
            row["created_at"],
            base_importance,
            access_count,
        )

    return result


def _validate_tags(tags: Optional[Iterable[str]]) -> List[str]:
    if tags is None:
        return []
    validated: List[str] = []
    for tag in tags:
        if not isinstance(tag, str):
            raise ValueError("Tags must be strings")
        stripped = tag.strip()
        if not stripped:
            raise ValueError("Tags cannot be empty strings")
        validated.append(stripped)
    return validated


def _enforce_tag_whitelist(tags: List[str]) -> None:
    from . import TAG_WHITELIST

    if not TAG_WHITELIST:
        return

    explicit = {tag for tag in TAG_WHITELIST if not tag.endswith(".*")}
    wildcards = [tag[:-2] for tag in TAG_WHITELIST if tag.endswith(".*")]

    for tag in tags:
        if tag in explicit:
            continue
        if any(tag == prefix or tag.startswith(prefix + ".") for prefix in wildcards):
            continue
        raise ValueError(f"Tag '{tag}' is not in the allowed tag list")


_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Cache for embedding models
_embedding_model_cache: Dict[str, Any] = {}


def _get_embedding_text(
    content: str,
    metadata: Optional[Dict[str, Any]],
    tags: List[str],
) -> str:
    """Combine content, metadata, and tags into a single text for embedding."""
    parts: List[str] = [content]

    if metadata:
        try:
            metadata_str = json.dumps(metadata, ensure_ascii=False)
        except (TypeError, ValueError):
            metadata_str = str(metadata)
        parts.append(metadata_str)

    if tags:
        parts.append(" ".join(tags))

    return " \n ".join(parts)


def _compute_embedding_tfidf(text: str) -> Dict[str, float]:
    """TF-IDF style bag-of-words embedding (default, no dependencies)."""
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return {}

    counts = Counter(tokens)
    total = sum(counts.values())
    if not total:
        return {}

    return {token: count / total for token, count in counts.items()}


def _compute_embedding_sentence_transformers(text: str) -> Dict[str, float]:
    """Use sentence-transformers for better semantic embeddings."""
    try:
        if "sentence_transformers" not in _embedding_model_cache:
            from sentence_transformers import SentenceTransformer

            # Use a small, fast model by default
            model_name = os.getenv("SENTENCE_TRANSFORMERS_MODEL", "all-MiniLM-L6-v2")
            _embedding_model_cache["sentence_transformers"] = SentenceTransformer(
                model_name
            )

        model = _embedding_model_cache["sentence_transformers"]
        embedding = model.encode(text, convert_to_numpy=True)

        # Convert numpy array to dict for storage (use indices as keys)
        return {str(i): float(val) for i, val in enumerate(embedding)}

    except ImportError:
        # Fallback to TF-IDF if sentence-transformers not available
        return _compute_embedding_tfidf(text)
    except Exception:
        # Fallback on any error
        return _compute_embedding_tfidf(text)


def _compute_embedding_openai(text: str) -> Dict[str, float]:
    """Use OpenAI embeddings API."""
    try:
        import openai

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # Fallback to TF-IDF if no API key
            return _compute_embedding_tfidf(text)

        if "openai_client" not in _embedding_model_cache:
            _embedding_model_cache["openai_client"] = openai.OpenAI(api_key=api_key)

        client = _embedding_model_cache["openai_client"]
        model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

        response = client.embeddings.create(
            input=text,
            model=model_name,
        )

        embedding = response.data[0].embedding

        # Convert to dict for storage
        return {str(i): float(val) for i, val in enumerate(embedding)}

    except ImportError:
        # Fallback to TF-IDF if openai not available
        return _compute_embedding_tfidf(text)
    except Exception:
        # Fallback on any error (API error, rate limit, etc.)
        return _compute_embedding_tfidf(text)


# ---------------------------------------------------------------------------
# Embedding cache functions
# ---------------------------------------------------------------------------

# Track cache statistics in memory
_embedding_cache_stats = {"hits": 0, "misses": 0}


def _compute_content_hash(text: str) -> str:
    """Compute SHA-256 hash of text for cache key."""
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _get_cached_embedding(
    conn: sqlite3.Connection,
    content_hash: str,
    model: str,
) -> Optional[Dict[str, float]]:
    """Get embedding from cache if exists and model matches."""
    if not EMBEDDING_CACHE_ENABLED:
        return None

    row = conn.execute(
        """
        SELECT embedding FROM embedding_cache
        WHERE content_hash = ? AND model = ?
        """,
        (content_hash, model),
    ).fetchone()

    if row:
        # Update access stats
        conn.execute(
            """
            UPDATE embedding_cache
            SET last_accessed = datetime('now'), access_count = access_count + 1
            WHERE content_hash = ?
            """,
            (content_hash,),
        )
        _embedding_cache_stats["hits"] += 1
        return _json_to_embedding(row["embedding"])

    _embedding_cache_stats["misses"] += 1
    return None


def _store_cached_embedding(
    conn: sqlite3.Connection,
    content_hash: str,
    embedding: Dict[str, float],
    model: str,
) -> None:
    """Store embedding in cache."""
    if not EMBEDDING_CACHE_ENABLED:
        return

    embedding_json = _embedding_to_json(embedding)
    conn.execute(
        """
        INSERT OR REPLACE INTO embedding_cache (content_hash, embedding, model)
        VALUES (?, ?, ?)
        """,
        (content_hash, embedding_json, model),
    )

    # Check if we need to evict old entries
    _enforce_cache_limit(conn)


def _enforce_cache_limit(conn: sqlite3.Connection) -> int:
    """Remove oldest entries if cache exceeds limit. Returns count of deleted entries."""
    if not EMBEDDING_CACHE_ENABLED or EMBEDDING_CACHE_MAX_ENTRIES <= 0:
        return 0

    count = conn.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0]
    if count <= EMBEDDING_CACHE_MAX_ENTRIES:
        return 0

    # Delete oldest 10% of entries
    delete_count = max(1, int(count * 0.1))
    conn.execute(
        """
        DELETE FROM embedding_cache
        WHERE content_hash IN (
            SELECT content_hash FROM embedding_cache
            ORDER BY last_accessed ASC
            LIMIT ?
        )
        """,
        (delete_count,),
    )
    return delete_count


def get_embedding_cache_stats(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Get embedding cache statistics."""
    total_entries = conn.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0]

    # Get oldest and newest entries
    oldest = conn.execute("SELECT MIN(last_accessed) FROM embedding_cache").fetchone()[
        0
    ]
    newest = conn.execute("SELECT MAX(last_accessed) FROM embedding_cache").fetchone()[
        0
    ]

    # Calculate hit rate
    total_requests = _embedding_cache_stats["hits"] + _embedding_cache_stats["misses"]
    hit_rate = (
        _embedding_cache_stats["hits"] / total_requests if total_requests > 0 else 0.0
    )

    return {
        "enabled": EMBEDDING_CACHE_ENABLED,
        "max_entries": EMBEDDING_CACHE_MAX_ENTRIES,
        "total_entries": total_entries,
        "cache_hits": _embedding_cache_stats["hits"],
        "cache_misses": _embedding_cache_stats["misses"],
        "hit_rate": round(hit_rate, 3),
        "oldest_entry": oldest,
        "newest_entry": newest,
    }


def clear_embedding_cache(conn: sqlite3.Connection) -> int:
    """Clear all entries from the embedding cache. Returns count of deleted entries."""
    count = conn.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0]
    conn.execute("DELETE FROM embedding_cache")
    conn.commit()
    # Reset stats
    _embedding_cache_stats["hits"] = 0
    _embedding_cache_stats["misses"] = 0
    return count


def _compute_embedding(
    content: str,
    metadata: Optional[Dict[str, Any]],
    tags: List[str],
    conn: Optional[sqlite3.Connection] = None,
) -> Dict[str, float]:
    """Compute embedding using configured backend with caching.

    Args:
        content: Memory content
        metadata: Memory metadata
        tags: Memory tags
        conn: Optional database connection for cache lookup/storage
    """
    text = _get_embedding_text(content, metadata, tags)

    # Try cache first if connection provided and cache enabled
    if conn is not None and EMBEDDING_CACHE_ENABLED:
        content_hash = _compute_content_hash(text)
        cached = _get_cached_embedding(conn, content_hash, EMBEDDING_MODEL)
        if cached is not None:
            return cached

    # Compute embedding
    if EMBEDDING_MODEL == "sentence-transformers":
        embedding = _compute_embedding_sentence_transformers(text)
    elif EMBEDDING_MODEL == "openai":
        embedding = _compute_embedding_openai(text)
    else:  # Default to tfidf
        embedding = _compute_embedding_tfidf(text)

    # Store in cache if connection provided
    if conn is not None and EMBEDDING_CACHE_ENABLED and embedding:
        content_hash = _compute_content_hash(text)
        _store_cached_embedding(conn, content_hash, embedding, EMBEDDING_MODEL)

    return embedding


# ---------------------------------------------------------------------------
# LLM-based memory comparison for deduplication
# ---------------------------------------------------------------------------


def _get_llm_client():
    """Get or create cached LLM client for comparison."""
    if not LLM_ENABLED:
        return None

    try:
        import openai

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None

        if "llm_client" not in _embedding_model_cache:
            base_url = os.getenv("OPENAI_BASE_URL")
            client_kwargs = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            _embedding_model_cache["llm_client"] = openai.OpenAI(**client_kwargs)

        return _embedding_model_cache["llm_client"]

    except ImportError:
        return None


def compare_memories_llm(
    content_a: str,
    content_b: str,
    metadata_a: Optional[Dict[str, Any]] = None,
    metadata_b: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Use LLM to semantically compare two memories for deduplication.

    Returns dict with:
        - verdict: "duplicate" | "similar" | "different"
        - confidence: 0.0-1.0
        - reasoning: Brief explanation
        - suggested_action: "merge" | "keep_both" | "review"
        - merge_suggestion: How to combine if merging

    Returns None if LLM is not available.
    """
    client = _get_llm_client()
    if not client:
        return None

    try:
        # Build comparison prompt
        prompt = f"""Compare these two memory entries and determine if they are duplicates.

Memory A:
{content_a}
{f"Metadata: {json.dumps(metadata_a)}" if metadata_a else ""}

Memory B:
{content_b}
{f"Metadata: {json.dumps(metadata_b)}" if metadata_b else ""}

Analyze whether these memories contain the same information (duplicates), related but distinct information (similar), or unrelated information (different).

Respond with JSON only (no markdown):
{{
  "verdict": "duplicate" | "similar" | "different",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation (1-2 sentences)",
  "suggested_action": "merge" | "keep_both" | "review",
  "merge_suggestion": "If verdict is duplicate, how to combine the content"
}}"""

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that compares text entries for semantic similarity. Always respond with valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=300,
        )

        result_text = response.choices[0].message.content.strip()
        # Parse JSON response
        result = json.loads(result_text)

        # Validate required fields
        if "verdict" not in result:
            result["verdict"] = "review"
        if "confidence" not in result:
            result["confidence"] = 0.5
        if "reasoning" not in result:
            result["reasoning"] = "No reasoning provided"
        if "suggested_action" not in result:
            result["suggested_action"] = "review"

        return result

    except json.JSONDecodeError:
        # LLM didn't return valid JSON
        return {
            "verdict": "review",
            "confidence": 0.0,
            "reasoning": "LLM response was not valid JSON",
            "suggested_action": "review",
        }
    except Exception as e:
        # API error, rate limit, etc.
        return {
            "verdict": "review",
            "confidence": 0.0,
            "reasoning": f"LLM error: {str(e)[:100]}",
            "suggested_action": "review",
        }


def find_duplicate_candidates(
    conn: "sqlite3.Connection",
    min_similarity: float = 0.7,
    max_similarity: float = 0.95,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Find memory pairs in similarity range that might be duplicates.

    Returns list of pairs with their similarity scores.
    """
    # Get all cross-refs from database
    cursor = conn.execute(
        "SELECT memory_id, related FROM memories_crossrefs WHERE related IS NOT NULL"
    )

    pairs_seen = set()
    candidates = []

    for row in cursor:
        memory_id = row[0]
        try:
            related = json.loads(row[1]) if row[1] else []
        except json.JSONDecodeError:
            continue

        for rel in related:
            if not rel:
                continue
            related_id = rel.get("id")
            score = rel.get("score", 0)

            if related_id is None:
                continue

            # Check if in similarity range
            if min_similarity <= score <= max_similarity:
                # Avoid duplicate pairs (A,B) and (B,A)
                pair_key = tuple(sorted([memory_id, related_id]))
                if pair_key not in pairs_seen:
                    pairs_seen.add(pair_key)
                    candidates.append(
                        {
                            "memory_a_id": pair_key[0],
                            "memory_b_id": pair_key[1],
                            "similarity_score": score,
                        }
                    )

    # Sort by similarity (highest first)
    candidates.sort(key=lambda x: x["similarity_score"], reverse=True)

    return candidates[:limit]


def _embedding_to_json(vector: Dict[str, float]) -> Optional[str]:
    if not vector:
        return None
    items = sorted(vector.items())
    return json.dumps(items, ensure_ascii=False)


def _json_to_embedding(data: Optional[str]) -> Dict[str, float]:
    if not data:
        return {}
    try:
        items = json.loads(data)
    except json.JSONDecodeError:
        return {}
    if isinstance(items, list):
        return {str(token): float(weight) for token, weight in items}
    return {}


def _embedding_norm(vector: Dict[str, float]) -> float:
    return math.sqrt(sum(weight * weight for weight in vector.values()))


def _cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    dot = 0.0
    for token, weight in vec_a.items():
        dot += weight * vec_b.get(token, 0.0)
    norm_a = _embedding_norm(vec_a)
    norm_b = _embedding_norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _upsert_embedding(
    conn: sqlite3.Connection,
    memory_id: int,
    vector: Dict[str, float],
) -> None:
    embedding_json = _embedding_to_json(vector)
    conn.execute(
        """
        INSERT INTO memories_embeddings(memory_id, embedding)
        VALUES(?, ?)
        ON CONFLICT(memory_id) DO UPDATE SET embedding=excluded.embedding
        """,
        (memory_id, embedding_json),
    )


def _delete_embedding(conn: sqlite3.Connection, memory_id: int) -> None:
    conn.execute("DELETE FROM memories_embeddings WHERE memory_id = ?", (memory_id,))


def _get_embeddings_for_ids(
    conn: sqlite3.Connection,
    memory_ids: List[int],
) -> Dict[int, Dict[str, float]]:
    if not memory_ids:
        return {}
    placeholders = ",".join("?" for _ in memory_ids)
    rows = conn.execute(
        f"SELECT memory_id, embedding FROM memories_embeddings WHERE memory_id IN ({placeholders})",
        memory_ids,
    ).fetchall()
    mapping: Dict[int, Dict[str, float]] = {}
    for row in rows:
        mapping[row["memory_id"]] = _json_to_embedding(row["embedding"])
    return mapping


def _search_by_vector(
    conn: sqlite3.Connection,
    vector_query: Dict[str, float],
    *,
    metadata_filters: Optional[Dict[str, Any]] = None,
    top_k: Optional[int] = 5,
    min_score: Optional[float] = None,
    exclude_ids: Optional[Iterable[int]] = None,
) -> List[Dict[str, Any]]:
    exclude_set = set(exclude_ids or [])

    candidates = list_memories(conn, query=None, metadata_filters=metadata_filters)
    filtered = [record for record in candidates if record["id"] not in exclude_set]

    ids = [record["id"] for record in filtered]
    embeddings = _get_embeddings_for_ids(conn, ids)

    results: List[Dict[str, Any]] = []
    for record in filtered:
        memory_id = record["id"]
        vector = embeddings.get(memory_id)
        if vector is None:
            vector = _compute_embedding(
                record["content"],
                record.get("metadata"),
                record.get("tags", []),
                conn,
            )
            _upsert_embedding(conn, memory_id, vector)
        score = _cosine_similarity(vector_query, vector)
        if min_score is not None and score < min_score:
            continue
        results.append({"score": score, "memory": record})

    results.sort(key=lambda entry: entry["score"], reverse=True)
    if top_k is not None:
        results = results[:top_k]
    return results


def _store_crossrefs(
    conn: sqlite3.Connection,
    memory_id: int,
    related: List[Dict[str, Any]],
) -> None:
    related_json = json.dumps(related, ensure_ascii=False) if related else None
    conn.execute(
        """
        INSERT INTO memories_crossrefs(memory_id, related)
        VALUES(?, ?)
        ON CONFLICT(memory_id) DO UPDATE SET related=excluded.related
        """,
        (memory_id, related_json),
    )


def _clear_crossrefs(conn: sqlite3.Connection, memory_id: int) -> None:
    conn.execute("DELETE FROM memories_crossrefs WHERE memory_id = ?", (memory_id,))


def get_crossrefs(conn: sqlite3.Connection, memory_id: int) -> List[Dict[str, Any]]:
    row = conn.execute(
        "SELECT related FROM memories_crossrefs WHERE memory_id = ?",
        (memory_id,),
    ).fetchone()
    if not row or not row["related"]:
        return []
    try:
        data = json.loads(row["related"])
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return data
    return []


def _update_crossrefs_for_memory(
    conn: sqlite3.Connection,
    memory_id: int,
    vector: Optional[Dict[str, float]] = None,
    top_k: int = 5,
    min_score: Optional[float] = None,
) -> List[Dict[str, Any]]:
    if vector is None:
        embeddings = _get_embeddings_for_ids(conn, [memory_id])
        vector = embeddings.get(memory_id)
        if vector is None:
            record = get_memory(conn, memory_id)
            if record is None:
                return []
            vector = _compute_embedding(
                record["content"],
                record.get("metadata"),
                record.get("tags", []),
                conn,
            )
            _upsert_embedding(conn, memory_id, vector)

    results = _search_by_vector(
        conn,
        vector,
        metadata_filters=None,
        top_k=top_k,
        min_score=min_score,
        exclude_ids=[memory_id],
    )

    related = [
        {"id": item["memory"]["id"], "score": item["score"], "edge_type": "related_to"}
        for item in results
    ]
    _store_crossrefs(conn, memory_id, related)
    return related


# Valid edge types for explicit links
EDGE_TYPES = {
    "related_to",
    "supersedes",
    "contradicts",
    "implements",
    "extends",
    "references",
}


def add_link(
    conn: sqlite3.Connection,
    from_id: int,
    to_id: int,
    edge_type: str = "references",
    bidirectional: bool = True,
) -> Dict[str, Any]:
    """Add an explicit link between two memories.

    Args:
        from_id: Source memory ID
        to_id: Target memory ID
        edge_type: Type of relationship (references, implements, supersedes, contradicts, extends)
        bidirectional: If True, also create reverse link

    Returns:
        Dict with status and created links
    """
    if edge_type not in EDGE_TYPES:
        raise ValueError(
            f"Invalid edge_type '{edge_type}'. Must be one of: {', '.join(sorted(EDGE_TYPES))}"
        )

    # Verify both memories exist
    from_mem = get_memory(conn, from_id)
    to_mem = get_memory(conn, to_id)
    if not from_mem:
        raise ValueError(f"Memory {from_id} not found")
    if not to_mem:
        raise ValueError(f"Memory {to_id} not found")

    links_created = []

    # Add link from -> to
    existing = get_crossrefs(conn, from_id)
    # Remove any existing link to the same target
    existing = [r for r in existing if r.get("id") != to_id]
    # Add new link
    existing.append({"id": to_id, "score": 1.0, "edge_type": edge_type})
    _store_crossrefs(conn, from_id, existing)
    links_created.append({"from": from_id, "to": to_id, "edge_type": edge_type})

    # Add reverse link if bidirectional
    if bidirectional:
        reverse_type = _get_reverse_edge_type(edge_type)
        existing_reverse = get_crossrefs(conn, to_id)
        existing_reverse = [r for r in existing_reverse if r.get("id") != from_id]
        existing_reverse.append(
            {"id": from_id, "score": 1.0, "edge_type": reverse_type}
        )
        _store_crossrefs(conn, to_id, existing_reverse)
        links_created.append({"from": to_id, "to": from_id, "edge_type": reverse_type})

    return {"status": "linked", "links": links_created}


def _get_reverse_edge_type(edge_type: str) -> str:
    """Get the reverse edge type for bidirectional links."""
    reverse_map = {
        "references": "referenced_by",
        "implements": "implemented_by",
        "supersedes": "superseded_by",
        "extends": "extended_by",
        "contradicts": "contradicts",  # symmetric
        "related_to": "related_to",  # symmetric
    }
    return reverse_map.get(edge_type, "related_to")


def remove_link(
    conn: sqlite3.Connection,
    from_id: int,
    to_id: int,
    bidirectional: bool = True,
) -> Dict[str, Any]:
    """Remove a link between two memories."""
    removed = []

    existing = get_crossrefs(conn, from_id)
    new_refs = [r for r in existing if r.get("id") != to_id]
    if len(new_refs) < len(existing):
        _store_crossrefs(conn, from_id, new_refs)
        removed.append({"from": from_id, "to": to_id})

    if bidirectional:
        existing_reverse = get_crossrefs(conn, to_id)
        new_refs_reverse = [r for r in existing_reverse if r.get("id") != from_id]
        if len(new_refs_reverse) < len(existing_reverse):
            _store_crossrefs(conn, to_id, new_refs_reverse)
            removed.append({"from": to_id, "to": from_id})

    return {"status": "unlinked", "removed": removed}


def detect_clusters(
    conn: sqlite3.Connection,
    min_cluster_size: int = 2,
    min_score: float = 0.3,
) -> List[Dict[str, Any]]:
    """Detect clusters of related memories using connected components.

    Args:
        min_cluster_size: Minimum memories to form a cluster
        min_score: Minimum similarity score to consider as connected

    Returns:
        List of clusters, each with member IDs and common tags
    """
    # Build adjacency graph from cross-references
    all_memories = list_memories(conn)
    memory_ids = {m["id"] for m in all_memories}
    memory_tags = {m["id"]: set(m.get("tags", [])) for m in all_memories}

    # Build graph edges
    edges: Dict[int, set] = {mid: set() for mid in memory_ids}
    for memory in all_memories:
        mid = memory["id"]
        refs = get_crossrefs(conn, mid)
        for ref in refs:
            ref_id = ref.get("id")
            score = ref.get("score", 0)
            if ref_id in memory_ids and score >= min_score:
                edges[mid].add(ref_id)
                edges[ref_id].add(mid)  # Make bidirectional for clustering

    # Find connected components using BFS
    visited: set = set()
    clusters: List[List[int]] = []

    for start_id in memory_ids:
        if start_id in visited:
            continue

        # BFS to find all connected nodes
        cluster: List[int] = []
        queue = [start_id]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            cluster.append(node)
            for neighbor in edges[node]:
                if neighbor not in visited:
                    queue.append(neighbor)

        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)

    # Format clusters with metadata
    result = []
    for i, cluster_ids in enumerate(clusters):
        # Find common tags
        all_tags = [memory_tags.get(mid, set()) for mid in cluster_ids]
        common_tags = set.intersection(*all_tags) if all_tags else set()

        # Find most common tags (even if not in all)
        tag_counts: Dict[str, int] = {}
        for tags in all_tags:
            for tag in tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        top_tags = sorted(tag_counts.keys(), key=lambda t: tag_counts[t], reverse=True)[
            :5
        ]

        result.append(
            {
                "cluster_id": i + 1,
                "size": len(cluster_ids),
                "memory_ids": sorted(cluster_ids),
                "common_tags": list(common_tags),
                "top_tags": top_tags,
            }
        )

    # Sort by size descending
    result.sort(key=lambda c: c["size"], reverse=True)
    return result


def _update_crossrefs(conn: sqlite3.Connection, memory_id: int) -> None:
    # Skip cross-reference computation for section memories
    record = get_memory(conn, memory_id)
    metadata = record.get("metadata") if record else None
    if metadata and metadata.get("type") == "section":
        return
    related = _update_crossrefs_for_memory(conn, memory_id)
    for item in related:
        _update_crossrefs_for_memory(conn, item["id"])


def rebuild_crossrefs(conn: sqlite3.Connection) -> int:
    rows = conn.execute("SELECT id, metadata FROM memories").fetchall()
    total = 0
    for row in rows:
        memory_id = row["id"]
        # Skip section memories - they don't need cross-references
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        if metadata.get("type") == "section":
            continue
        _update_crossrefs_for_memory(conn, memory_id)
        total += 1
    conn.commit()
    return total


def update_crossrefs(conn: sqlite3.Connection, memory_id: int) -> None:
    _update_crossrefs(conn, memory_id)


def _remove_memory_from_crossrefs(conn: sqlite3.Connection, memory_id: int) -> None:
    rows = conn.execute("SELECT memory_id, related FROM memories_crossrefs").fetchall()
    for row in rows:
        related = []
        if row["related"]:
            try:
                related = json.loads(row["related"])
            except json.JSONDecodeError:
                related = []
        filtered = [entry for entry in related if entry.get("id") != memory_id]
        if len(filtered) != len(related):
            _store_crossrefs(conn, row["memory_id"], filtered)


def add_memory(
    conn: sqlite3.Connection,
    *,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    tier: Optional[str] = None,
    expires_at: Optional[str] = None,
    workspace: Optional[str] = None,
) -> Dict[str, Any]:
    # Validate and normalize content (trim, length check)
    content = _validate_content(content)

    # Validate tier and normalize expires_at
    validated_tier = _validate_tier(tier)
    expires_at = _normalize_expires_at(expires_at)
    workspace = workspace or DEFAULT_WORKSPACE

    # Auto-detect memory type (issue/todo) from content if not explicitly set
    metadata, tags = _apply_auto_detection(content, metadata, tags)

    validated_tags = _validate_tags(tags)
    _enforce_tag_whitelist(validated_tags)
    tags_json = json.dumps(validated_tags, ensure_ascii=False)

    # Two-pass approach for images:
    # 1. Insert memory first to get ID (needed for R2 image keys)
    # 2. Process metadata with memory_id, then update the record

    # Check if metadata has images that need processing
    has_images = (
        metadata is not None
        and isinstance(metadata.get("images"), list)
        and len(metadata.get("images", [])) > 0
    )

    # Get sync version for this change
    sync_version = _get_next_sync_version(conn)

    if has_images:
        # First pass: insert without processed images to get memory_id
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        cur = conn.execute(
            "INSERT INTO memories (content, metadata, tags, created_at, tier, expires_at, sync_version, workspace) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                content,
                None,
                tags_json,
                now,
                validated_tier,
                expires_at,
                sync_version,
                workspace,
            ),
        )
        memory_id = cur.lastrowid

        # Second pass: process metadata with memory_id (uploads images to R2)
        prepared_metadata = _prepare_metadata(metadata, memory_id=memory_id)
        metadata_json = (
            json.dumps(prepared_metadata, ensure_ascii=False)
            if prepared_metadata
            else None
        )

        # Update the record with processed metadata
        conn.execute(
            "UPDATE memories SET metadata = ? WHERE id = ?",
            (metadata_json, memory_id),
        )
    else:
        # No images - single pass
        prepared_metadata = _prepare_metadata(metadata)
        metadata_json = (
            json.dumps(prepared_metadata, ensure_ascii=False)
            if prepared_metadata
            else None
        )
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        cur = conn.execute(
            "INSERT INTO memories (content, metadata, tags, created_at, tier, expires_at, sync_version, workspace) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                content,
                metadata_json,
                tags_json,
                now,
                validated_tier,
                expires_at,
                sync_version,
                workspace,
            ),
        )
        memory_id = cur.lastrowid

    _fts_upsert(conn, memory_id, content, metadata_json, tags_json)
    vector = _compute_embedding(content, prepared_metadata, validated_tags, conn)
    _upsert_embedding(conn, memory_id, vector)
    _update_crossrefs(conn, memory_id)
    conn.commit()
    _emit_event(conn, memory_id, validated_tags)
    return get_memory(conn, memory_id)


def add_memories(
    conn: sqlite3.Connection,
    entries: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    embeddings: List[Dict[str, float]] = []
    prepared: List[
        tuple[str, Optional[str], Optional[str], str, Optional[str], int]
    ] = []

    for entry in entries:
        if "content" not in entry:
            raise ValueError("Each batch entry must include 'content'")
        content = str(entry["content"]).strip()
        metadata = entry.get("metadata")
        tags = entry.get("tags") or []
        tier = _validate_tier(entry.get("tier"))
        expires_at = _normalize_expires_at(entry.get("expires_at"))
        # Auto-detect memory type (issue/todo) from content if not explicitly set
        metadata, tags = _apply_auto_detection(content, metadata, tags)
        prepared_metadata = _prepare_metadata(metadata)
        validated_tags = _validate_tags(tags)
        _enforce_tag_whitelist(validated_tags)
        metadata_json = (
            json.dumps(prepared_metadata, ensure_ascii=False)
            if prepared_metadata
            else None
        )
        tags_json = json.dumps(validated_tags, ensure_ascii=False)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        # Get sync version for each entry
        sync_version = _get_next_sync_version(conn)
        prepared.append(
            (content, metadata_json, tags_json, now, tier, expires_at, sync_version)
        )
        rows.append(
            {
                "content": content,
                "metadata_json": metadata_json,
                "tags_json": tags_json,
                "validated_tags": validated_tags,
            }
        )
        embeddings.append(
            _compute_embedding(content, prepared_metadata, validated_tags, conn)
        )

    if not prepared:
        return []

    cur = conn.executemany(
        "INSERT INTO memories (content, metadata, tags, created_at, tier, expires_at, sync_version) VALUES (?, ?, ?, ?, ?, ?, ?)",
        prepared,
    )

    # SQLite returns the cursor of the last execute; capture inserted IDs manually
    start_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    inserted: List[int] = list(range(start_id - len(prepared) + 1, start_id + 1))

    for memory_id, entry, vector in zip(inserted, rows, embeddings):
        _fts_upsert(
            conn,
            memory_id,
            entry["content"],
            entry["metadata_json"],
            entry["tags_json"],
        )
        _upsert_embedding(conn, memory_id, vector)
        _update_crossrefs(conn, memory_id)

    conn.commit()

    # Emit events for memories with trigger tag
    for memory_id, entry in zip(inserted, rows):
        _emit_event(conn, memory_id, entry["validated_tags"])

    return [get_memory(conn, memory_id) for memory_id in inserted]


def get_memory(
    conn: sqlite3.Connection,
    memory_id: int,
    track_access: bool = False,
) -> Optional[Dict[str, Any]]:
    """Retrieve a single memory by ID.

    Args:
        conn: Database connection
        memory_id: ID of memory to retrieve
        track_access: If True, increment access count and update last_accessed

    Returns:
        Memory dict or None if not found
    """
    row = conn.execute(
        """SELECT id, content, metadata, tags, created_at, updated_at,
                  importance, last_accessed, access_count, tier, expires_at, workspace
           FROM memories WHERE id = ?""",
        (memory_id,),
    ).fetchone()
    if not row:
        return None

    if track_access:
        _track_access(conn, memory_id)
        conn.commit()

    record = _serialise_row(row)
    record["related"] = get_crossrefs(conn, memory_id)
    return record


def update_memory(
    conn: sqlite3.Connection,
    memory_id: int,
    *,
    content: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    tier: Optional[str] = None,
    expires_at: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Update an existing memory. Only provided fields are updated."""
    # First check if memory exists
    existing = get_memory(conn, memory_id)
    if not existing:
        return None

    # Determine what to update
    new_content = content.strip() if content is not None else existing["content"]
    new_metadata = (
        _prepare_metadata(metadata)
        if metadata is not None
        else existing.get("metadata")
    )
    new_tags = _validate_tags(tags) if tags is not None else existing.get("tags", [])
    new_tier = (
        _validate_tier(tier) if tier is not None else existing.get("tier", DEFAULT_TIER)
    )
    new_expires_at = (
        _normalize_expires_at(expires_at)
        if expires_at is not None
        else existing.get("expires_at")
    )

    if tags is not None:
        _enforce_tag_whitelist(new_tags)

    # Check if content actually changed (affects whether we need to recompute embeddings)
    content_changed = content is not None and new_content != existing["content"]

    # Serialize for storage
    metadata_json = (
        json.dumps(new_metadata, ensure_ascii=False) if new_metadata else None
    )
    tags_json = json.dumps(new_tags, ensure_ascii=False)

    # Get new sync version for this update
    sync_version = _get_next_sync_version(conn)

    # Update the memory
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    cur = conn.execute(
        "UPDATE memories SET content = ?, metadata = ?, tags = ?, tier = ?, expires_at = ?, sync_version = ?, updated_at = ? WHERE id = ?",
        (
            new_content,
            metadata_json,
            tags_json,
            new_tier,
            new_expires_at,
            sync_version,
            now,
            memory_id,
        ),
    )

    # Verify the update affected a row (helps catch D1 issues)
    if hasattr(cur, "rowcount") and cur.rowcount == 0:
        # Row wasn't updated - this shouldn't happen since we checked existence
        raise RuntimeError(f"UPDATE affected 0 rows for memory {memory_id}")

    # Only recompute expensive operations if content changed
    # Metadata-only updates (status, tags, etc.) don't need embedding/crossref recalc
    if content_changed:
        # Update FTS index
        _fts_upsert(conn, memory_id, new_content, metadata_json, tags_json)

        # Update embeddings (calls OpenAI API - ~1-2 sec)
        vector = _compute_embedding(new_content, new_metadata, new_tags, conn)
        _upsert_embedding(conn, memory_id, vector)

        # Skip cross-references update - too expensive for D1 HTTP API (~15 sec)
        # Cross-refs remain valid enough until manual rebuild via memory_rebuild_crossrefs

    conn.commit()
    _emit_event(conn, memory_id, new_tags)

    # Return the data we just wrote instead of reading back from DB
    # This avoids D1 read replica lag issues where reads immediately
    # after writes might return stale data from a read replica
    result = {
        "id": memory_id,
        "content": new_content,
        "metadata": _present_metadata(new_metadata) if new_metadata else None,
        "tags": new_tags,
        "created_at": existing.get("created_at"),
        "updated_at": now,
    }

    # Preserve importance fields from existing record
    if "importance" in existing:
        result["importance"] = existing["importance"]
        result["access_count"] = existing.get("access_count", 0)
        result["last_accessed"] = existing.get("last_accessed")
        result["importance_score"] = existing.get("importance_score")

    # Get crossrefs - these were just updated so might also be stale,
    # but the semantic content matters more for consistency
    result["related"] = get_crossrefs(conn, memory_id)

    return result


def delete_memory(conn: sqlite3.Connection, memory_id: int) -> bool:
    # Get content preview before deletion for sync tracking
    row = conn.execute(
        "SELECT content FROM memories WHERE id = ?", (memory_id,)
    ).fetchone()
    if not row:
        return False

    content_preview = row[0][:100] if row[0] else None

    # Clean up R2 images before deleting memory
    import logging

    from .image_storage import get_image_storage_instance

    image_storage = get_image_storage_instance()
    if image_storage:
        try:
            deleted_images = image_storage.delete_memory_images(memory_id)
            if deleted_images > 0:
                logging.getLogger(__name__).info(
                    f"Deleted {deleted_images} R2 images for memory {memory_id}"
                )
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"Failed to delete R2 images for memory {memory_id}: {e}"
            )

    # Track deletion for sync
    sync_version = _get_next_sync_version(conn)
    conn.execute(
        "INSERT OR REPLACE INTO deleted_memories (memory_id, content_preview, sync_version) VALUES (?, ?, ?)",
        (memory_id, content_preview, sync_version),
    )

    _fts_delete(conn, memory_id)
    _delete_embedding(conn, memory_id)
    _clear_crossrefs(conn, memory_id)
    _remove_memory_from_crossrefs(conn, memory_id)
    cur = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    conn.commit()
    return cur.rowcount > 0


def delete_memories(conn: sqlite3.Connection, memory_ids: Iterable[int]) -> int:
    ids = list(memory_ids)
    if not ids:
        return 0

    # Get content previews before deletion for sync tracking
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT id, content FROM memories WHERE id IN ({placeholders})",
        ids,
    ).fetchall()

    # Track deletions for sync
    for row in rows:
        memory_id = row[0]
        content_preview = row[1][:100] if row[1] else None
        sync_version = _get_next_sync_version(conn)
        conn.execute(
            "INSERT OR REPLACE INTO deleted_memories (memory_id, content_preview, sync_version) VALUES (?, ?, ?)",
            (memory_id, content_preview, sync_version),
        )

    # Clean up R2 images for all memories
    import logging

    from .image_storage import get_image_storage_instance

    image_storage = get_image_storage_instance()
    if image_storage:
        for memory_id in ids:
            try:
                image_storage.delete_memory_images(memory_id)
            except Exception as e:
                logging.getLogger(__name__).warning(
                    f"Failed to delete R2 images for memory {memory_id}: {e}"
                )

    for memory_id in ids:
        _fts_delete(conn, memory_id)
        _delete_embedding(conn, memory_id)
        _clear_crossrefs(conn, memory_id)
        _remove_memory_from_crossrefs(conn, memory_id)
    conn.execute(
        f"DELETE FROM memories WHERE id IN ({placeholders})",
        ids,
    )
    conn.commit()
    return len(ids)


def _parse_date_filter(date_str: str) -> str:
    """Parse date string to ISO format. Supports ISO dates and relative formats like '7d', '1m', '1y'."""
    if not date_str:
        return date_str

    # Try ISO format first
    try:
        parsed = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return parsed.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        pass

    # Try relative formats: 7d, 1m, 1y, etc.
    match = re.match(r"^(\d+)([dmyDMY])$", date_str.strip())
    if match:
        value = int(match.group(1))
        unit = match.group(2).lower()

        now = datetime.now(timezone.utc)
        if unit == "d":
            target = now - timedelta(days=value)
        elif unit == "m":
            target = now - timedelta(days=value * 30)  # Approximate
        elif unit == "y":
            target = now - timedelta(days=value * 365)  # Approximate
        else:
            raise ValueError(f"Unknown time unit: {unit}")

        return target.strftime("%Y-%m-%d %H:%M:%S")

    raise ValueError(f"Invalid date format: {date_str}")


def list_memories(
    conn: sqlite3.Connection,
    query: Optional[str] = None,
    metadata_filters: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = 0,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    tags_any: Optional[List[str]] = None,
    tags_all: Optional[List[str]] = None,
    tags_none: Optional[List[str]] = None,
    sort_by_importance: bool = False,
    workspace: Optional[str] = None,
    workspaces: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """List memories with optional filtering.

    Args:
        workspace: Filter to a single workspace (None = all workspaces)
        workspaces: Filter to multiple workspaces (None = all workspaces)
                   If both workspace and workspaces are provided, workspace is ignored.
    """
    validated_filters = _validate_metadata_filters(metadata_filters)

    rows: List[sqlite3.Row]

    # Parse date filters
    parsed_date_from = _parse_date_filter(date_from) if date_from else None
    parsed_date_to = _parse_date_filter(date_to) if date_to else None

    # Build date filter clauses (one with alias 'm.' for FTS, one without for regular queries)
    date_clause_fts = ""  # For FTS queries using alias 'm'
    date_clause_plain = ""  # For non-FTS queries
    date_params = []

    if parsed_date_from:
        date_clause_fts += " AND m.created_at >= ?"
        date_clause_plain += " AND created_at >= ?"
        date_params.append(parsed_date_from)
    if parsed_date_to:
        date_clause_fts += " AND m.created_at <= ?"
        date_clause_plain += " AND created_at <= ?"
        date_params.append(parsed_date_to)

    # Build LIMIT/OFFSET clause
    limit_clause = ""
    limit_params = []
    if limit is not None:
        limit_clause = " LIMIT ?"
        limit_params.append(limit)
        if offset:
            limit_clause += " OFFSET ?"
            limit_params.append(offset)

    # Column list including importance, tier, and workspace fields
    cols_fts = "m.id, m.content, m.metadata, m.tags, m.created_at, m.updated_at, m.importance, m.last_accessed, m.access_count, m.tier, m.expires_at, m.workspace"
    cols_plain = "id, content, metadata, tags, created_at, updated_at, importance, last_accessed, access_count, tier, expires_at, workspace"

    # Build workspace filter
    workspace_clause_fts = ""
    workspace_clause_plain = ""
    workspace_params: List[str] = []

    if workspaces:
        placeholders = ",".join("?" * len(workspaces))
        workspace_clause_fts = f" AND COALESCE(m.workspace, ?) IN ({placeholders})"
        workspace_clause_plain = f" AND COALESCE(workspace, ?) IN ({placeholders})"
        workspace_params = [DEFAULT_WORKSPACE, *workspaces]
    elif workspace:
        workspace_clause_fts = " AND COALESCE(m.workspace, ?) = ?"
        workspace_clause_plain = " AND COALESCE(workspace, ?) = ?"
        workspace_params = [DEFAULT_WORKSPACE, workspace]

    # Order clause - use importance_score calculation or created_at
    order_fts = "m.created_at DESC"
    order_plain = "created_at DESC"

    if query and _fts_enabled(conn):
        # Use full-text search when available. Fall back to LIKE if the query fails.
        try:
            rows = conn.execute(
                f"""
                SELECT {cols_fts}
                FROM memories m
                JOIN memories_fts f ON m.id = f.rowid
                WHERE f MATCH ?{date_clause_fts}{workspace_clause_fts}
                ORDER BY {order_fts}{limit_clause}
                """,
                (query, *date_params, *workspace_params, *limit_params),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
    elif query:
        pattern = f"%{query}%"
        rows = conn.execute(
            f"""
            SELECT {cols_plain}
            FROM memories
            WHERE (content LIKE ? OR tags LIKE ? OR metadata LIKE ?){date_clause_plain}{workspace_clause_plain}
            ORDER BY {order_plain}{limit_clause}
            """,
            (pattern, pattern, pattern, *date_params, *workspace_params, *limit_params),
        ).fetchall()
    else:
        where_clause = " WHERE 1=1" + date_clause_plain + workspace_clause_plain
        rows = conn.execute(
            f"SELECT {cols_plain} FROM memories{where_clause} ORDER BY {order_plain}{limit_clause}",
            tuple([*date_params, *workspace_params, *limit_params]),
        ).fetchall()

    # If the FTS search yielded nothing because of an SQLite error (e.g. malformed query)
    # fall back to a LIKE search for resilience.
    if query and _fts_enabled(conn) and not rows:
        pattern = f"%{query}%"
        rows = conn.execute(
            f"""
            SELECT {cols_plain}
            FROM memories
            WHERE (content LIKE ? OR tags LIKE ? OR metadata LIKE ?){date_clause_plain}{workspace_clause_plain}
            ORDER BY {order_plain}{limit_clause}
            """,
            (pattern, pattern, pattern, *date_params, *workspace_params, *limit_params),
        ).fetchall()

    records: List[Dict[str, Any]] = []
    for row in rows:
        record = _serialise_row(row)
        if validated_filters and not _metadata_matches_filters(
            record.get("metadata"), validated_filters
        ):
            continue

        # Apply tag filters
        record_tags = set(record.get("tags", []))

        # tags_any: match if ANY of the specified tags are present (OR logic)
        if tags_any:
            if not any(tag in record_tags for tag in tags_any):
                continue

        # tags_all: match only if ALL of the specified tags are present (AND logic)
        if tags_all:
            if not all(tag in record_tags for tag in tags_all):
                continue

        # tags_none: exclude if ANY of the specified tags are present (NOT logic)
        if tags_none:
            if any(tag in record_tags for tag in tags_none):
                continue

        records.append(record)

    # Sort by importance score if requested
    if sort_by_importance:
        records.sort(key=lambda r: r.get("importance_score", 0.0), reverse=True)

    return records


def collect_all_tags(conn: sqlite3.Connection) -> List[str]:
    tags: set[str] = set()
    rows = conn.execute("SELECT tags FROM memories")
    for (tags_json,) in rows:
        if not tags_json:
            continue
        try:
            parsed = json.loads(tags_json)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            for tag in parsed:
                if isinstance(tag, str) and tag.strip():
                    tags.add(tag.strip())
    return sorted(tags)


def find_invalid_tag_entries(
    conn: sqlite3.Connection,
    allowlist: Iterable[str],
) -> List[Dict[str, Any]]:
    allowed = set(allowlist)
    if not allowed:
        return []

    explicit = {tag for tag in allowed if not tag.endswith(".*")}
    wildcards = [tag[:-2] for tag in allowed if tag.endswith(".*")]

    invalid: List[Dict[str, Any]] = []
    rows = conn.execute("SELECT id, tags FROM memories")
    for memory_id, tags_json in rows:
        if not tags_json:
            continue
        try:
            parsed = json.loads(tags_json)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, list):
            continue
        bad: List[str] = []
        for tag in parsed:
            if not isinstance(tag, str):
                continue
            if tag in explicit:
                continue
            if any(
                tag == prefix or tag.startswith(prefix + ".") for prefix in wildcards
            ):
                continue
            bad.append(tag)
        if bad:
            invalid.append({"id": memory_id, "invalid_tags": bad})
    return invalid


def semantic_search(
    conn: sqlite3.Connection,
    query: str,
    *,
    metadata_filters: Optional[Dict[str, Any]] = None,
    top_k: Optional[int] = 5,
    min_score: Optional[float] = None,
    auto_rebuild: bool = True,
) -> List[Dict[str, Any]]:
    """Perform semantic search using vector embeddings.

    Args:
        conn: Database connection
        query: Search query text
        metadata_filters: Optional metadata filters
        top_k: Maximum number of results
        min_score: Minimum similarity score threshold
        auto_rebuild: If True, automatically rebuild embeddings on model mismatch

    Returns:
        List of results with score and memory
    """
    # Check for embedding model mismatch and rebuild if needed
    if auto_rebuild and _check_embedding_model_mismatch(conn):
        import sys

        print(
            f"[memora] Embedding model changed: rebuilding embeddings with '{EMBEDDING_MODEL}'...",
            file=sys.stderr,
        )
        rebuild_embeddings(conn)

    vector_query = _compute_embedding(query, None, [], conn)
    if not vector_query:
        return []
    return _search_by_vector(
        conn,
        vector_query,
        metadata_filters=metadata_filters,
        top_k=top_k,
        min_score=min_score,
    )


# Valid fusion methods for hybrid search
FUSION_METHODS = {"rrf", "weighted"}


def hybrid_search(
    conn: sqlite3.Connection,
    query: str,
    *,
    semantic_weight: float = 0.6,
    fusion_method: str = "rrf",
    top_k: int = 10,
    min_score: float = 0.0,
    metadata_filters: Optional[Dict[str, Any]] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    tags_any: Optional[List[str]] = None,
    tags_all: Optional[List[str]] = None,
    tags_none: Optional[List[str]] = None,
    auto_rebuild: bool = True,
) -> List[Dict[str, Any]]:
    """Combine FTS keyword search and semantic vector search.

    Args:
        conn: Database connection
        query: Search query text
        semantic_weight: Weight for semantic results (0-1). Keyword weight = 1 - semantic_weight.
        fusion_method: How to combine results:
            - "rrf": Reciprocal Rank Fusion (position-based, default)
            - "weighted": Direct score weighting (semantic_weight * vector_score + keyword_weight * text_score)
        top_k: Maximum number of results to return
        min_score: Minimum combined score threshold
        metadata_filters: Optional metadata filters
        date_from: Optional date filter (ISO format or relative like "7d", "1m", "1y")
        date_to: Optional date filter
        tags_any: Match memories with ANY of these tags
        tags_all: Match memories with ALL of these tags
        tags_none: Exclude memories with ANY of these tags
        auto_rebuild: If True, automatically rebuild embeddings on model mismatch

    Returns:
        List of memories with combined scores, sorted by relevance
    """
    if not query or not query.strip():
        return []

    # Validate fusion method
    if fusion_method not in FUSION_METHODS:
        raise ValueError(
            f"Invalid fusion_method '{fusion_method}'. Must be one of: {', '.join(sorted(FUSION_METHODS))}"
        )

    # Clamp semantic_weight to valid range
    semantic_weight = max(0.0, min(1.0, semantic_weight))
    keyword_weight = 1.0 - semantic_weight

    # 1. Get semantic search results (fetch more than top_k for better fusion)
    semantic_results = semantic_search(
        conn,
        query,
        metadata_filters=metadata_filters,
        top_k=top_k * 3,
        min_score=None,  # Get all results, filter after fusion
        auto_rebuild=auto_rebuild,
    )

    # 2. Get keyword search results
    keyword_results = list_memories(
        conn,
        query=query,
        metadata_filters=metadata_filters,
        limit=top_k * 3,
        offset=0,
        date_from=date_from,
        date_to=date_to,
        tags_any=tags_any,
        tags_all=tags_all,
        tags_none=tags_none,
    )

    scores: Dict[int, float] = {}
    memories_by_id: Dict[int, Dict[str, Any]] = {}

    if fusion_method == "weighted":
        # Weighted fusion: direct score combination
        # semantic_weight * vector_score + keyword_weight * text_score
        # Note: keyword results from FTS don't have scores, so we assign based on rank

        # Score semantic results using actual similarity scores
        for result in semantic_results:
            memory = result.get("memory", result)
            memory_id = memory["id"]
            memories_by_id[memory_id] = memory
            semantic_score = result.get("score", 0.0)
            scores[memory_id] = semantic_weight * semantic_score

        # Score keyword results - assign decreasing scores based on rank
        # FTS results are already sorted by relevance, so rank 0 = most relevant
        num_keyword = len(keyword_results)
        for rank, memory in enumerate(keyword_results):
            memory_id = memory["id"]
            memories_by_id[memory_id] = memory
            # Convert rank to a 0-1 score (rank 0 gets 1.0, last rank gets ~0)
            if num_keyword > 1:
                keyword_score = (
                    1.0 - (rank / (num_keyword - 1)) * 0.9
                )  # Range: 1.0 to 0.1
            else:
                keyword_score = 1.0
            if memory_id in scores:
                scores[memory_id] += keyword_weight * keyword_score
            else:
                scores[memory_id] = keyword_weight * keyword_score

    else:
        # RRF fusion (default): Reciprocal Rank Fusion
        # RRF score = sum(1 / (k + rank)) where k is a constant (typically 60)
        rrf_k = 60

        # Score semantic results
        for rank, result in enumerate(semantic_results):
            memory = result.get("memory", result)
            memory_id = memory["id"]
            memories_by_id[memory_id] = memory
            semantic_score = result.get("score", 0.0)
            # Combine RRF with original semantic score for better ranking
            rrf_contribution = semantic_weight / (rrf_k + rank)
            score_boost = (
                semantic_weight * semantic_score * 0.1
            )  # Small boost from actual similarity
            scores[memory_id] = (
                scores.get(memory_id, 0) + rrf_contribution + score_boost
            )

        # Score keyword results
        for rank, memory in enumerate(keyword_results):
            memory_id = memory["id"]
            memories_by_id[memory_id] = memory
            rrf_contribution = keyword_weight / (rrf_k + rank)
            scores[memory_id] = scores.get(memory_id, 0) + rrf_contribution

    # 4. Sort by combined score and apply filters
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    results: List[Dict[str, Any]] = []
    for memory_id in sorted_ids:
        if len(results) >= top_k:
            break

        score = scores[memory_id]
        if score < min_score:
            continue

        memory = memories_by_id[memory_id]
        results.append(
            {
                "score": round(score, 4),
                "memory": memory,
            }
        )

    return results


def _get_stored_embedding_model(conn: sqlite3.Connection) -> Optional[str]:
    """Get the embedding model name stored in the database."""
    row = conn.execute(
        "SELECT value FROM memories_meta WHERE key = 'embedding_model'"
    ).fetchone()
    return row["value"] if row else None


def _set_stored_embedding_model(conn: sqlite3.Connection, model: str) -> None:
    """Store the embedding model name in the database."""
    conn.execute(
        """
        INSERT INTO memories_meta (key, value) VALUES ('embedding_model', ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (model,),
    )
    conn.commit()


def _check_embedding_model_mismatch(conn: sqlite3.Connection) -> bool:
    """Check if current embedding model differs from stored model.

    Returns True if mismatch detected (rebuild needed).
    """
    stored = _get_stored_embedding_model(conn)
    if stored is None:
        # No model stored yet - check if embeddings exist
        count = conn.execute("SELECT COUNT(*) FROM memories_embeddings").fetchone()[0]
        if count > 0:
            # Embeddings exist but no model recorded - assume mismatch
            return True
        return False
    return stored != EMBEDDING_MODEL


def rebuild_embeddings(conn: sqlite3.Connection) -> int:
    """Rebuild all embeddings using current EMBEDDING_MODEL."""
    rows = conn.execute("SELECT id, content, metadata, tags FROM memories").fetchall()
    updated = 0
    for row in rows:
        memory_id = row["id"]
        metadata = json.loads(row["metadata"]) if row["metadata"] else None
        tags = json.loads(row["tags"]) if row["tags"] else []
        vector = _compute_embedding(row["content"], metadata, tags, conn)
        _upsert_embedding(conn, memory_id, vector)
        updated += 1
    # Store the model name after successful rebuild
    _set_stored_embedding_model(conn, EMBEDDING_MODEL)
    conn.commit()
    return updated


def calculate_importance(
    created_at: str,
    base_importance: float = 1.0,
    access_count: int = 0,
    half_life_days: int = 30,
) -> float:
    """Calculate importance score with time decay and access boost.

    Score = base_importance * recency_factor * access_factor

    Args:
        created_at: ISO datetime string of when memory was created
        base_importance: Base importance value (default 1.0)
        access_count: Number of times memory has been accessed
        half_life_days: Days until importance decays to half (default 30)

    Returns:
        Calculated importance score
    """
    base = base_importance if base_importance is not None else 1.0

    # Recency decay (exponential, half-life = half_life_days)
    try:
        # Handle datetime with or without timezone/microseconds
        created_str = created_at.replace("Z", "+00:00") if created_at else None
        if created_str:
            # Try parsing as full datetime first
            try:
                created = datetime.fromisoformat(created_str)
            except ValueError:
                # Try simpler format
                created = datetime.strptime(created_str[:19], "%Y-%m-%d %H:%M:%S")
            age_days = (datetime.now() - created.replace(tzinfo=None)).days
            recency = 0.5 ** (age_days / half_life_days) if age_days >= 0 else 1.0
        else:
            recency = 1.0
    except (ValueError, TypeError):
        recency = 1.0

    # Access boost (logarithmic to prevent runaway scores)
    access = access_count if access_count is not None else 0
    access_factor = 1 + math.log(access + 1) * 0.1

    return round(base * recency * access_factor, 4)


def _track_access(conn: sqlite3.Connection, memory_id: int) -> None:
    """Update access tracking for a memory (last_accessed and access_count)."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        """
        UPDATE memories
        SET access_count = COALESCE(access_count, 0) + 1,
            last_accessed = ?
        WHERE id = ?
        """,
        (now, memory_id),
    )
    # Don't commit here - let caller manage transaction


def boost_memory(
    conn: sqlite3.Connection,
    memory_id: int,
    boost_amount: float = 0.5,
) -> Optional[Dict[str, Any]]:
    """Boost a memory's base importance score.

    Args:
        conn: Database connection
        memory_id: ID of memory to boost
        boost_amount: Amount to add to base importance (default 0.5)

    Returns:
        Updated memory dict or None if not found
    """
    # First check if memory exists
    row = conn.execute(
        "SELECT importance FROM memories WHERE id = ?",
        (memory_id,),
    ).fetchone()

    if not row:
        return None

    current = row["importance"] if row["importance"] is not None else 1.0
    new_importance = current + boost_amount

    conn.execute(
        "UPDATE memories SET importance = ? WHERE id = ?",
        (new_importance, memory_id),
    )
    conn.commit()

    return get_memory(conn, memory_id)


def get_statistics(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Gather statistics about stored memories."""
    stats: Dict[str, Any] = {}

    # Total count
    total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    stats["total_memories"] = total

    # Tag statistics
    tag_counts: Dict[str, int] = {}
    rows = conn.execute("SELECT tags FROM memories").fetchall()
    for (tags_json,) in rows:
        if tags_json:
            try:
                tags = json.loads(tags_json)
                if isinstance(tags, list):
                    for tag in tags:
                        if isinstance(tag, str):
                            tag_counts[tag] = tag_counts.get(tag, 0) + 1
            except json.JSONDecodeError:
                pass

    stats["tag_counts"] = dict(
        sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
    )
    stats["unique_tags"] = len(tag_counts)

    # Section statistics
    section_counts: Dict[str, int] = {}
    subsection_counts: Dict[str, int] = {}
    rows = conn.execute("SELECT metadata FROM memories").fetchall()
    for (metadata_json,) in rows:
        if metadata_json:
            try:
                metadata = json.loads(metadata_json)
                if isinstance(metadata, dict):
                    section = metadata.get("section")
                    if section:
                        section_counts[section] = section_counts.get(section, 0) + 1
                    subsection = metadata.get("subsection")
                    if subsection:
                        subsection_counts[subsection] = (
                            subsection_counts.get(subsection, 0) + 1
                        )
            except json.JSONDecodeError:
                pass

    stats["section_counts"] = dict(
        sorted(section_counts.items(), key=lambda x: x[1], reverse=True)
    )
    stats["subsection_counts"] = dict(
        sorted(subsection_counts.items(), key=lambda x: x[1], reverse=True)
    )

    # Date-based statistics (memories per month)
    monthly_counts: Dict[str, int] = {}
    rows = conn.execute("SELECT created_at FROM memories").fetchall()
    for (created_at,) in rows:
        if created_at:
            try:
                # Extract YYYY-MM from timestamp
                month = created_at[:7]  # "2025-09"
                monthly_counts[month] = monthly_counts.get(month, 0) + 1
            except (IndexError, TypeError):
                pass

    stats["monthly_counts"] = dict(sorted(monthly_counts.items()))

    # Cross-reference statistics (most connected memories)
    crossref_counts: List[tuple[int, int]] = []
    rows = conn.execute("SELECT memory_id, related FROM memories_crossrefs").fetchall()
    for memory_id, related_json in rows:
        if related_json:
            try:
                related = json.loads(related_json)
                if isinstance(related, list):
                    crossref_counts.append((memory_id, len(related)))
            except json.JSONDecodeError:
                pass

    # Sort by count and take top 10
    crossref_counts.sort(key=lambda x: x[1], reverse=True)
    stats["most_connected"] = [
        {"memory_id": memory_id, "connections": count}
        for memory_id, count in crossref_counts[:10]
    ]

    # Date range
    date_range = conn.execute(
        "SELECT MIN(created_at), MAX(created_at) FROM memories"
    ).fetchone()
    if date_range and date_range[0]:
        stats["date_range"] = {
            "oldest": date_range[0],
            "newest": date_range[1],
        }

    # Embedding cache statistics
    if EMBEDDING_CACHE_ENABLED:
        stats["embedding_cache"] = get_embedding_cache_stats(conn)

    return stats


# ---------------------------------------------------------------------------
# Cross-Session Memory Sharing Functions
# ---------------------------------------------------------------------------


def share_memory(
    conn: sqlite3.Connection,
    memory_id: int,
    source_agent: Optional[str] = None,
    target_agents: Optional[List[str]] = None,
    message: Optional[str] = None,
) -> Dict[str, Any]:
    """Share a memory with other agents/sessions.

    Args:
        conn: Database connection
        memory_id: ID of memory to share
        source_agent: Agent ID of the sharer
        target_agents: List of agent IDs to notify (None = broadcast to all)
        message: Optional message to include with the share

    Returns:
        Dictionary with share event details
    """
    # Verify memory exists
    memory = get_memory(conn, memory_id)
    if not memory:
        raise ValueError(f"Memory {memory_id} not found")

    # Add shared-cache tag if not already present
    tags = memory.get("tags", [])
    if "shared-cache" not in tags:
        tags = tags + ["shared-cache"]
        update_memory(conn, memory_id, tags=tags)

    # Create share event
    target_agents_json = json.dumps(target_agents) if target_agents else None
    cur = conn.execute(
        """
        INSERT INTO share_events (memory_id, source_agent, target_agents, message)
        VALUES (?, ?, ?, ?)
        """,
        (memory_id, source_agent, target_agents_json, message),
    )
    event_id = cur.lastrowid

    # Also emit a regular event for the events poll system
    _emit_event(conn, memory_id, ["shared-cache", "memory_shared"])

    conn.commit()

    return {
        "event_id": event_id,
        "memory_id": memory_id,
        "source_agent": source_agent,
        "target_agents": target_agents,
        "message": message,
        "shared_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
    }


def get_shared_memories(
    conn: sqlite3.Connection,
    agent_id: Optional[str] = None,
    since_timestamp: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """Get memories shared with an agent.

    Args:
        conn: Database connection
        agent_id: Agent to check shares for (None = all shares)
        since_timestamp: Only get shares after this timestamp
        limit: Maximum number of results

    Returns:
        Dictionary with count and list of share events
    """
    query = """
        SELECT se.id, se.memory_id, se.source_agent, se.target_agents,
               se.message, se.created_at, se.acknowledged_by,
               m.content, m.metadata, m.tags
        FROM share_events se
        JOIN memories m ON se.memory_id = m.id
        WHERE 1=1
    """
    params: List[Any] = []

    # Filter by target agent if specified
    if agent_id:
        # Match if target_agents is NULL (broadcast) or contains the agent
        query += " AND (se.target_agents IS NULL OR se.target_agents LIKE ?)"
        params.append(f'%"{agent_id}"%')

    # Filter by timestamp
    if since_timestamp:
        query += " AND se.created_at > ?"
        params.append(since_timestamp)

    query += " ORDER BY se.created_at DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()

    results = []
    for row in rows:
        target_agents = (
            json.loads(row["target_agents"]) if row["target_agents"] else None
        )
        acknowledged_by = (
            json.loads(row["acknowledged_by"]) if row["acknowledged_by"] else []
        )
        metadata = json.loads(row["metadata"]) if row["metadata"] else None
        tags = json.loads(row["tags"]) if row["tags"] else []

        results.append(
            {
                "event_id": row["id"],
                "memory_id": row["memory_id"],
                "source_agent": row["source_agent"],
                "target_agents": target_agents,
                "message": row["message"],
                "shared_at": row["created_at"],
                "acknowledged_by": acknowledged_by,
                "memory": {
                    "id": row["memory_id"],
                    "content": row["content"],
                    "metadata": metadata,
                    "tags": tags,
                },
            }
        )

    return {
        "count": len(results),
        "shares": results,
    }


def acknowledge_share(
    conn: sqlite3.Connection,
    event_id: int,
    agent_id: str,
) -> Dict[str, Any]:
    """Acknowledge receipt of a shared memory.

    Args:
        conn: Database connection
        event_id: Share event ID to acknowledge
        agent_id: Agent acknowledging the share

    Returns:
        Dictionary with acknowledgment status
    """
    row = conn.execute(
        "SELECT acknowledged_by FROM share_events WHERE id = ?",
        (event_id,),
    ).fetchone()

    if not row:
        return {"error": "Share event not found", "event_id": event_id}

    acknowledged_by = (
        json.loads(row["acknowledged_by"]) if row["acknowledged_by"] else []
    )

    if agent_id not in acknowledged_by:
        acknowledged_by.append(agent_id)
        conn.execute(
            "UPDATE share_events SET acknowledged_by = ? WHERE id = ?",
            (json.dumps(acknowledged_by), event_id),
        )
        conn.commit()

    return {
        "event_id": event_id,
        "agent_id": agent_id,
        "acknowledged": True,
        "all_acknowledged_by": acknowledged_by,
    }


# ---------------------------------------------------------------------------
# Sync Version Tracking Functions
# ---------------------------------------------------------------------------


def get_current_sync_version(conn: sqlite3.Connection) -> int:
    """Get the current global sync version."""
    row = conn.execute(
        "SELECT value FROM sync_metadata WHERE key = 'global_version'"
    ).fetchone()
    return int(row[0]) if row else 0


def get_agent_sync_state(conn: sqlite3.Connection, agent_id: str) -> Dict[str, Any]:
    """Get the sync state for a specific agent."""
    row = conn.execute(
        "SELECT last_sync_version, last_sync_at FROM sync_state WHERE agent_id = ?",
        (agent_id,),
    ).fetchone()
    if row:
        return {
            "agent_id": agent_id,
            "last_sync_version": row[0],
            "last_sync_at": row[1],
        }
    return {
        "agent_id": agent_id,
        "last_sync_version": 0,
        "last_sync_at": None,
    }


def update_agent_sync_state(
    conn: sqlite3.Connection, agent_id: str, sync_version: int
) -> None:
    """Update the sync state for an agent after a successful sync."""
    conn.execute(
        """
        INSERT INTO sync_state (agent_id, last_sync_version, last_sync_at)
        VALUES (?, ?, datetime('now'))
        ON CONFLICT(agent_id) DO UPDATE SET
            last_sync_version = excluded.last_sync_version,
            last_sync_at = excluded.last_sync_at
        """,
        (agent_id, sync_version),
    )
    conn.commit()


def sync_delta(
    conn: sqlite3.Connection,
    since_version: int,
    include_deleted: bool = True,
    agent_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Get all changes since a version number for delta sync.

    Args:
        conn: Database connection
        since_version: Get changes after this version (exclusive)
        include_deleted: Include deleted memory records
        agent_id: If provided, update agent's sync state after query

    Returns:
        Dictionary with current_version and list of changes
    """
    current_version = get_current_sync_version(conn)

    changes: List[Dict[str, Any]] = []

    # Get created/updated memories
    rows = conn.execute(
        """
        SELECT id, content, metadata, tags, created_at, updated_at,
               tier, expires_at, importance, sync_version
        FROM memories
        WHERE sync_version > ?
        ORDER BY sync_version ASC
        """,
        (since_version,),
    ).fetchall()

    for row in rows:
        metadata = json.loads(row["metadata"]) if row["metadata"] else None
        tags = json.loads(row["tags"]) if row["tags"] else []

        # Determine if this is a create or update based on created_at vs updated_at
        action = "update" if row["updated_at"] else "create"

        changes.append(
            {
                "action": action,
                "sync_version": row["sync_version"],
                "memory": {
                    "id": row["id"],
                    "content": row["content"],
                    "metadata": metadata,
                    "tags": tags,
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "tier": row["tier"] or DEFAULT_TIER,
                    "expires_at": row["expires_at"],
                    "importance": row["importance"] or 1.0,
                },
            }
        )

    # Get deleted memories
    if include_deleted:
        deleted_rows = conn.execute(
            """
            SELECT memory_id, content_preview, deleted_at, sync_version
            FROM deleted_memories
            WHERE sync_version > ?
            ORDER BY sync_version ASC
            """,
            (since_version,),
        ).fetchall()

        for row in deleted_rows:
            changes.append(
                {
                    "action": "delete",
                    "sync_version": row["sync_version"],
                    "memory_id": row["memory_id"],
                    "content_preview": row["content_preview"],
                    "deleted_at": row["deleted_at"],
                }
            )

    # Sort all changes by sync_version
    changes.sort(key=lambda x: x["sync_version"])

    # Update agent sync state if agent_id provided
    if agent_id and changes:
        update_agent_sync_state(conn, agent_id, current_version)

    return {
        "current_version": current_version,
        "since_version": since_version,
        "change_count": len(changes),
        "changes": changes,
    }


def cleanup_deleted_memories(
    conn: sqlite3.Connection, older_than_days: int = 30
) -> int:
    """Remove old deleted memory records that are no longer needed for sync.

    Args:
        conn: Database connection
        older_than_days: Delete records older than this many days

    Returns:
        Number of records deleted
    """
    cur = conn.execute(
        """
        DELETE FROM deleted_memories
        WHERE deleted_at < datetime('now', ?)
        """,
        (f"-{older_than_days} days",),
    )
    conn.commit()
    return cur.rowcount


def export_memories(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Export all memories to a JSON-serializable list.

    Includes tier and expires_at fields for lossless backup/restore.
    """
    rows = conn.execute(
        "SELECT id, content, metadata, tags, created_at, updated_at, tier, expires_at, importance FROM memories ORDER BY id"
    ).fetchall()

    exported: List[Dict[str, Any]] = []
    for row in rows:
        metadata = row["metadata"]
        tags = row["tags"]
        entry = {
            "id": row["id"],
            "content": row["content"],
            "metadata": json.loads(metadata) if metadata else None,
            "tags": json.loads(tags) if tags else [],
            "created_at": row["created_at"],
        }
        # Include optional fields if present
        if row["updated_at"]:
            entry["updated_at"] = row["updated_at"]
        if row["tier"] and row["tier"] != DEFAULT_TIER:
            entry["tier"] = row["tier"]
        if row["expires_at"]:
            entry["expires_at"] = row["expires_at"]
        if row["importance"] and row["importance"] != 1.0:
            entry["importance"] = row["importance"]
        exported.append(entry)

    return exported


def import_memories(
    conn: sqlite3.Connection,
    data: List[Dict[str, Any]],
    strategy: str = "append",
) -> Dict[str, Any]:
    """Import memories from a JSON list.

    Args:
        conn: Database connection
        data: List of memory dictionaries
        strategy: "replace" (clear all first), "merge" (skip duplicates), "append" (add all)

    Returns:
        Dictionary with import statistics
    """
    if strategy not in ("replace", "merge", "append"):
        raise ValueError("strategy must be 'replace', 'merge', or 'append'")

    # Replace: clear database first
    if strategy == "replace":
        conn.execute("DELETE FROM memories")
        conn.execute("DELETE FROM memories_fts")
        conn.execute("DELETE FROM memories_embeddings")
        conn.execute("DELETE FROM memories_crossrefs")
        conn.commit()

    imported = 0
    skipped = 0
    errors = []

    # Get existing content hashes for merge strategy
    existing_contents: set[str] = set()
    if strategy == "merge":
        rows = conn.execute("SELECT content FROM memories").fetchall()
        existing_contents = {row["content"] for row in rows}

    for idx, entry in enumerate(data):
        try:
            content = entry.get("content", "").strip()
            if not content:
                errors.append({"index": idx, "error": "Missing content"})
                continue

            # Skip duplicates in merge mode
            if strategy == "merge" and content in existing_contents:
                skipped += 1
                continue

            metadata = entry.get("metadata")
            tags = entry.get("tags", [])
            created_at = entry.get("created_at")
            tier = _validate_tier(entry.get("tier"))
            expires_at = _normalize_expires_at(entry.get("expires_at"))
            importance = entry.get("importance", 1.0)

            # Prepare data
            prepared_metadata = _prepare_metadata(metadata) if metadata else None
            validated_tags = _validate_tags(tags)
            _enforce_tag_whitelist(validated_tags)

            metadata_json = (
                json.dumps(prepared_metadata, ensure_ascii=False)
                if prepared_metadata
                else None
            )
            tags_json = json.dumps(validated_tags, ensure_ascii=False)

            # Insert with all fields preserved
            if created_at:
                cur = conn.execute(
                    "INSERT INTO memories (content, metadata, tags, created_at, tier, expires_at, importance) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        content,
                        metadata_json,
                        tags_json,
                        created_at,
                        tier,
                        expires_at,
                        importance,
                    ),
                )
            else:
                cur = conn.execute(
                    "INSERT INTO memories (content, metadata, tags, tier, expires_at, importance) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        content,
                        metadata_json,
                        tags_json,
                        tier,
                        expires_at,
                        importance,
                    ),
                )

            memory_id = cur.lastrowid

            # Update FTS and embeddings
            _fts_upsert(conn, memory_id, content, metadata_json, tags_json)
            vector = _compute_embedding(
                content, prepared_metadata, validated_tags, conn
            )
            _upsert_embedding(conn, memory_id, vector)

            imported += 1

        except Exception as exc:
            errors.append({"index": idx, "error": str(exc)})

    conn.commit()

    # Rebuild cross-references after import
    if imported > 0:
        rebuild_crossrefs(conn)

    return {
        "imported": imported,
        "skipped": skipped,
        "errors": errors[:10],  # Limit error list to first 10
        "total_errors": len(errors),
    }


def poll_events(
    conn: sqlite3.Connection,
    since_timestamp: Optional[str] = None,
    tags_filter: Optional[List[str]] = None,
    unconsumed_only: bool = True,
) -> List[Dict[str, Any]]:
    """Poll for memory events."""
    query = (
        "SELECT id, memory_id, tags, timestamp, consumed FROM memories_events WHERE 1=1"
    )
    params: List[Any] = []

    if unconsumed_only:
        query += " AND consumed = 0"

    if since_timestamp:
        query += " AND timestamp > ?"
        params.append(since_timestamp)

    if tags_filter:
        # Check if any of the filter tags are in the event's tags JSON array
        tag_conditions = " OR ".join(
            ["json_extract(tags, '$') LIKE ?" for _ in tags_filter]
        )
        query += f" AND ({tag_conditions})"
        for tag in tags_filter:
            params.append(f'%"{tag}"%')

    query += " ORDER BY timestamp DESC"

    rows = conn.execute(query, params).fetchall()

    events = []
    for row in rows:
        events.append(
            {
                "id": row["id"],
                "memory_id": row["memory_id"],
                "tags": json.loads(row["tags"]) if row["tags"] else [],
                "timestamp": row["timestamp"],
                "consumed": bool(row["consumed"]),
            }
        )

    return events


def clear_events(conn: sqlite3.Connection, event_ids: List[int]) -> int:
    """Mark events as consumed."""
    if not event_ids:
        return 0

    placeholders = ",".join(["?" for _ in event_ids])
    conn.execute(
        f"UPDATE memories_events SET consumed = 1 WHERE id IN ({placeholders})",
        event_ids,
    )
    conn.commit()
    return len(event_ids)


def cleanup_expired_memories(
    conn: sqlite3.Connection,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Delete expired daily tier memories.

    Removes memories where:
    - tier = 'daily'
    - expires_at is not null
    - expires_at < current time

    Args:
        conn: Database connection
        dry_run: If True, only report what would be deleted without actually deleting

    Returns:
        Dictionary with count of deleted memories and their IDs
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    # Find expired memories
    rows = conn.execute(
        """
        SELECT id, content, expires_at, created_at
        FROM memories
        WHERE tier = 'daily'
          AND expires_at IS NOT NULL
          AND expires_at < ?
        ORDER BY expires_at ASC
        """,
        (now,),
    ).fetchall()

    expired_ids = [row["id"] for row in rows]
    expired_details = [
        {
            "id": row["id"],
            "preview": row["content"][:80] + "..."
            if len(row["content"]) > 80
            else row["content"],
            "expires_at": row["expires_at"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]

    if dry_run:
        return {
            "dry_run": True,
            "would_delete": len(expired_ids),
            "expired_memories": expired_details,
        }

    # Actually delete the expired memories
    deleted_count = 0
    for memory_id in expired_ids:
        if delete_memory(conn, memory_id):
            deleted_count += 1

    return {
        "deleted": deleted_count,
        "expired_memory_ids": expired_ids,
        "details": expired_details,
    }


# ---------------------------------------------------------------------------
# Session Transcript Indexing
# ---------------------------------------------------------------------------


def index_conversation(
    conn: sqlite3.Connection,
    messages: List[Dict[str, Any]],
    session_id: Optional[str] = None,
    title: Optional[str] = None,
    chunk_size: int = 10,
    overlap: int = 2,
    tags: Optional[List[str]] = None,
    create_memories: bool = True,
) -> Dict[str, Any]:
    """
    Index a conversation for semantic search.

    Chunks the conversation and optionally creates memory entries for each chunk
    that can be found via semantic search.

    Args:
        conn: Database connection
        messages: List of message dicts with 'role', 'content', and optional 'timestamp'
        session_id: Unique session identifier (auto-generated if not provided)
        title: Optional title for the session
        chunk_size: Messages per chunk (default: 10)
        overlap: Overlap between chunks (default: 2)
        tags: Additional tags to apply to created memories
        create_memories: If True, create memory entries for each chunk (default: True)

    Returns:
        Dict with session_id, chunks_created, memories_created, message_count
    """
    if not messages:
        return {"error": "no_messages", "message": "No messages to index"}

    # Generate session_id if not provided
    if not session_id:
        import uuid
        from datetime import datetime

        session_id = (
            f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        )

    # Create chunks
    chunks = chunk_conversation(messages, chunk_size=chunk_size, overlap=overlap)

    # Prepare base tags
    base_tags = ["session", f"session/{session_id}"]
    if tags:
        base_tags.extend(tags)

    # Store session metadata
    metadata_json = json.dumps(
        {
            "chunk_size": chunk_size,
            "overlap": overlap,
        }
    )

    conn.execute(
        """
        INSERT OR REPLACE INTO sessions
        (session_id, title, message_count, chunk_count, metadata, updated_at, last_indexed_at)
        VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now'))
        """,
        (session_id, title, len(messages), len(chunks), metadata_json),
    )

    # Store chunks and optionally create memories
    memories_created = []
    for chunk in chunks:
        chunk_index = chunk.get("chunk_index", 0)
        message_range = chunk.get("message_range", (0, 0))
        content = chunk["content"]

        memory_id = None
        if create_memories:
            # Create a memory for this chunk
            chunk_tags = base_tags + [f"chunk/{chunk_index}"]
            memory = add_memory(
                conn,
                content=content,
                tags=chunk_tags,
                metadata={
                    "type": "session_chunk",
                    "session_id": session_id,
                    "chunk_index": chunk_index,
                    "message_range": list(message_range),
                },
            )
            memory_id = memory["id"]
            memories_created.append(memory_id)

        # Store chunk reference
        conn.execute(
            """
            INSERT OR REPLACE INTO session_chunks
            (session_id, chunk_index, content, message_start, message_end, memory_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                chunk_index,
                content,
                message_range[0],
                message_range[1],
                memory_id,
            ),
        )

    conn.commit()

    return {
        "session_id": session_id,
        "title": title,
        "message_count": len(messages),
        "chunks_created": len(chunks),
        "memories_created": memories_created,
    }


def index_conversation_delta(
    conn: sqlite3.Connection,
    session_id: str,
    new_messages: List[Dict[str, Any]],
    chunk_size: int = 10,
    overlap: int = 2,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Incrementally index new messages for an existing session.

    Only indexes messages that haven't been indexed yet (delta-based).

    Args:
        conn: Database connection
        session_id: Existing session identifier
        new_messages: New messages to add and index
        chunk_size: Messages per chunk (default: 10)
        overlap: Overlap between chunks (default: 2)
        tags: Additional tags for new memories

    Returns:
        Dict with new_chunks_created, new_memories_created, total_message_count
    """
    # Get existing session info
    session = conn.execute(
        "SELECT message_count, chunk_count FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()

    if not session:
        return {"error": "session_not_found", "session_id": session_id}

    existing_message_count = session[0]
    existing_chunk_count = session[1]

    if not new_messages:
        return {
            "session_id": session_id,
            "new_chunks_created": 0,
            "new_memories_created": [],
            "total_message_count": existing_message_count,
        }

    # Get existing chunks to find last indexed content
    last_chunk = conn.execute(
        """
        SELECT message_end FROM session_chunks
        WHERE session_id = ?
        ORDER BY chunk_index DESC LIMIT 1
        """,
        (session_id,),
    ).fetchone()

    # Prepare base tags
    base_tags = ["session", f"session/{session_id}"]
    if tags:
        base_tags.extend(tags)

    # Create chunks for new messages
    chunks = chunk_conversation(new_messages, chunk_size=chunk_size, overlap=overlap)

    # Adjust chunk indices and message ranges
    new_chunk_start = existing_chunk_count
    message_offset = existing_message_count

    memories_created = []
    for i, chunk in enumerate(chunks):
        chunk_index = new_chunk_start + i
        original_range = chunk.get("message_range", (0, 0))
        # Adjust message range to global indices
        message_range = (
            message_offset + original_range[0],
            message_offset + original_range[1],
        )
        content = chunk["content"]

        # Create a memory for this chunk
        chunk_tags = base_tags + [f"chunk/{chunk_index}"]
        memory = add_memory(
            conn,
            content=content,
            tags=chunk_tags,
            metadata={
                "type": "session_chunk",
                "session_id": session_id,
                "chunk_index": chunk_index,
                "message_range": list(message_range),
                "is_delta": True,
            },
        )
        memory_id = memory["id"]
        memories_created.append(memory_id)

        # Store chunk reference
        conn.execute(
            """
            INSERT INTO session_chunks
            (session_id, chunk_index, content, message_start, message_end, memory_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                chunk_index,
                content,
                message_range[0],
                message_range[1],
                memory_id,
            ),
        )

    # Update session metadata
    new_total_messages = existing_message_count + len(new_messages)
    new_total_chunks = existing_chunk_count + len(chunks)

    conn.execute(
        """
        UPDATE sessions
        SET message_count = ?, chunk_count = ?, updated_at = datetime('now'), last_indexed_at = datetime('now')
        WHERE session_id = ?
        """,
        (new_total_messages, new_total_chunks, session_id),
    )

    conn.commit()

    return {
        "session_id": session_id,
        "new_chunks_created": len(chunks),
        "new_memories_created": memories_created,
        "total_message_count": new_total_messages,
        "total_chunk_count": new_total_chunks,
    }


def get_session(conn: sqlite3.Connection, session_id: str) -> Optional[Dict[str, Any]]:
    """Get session metadata by ID."""
    row = conn.execute(
        """
        SELECT session_id, title, created_at, updated_at, last_indexed_at,
               message_count, chunk_count, metadata
        FROM sessions WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()

    if not row:
        return None

    return {
        "session_id": row[0],
        "title": row[1],
        "created_at": row[2],
        "updated_at": row[3],
        "last_indexed_at": row[4],
        "message_count": row[5],
        "chunk_count": row[6],
        "metadata": json.loads(row[7]) if row[7] else {},
    }


def list_sessions(
    conn: sqlite3.Connection,
    limit: Optional[int] = None,
    offset: int = 0,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List all indexed sessions."""
    query = """
        SELECT session_id, title, created_at, updated_at, last_indexed_at,
               message_count, chunk_count
        FROM sessions
        WHERE 1=1
    """
    params: List[Any] = []

    if date_from:
        query += " AND created_at >= ?"
        params.append(date_from)
    if date_to:
        query += " AND created_at <= ?"
        params.append(date_to)

    query += " ORDER BY updated_at DESC"

    if limit:
        query += " LIMIT ?"
        params.append(limit)
    if offset:
        query += " OFFSET ?"
        params.append(offset)

    rows = conn.execute(query, params).fetchall()

    return [
        {
            "session_id": row[0],
            "title": row[1],
            "created_at": row[2],
            "updated_at": row[3],
            "last_indexed_at": row[4],
            "message_count": row[5],
            "chunk_count": row[6],
        }
        for row in rows
    ]


def search_sessions(
    conn: sqlite3.Connection,
    query: str,
    session_ids: Optional[List[str]] = None,
    top_k: int = 10,
    min_score: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Search across indexed session chunks using semantic search.

    Args:
        conn: Database connection
        query: Search query
        session_ids: Optional list of session IDs to search within
        top_k: Maximum results to return
        min_score: Minimum similarity score

    Returns:
        List of search results with session context
    """
    # Use the existing semantic search but filter by session tags
    metadata_filters = {"type": "session_chunk"}

    results = semantic_search(
        conn,
        query,
        top_k=top_k * 2,
        min_score=min_score,  # Get more to filter
    )

    # Filter and enhance with session context
    session_results = []
    for result in results:
        memory = result.get("memory", {})
        metadata = memory.get("metadata", {})

        # Check if this is a session chunk
        if metadata.get("type") != "session_chunk":
            continue

        # Filter by session_ids if specified
        mem_session_id = metadata.get("session_id")
        if session_ids and mem_session_id not in session_ids:
            continue

        # Get session info
        session = get_session(conn, mem_session_id)

        session_results.append(
            {
                "score": result.get("score", 0),
                "memory_id": memory.get("id"),
                "content": memory.get("content"),
                "session_id": mem_session_id,
                "session_title": session.get("title") if session else None,
                "chunk_index": metadata.get("chunk_index"),
                "message_range": metadata.get("message_range"),
            }
        )

        if len(session_results) >= top_k:
            break

    return session_results


def delete_session(conn: sqlite3.Connection, session_id: str) -> Dict[str, Any]:
    """
    Delete a session and all its associated chunks and memories.

    Args:
        conn: Database connection
        session_id: Session to delete

    Returns:
        Dict with deleted counts
    """
    # Get memory IDs associated with this session
    rows = conn.execute(
        "SELECT memory_id FROM session_chunks WHERE session_id = ? AND memory_id IS NOT NULL",
        (session_id,),
    ).fetchall()

    memory_ids = [row[0] for row in rows]

    # Delete memories
    deleted_memories = 0
    for memory_id in memory_ids:
        if delete_memory(conn, memory_id):
            deleted_memories += 1

    # Delete chunks (cascades from session delete due to FK)
    conn.execute("DELETE FROM session_chunks WHERE session_id = ?", (session_id,))

    # Delete session
    result = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    session_deleted = result.rowcount > 0

    conn.commit()

    return {
        "session_id": session_id,
        "session_deleted": session_deleted,
        "chunks_deleted": len(memory_ids),
        "memories_deleted": deleted_memories,
    }


# ---------------------------------------------------------------------------
# Identity Links (Entity Unification)
# ---------------------------------------------------------------------------

# Valid entity types for identities
VALID_ENTITY_TYPES = {"person", "organization", "project", "tool", "concept", "other"}


def create_identity(
    conn: sqlite3.Connection,
    canonical_id: str,
    display_name: str,
    entity_type: str = "person",
    aliases: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a canonical identity with optional aliases.

    An identity represents a unique entity (person, organization, etc.)
    that can be referenced across multiple memories.

    Args:
        conn: Database connection
        canonical_id: Unique identifier (e.g., "user:ronaldo", "org:acme")
        display_name: Human-readable name
        entity_type: Type of entity (person, organization, project, tool, concept, other)
        aliases: Alternative names/IDs for this identity
        metadata: Additional metadata

    Returns:
        Created identity dict with canonical_id, display_name, entity_type, aliases
    """
    if entity_type not in VALID_ENTITY_TYPES:
        raise ValueError(
            f"Invalid entity_type '{entity_type}'. Must be one of: {', '.join(sorted(VALID_ENTITY_TYPES))}"
        )

    metadata_json = json.dumps(metadata or {})

    try:
        conn.execute(
            """
            INSERT INTO identities (canonical_id, display_name, entity_type, metadata)
            VALUES (?, ?, ?, ?)
            """,
            (canonical_id, display_name, entity_type, metadata_json),
        )
    except sqlite3.IntegrityError:
        raise ValueError(f"Identity '{canonical_id}' already exists")

    # Add aliases if provided
    aliases_added = []
    if aliases:
        for alias in aliases:
            try:
                conn.execute(
                    """
                    INSERT INTO identity_aliases (alias, canonical_id, source)
                    VALUES (?, ?, 'manual')
                    """,
                    (alias, canonical_id),
                )
                aliases_added.append(alias)
            except sqlite3.IntegrityError:
                # Alias already exists, skip
                pass

    conn.commit()

    return {
        "canonical_id": canonical_id,
        "display_name": display_name,
        "entity_type": entity_type,
        "aliases": aliases_added,
        "metadata": metadata or {},
    }


def get_identity(
    conn: sqlite3.Connection, canonical_id: str
) -> Optional[Dict[str, Any]]:
    """Get an identity by its canonical ID."""
    row = conn.execute(
        """
        SELECT canonical_id, display_name, entity_type, metadata, created_at, updated_at
        FROM identities WHERE canonical_id = ?
        """,
        (canonical_id,),
    ).fetchone()

    if not row:
        return None

    # Get aliases
    alias_rows = conn.execute(
        "SELECT alias, source FROM identity_aliases WHERE canonical_id = ?",
        (canonical_id,),
    ).fetchall()

    return {
        "canonical_id": row[0],
        "display_name": row[1],
        "entity_type": row[2],
        "metadata": json.loads(row[3]) if row[3] else {},
        "created_at": row[4],
        "updated_at": row[5],
        "aliases": [{"alias": a[0], "source": a[1]} for a in alias_rows],
    }


def resolve_identity(conn: sqlite3.Connection, alias_or_id: str) -> Optional[str]:
    """
    Resolve an alias or canonical ID to the canonical ID.

    Args:
        conn: Database connection
        alias_or_id: Either a canonical_id or an alias

    Returns:
        The canonical_id if found, None otherwise
    """
    # First check if it's a canonical ID
    row = conn.execute(
        "SELECT canonical_id FROM identities WHERE canonical_id = ?",
        (alias_or_id,),
    ).fetchone()
    if row:
        return row[0]

    # Check aliases
    row = conn.execute(
        "SELECT canonical_id FROM identity_aliases WHERE alias = ?",
        (alias_or_id,),
    ).fetchone()
    if row:
        return row[0]

    return None


def add_identity_alias(
    conn: sqlite3.Connection,
    canonical_id: str,
    alias: str,
    source: str = "manual",
) -> Dict[str, Any]:
    """
    Add an alias to an existing identity.

    Args:
        conn: Database connection
        canonical_id: The identity to add the alias to
        alias: The new alias
        source: Source of the alias (e.g., "github", "email", "manual")

    Returns:
        Dict with the added alias info
    """
    # Verify identity exists
    if not get_identity(conn, canonical_id):
        raise ValueError(f"Identity '{canonical_id}' not found")

    try:
        conn.execute(
            """
            INSERT INTO identity_aliases (alias, canonical_id, source)
            VALUES (?, ?, ?)
            """,
            (alias, canonical_id, source),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        raise ValueError(f"Alias '{alias}' already exists")

    return {"alias": alias, "canonical_id": canonical_id, "source": source}


def link_memory_to_identity(
    conn: sqlite3.Connection,
    memory_id: int,
    identity_id: str,
    mention_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Link a memory to an identity.

    This creates a bidirectional relationship - you can find memories by identity
    and identities mentioned in a memory.

    Args:
        conn: Database connection
        memory_id: The memory to link
        identity_id: The identity (canonical_id or alias) to link to
        mention_text: Optional text that triggered the link (e.g., "@ronaldo")

    Returns:
        Dict with the link info
    """
    # Resolve identity (could be alias)
    canonical_id = resolve_identity(conn, identity_id)
    if not canonical_id:
        raise ValueError(f"Identity '{identity_id}' not found")

    # Verify memory exists
    memory = get_memory(conn, memory_id)
    if not memory:
        raise ValueError(f"Memory {memory_id} not found")

    try:
        conn.execute(
            """
            INSERT INTO memory_identity_links (memory_id, identity_id, mention_text)
            VALUES (?, ?, ?)
            """,
            (memory_id, canonical_id, mention_text),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        # Link already exists
        pass

    return {
        "memory_id": memory_id,
        "identity_id": canonical_id,
        "mention_text": mention_text,
    }


def unlink_memory_from_identity(
    conn: sqlite3.Connection,
    memory_id: int,
    identity_id: str,
) -> Dict[str, Any]:
    """Remove a link between a memory and an identity."""
    canonical_id = resolve_identity(conn, identity_id)
    if not canonical_id:
        canonical_id = identity_id  # Use as-is if not found

    result = conn.execute(
        "DELETE FROM memory_identity_links WHERE memory_id = ? AND identity_id = ?",
        (memory_id, canonical_id),
    )
    conn.commit()

    return {
        "memory_id": memory_id,
        "identity_id": canonical_id,
        "removed": result.rowcount > 0,
    }


def get_memories_by_identity(
    conn: sqlite3.Connection,
    identity_id: str,
    include_aliases: bool = True,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Find all memories linked to an identity.

    Args:
        conn: Database connection
        identity_id: The identity (canonical_id or alias)
        include_aliases: If True, also search by aliases
        limit: Maximum memories to return

    Returns:
        List of memories with their link info
    """
    canonical_id = resolve_identity(conn, identity_id)
    if not canonical_id:
        return []

    query = """
        SELECT m.id, m.content, m.metadata, m.tags, m.created_at,
               mil.mention_text, mil.created_at as link_created_at
        FROM memories m
        JOIN memory_identity_links mil ON m.id = mil.memory_id
        WHERE mil.identity_id = ?
        ORDER BY m.created_at DESC
    """
    params: List[Any] = [canonical_id]

    if limit:
        query += " LIMIT ?"
        params.append(limit)

    rows = conn.execute(query, params).fetchall()

    return [
        {
            "id": row[0],
            "content": row[1],
            "metadata": json.loads(row[2]) if row[2] else {},
            "tags": json.loads(row[3]) if row[3] else [],
            "created_at": row[4],
            "mention_text": row[5],
            "link_created_at": row[6],
        }
        for row in rows
    ]


def get_identities_in_memory(
    conn: sqlite3.Connection,
    memory_id: int,
) -> List[Dict[str, Any]]:
    """
    Get all identities linked to a memory.

    Args:
        conn: Database connection
        memory_id: The memory ID

    Returns:
        List of identities with their link info
    """
    rows = conn.execute(
        """
        SELECT i.canonical_id, i.display_name, i.entity_type,
               mil.mention_text, mil.created_at as link_created_at
        FROM identities i
        JOIN memory_identity_links mil ON i.canonical_id = mil.identity_id
        WHERE mil.memory_id = ?
        ORDER BY mil.created_at
        """,
        (memory_id,),
    ).fetchall()

    return [
        {
            "canonical_id": row[0],
            "display_name": row[1],
            "entity_type": row[2],
            "mention_text": row[3],
            "link_created_at": row[4],
        }
        for row in rows
    ]


def list_identities(
    conn: sqlite3.Connection,
    entity_type: Optional[str] = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    List all identities.

    Args:
        conn: Database connection
        entity_type: Filter by entity type
        limit: Maximum identities to return
        offset: Offset for pagination

    Returns:
        List of identities with memory counts
    """
    query = """
        SELECT i.canonical_id, i.display_name, i.entity_type, i.created_at,
               COUNT(mil.memory_id) as memory_count
        FROM identities i
        LEFT JOIN memory_identity_links mil ON i.canonical_id = mil.identity_id
    """
    params: List[Any] = []

    if entity_type:
        query += " WHERE i.entity_type = ?"
        params.append(entity_type)

    query += " GROUP BY i.canonical_id ORDER BY i.display_name"

    if limit:
        query += " LIMIT ?"
        params.append(limit)
    if offset:
        query += " OFFSET ?"
        params.append(offset)

    rows = conn.execute(query, params).fetchall()

    return [
        {
            "canonical_id": row[0],
            "display_name": row[1],
            "entity_type": row[2],
            "created_at": row[3],
            "memory_count": row[4],
        }
        for row in rows
    ]


def update_identity(
    conn: sqlite3.Connection,
    canonical_id: str,
    display_name: Optional[str] = None,
    entity_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Update an identity's properties."""
    identity = get_identity(conn, canonical_id)
    if not identity:
        return None

    updates = []
    params: List[Any] = []

    if display_name is not None:
        updates.append("display_name = ?")
        params.append(display_name)

    if entity_type is not None:
        if entity_type not in VALID_ENTITY_TYPES:
            raise ValueError(f"Invalid entity_type '{entity_type}'")
        updates.append("entity_type = ?")
        params.append(entity_type)

    if metadata is not None:
        updates.append("metadata = ?")
        params.append(json.dumps(metadata))

    if updates:
        updates.append("updated_at = datetime('now')")
        params.append(canonical_id)
        conn.execute(
            f"UPDATE identities SET {', '.join(updates)} WHERE canonical_id = ?",
            params,
        )
        conn.commit()

    return get_identity(conn, canonical_id)


def delete_identity(conn: sqlite3.Connection, canonical_id: str) -> Dict[str, Any]:
    """
    Delete an identity and all its links.

    Args:
        conn: Database connection
        canonical_id: Identity to delete

    Returns:
        Dict with deletion info
    """
    # Count links before deletion
    link_count = conn.execute(
        "SELECT COUNT(*) FROM memory_identity_links WHERE identity_id = ?",
        (canonical_id,),
    ).fetchone()[0]

    alias_count = conn.execute(
        "SELECT COUNT(*) FROM identity_aliases WHERE canonical_id = ?",
        (canonical_id,),
    ).fetchone()[0]

    # Delete (cascades to aliases and links)
    result = conn.execute(
        "DELETE FROM identities WHERE canonical_id = ?",
        (canonical_id,),
    )
    conn.commit()

    return {
        "canonical_id": canonical_id,
        "deleted": result.rowcount > 0,
        "links_removed": link_count,
        "aliases_removed": alias_count,
    }


def search_identities(
    conn: sqlite3.Connection,
    query: str,
    entity_type: Optional[str] = None,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Search identities by name or alias.

    Args:
        conn: Database connection
        query: Search query (matches display_name, canonical_id, or aliases)
        entity_type: Optional filter by entity type
        limit: Maximum results

    Returns:
        List of matching identities
    """
    search_pattern = f"%{query}%"

    sql = """
        SELECT DISTINCT i.canonical_id, i.display_name, i.entity_type, i.created_at
        FROM identities i
        LEFT JOIN identity_aliases ia ON i.canonical_id = ia.canonical_id
        WHERE (i.display_name LIKE ? OR i.canonical_id LIKE ? OR ia.alias LIKE ?)
    """
    params: List[Any] = [search_pattern, search_pattern, search_pattern]

    if entity_type:
        sql += " AND i.entity_type = ?"
        params.append(entity_type)

    sql += " ORDER BY i.display_name LIMIT ?"
    params.append(limit)

    rows = conn.execute(sql, params).fetchall()

    return [
        {
            "canonical_id": row[0],
            "display_name": row[1],
            "entity_type": row[2],
            "created_at": row[3],
        }
        for row in rows
    ]


# ============================================================================
# Workspace Management
# ============================================================================


def list_workspaces(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """
    List all workspaces with memory counts.

    Returns:
        List of workspaces with name and memory_count
    """
    _ensure_workspace_column(conn)

    rows = conn.execute(
        """
        SELECT
            COALESCE(workspace, ?) as workspace_name,
            COUNT(*) as memory_count,
            MIN(created_at) as first_memory,
            MAX(created_at) as last_memory
        FROM memories
        GROUP BY COALESCE(workspace, ?)
        ORDER BY memory_count DESC
        """,
        (DEFAULT_WORKSPACE, DEFAULT_WORKSPACE),
    ).fetchall()

    return [
        {
            "workspace": row[0],
            "memory_count": row[1],
            "first_memory": row[2],
            "last_memory": row[3],
        }
        for row in rows
    ]


def get_workspace_stats(conn: sqlite3.Connection, workspace: str) -> Dict[str, Any]:
    """
    Get statistics for a specific workspace.

    Args:
        conn: Database connection
        workspace: Workspace name

    Returns:
        Dictionary with workspace statistics
    """
    _ensure_workspace_column(conn)

    # Basic memory count
    row = conn.execute(
        """
        SELECT
            COUNT(*) as total_memories,
            SUM(CASE WHEN tier = 'daily' THEN 1 ELSE 0 END) as daily_memories,
            SUM(CASE WHEN tier = 'permanent' OR tier IS NULL THEN 1 ELSE 0 END) as permanent_memories,
            MIN(created_at) as first_memory,
            MAX(created_at) as last_memory,
            AVG(importance) as avg_importance
        FROM memories
        WHERE COALESCE(workspace, ?) = ?
        """,
        (DEFAULT_WORKSPACE, workspace),
    ).fetchone()

    if not row or row[0] == 0:
        return {
            "workspace": workspace,
            "exists": False,
            "total_memories": 0,
        }

    # Tag distribution
    tag_rows = conn.execute(
        """
        SELECT tags FROM memories
        WHERE COALESCE(workspace, ?) = ? AND tags IS NOT NULL
        """,
        (DEFAULT_WORKSPACE, workspace),
    ).fetchall()

    tag_counts: Dict[str, int] = {}
    for tr in tag_rows:
        if tr[0]:
            tags = json.loads(tr[0])
            for tag in tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

    # Top tags
    top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "workspace": workspace,
        "exists": True,
        "total_memories": row[0],
        "daily_memories": row[1] or 0,
        "permanent_memories": row[2] or 0,
        "first_memory": row[3],
        "last_memory": row[4],
        "avg_importance": round(row[5], 3) if row[5] else 1.0,
        "top_tags": [{"tag": t[0], "count": t[1]} for t in top_tags],
    }


def move_memories_to_workspace(
    conn: sqlite3.Connection,
    memory_ids: List[int],
    target_workspace: str,
) -> Dict[str, Any]:
    """
    Move memories to a different workspace.

    Args:
        conn: Database connection
        memory_ids: List of memory IDs to move
        target_workspace: Destination workspace

    Returns:
        Dictionary with move results
    """
    _ensure_workspace_column(conn)

    if not memory_ids:
        return {"moved": 0, "not_found": []}

    placeholders = ",".join("?" * len(memory_ids))

    # Check which IDs exist
    existing = conn.execute(
        f"SELECT id FROM memories WHERE id IN ({placeholders})",
        memory_ids,
    ).fetchall()
    existing_ids = {row[0] for row in existing}
    not_found = [mid for mid in memory_ids if mid not in existing_ids]

    # Update workspace for existing memories
    if existing_ids:
        existing_list = list(existing_ids)
        existing_placeholders = ",".join("?" * len(existing_list))
        sync_version = _get_next_sync_version(conn)
        cur = conn.execute(
            f"""
            UPDATE memories
            SET workspace = ?, sync_version = ?
            WHERE id IN ({existing_placeholders})
            """,
            [target_workspace, sync_version] + existing_list,
        )
        conn.commit()
        moved = cur.rowcount
    else:
        moved = 0

    return {
        "moved": moved,
        "target_workspace": target_workspace,
        "not_found": not_found,
    }


def delete_workspace(
    conn: sqlite3.Connection,
    workspace: str,
    delete_memories: bool = False,
) -> Dict[str, Any]:
    """
    Delete a workspace. Optionally delete all memories in it.

    Args:
        conn: Database connection
        workspace: Workspace to delete
        delete_memories: If True, delete all memories. If False, move to default.

    Returns:
        Dictionary with deletion results
    """
    _ensure_workspace_column(conn)

    if workspace == DEFAULT_WORKSPACE:
        return {
            "error": "cannot_delete_default",
            "message": "Cannot delete the default workspace",
        }

    # Count memories in workspace
    count = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE workspace = ?",
        (workspace,),
    ).fetchone()[0]

    if delete_memories:
        # Delete all memories in workspace
        ids = [
            row[0]
            for row in conn.execute(
                "SELECT id FROM memories WHERE workspace = ?", (workspace,)
            ).fetchall()
        ]

        deleted_memories = 0
        for memory_id in ids:
            if delete_memory(conn, memory_id):
                deleted_memories += 1

        return {
            "workspace": workspace,
            "deleted": True,
            "memories_deleted": deleted_memories,
        }

    # Move memories to default workspace
    if count:
        sync_version = _get_next_sync_version(conn)
        conn.execute(
            "UPDATE memories SET workspace = ?, sync_version = ? WHERE workspace = ?",
            (DEFAULT_WORKSPACE, sync_version, workspace),
        )
        conn.commit()

    return {
        "workspace": workspace,
        "deleted": True,
        "memories_moved_to_default": count,
    }
