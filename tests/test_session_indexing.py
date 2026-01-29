"""Tests for session transcript indexing functionality."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from memora.storage import (
    chunk_conversation,
    connect,
    delete_session,
    format_conversation_chunk,
    get_session,
    index_conversation,
    index_conversation_delta,
    list_sessions,
    search_sessions,
)

# Sample conversation for testing
SAMPLE_MESSAGES = [
    {"role": "user", "content": "Hello, I need help with authentication."},
    {
        "role": "assistant",
        "content": "I'd be happy to help! What authentication method are you considering?",
    },
    {
        "role": "assistant",
        "content": "I'd be happy to help! What authentication method are you considering?",
    },
    {"role": "user", "content": "I'm thinking about using JWT tokens."},
    {
        "role": "assistant",
        "content": "JWT tokens are a great choice for stateless authentication.",
    },
    {"role": "user", "content": "Can you show me an example implementation?"},
    {"role": "assistant", "content": "Sure! Here's a basic JWT implementation..."},
    {"role": "user", "content": "What about refresh tokens?"},
    {"role": "assistant", "content": "Refresh tokens extend the session securely..."},
    {"role": "user", "content": "How do I store them safely?"},
    {"role": "assistant", "content": "Use HTTP-only cookies for maximum security."},
    {"role": "user", "content": "Thanks, that's very helpful!"},
    {
        "role": "assistant",
        "content": "You're welcome! Let me know if you have more questions.",
    },
]


class TestChunkConversation:
    """Tests for the chunk_conversation function."""

    def test_empty_messages(self):
        """Empty message list should return empty chunks."""
        result = chunk_conversation([])
        assert result == []

    def test_single_message(self):
        """Single message should create one chunk."""
        messages = [{"role": "user", "content": "Hello"}]
        chunks = chunk_conversation(messages, chunk_size=10)

        assert len(chunks) == 1
        assert chunks[0]["chunk_index"] == 0
        assert "user: Hello" in chunks[0]["content"]

    def test_messages_within_chunk_size(self):
        """Messages within chunk_size should create one chunk."""
        messages = SAMPLE_MESSAGES[:5]
        chunks = chunk_conversation(messages, chunk_size=10)

        assert len(chunks) == 1
        assert chunks[0]["message_range"] == (0, 4)

    def test_overlapping_chunks(self):
        """Multiple chunks should overlap correctly."""
        chunks = chunk_conversation(SAMPLE_MESSAGES, chunk_size=5, overlap=2)

        # With 12 messages, chunk_size=5, overlap=2:
        # Chunk 0: messages 0-4 (5 messages)
        # Chunk 1: messages 3-7 (starts at 5-2=3)
        # Chunk 2: messages 6-10
        # Chunk 3: messages 9-11
        assert len(chunks) >= 3

        # Verify overlap
        if len(chunks) >= 2:
            chunk0_end = chunks[0]["message_range"][1]
            chunk1_start = chunks[1]["message_range"][0]
            # There should be overlap
            assert chunk0_end >= chunk1_start

    def test_chunk_content_format(self):
        """Chunk content should be properly formatted."""
        messages = [
            {"role": "user", "content": "Test message"},
            {"role": "assistant", "content": "Response message"},
        ]
        chunks = chunk_conversation(messages, chunk_size=10)

        content = chunks[0]["content"]
        assert "user: Test message" in content
        assert "assistant: Response message" in content

    def test_chunk_with_timestamps(self):
        """Chunks should include timestamps when available."""
        messages = [
            {"role": "user", "content": "Hello", "timestamp": "2026-01-28T10:00:00"},
            {"role": "assistant", "content": "Hi!", "timestamp": "2026-01-28T10:00:05"},
        ]
        chunks = chunk_conversation(messages, chunk_size=10)

        content = chunks[0]["content"]
        assert "[2026-01-28T10:00:00]" in content
        assert "[2026-01-28T10:00:05]" in content

    def test_chunk_metadata(self):
        """Chunks should include metadata when requested."""
        chunks = chunk_conversation(
            SAMPLE_MESSAGES, chunk_size=5, include_metadata=True
        )

        for i, chunk in enumerate(chunks):
            assert "chunk_index" in chunk
            assert chunk["chunk_index"] == i
            assert "total_chunks" in chunk
            assert "message_range" in chunk


class TestFormatConversationChunk:
    """Tests for the format_conversation_chunk helper."""

    def test_basic_formatting(self):
        """Messages should be formatted as role: content."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = format_conversation_chunk(messages)

        assert result == "user: Hello\nassistant: Hi!"

    def test_with_timestamps(self):
        """Timestamps should be included when present and enabled."""
        messages = [
            {"role": "user", "content": "Hello", "timestamp": "10:00"},
        ]
        result = format_conversation_chunk(messages, include_timestamps=True)
        assert "[10:00]" in result

    def test_without_timestamps(self):
        """Timestamps should be excluded when disabled."""
        messages = [
            {"role": "user", "content": "Hello", "timestamp": "10:00"},
        ]
        result = format_conversation_chunk(messages, include_timestamps=False)
        assert "[10:00]" not in result

    def test_custom_separator(self):
        """Custom separator should be used."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = format_conversation_chunk(messages, separator=" | ")
        assert result == "user: Hello | assistant: Hi!"


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        # Set environment variables for test
        import os

        old_uri = os.environ.get("MEMORA_STORAGE_URI")
        old_allow_any = os.environ.get("MEMORA_ALLOW_ANY_TAG")

        os.environ["MEMORA_STORAGE_URI"] = f"file://{db_path}"
        os.environ["MEMORA_ALLOW_ANY_TAG"] = "1"  # Allow any tag for tests

        # Force reload of modules to pick up new settings
        import importlib

        import memora
        from memora import storage

        importlib.reload(memora)
        importlib.reload(storage)

        conn = storage.connect()
        yield conn

        conn.close()

        # Restore environment
        if old_uri:
            os.environ["MEMORA_STORAGE_URI"] = old_uri
        else:
            os.environ.pop("MEMORA_STORAGE_URI", None)

        if old_allow_any:
            os.environ["MEMORA_ALLOW_ANY_TAG"] = old_allow_any
        else:
            os.environ.pop("MEMORA_ALLOW_ANY_TAG", None)


class TestIndexConversation:
    """Tests for conversation indexing with database."""

    def test_index_basic_conversation(self, temp_db):
        """Index a basic conversation."""
        result = index_conversation(
            temp_db,
            messages=SAMPLE_MESSAGES[:4],
            title="Test Conversation",
            chunk_size=10,
        )

        assert "session_id" in result
        assert result["message_count"] == 4
        assert result["chunks_created"] >= 1
        assert result["title"] == "Test Conversation"

    def test_auto_generate_session_id(self, temp_db):
        """Session ID should be auto-generated if not provided."""
        result = index_conversation(temp_db, messages=SAMPLE_MESSAGES[:2])

        assert result["session_id"].startswith("session-")

    def test_custom_session_id(self, temp_db):
        """Custom session ID should be used."""
        result = index_conversation(
            temp_db,
            messages=SAMPLE_MESSAGES[:2],
            session_id="my-custom-session",
        )

        assert result["session_id"] == "my-custom-session"

    def test_memories_created(self, temp_db):
        """Memories should be created for each chunk."""
        result = index_conversation(
            temp_db,
            messages=SAMPLE_MESSAGES,
            chunk_size=5,
            create_memories=True,
        )

        assert len(result["memories_created"]) >= 1

    def test_no_memories_when_disabled(self, temp_db):
        """No memories should be created when disabled."""
        result = index_conversation(
            temp_db,
            messages=SAMPLE_MESSAGES[:2],
            create_memories=False,
        )

        assert result["memories_created"] == []

    def test_custom_tags(self, temp_db):
        """Custom tags should be applied to memories."""
        result = index_conversation(
            temp_db,
            messages=SAMPLE_MESSAGES[:2],
            tags=["auth", "jwt"],
        )

        # Get the memory and verify tags
        if result["memories_created"]:
            from memora.storage import get_memory

            memory = get_memory(temp_db, result["memories_created"][0])
            assert "auth" in memory["tags"]
            assert "jwt" in memory["tags"]


class TestIndexConversationDelta:
    """Tests for incremental conversation indexing."""

    def test_delta_index_new_messages(self, temp_db):
        """New messages should be added to existing session."""
        # First, index initial messages
        initial = index_conversation(
            temp_db,
            messages=SAMPLE_MESSAGES[:4],
            session_id="delta-test",
            chunk_size=5,
        )

        # Add more messages
        delta = index_conversation_delta(
            temp_db,
            session_id="delta-test",
            new_messages=SAMPLE_MESSAGES[4:8],
            chunk_size=5,
        )

        assert delta["new_chunks_created"] >= 1
        assert delta["total_message_count"] == 8

    def test_delta_nonexistent_session(self, temp_db):
        """Delta index should fail for nonexistent session."""
        result = index_conversation_delta(
            temp_db,
            session_id="nonexistent-session",
            new_messages=SAMPLE_MESSAGES[:2],
        )

        assert "error" in result
        assert result["error"] == "session_not_found"


class TestSessionManagement:
    """Tests for session listing and retrieval."""

    def test_get_session(self, temp_db):
        """Get session metadata by ID."""
        # Create a session
        index_conversation(
            temp_db,
            messages=SAMPLE_MESSAGES[:4],
            session_id="get-test",
            title="Test Session",
        )

        session = get_session(temp_db, "get-test")

        assert session is not None
        assert session["session_id"] == "get-test"
        assert session["title"] == "Test Session"
        assert session["message_count"] == 4

    def test_get_nonexistent_session(self, temp_db):
        """Getting nonexistent session should return None."""
        session = get_session(temp_db, "nonexistent")
        assert session is None

    def test_list_sessions(self, temp_db):
        """List all indexed sessions."""
        # Create multiple sessions
        for i in range(3):
            index_conversation(
                temp_db,
                messages=SAMPLE_MESSAGES[:2],
                session_id=f"list-test-{i}",
                title=f"Session {i}",
            )

        sessions = list_sessions(temp_db)

        assert len(sessions) >= 3

    def test_list_sessions_with_limit(self, temp_db):
        """List sessions with limit."""
        for i in range(5):
            index_conversation(
                temp_db,
                messages=SAMPLE_MESSAGES[:2],
                session_id=f"limit-test-{i}",
            )

        sessions = list_sessions(temp_db, limit=2)
        assert len(sessions) == 2

    def test_delete_session(self, temp_db):
        """Delete session and its memories."""
        # Create a session with memories
        result = index_conversation(
            temp_db,
            messages=SAMPLE_MESSAGES[:4],
            session_id="delete-test",
            create_memories=True,
        )

        memories_count = len(result["memories_created"])

        # Delete it
        delete_result = delete_session(temp_db, "delete-test")

        assert delete_result["session_deleted"] is True
        assert delete_result["memories_deleted"] == memories_count

        # Verify it's gone
        session = get_session(temp_db, "delete-test")
        assert session is None
