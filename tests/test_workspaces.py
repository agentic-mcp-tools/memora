"""Tests for multi-workspace collections (RML-975)."""

from __future__ import annotations

import importlib
import os
import tempfile
from pathlib import Path

import pytest

# Import DEFAULT_WORKSPACE at module level for assertions
from memora.storage import DEFAULT_WORKSPACE


@pytest.fixture
def db_conn():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        # Save old env
        old_uri = os.environ.get("MEMORA_STORAGE_URI")
        old_allow_any = os.environ.get("MEMORA_ALLOW_ANY_TAG")

        # Set test environment
        os.environ["MEMORA_STORAGE_URI"] = f"file://{db_path}"
        os.environ["MEMORA_ALLOW_ANY_TAG"] = "1"

        # Force reload to pick up new settings
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


# Import after fixture definition
from memora.storage import (
    add_memory,
    delete_workspace,
    get_memory,
    get_workspace_stats,
    list_memories,
    list_workspaces,
    move_memories_to_workspace,
)


class TestWorkspaceColumn:
    """Test workspace column and defaults."""

    def test_memory_created_with_default_workspace(self, db_conn):
        """Memories without explicit workspace use default."""
        mem = add_memory(
            db_conn,
            content="Test memory without workspace",
            tags=["test"],
        )
        assert mem["workspace"] == DEFAULT_WORKSPACE

    def test_memory_created_with_custom_workspace(self, db_conn):
        """Memories can be created in a specific workspace."""
        mem = add_memory(
            db_conn,
            content="Test memory in project-alpha",
            tags=["test"],
            workspace="project-alpha",
        )
        assert mem["workspace"] == "project-alpha"

    def test_get_memory_includes_workspace(self, db_conn):
        """get_memory returns workspace field."""
        mem = add_memory(
            db_conn,
            content="Test memory",
            workspace="my-project",
        )
        retrieved = get_memory(db_conn, mem["id"])
        assert retrieved is not None
        assert retrieved["workspace"] == "my-project"


class TestListMemoriesWorkspaceFilter:
    """Test filtering memories by workspace."""

    def test_list_memories_filter_single_workspace(self, db_conn):
        """Filter to a single workspace."""
        # Create memories in different workspaces
        add_memory(db_conn, content="Memory A", workspace="project-a")
        add_memory(db_conn, content="Memory B", workspace="project-a")
        add_memory(db_conn, content="Memory C", workspace="project-b")
        add_memory(db_conn, content="Memory D")  # default workspace

        # Filter to project-a
        results = list_memories(db_conn, workspace="project-a")
        assert len(results) == 2
        assert all(m["workspace"] == "project-a" for m in results)

    def test_list_memories_default_includes_null_workspace(self, db_conn):
        """Default workspace filter should include legacy NULL workspace rows."""
        mem = add_memory(db_conn, content="Legacy memory")
        db_conn.execute(
            "UPDATE memories SET workspace = NULL WHERE id = ?",
            (mem["id"],),
        )
        db_conn.commit()

        results = list_memories(db_conn, workspace=DEFAULT_WORKSPACE)
        ids = {m["id"] for m in results}
        assert mem["id"] in ids

    def test_list_memories_filter_multiple_workspaces(self, db_conn):
        """Filter to multiple workspaces."""
        add_memory(db_conn, content="Memory A", workspace="project-a")
        add_memory(db_conn, content="Memory B", workspace="project-b")
        add_memory(db_conn, content="Memory C", workspace="project-c")

        # Filter to project-a and project-b
        results = list_memories(db_conn, workspaces=["project-a", "project-b"])
        assert len(results) == 2
        workspaces = {m["workspace"] for m in results}
        assert workspaces == {"project-a", "project-b"}

    def test_list_memories_no_workspace_filter_returns_all(self, db_conn):
        """Without workspace filter, return all memories."""
        add_memory(db_conn, content="Memory A", workspace="project-a")
        add_memory(db_conn, content="Memory B", workspace="project-b")
        add_memory(db_conn, content="Memory C")  # default

        results = list_memories(db_conn)
        assert len(results) == 3

    def test_list_memories_workspace_with_query(self, db_conn):
        """Workspace filter combined with text query."""
        add_memory(db_conn, content="Important task", workspace="work")
        add_memory(db_conn, content="Important note", workspace="personal")
        add_memory(db_conn, content="Random stuff", workspace="work")

        results = list_memories(db_conn, query="Important", workspace="work")
        assert len(results) == 1
        assert results[0]["workspace"] == "work"
        assert "Important" in results[0]["content"]


class TestListWorkspaces:
    """Test listing workspaces."""

    def test_list_workspaces_empty_db(self, db_conn):
        """Empty database returns empty list."""
        workspaces = list_workspaces(db_conn)
        assert workspaces == []

    def test_list_workspaces_single_workspace(self, db_conn):
        """Single workspace is listed."""
        add_memory(db_conn, content="Memory 1", workspace="my-project")
        add_memory(db_conn, content="Memory 2", workspace="my-project")

        workspaces = list_workspaces(db_conn)
        assert len(workspaces) == 1
        assert workspaces[0]["workspace"] == "my-project"
        assert workspaces[0]["memory_count"] == 2

    def test_list_workspaces_multiple_workspaces(self, db_conn):
        """Multiple workspaces are listed with counts."""
        add_memory(db_conn, content="Memory 1", workspace="project-a")
        add_memory(db_conn, content="Memory 2", workspace="project-a")
        add_memory(db_conn, content="Memory 3", workspace="project-b")
        add_memory(db_conn, content="Memory 4")  # default workspace

        workspaces = list_workspaces(db_conn)
        assert len(workspaces) == 3

        # Should be ordered by memory_count descending
        assert workspaces[0]["memory_count"] == 2

        # Check all workspaces present
        workspace_names = {w["workspace"] for w in workspaces}
        assert workspace_names == {"project-a", "project-b", DEFAULT_WORKSPACE}

    def test_list_workspaces_includes_timestamps(self, db_conn):
        """Workspaces include first and last memory timestamps."""
        add_memory(db_conn, content="First", workspace="project")
        add_memory(db_conn, content="Last", workspace="project")

        workspaces = list_workspaces(db_conn)
        assert len(workspaces) == 1
        assert "first_memory" in workspaces[0]
        assert "last_memory" in workspaces[0]


class TestGetWorkspaceStats:
    """Test workspace statistics."""

    def test_get_workspace_stats_nonexistent(self, db_conn):
        """Nonexistent workspace returns exists=False."""
        stats = get_workspace_stats(db_conn, "nonexistent")
        assert stats["exists"] is False
        assert stats["total_memories"] == 0

    def test_get_workspace_stats_basic(self, db_conn):
        """Basic workspace stats."""
        add_memory(db_conn, content="Memory 1", workspace="project", tags=["tag-a"])
        add_memory(
            db_conn, content="Memory 2", workspace="project", tags=["tag-a", "tag-b"]
        )
        add_memory(db_conn, content="Memory 3", workspace="project", tags=["tag-b"])

        stats = get_workspace_stats(db_conn, "project")
        assert stats["exists"] is True
        assert stats["total_memories"] == 3
        assert stats["permanent_memories"] == 3
        assert stats["daily_memories"] == 0

    def test_get_workspace_stats_top_tags(self, db_conn):
        """Stats include top tags."""
        add_memory(db_conn, content="Memory 1", workspace="project", tags=["common"])
        add_memory(db_conn, content="Memory 2", workspace="project", tags=["common"])
        add_memory(db_conn, content="Memory 3", workspace="project", tags=["rare"])

        stats = get_workspace_stats(db_conn, "project")
        top_tags = stats["top_tags"]
        assert len(top_tags) >= 1
        # common should have count 2
        common_tag = next((t for t in top_tags if t["tag"] == "common"), None)
        assert common_tag is not None
        assert common_tag["count"] == 2

    def test_get_workspace_stats_with_daily_tier(self, db_conn):
        """Stats track daily vs permanent memories."""
        add_memory(db_conn, content="Permanent", workspace="project")
        add_memory(db_conn, content="Daily", workspace="project", tier="daily")

        stats = get_workspace_stats(db_conn, "project")
        assert stats["permanent_memories"] == 1
        assert stats["daily_memories"] == 1


class TestMoveMemoriesToWorkspace:
    """Test moving memories between workspaces."""

    def test_move_memories_basic(self, db_conn):
        """Move memories to a new workspace."""
        mem1 = add_memory(db_conn, content="Memory 1", workspace="source")
        mem2 = add_memory(db_conn, content="Memory 2", workspace="source")

        result = move_memories_to_workspace(
            db_conn,
            memory_ids=[mem1["id"], mem2["id"]],
            target_workspace="destination",
        )

        assert result["moved"] == 2
        assert result["target_workspace"] == "destination"
        assert result["not_found"] == []

        # Verify memories moved
        retrieved1 = get_memory(db_conn, mem1["id"])
        retrieved2 = get_memory(db_conn, mem2["id"])
        assert retrieved1["workspace"] == "destination"
        assert retrieved2["workspace"] == "destination"

    def test_move_memories_not_found(self, db_conn):
        """Moving nonexistent memories reports not_found."""
        mem = add_memory(db_conn, content="Real memory")

        result = move_memories_to_workspace(
            db_conn,
            memory_ids=[mem["id"], 9999, 8888],
            target_workspace="new-ws",
        )

        assert result["moved"] == 1
        assert set(result["not_found"]) == {9999, 8888}

    def test_move_memories_empty_list(self, db_conn):
        """Moving empty list returns immediately."""
        result = move_memories_to_workspace(
            db_conn,
            memory_ids=[],
            target_workspace="new-ws",
        )
        assert result["moved"] == 0


class TestDeleteWorkspace:
    """Test deleting workspaces."""

    def test_delete_workspace_cannot_delete_default(self, db_conn):
        """Cannot delete the default workspace."""
        result = delete_workspace(db_conn, DEFAULT_WORKSPACE)
        assert "error" in result
        assert result["error"] == "cannot_delete_default"

    def test_delete_workspace_move_to_default(self, db_conn):
        """Delete workspace, moving memories to default."""
        mem1 = add_memory(db_conn, content="Memory 1", workspace="old-project")
        mem2 = add_memory(db_conn, content="Memory 2", workspace="old-project")

        result = delete_workspace(db_conn, "old-project", delete_memories=False)

        assert result["deleted"] is True
        assert result["memories_moved_to_default"] == 2

        # Verify memories moved to default
        retrieved1 = get_memory(db_conn, mem1["id"])
        assert retrieved1["workspace"] == DEFAULT_WORKSPACE

    def test_delete_workspace_with_memories(self, db_conn):
        """Delete workspace including all memories."""
        mem1 = add_memory(db_conn, content="Memory 1", workspace="to-delete")
        mem2 = add_memory(db_conn, content="Memory 2", workspace="to-delete")
        # Keep one in another workspace
        mem3 = add_memory(db_conn, content="Memory 3", workspace="keep-this")

        result = delete_workspace(db_conn, "to-delete", delete_memories=True)

        assert result["deleted"] is True
        assert result["memories_deleted"] == 2

        # Verify memories deleted
        assert get_memory(db_conn, mem1["id"]) is None
        assert get_memory(db_conn, mem2["id"]) is None
        # Other workspace unchanged
        assert get_memory(db_conn, mem3["id"]) is not None

    def test_delete_empty_workspace(self, db_conn):
        """Delete workspace with no memories."""
        # First create then delete
        add_memory(db_conn, content="Temp", workspace="empty-ws")
        # Move it away
        memories = list_memories(db_conn, workspace="empty-ws")
        move_memories_to_workspace(db_conn, [m["id"] for m in memories], "other")

        result = delete_workspace(db_conn, "empty-ws", delete_memories=True)
        assert result["deleted"] is True
        assert result["memories_deleted"] == 0


class TestWorkspaceIntegration:
    """Integration tests for workspace features."""

    def test_create_list_filter_workflow(self, db_conn):
        """Full workflow: create in workspace, list, filter."""
        # Create memories in multiple workspaces
        add_memory(
            db_conn, content="Project A task 1", workspace="project-a", tags=["task"]
        )
        add_memory(
            db_conn, content="Project A task 2", workspace="project-a", tags=["task"]
        )
        add_memory(
            db_conn, content="Project B note", workspace="project-b", tags=["note"]
        )
        add_memory(
            db_conn, content="Personal idea", workspace="personal", tags=["idea"]
        )

        # List all workspaces
        workspaces = list_workspaces(db_conn)
        assert len(workspaces) == 3

        # Filter to project-a
        project_a_memories = list_memories(db_conn, workspace="project-a")
        assert len(project_a_memories) == 2
        assert all("Project A" in m["content"] for m in project_a_memories)

        # Get stats
        stats = get_workspace_stats(db_conn, "project-a")
        assert stats["total_memories"] == 2

    def test_move_and_delete_workflow(self, db_conn):
        """Workflow: move memories between workspaces, then delete."""
        # Create in project workspace
        mem1 = add_memory(db_conn, content="Active task", workspace="active-project")
        mem2 = add_memory(db_conn, content="Completed task", workspace="active-project")

        # Archive completed task
        move_memories_to_workspace(db_conn, [mem2["id"]], "archive")

        # Verify
        active = list_memories(db_conn, workspace="active-project")
        archived = list_memories(db_conn, workspace="archive")
        assert len(active) == 1
        assert len(archived) == 1

        # Delete archive
        delete_workspace(db_conn, "archive", delete_memories=True)

        # Verify archived memory gone
        assert get_memory(db_conn, mem2["id"]) is None
        # Active still there
        assert get_memory(db_conn, mem1["id"]) is not None
