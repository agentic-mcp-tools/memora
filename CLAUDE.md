# Memora

## Session Start

At the beginning of each session, proactively search memories for context related to the current project or working directory using `memory_hybrid_search`. Briefly summarize relevant findings to establish context.

## Memory Search

When the user asks about past work, stored knowledge, or previously discussed topics:
1. Use `memory_hybrid_search` to find relevant memories
2. Use `memory_semantic_search` for pure meaning-based lookup
3. Summarize findings and cite memory IDs (e.g., "Memory #51 shows...")

For research/recall questions, always search memories first before answering.

---

## New Features (January 2026)

### Memory Tiering

Organize memories by lifecycle:

| Tier | Behavior | Use Case |
|------|----------|----------|
| `permanent` | Never expires (default) | Important knowledge, decisions |
| `daily` | Auto-expires after TTL | Session context, scratch notes |

```python
# Create auto-expiring memory (default 24h)
memory_create_daily(content="Today's standup notes", ttl_hours=24)

# Clean up expired memories
memory_cleanup_expired()

# Promote daily to permanent before expiration
memory_promote_to_permanent(memory_id=42)
```

### Workspace Isolation

Isolate memories by project:

```python
# Create in workspace
memory_create(content="API architecture", workspace="ibvi-api")

# List filtered by workspace
memory_list(workspace="ibvi-api")

# Search in multiple workspaces
memory_hybrid_search(query="auth", workspaces=["ibvi-api", "mbras-web"])

# Management
memory_workspace_list()           # List all with counts
memory_workspace_stats("ibvi")    # Detailed stats
memory_workspace_move([1,2], "archive")  # Move memories
memory_workspace_delete("old-project")   # Delete workspace
```

### Identity Links

Unify entity references across memories:

```python
# Create canonical identity
memory_identity_create(
    canonical_id="user:ronaldo",
    display_name="Ronaldo Lima",
    entity_type="person",
    aliases=["@ronaldo", "limaronaldo", "ron@email.com"]
)

# Link memories to identities
memory_identity_link(memory_id=42, identity_id="user:ronaldo")

# Search by identity (finds all linked memories regardless of alias)
memory_search_by_identity(identity_id="user:ronaldo")

# Get identities mentioned in a memory
memory_get_identities(memory_id=42)
```

### Session Transcript Indexing

Index conversations with chunking for semantic search:

```python
# Index full conversation
memory_index_conversation(
    messages=[
        {"role": "user", "content": "How do I implement auth?"},
        {"role": "assistant", "content": "Use JWT tokens..."},
    ],
    session_id="session-123",
    chunk_size=10,
    overlap=2
)

# Add new messages incrementally
memory_index_conversation_delta(
    session_id="session-123",
    new_messages=[...]
)

# Search across sessions
memory_session_search(query="authentication JWT")

# Manage sessions
memory_session_list()
memory_session_get("session-123")
memory_session_delete("session-123")
```

### Content Utilities

```python
# Soft trim: preserves head (60%) and tail (30%) with ellipsis
memory_soft_trim(memory_id=42, max_length=500)
# Returns: "Start of content...[800 chars truncated]...end of content"

# Compact list for browsing
memory_list_compact(limit=50)
# Returns minimal fields: id, preview, tags, created_at
```

### Embedding Cache

Reduce API calls with LRU caching:

```python
# Check cache performance
memory_embedding_cache_stats()
# Returns: {"hits": 1234, "misses": 56, "hit_rate": 0.96}

# Clear cache (useful when changing models)
memory_embedding_cache_clear()
```

Configure via environment:
- `MEMORA_EMBEDDING_CACHE=true` (default)
- `MEMORA_EMBEDDING_CACHE_SIZE=50000` (max entries)

### Multi-Agent Sync

Synchronize across agents:

```python
# Get current version
memory_sync_version()

# Get changes since version (delta sync)
memory_sync_delta(since_version=35, agent_id="agent-1")

# Share memory with other agents
memory_share(memory_id=42, source_agent="agent-1", target_agents=["agent-2"])

# Poll for shared memories
memory_shared_poll(agent_id="agent-2")

# Acknowledge receipt
memory_share_ack(event_id=123, agent_id="agent-2")
```

### Project Context Scanning

Auto-discover and index AI instruction files:

```python
# Scan project for CLAUDE.md, .cursorrules, etc.
memory_scan_project(path="/path/to/project")

# Get indexed context
memory_get_project_context(path="/path/to/project")

# List discovered files
memory_list_instruction_files(path="/path/to/project")
```

Supported files: `CLAUDE.md`, `AGENTS.md`, `.cursorrules`, `.github/copilot-instructions.md`, `GEMINI.md`, `.aider.conf.yml`, `CONVENTIONS.md`, `CODING_GUIDELINES.md`, `.windsurfrules`

---

## Complete MCP Tool Reference

### Core Operations
| Tool | Description |
|------|-------------|
| `memory_create` | Create memory with content, tags, metadata, tier, workspace |
| `memory_get` | Get memory by ID |
| `memory_update` | Update content, tags, metadata, tier |
| `memory_delete` | Delete memory |
| `memory_list` | List with filters (query, tags, dates, workspace) |
| `memory_list_compact` | Compact list for browsing |

### Search
| Tool | Description |
|------|-------------|
| `memory_semantic_search` | Vector similarity search |
| `memory_hybrid_search` | Combined keyword + semantic with RRF |
| `memory_related` | Get cross-referenced memories |
| `memory_clusters` | Detect memory clusters |

### Tiering
| Tool | Description |
|------|-------------|
| `memory_create_daily` | Create auto-expiring memory |
| `memory_promote_to_permanent` | Convert daily to permanent |
| `memory_cleanup_expired` | Delete expired memories |

### Workspaces
| Tool | Description |
|------|-------------|
| `memory_workspace_list` | List all workspaces |
| `memory_workspace_stats` | Detailed statistics |
| `memory_workspace_move` | Move memories |
| `memory_workspace_delete` | Delete workspace |

### Identities
| Tool | Description |
|------|-------------|
| `memory_identity_create` | Create canonical identity |
| `memory_identity_get` | Get by ID |
| `memory_identity_update` | Update properties |
| `memory_identity_delete` | Delete identity |
| `memory_identity_list` | List all |
| `memory_identity_search` | Search by name/alias |
| `memory_identity_add_alias` | Add alias |
| `memory_identity_link` | Link memory to identity |
| `memory_identity_unlink` | Remove link |
| `memory_search_by_identity` | Find memories by identity |
| `memory_get_identities` | Get identities in memory |

### Sessions
| Tool | Description |
|------|-------------|
| `memory_index_conversation` | Index with chunking |
| `memory_index_conversation_delta` | Incremental indexing |
| `memory_session_get` | Get session metadata |
| `memory_session_list` | List sessions |
| `memory_session_search` | Search across sessions |
| `memory_session_delete` | Delete session |

### Sync
| Tool | Description |
|------|-------------|
| `memory_sync_version` | Current version |
| `memory_sync_delta` | Changes since version |
| `memory_sync_state` | Agent sync state |
| `memory_share` | Share with agents |
| `memory_shared_poll` | Poll for shares |
| `memory_share_ack` | Acknowledge share |

### Utilities
| Tool | Description |
|------|-------------|
| `memory_soft_trim` | Truncated content |
| `memory_embedding_cache_stats` | Cache statistics |
| `memory_embedding_cache_clear` | Clear cache |
| `memory_scan_project` | Index AI instruction files |
| `memory_get_project_context` | Get indexed context |
| `memory_list_instruction_files` | List discovered files |
