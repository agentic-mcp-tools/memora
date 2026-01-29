<h1 align="center"><img src="media/memora_new.gif" width="60" alt="Memora Logo" align="absmiddle"> Memora</h1>

<p align="center">
<b>Give your AI agents persistent memory</b><br>
A lightweight MCP server for semantic memory storage, knowledge graphs, and cross-session context.
</p>

<p align="center">
<a href="https://github.com/agentic-mcp-tools/memora/releases"><img src="https://img.shields.io/github/v/tag/agentic-mcp-tools/memora?label=version&color=blue" alt="Version"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
<a href="https://github.com/thedotmack/awesome-claude-code"><img src="https://awesome.re/mentioned-badge.svg" alt="Mentioned in Awesome Claude Code"></a>
</p>

<p align="center">
<img src="media/demo.gif" alt="Memora Demo" width="800">
</p>

<p align="center">
<b><a href="#features">Features</a></b> · <b><a href="#install">Install</a></b> · <b><a href="#usage">Usage</a></b> · <b><a href="#configuration">Config</a></b> · <b><a href="#live-graph-server">Live Graph</a></b> · <b><a href="#semantic-search--embeddings">Semantic Search</a></b> · <b><a href="#llm-deduplication">LLM Deduplication</a></b>
</p>

## Features

- 💾 **Persistent Storage** - SQLite-backed database with optional cloud sync (S3, GCS, Azure)
- 🔍 **Semantic Search** - Vector embeddings (TF-IDF, sentence-transformers, or OpenAI)
- 🤖 **LLM Deduplication** - Find and merge duplicate memories with AI-powered comparison
- ⚡ **Memory Automation** - Structured tools for TODOs, issues, and section placeholders
- 🔗 **Memory Linking** - Typed edges, importance boosting, and cluster detection
- 🗂️ **Memory Tiering** - Daily (auto-expiring) vs permanent memories with configurable TTL
- 📁 **Multi-Workspace** - Isolate memories by project with workspace management tools
- 👤 **Identity Links** - Entity unification with canonical IDs, aliases, and cross-memory linking
- 💬 **Session Indexing** - Index conversation transcripts with chunking and delta updates
- ✂️ **Soft Trim** - Condensed memory views preserving head/tail with smart truncation
- 📡 **Event Notifications** - Poll-based system for inter-agent communication
- 🎯 **Advanced Queries** - Full-text search, date ranges, tag filters (AND/OR/NOT)
- 🔀 **Cross-references** - Auto-linked related memories based on similarity
- 📂 **Hierarchical Organization** - Explore memories by section/subsection
- 📦 **Export/Import** - Backup and restore with merge strategies
- 🕸️ **Knowledge Graph** - Interactive HTML visualization with Mermaid diagram rendering
- 🌐 **Live Graph Server** - Auto-starts HTTP server for remote access via SSH
- 📊 **Statistics & Analytics** - Tag usage, trends, and connection insights

## Install

```bash
pip install git+https://github.com/agentic-mcp-tools/memora.git
```

Includes cloud storage (S3/R2) and OpenAI embeddings out of the box.

```bash
# Optional: local embeddings (offline, ~2GB for PyTorch)
pip install "memora[local]" @ git+https://github.com/agentic-mcp-tools/memora.git
```

<details id="usage">
<summary><big><big><strong>Usage</strong></big></big></summary>

The server runs automatically when configured in Claude Code. Manual invocation:

```bash
# Default (stdio mode for MCP)
memora-server

# With graph visualization server
memora-server --graph-port 8765

# HTTP transport (alternative to stdio)
memora-server --transport streamable-http --host 127.0.0.1 --port 8080
```

</details>

<details id="configuration">
<summary><big><big><strong>Configuration</strong></big></big></summary>

### Claude Code

Add to `.mcp.json` in your project root:

**Local DB:**
```json
{
  "mcpServers": {
    "memora": {
      "command": "memora-server",
      "args": [],
      "env": {
        "MEMORA_DB_PATH": "~/.local/share/memora/memories.db",
        "MEMORA_ALLOW_ANY_TAG": "1",
        "MEMORA_GRAPH_PORT": "8765"
      }
    }
  }
}
```

**Cloud DB (Cloudflare D1) - Recommended:**
```json
{
  "mcpServers": {
    "memora": {
      "command": "memora-server",
      "args": ["--no-graph"],
      "env": {
        "MEMORA_STORAGE_URI": "d1://<account-id>/<database-id>",
        "CLOUDFLARE_API_TOKEN": "<your-api-token>",
        "MEMORA_ALLOW_ANY_TAG": "1"
      }
    }
  }
}
```

With D1, use `--no-graph` to disable the local visualization server. Instead, use the hosted graph at your Cloudflare Pages URL (see [Cloud Graph](#cloud-graph)).

**Cloud DB (S3/R2) - Sync mode:**
```json
{
  "mcpServers": {
    "memora": {
      "command": "memora-server",
      "args": [],
      "env": {
        "AWS_PROFILE": "memora",
        "AWS_ENDPOINT_URL": "https://<account-id>.r2.cloudflarestorage.com",
        "MEMORA_STORAGE_URI": "s3://memories/memories.db",
        "MEMORA_CLOUD_ENCRYPT": "true",
        "MEMORA_ALLOW_ANY_TAG": "1",
        "MEMORA_GRAPH_PORT": "8765"
      }
    }
  }
}
```

### Codex CLI

Add to `~/.codex/config.toml`:

```toml
[mcp_servers.memora]
  command = "memora-server"  # or full path: /path/to/bin/memora-server
  args = ["--no-graph"]
  env = {
    AWS_PROFILE = "memora",
    AWS_ENDPOINT_URL = "https://<account-id>.r2.cloudflarestorage.com",
    MEMORA_STORAGE_URI = "s3://memories/memories.db",
    MEMORA_CLOUD_ENCRYPT = "true",
    MEMORA_ALLOW_ANY_TAG = "1",
  }
```

</details>

<details id="environment-variables">
<summary><big><big><strong>Environment Variables</strong></big></big></summary>

| Variable               | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `MEMORA_DB_PATH`       | Local SQLite database path (default: `~/.local/share/memora/memories.db`)  |
| `MEMORA_STORAGE_URI`   | Storage URI: `d1://<account>/<db-id>` (D1) or `s3://bucket/memories.db` (S3/R2) |
| `CLOUDFLARE_API_TOKEN` | API token for D1 database access (required for `d1://` URI)                |
| `MEMORA_CLOUD_ENCRYPT` | Encrypt database before uploading to cloud (`true`/`false`)                |
| `MEMORA_CLOUD_COMPRESS`| Compress database before uploading to cloud (`true`/`false`)               |
| `MEMORA_CACHE_DIR`     | Local cache directory for cloud-synced database                            |
| `MEMORA_ALLOW_ANY_TAG` | Allow any tag without validation against allowlist (`1` to enable)         |
| `MEMORA_TAG_FILE`      | Path to file containing allowed tags (one per line)                        |
| `MEMORA_TAGS`          | Comma-separated list of allowed tags                                       |
| `MEMORA_GRAPH_PORT`    | Port for the knowledge graph visualization server (default: `8765`)        |
| `MEMORA_EMBEDDING_MODEL` | Embedding backend: `tfidf` (default), `sentence-transformers`, or `openai` |
| `SENTENCE_TRANSFORMERS_MODEL` | Model for sentence-transformers (default: `all-MiniLM-L6-v2`)        |
| `OPENAI_API_KEY`       | API key for OpenAI embeddings and LLM deduplication                        |
| `OPENAI_BASE_URL`      | Base URL for OpenAI-compatible APIs (OpenRouter, Azure, etc.)              |
| `OPENAI_EMBEDDING_MODEL` | OpenAI embedding model (default: `text-embedding-3-small`)               |
| `MEMORA_LLM_ENABLED`   | Enable LLM-powered deduplication comparison (`true`/`false`, default: `true`) |
| `MEMORA_LLM_MODEL`     | Model for deduplication comparison (default: `gpt-4o-mini`)                |
| `MEMORA_EMBEDDING_CACHE` | Enable embedding cache (`true`/`false`, default: `true`)                 |
| `MEMORA_EMBEDDING_CACHE_SIZE` | Max entries in embedding cache (default: `50000`)                   |
| `AWS_PROFILE`          | AWS credentials profile from `~/.aws/credentials` (useful for R2)          |
| `AWS_ENDPOINT_URL`     | S3-compatible endpoint for R2/MinIO                                        |
| `R2_PUBLIC_DOMAIN`     | Public domain for R2 image URLs                                            |

</details>

<details id="semantic-search--embeddings">
<summary><big><big><strong>Semantic Search & Embeddings</strong></big></big></summary>

Memora supports three embedding backends:

| Backend | Install | Quality | Speed |
|---------|---------|---------|-------|
| `openai` (default) | Included | High quality | API latency |
| `sentence-transformers` | `pip install memora[local]` | Good, runs offline | Medium |
| `tfidf` | Included | Basic keyword matching | Fast |

**Automatic:** Embeddings and cross-references are computed automatically when you `memory_create`, `memory_update`, or `memory_create_batch`.

**Manual rebuild required** when:
- Changing `MEMORA_EMBEDDING_MODEL` after memories exist
- Switching to a different sentence-transformers model

```bash
# After changing embedding model, rebuild all embeddings
memory_rebuild_embeddings

# Then rebuild cross-references to update the knowledge graph
memory_rebuild_crossrefs
```

</details>

<details id="live-graph-server">
<summary><big><big><strong>Live Graph Server</strong></big></big></summary>

A built-in HTTP server starts automatically with the MCP server, serving an interactive knowledge graph visualization.

<table>
<tr>
<td align="center"><img src="media/ui_details.png" alt="Details Panel" width="400"><br><em>Details Panel</em></td>
<td align="center"><img src="media/ui_timeline.png" alt="Timeline Panel" width="400"><br><em>Timeline Panel</em></td>
</tr>
</table>

**Access locally:**
```
http://localhost:8765/graph
```

**Remote access via SSH:**
```bash
ssh -L 8765:localhost:8765 user@remote
# Then open http://localhost:8765/graph in your browser
```

**Configuration:**
```json
{
  "env": {
    "MEMORA_GRAPH_PORT": "8765"
  }
}
```

To disable: add `"--no-graph"` to args in your MCP config.

### Graph UI Features

- **Details Panel** - View memory content, metadata, tags, and related memories
- **Timeline Panel** - Browse memories chronologically, click to highlight in graph
- **Time Slider** - Filter memories by date range, drag to explore history
- **Real-time Updates** - Graph and timeline update via SSE when memories change
- **Filters** - Tag/section dropdowns, zoom controls
- **Mermaid Rendering** - Code blocks render as diagrams

### Node Colors

- 🟣 **Tags** - Purple shades by tag
- 🔴 **Issues** - Red (open), Orange (in progress), Green (resolved), Gray (won't fix)
- 🔵 **TODOs** - Blue (open), Orange (in progress), Green (completed), Red (blocked)

Node size reflects connection count.

</details>

<details id="neovim-integration">
<summary><big><big><strong>Neovim Integration</strong></big></big></summary>

Browse memories directly in Neovim with Telescope. Copy the plugin to your config:

```bash
# For kickstart.nvim / lazy.nvim
cp nvim/memora.lua ~/.config/nvim/lua/kickstart/plugins/
```

**Usage:** Press `<leader>sm` to open the memory browser with fuzzy search and preview.

Requires: `telescope.nvim`, `plenary.nvim`, and `memora` installed in your Python environment.

</details>

<details id="knowledge-graph-export">
<summary><big><big><strong>Knowledge Graph Export (Optional)</strong></big></big></summary>

For offline viewing, export memories as a static HTML file:

```python
memory_export_graph(output_path="~/memories_graph.html", min_score=0.25)
```

This is optional - the Live Graph Server provides the same visualization with real-time updates.

</details>

<details id="cloud-graph">
<summary><big><big><strong>Cloud Graph (Recommended for D1)</strong></big></big></summary>

When using Cloudflare D1 as your database, the graph visualization is hosted on Cloudflare Pages - no local server needed.

**Benefits:**
- Access from anywhere (no SSH tunneling)
- Real-time updates via WebSocket
- Multi-database support via `?db=` parameter
- Secure access with Cloudflare Zero Trust

**Setup:**

1. **Create D1 database:**
   ```bash
   npx wrangler d1 create memora-graph
   npx wrangler d1 execute memora-graph --file=memora-graph/schema.sql
   ```

2. **Deploy Pages:**
   ```bash
   cd memora-graph
   npx wrangler pages deploy ./public --project-name=memora-graph
   ```

3. **Configure bindings** in Cloudflare Dashboard:
   - Pages → memora-graph → Settings → Bindings
   - Add D1: `DB_MEMORA` → your database
   - Add R2: `R2_MEMORA` → your bucket (for images)

4. **Configure MCP** with D1 URI:
   ```json
   {
     "env": {
       "MEMORA_STORAGE_URI": "d1://<account-id>/<database-id>",
       "CLOUDFLARE_API_TOKEN": "<your-token>"
     }
   }
   ```

**Access:** `https://memora-graph.pages.dev`

**Secure with Zero Trust:**
1. Cloudflare Dashboard → Zero Trust → Access → Applications
2. Add application for `memora-graph.pages.dev`
3. Create policy with allowed emails
4. Pages → Settings → Enable Access Policy

See [`memora-graph/`](memora-graph/) for detailed setup and multi-database configuration.

</details>

<details id="llm-deduplication">
<summary><big><big><strong>LLM Deduplication</strong></big></big></summary>

Find and merge duplicate memories using AI-powered semantic comparison:

```python
# Find potential duplicates (uses cross-refs + optional LLM analysis)
memory_find_duplicates(min_similarity=0.7, max_similarity=0.95, limit=10, use_llm=True)

# Merge duplicates (append, prepend, or replace strategies)
memory_merge(source_id=123, target_id=456, merge_strategy="append")
```

**LLM Comparison** analyzes memory pairs and returns:
- `verdict`: "duplicate", "similar", or "different"
- `confidence`: 0.0-1.0 score
- `reasoning`: Brief explanation
- `suggested_action`: "merge", "keep_both", or "review"

Works with any OpenAI-compatible API (OpenAI, OpenRouter, Azure, etc.) via `OPENAI_BASE_URL`.

</details>

<details id="memory-automation-tools">
<summary><big><big><strong>Memory Automation Tools</strong></big></big></summary>

Structured tools for common memory types:

```python
# Create a TODO with status and priority
memory_create_todo(content="Implement feature X", status="open", priority="high", category="backend")

# Create an issue with severity
memory_create_issue(content="Bug in login flow", status="open", severity="major", component="auth")

# Create a section placeholder (hidden from graph)
memory_create_section(content="Architecture", section="docs", subsection="api")
```

</details>

<details id="memory-linking">
<summary><big><big><strong>Memory Linking</strong></big></big></summary>

Manage relationships between memories:

```python
# Create typed edges between memories
memory_link(from_id=1, to_id=2, edge_type="implements", bidirectional=True)

# Edge types: references, implements, supersedes, extends, contradicts, related_to

# Remove links
memory_unlink(from_id=1, to_id=2)

# Boost memory importance for ranking
memory_boost(memory_id=42, boost_amount=0.5)

# Detect clusters of related memories
memory_clusters(min_cluster_size=2, min_score=0.3)
```

</details>

## Memory Tiering

Organize memories by lifecycle with automatic expiration:

```python
# Create a daily/ephemeral memory (auto-expires after 24h)
memory_create_daily(
    content="Today's meeting notes",
    tags=["meeting"],
    ttl_hours=24  # Custom TTL (default: 24)
)

# Create a permanent memory (default tier)
memory_create(content="Architecture decision", tier="permanent")

# Cleanup expired memories manually
memory_cleanup_expired()  # Returns count of cleaned memories
```

| Tier | Behavior | Use Case |
|------|----------|----------|
| `permanent` | Never expires (default) | Important knowledge, decisions |
| `daily` | Auto-expires after TTL | Session context, scratch notes |

## Multi-Workspace Collections

Isolate memories into separate workspaces for multi-project organization:

```python
# Create memory in a specific workspace
memory_create(content="Project A feature", workspace="project-a")

# List memories filtered by workspace
memory_list(workspace="project-a")
memory_list(workspaces=["project-a", "project-b"])  # Multiple workspaces

# List all workspaces with memory counts
memory_workspace_list()
# Returns: [{"workspace": "project-a", "memory_count": 42, "first_memory": "...", "last_memory": "..."}]

# Get detailed workspace statistics
memory_workspace_stats(workspace="project-a")
# Returns: total_memories, daily/permanent counts, top_tags, avg_importance

# Move memories between workspaces
memory_workspace_move(memory_ids=[1, 2, 3], target_workspace="archive")

# Delete a workspace (moves memories to default, or deletes all)
memory_workspace_delete(workspace="old-project", delete_memories=False)  # Move to default
memory_workspace_delete(workspace="temp", delete_memories=True)  # Delete all
```

## Identity Links (Entity Unification)

Unify references to the same entity across memories:

```python
# Create a canonical identity
memory_identity_create(
    canonical_id="user:ronaldo",
    display_name="Ronaldo Lima",
    entity_type="person",  # person, organization, project, tool, concept, other
    aliases=["@ronaldo", "limaronaldo", "ronaldo@email.com"]
)

# Resolve an alias to its canonical identity
memory_identity_get(canonical_id="user:ronaldo")

# Add more aliases
memory_identity_add_alias(
    canonical_id="user:ronaldo",
    alias="rlima",
    source="github"
)

# Link memories to identities
memory_identity_link(memory_id=42, identity_id="user:ronaldo", mention_text="Ronaldo")
memory_identity_unlink(memory_id=42, identity_id="user:ronaldo")

# Search memories by identity (includes all aliases)
memory_search_by_identity(identity_id="user:ronaldo", include_aliases=True)

# Get all identities mentioned in a memory
memory_get_identities(memory_id=42)

# List and search identities
memory_identity_list(entity_type="person", limit=10)
memory_identity_search(query="ronaldo")
```

## Session Transcript Indexing

Index conversation transcripts for semantic search across sessions:

```python
# Index a conversation with automatic chunking
memory_index_conversation(
    messages=[
        {"role": "user", "content": "How do I implement auth?"},
        {"role": "assistant", "content": "You can use JWT tokens..."},
        # ... more messages
    ],
    session_id="session-abc123",  # Optional, auto-generated if not provided
    chunk_size=10,     # Messages per chunk (default: 10)
    overlap=2,         # Overlapping messages between chunks (default: 2)
    create_memories=True,  # Also create searchable memories (default: True)
    tags=["session", "auth-discussion"]
)

# Incrementally index new messages (delta indexing)
memory_index_conversation_delta(
    session_id="session-abc123",
    new_messages=[
        {"role": "user", "content": "What about refresh tokens?"},
        {"role": "assistant", "content": "Refresh tokens extend..."}
    ]
)

# Search across session transcripts
memory_session_search(query="authentication JWT", limit=10)

# Get session details
memory_session_get(session_id="session-abc123")

# List all sessions
memory_session_list(limit=20)

# Delete a session and its chunks
memory_session_delete(session_id="session-abc123")
```

## Soft Trim & Content Preview

Get condensed views of memory content:

```python
# Soft trim: preserves head and tail with ellipsis in middle
memory_soft_trim(
    memory_id=42,
    max_length=500,    # Target length (default: 500)
    head_ratio=0.6,    # Portion for head (default: 0.6)
    tail_ratio=0.3     # Portion for tail (default: 0.3)
)
# Returns: "Start of content...[1234 chars truncated]...end of content"

# Compact list (preview only, for browsing)
memory_list_compact(limit=50)
# Returns minimal fields: id, preview (first 80 chars), tags, created_at
```

## Multi-Agent Sync

Synchronize memories across multiple agents with version tracking:

```python
# Get current sync version
memory_sync_version()
# Returns: {"current_version": 42}

# Get changes since a specific version (delta sync)
memory_sync_delta(
    since_version=35,        # Get changes after this version
    include_deleted=True,    # Include deletion records
    agent_id="agent-1"       # Track sync state per agent
)
# Returns: {"changes": [...], "current_version": 42, "has_more": False}

# Get sync state for an agent
memory_sync_state(agent_id="agent-1")
# Returns: {"agent_id": "agent-1", "last_sync_version": 35, "last_sync_at": "..."}

# Cleanup old deletion records
memory_sync_cleanup(older_than_days=30)
```

## Memory Sharing (Inter-Agent Communication)

Share memories between agents with acknowledgment tracking:

```python
# Share a memory with specific agents
memory_share(
    memory_id=42,
    source_agent="agent-1",
    target_agents=["agent-2", "agent-3"],  # None = broadcast to all
    message="Check out this important finding"
)

# Poll for shared memories
memory_shared_poll(
    agent_id="agent-2",           # Filter to your agent
    since_timestamp="2024-01-01", # Only new shares
    limit=50
)

# Acknowledge receipt of a shared memory
memory_share_ack(event_id=123, agent_id="agent-2")
```

## Event System

Poll-based event notifications for inter-agent communication:

```python
# Poll for events (e.g., shared-cache notifications)
memory_events_poll(
    since_timestamp="2024-01-01T00:00:00",
    tags_filter=["shared-cache"],
    unconsumed_only=True
)

# Mark events as consumed
memory_events_clear(event_ids=[1, 2, 3])
```

## Project Context Scanning

Automatically discover and index AI instruction files:

```python
# Scan for AI instruction files (CLAUDE.md, .cursorrules, etc.)
memory_scan_project(
    path="/path/to/project",    # Default: current directory
    extract_sections=True,      # Parse markdown sections
    scan_parents=False,         # Also scan parent directories
    force_rescan=False          # Force update even if unchanged
)

# Get previously indexed project context
memory_get_project_context(
    path="/path/to/project",
    include_sections=True
)

# List discovered instruction files (without indexing)
memory_list_instruction_files(path="/path/to/project")
```

**Supported files:** `CLAUDE.md`, `AGENTS.md`, `.cursorrules`, `.github/copilot-instructions.md`, `GEMINI.md`, `.aider.conf.yml`, `CONVENTIONS.md`, `CODING_GUIDELINES.md`, `.windsurfrules`

## Embedding Cache

Optimize embedding computation with caching:

```python
# Get cache statistics
memory_embedding_cache_stats()
# Returns: {"enabled": True, "hits": 1234, "misses": 56, "hit_rate": 0.96, ...}

# Clear the cache (useful when changing models)
memory_embedding_cache_clear()
```

**Configuration:**
```bash
MEMORA_EMBEDDING_CACHE=true           # Enable/disable (default: true)
MEMORA_EMBEDDING_CACHE_SIZE=50000     # Max entries (default: 50000)
```

## Image Storage (R2)

Store and manage images in Cloudflare R2:

```python
# Upload an image file
memory_upload_image(
    file_path="/path/to/image.png",
    memory_id=42,
    image_index=0,
    caption="Screenshot of the error"
)

# Migrate existing base64 images to R2
memory_migrate_images(dry_run=True)   # Preview changes
memory_migrate_images(dry_run=False)  # Actually migrate
```

## CLI Commands

```bash
# Start the server
memora-server

# Force sync with cloud storage
memora-server sync-pull    # Download from cloud
memora-server sync-push    # Upload to cloud
memora-server sync-status  # Show sync status

# Migrate images to R2
memora-server migrate-images --dry-run
memora-server migrate-images

# Show storage backend info
memora-server info
```

## Complete MCP Tool Reference

### Core Memory Operations
| Tool | Description |
|------|-------------|
| `memory_create` | Create a new memory with content, tags, metadata, tier, workspace |
| `memory_get` | Retrieve a memory by ID |
| `memory_update` | Update content, tags, metadata, tier, or expiration |
| `memory_delete` | Delete a memory |
| `memory_list` | List memories with filters (query, tags, dates, workspace) |
| `memory_list_compact` | List with minimal fields for browsing |
| `memory_create_batch` | Create multiple memories at once |
| `memory_delete_batch` | Delete multiple memories by IDs |

### Specialized Memory Types
| Tool | Description |
|------|-------------|
| `memory_create_todo` | Create a TODO with priority and category |
| `memory_create_issue` | Create an issue/bug with severity and status |
| `memory_create_section` | Create organizational section headers |
| `memory_create_daily` | Create auto-expiring daily memory |
| `memory_promote_to_permanent` | Convert daily memory to permanent |
| `memory_cleanup_expired` | Delete expired daily memories |

### Search & Discovery
| Tool | Description |
|------|-------------|
| `memory_semantic_search` | Vector similarity search |
| `memory_hybrid_search` | Combined keyword + semantic search with RRF |
| `memory_related` | Get cross-referenced memories |
| `memory_clusters` | Detect clusters of related memories |

### Organization
| Tool | Description |
|------|-------------|
| `memory_tags` | List allowed tags |
| `memory_tag_hierarchy` | Get tags as namespace tree |
| `memory_validate_tags` | Check tags against allowlist |
| `memory_hierarchy` | Browse memories by section/subsection |

### Workspace Management
| Tool | Description |
|------|-------------|
| `memory_workspace_list` | List all workspaces with counts |
| `memory_workspace_stats` | Get detailed workspace statistics |
| `memory_workspace_move` | Move memories between workspaces |
| `memory_workspace_delete` | Delete workspace (move or delete memories) |

### Identity Management
| Tool | Description |
|------|-------------|
| `memory_identity_create` | Create canonical identity with aliases |
| `memory_identity_get` | Get identity by canonical ID |
| `memory_identity_update` | Update identity properties |
| `memory_identity_delete` | Delete identity and links |
| `memory_identity_list` | List all identities |
| `memory_identity_search` | Search identities by name/alias |
| `memory_identity_add_alias` | Add alias to identity |
| `memory_identity_link` | Link memory to identity |
| `memory_identity_unlink` | Remove memory-identity link |
| `memory_search_by_identity` | Find memories by identity |
| `memory_get_identities` | Get identities in a memory |

### Session Indexing
| Tool | Description |
|------|-------------|
| `memory_index_conversation` | Index conversation with chunking |
| `memory_index_conversation_delta` | Incrementally index new messages |
| `memory_session_get` | Get session metadata |
| `memory_session_list` | List all indexed sessions |
| `memory_session_search` | Semantic search across sessions |
| `memory_session_delete` | Delete session and chunks |

### Linking & Relationships
| Tool | Description |
|------|-------------|
| `memory_link` | Create typed link between memories |
| `memory_unlink` | Remove link between memories |
| `memory_boost` | Increase memory importance score |

### Maintenance
| Tool | Description |
|------|-------------|
| `memory_rebuild_embeddings` | Recompute all embeddings |
| `memory_rebuild_crossrefs` | Recompute cross-references |
| `memory_find_duplicates` | Find potential duplicate pairs |
| `memory_merge` | Merge two memories into one |
| `memory_stats` | Get database statistics |

### Sync & Sharing
| Tool | Description |
|------|-------------|
| `memory_sync_version` | Get current sync version |
| `memory_sync_delta` | Get changes since version |
| `memory_sync_state` | Get agent sync state |
| `memory_sync_cleanup` | Clean old deletion records |
| `memory_share` | Share memory with agents |
| `memory_shared_poll` | Poll for shared memories |
| `memory_share_ack` | Acknowledge shared memory |

### Events & Export
| Tool | Description |
|------|-------------|
| `memory_events_poll` | Poll for events |
| `memory_events_clear` | Mark events consumed |
| `memory_export` | Export all memories to JSON |
| `memory_import` | Import memories from JSON |
| `memory_export_graph` | Export as HTML knowledge graph |

### Utilities
| Tool | Description |
|------|-------------|
| `memory_soft_trim` | Get truncated memory content |
| `memory_upload_image` | Upload image to R2 |
| `memory_migrate_images` | Migrate base64 to R2 |
| `memory_embedding_cache_stats` | Get cache statistics |
| `memory_embedding_cache_clear` | Clear embedding cache |
| `memory_scan_project` | Index AI instruction files |
| `memory_get_project_context` | Get indexed project context |
| `memory_list_instruction_files` | List discovered AI files |

## Contributing

Contributions welcome! Please open an issue or PR.

## License

MIT License - see [LICENSE](LICENSE) for details.
