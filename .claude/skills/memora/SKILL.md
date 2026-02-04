---
name: memora
description: Persistent memory and knowledge management. Use at session start to load context, when the user asks about past work or stored knowledge, and when saving important information for future sessions.
user-invocable: false
---

## Session Start

At the beginning of each session, proactively search memories for context related to the current project or working directory using `memory_hybrid_search`. Briefly summarize relevant findings to establish context.

## Memory Search

When the user asks about past work, stored knowledge, or previously discussed topics:

1. Use `memory_hybrid_search` to find relevant memories
2. Use `memory_semantic_search` for pure meaning-based lookup
3. Summarize findings and cite memory IDs (e.g., "Memory #42 shows...")

For research/recall questions, always search memories first before answering.

## Saving Memories

When the user asks to remember something, or when important decisions, patterns, or context emerge during a session:

1. Use `memory_create` for general knowledge and notes
2. Use `memory_create_todo` for tasks and action items
3. Use `memory_create_issue` for bugs and problems
4. Use appropriate tags to organize memories

## Avoiding Duplicates

Before creating a new memory, use `memory_hybrid_search` to check if similar content already exists. If a near-match is found, use `memory_update` to extend the existing memory instead of creating a duplicate.

## Linking Related Memories

When creating or finding memories that relate to each other, use `memory_link` to establish explicit connections. This strengthens the knowledge graph and improves future recall.
