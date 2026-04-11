# Memento

**Any model, same memory.** A bitemporal knowledge graph that gives AI agents persistent, structured memory across LLM providers, clients, and conversations.

Most AI memory systems dump text into a vector store and retrieve by similarity. Memento builds a **knowledge graph** — resolving entities, detecting contradictions, tracking time, and composing answers from structured relationships rather than raw chunks.

Works with any MCP-compatible client (Claude Desktop, Cursor, Claude Code, Cline, Windsurf, OpenClaw, Continue.dev) and any LLM backend (Claude, GPT, Gemini, Llama, Mistral, Ollama, or any OpenAI-compatible endpoint).

**90.8% overall accuracy, 92.2% task average on [LongMemEval](BENCHMARKS.md)** (500 questions, GPT-4o judge) — a benchmark for long-term conversational memory covering temporal reasoning, knowledge updates, multi-session recall, and preference tracking.

## Quick Start

### MCP Server

```bash
pip install memento-memory[anthropic]
export ANTHROPIC_API_KEY=your-key
memento-mcp
```

Add to your MCP client config (e.g., Claude Desktop `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "memento": {
      "command": "memento-mcp",
      "env": { "ANTHROPIC_API_KEY": "your-key" }
    }
  }
}
```

That's it. The agent now has persistent memory and calls `memory_ingest` to store facts and `memory_recall` to retrieve them. Every MCP client on the same machine shares the same knowledge graph.

### Python API

```python
from memento import MemoryStore

store = MemoryStore()

# Ingest — extracts entities, resolves against the graph, detects contradictions
store.ingest("John Smith is VP of Sales at Alpha Corp.")
store.ingest("Alpha Corp is acquiring Beta Inc.")

# Recall — graph traversal + ranking + context budgeting
memory = store.recall("What should I know about John?")
print(memory.text)
# ## John Smith (person)
# - title: VP of Sales
# - → [works_at] Alpha Corp
#
# ## Alpha Corp (organization)
# - → [acquiring] Beta Inc

# Point-in-time queries
memory = store.recall("Where was John in January?", as_of="2025-01-31T00:00:00Z")

# Direct manipulation
store.correct(entity_id, "title", "VP of Sales", reason="Promoted")
store.forget(entity_id=entity_id)
store.merge(entity_a_id, entity_b_id)

# Introspection
conflicts = store.conflicts()
health = store.health()
entities = store.entity_list()

# Privacy
export = store.export_entity_data(entity_id)
chain = store.audit_belief(entity_id, "title")
receipt = store.hard_delete(entity_id)

# Consolidation
store.consolidate()

# Session tracking (scratchpad with coreference)
session = store.start_session()
session.on_turn("I met John Smith today.")
session.on_turn("He mentioned a new project.")
session.end()  # Flushes through ingestion pipeline
```

## LLM Providers

Memento is provider-agnostic. Swap the backend via config — no code changes.

| Provider | Install | Config |
|---|---|---|
| **Anthropic** | `pip install memento-memory[anthropic]` | `ANTHROPIC_API_KEY` |
| **OpenAI** | `pip install memento-memory[openai]` | `OPENAI_API_KEY`, `MEMENTO_LLM_PROVIDER=openai` |
| **Google Gemini** | `pip install memento-memory[gemini]` | `GOOGLE_API_KEY`, `MEMENTO_LLM_PROVIDER=gemini` |
| **Ollama** (fully local) | `pip install memento-memory[openai]` | `MEMENTO_LLM_PROVIDER=ollama` |
| **Any OpenAI-compatible** | `pip install memento-memory[openai]` | `MEMENTO_LLM_PROVIDER=openai-compatible`, `MEMENTO_LLM_BASE_URL=...` |

## How It Works

```
Agent / LLM
  │ query              │ ingest
  ▼                    ▼
Retrieval Engine    Ingestion Pipeline
  │                    │
  ▼                    ▼
Temporal Knowledge Graph (SQLite)
  │
  ├── Consolidation Engine (decay, dedup, prune)
  ├── Verbatim Fallback (FTS5 + vector search)
  └── Privacy Layer (export, audit, hard delete)
```

- **Entity resolution** — "John," "John Smith," and "the Alpha guy" become one node. Tiered matching: exact/fuzzy/phonetic (cheap) before embedding similarity and LLM tiebreaker (expensive).
- **Contradiction detection** — flags when new facts conflict with existing ones
- **Bitemporal model** — every fact tracks when it was true (valid time) and when the system learned it (transaction time)
- **Immutable history** — facts are never deleted, only superseded. Full audit trail.
- **Verbatim fallback** — raw text stored alongside the graph, so extraction loss doesn't mean information loss
- **Compositional retrieval** — "What should I know before my meeting with John?" traverses the graph, not just retrieves chunks
- **Confidence decay** — multiplicative decay prevents artificial confidence floors from repeated confirmations
- **Consolidation** — background engine decays stale info, merges duplicates, prunes orphans

## Benchmarks

**90.8% overall accuracy on LongMemEval** (500 questions across 6 categories):

| Category | Accuracy |
|---|--:|
| Single-session (assistant) | 98.2% |
| Single-session (user) | 97.1% |
| Single-session (preference) | 93.3% |
| Temporal reasoning | 89.5% |
| Knowledge update | 88.5% |
| Multi-session | 86.5% |
| **Task-averaged** | **92.2%** |

Full methodology and reproduction steps: [BENCHMARKS.md](BENCHMARKS.md)

## CLI

Admin and introspection tools for the knowledge graph:

```bash
memento entities                        # List all entities
memento entity <id>                     # Show entity details
memento history <id> <key>              # Property history over time
memento snapshot <id> --as-of 2025-06   # Point-in-time view
memento stats                           # Graph statistics
memento merge <id_a> <id_b>             # Merge two entities
memento consolidate                     # Run maintenance pass
memento export <id>                     # GDPR data export (JSON)
memento audit <id> <key>                # Trace a belief to its source
memento delete <id> --hard              # Hard delete with receipt
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `MEMENTO_LLM_PROVIDER` | auto-detect | `anthropic`, `openai`, `gemini`, `ollama` |
| `MEMENTO_LLM_API_KEY` | — | API key (or use provider-specific env vars) |
| `MEMENTO_LLM_BASE_URL` | — | For Ollama/vLLM endpoints |
| `MEMENTO_DB_PATH` | `~/.memento/memento.db` | SQLite database path |
| `MEMENTO_EMBEDDING_PROVIDER` | `sentence-transformers` | `sentence-transformers` or `openai` |
| `ANTHROPIC_API_KEY` | — | Anthropic-specific key |
| `OPENAI_API_KEY` | — | OpenAI-specific key |
| `GOOGLE_API_KEY` | — | Gemini-specific key |

## License

MIT
