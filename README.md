# Memento

Temporal knowledge graph memory for AI agents. Goes beyond flat vector stores with contradiction detection, temporal versioning, entity resolution, compositional retrieval, and principled forgetting.

## Why Memento?

Most AI memory systems dump text into a vector store and retrieve by similarity. Memento builds a **knowledge graph** that:

- **Resolves entities** — "John," "John Smith," and "the Acme guy" become one node
- **Detects contradictions** — flags when new facts conflict with existing ones
- **Tracks time** — every fact has valid time and recorded time (bitemporal model)
- **Composes answers** — "What should I know before my meeting with John?" traverses the graph, not just retrieves chunks
- **Improves over time** — consolidation engine decays stale info, merges duplicates, prunes orphans
- **Never loses data** — verbatim text stored alongside the graph as a fallback

## Quick Start

### MCP Server (Claude Desktop, Cursor, Claude Code)

```bash
pip install memento-memory[anthropic]
export ANTHROPIC_API_KEY=your-key
memento-mcp
```

Add to Claude Desktop config (`claude_desktop_config.json`):
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

### Python API

```python
from memento import MemoryStore

store = MemoryStore()

store.ingest("John Smith is VP of Sales at Acme Corp.")
store.ingest("Acme Corp is acquiring Beta Inc.")

memory = store.recall("What should I know about John?")
print(memory.text)
# ## John Smith (person)
# - title: VP of Sales
# - → [works_at] Acme Corp
#
# ## Acme Corp (organization)
# - → [acquiring] Beta Inc
```

### With OpenAI

```bash
pip install memento-memory[openai]
export MEMENTO_LLM_PROVIDER=openai
export OPENAI_API_KEY=your-key
```

### With Ollama (fully local, no API keys)

```bash
pip install memento-memory[openai]
export MEMENTO_LLM_PROVIDER=ollama
# Assumes Ollama running at localhost:11434
```

## Supported LLM Providers

| Provider | Install | Config |
|----------|---------|--------|
| **Anthropic** (Claude) | `pip install memento-memory[anthropic]` | `ANTHROPIC_API_KEY` |
| **OpenAI** (GPT-4) | `pip install memento-memory[openai]` | `OPENAI_API_KEY`, `MEMENTO_LLM_PROVIDER=openai` |
| **Google** (Gemini) | `pip install memento-memory[gemini]` | `GOOGLE_API_KEY`, `MEMENTO_LLM_PROVIDER=gemini` |
| **Ollama** (local) | `pip install memento-memory[openai]` | `MEMENTO_LLM_PROVIDER=ollama` |
| **Any OpenAI-compatible** | `pip install memento-memory[openai]` | `MEMENTO_LLM_PROVIDER=openai-compatible`, `MEMENTO_LLM_BASE_URL=...` |

## CLI

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

## Python API

```python
from memento import MemoryStore

store = MemoryStore()

# Ingest (runs entity extraction + resolution + conflict detection)
result = store.ingest("John moved to Austin last month.")

# Recall (graph traversal + ranking + context budgeting)
memory = store.recall("John's location", token_budget=500)
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

## Architecture

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

**Key design decisions:**
- **Bitemporal model** — every fact tracks when it was true (valid time) and when the system learned it (transaction time)
- **Immutable history** — facts are never deleted, only superseded. Full audit trail.
- **Verbatim fallback** — raw text stored alongside the graph, so extraction loss doesn't mean information loss
- **Multiplicative confidence decay** — prevents artificial confidence floors from repeated confirmations
- **Entity resolution in tiers** — exact/fuzzy/phonetic matching (cheap) before embedding similarity and LLM tiebreaker (expensive)
- **Provider-agnostic** — swap LLM providers via config, no code changes

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMENTO_LLM_PROVIDER` | auto-detect | `anthropic`, `openai`, `gemini`, `ollama` |
| `MEMENTO_LLM_API_KEY` | — | API key (or use provider-specific vars) |
| `MEMENTO_LLM_BASE_URL` | — | For Ollama/vLLM endpoints |
| `MEMENTO_DB_PATH` | `~/.memento/memento.db` | SQLite database path |
| `MEMENTO_EMBEDDING_PROVIDER` | `sentence-transformers` | `sentence-transformers` or `openai` |
| `ANTHROPIC_API_KEY` | — | Anthropic-specific key |
| `OPENAI_API_KEY` | — | OpenAI-specific key |
| `GOOGLE_API_KEY` | — | Gemini-specific key |

## License

MIT
