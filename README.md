# Memento

![PyPI - Version](https://img.shields.io/pypi/v/memento-memory?v=1)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/memento-memory?v=1)
![GitHub License](https://img.shields.io/github/license/shane-farkas/memento-memory)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/shane-farkas/memento-memory/test.yml?label=tests)

**Any model, same memory.** A bitemporal knowledge graph (tracking when facts were true vs. when they were learned) that gives AI agents persistent, structured memory across LLM providers, clients, and conversations.

Most AI memory systems dump text into a vector store and retrieve by similarity. Memento builds a **knowledge graph** that resolves entities, detects contradictions, tracks time, and composes answers from structured relationships rather than raw chunks.

Works with any MCP-compatible client (Claude Desktop, Cursor, Claude Code, Cline, Windsurf, OpenClaw, Continue.dev) and any LLM backend (Claude, GPT, Gemini, Llama, Mistral, Ollama, or any OpenAI-compatible endpoint).

**90.8% overall accuracy, 92.2% task average on [LongMemEval](BENCHMARKS.md)** (500 questions, GPT-4o judge) — a benchmark for long-term conversational memory covering temporal reasoning, knowledge updates, multi-session recall, and preference tracking.

## Quick Start

### MCP Server

```bash
pip install memento-memory[anthropic]
export ANTHROPIC_API_KEY=your-key
memento-mcp
```

Set your API key as an environment variable in your shell profile (e.g. `~/.zshrc`, `~/.bashrc`, or Windows system environment variables) rather than hardcoding it in config files:

```bash
export ANTHROPIC_API_KEY=your-key
```

Then add to your MCP client config (e.g., Claude Desktop `claude_desktop_config.json`), referencing the variable with `${...}` expansion:

```json
{
  "mcpServers": {
    "memento": {
      "command": "memento-mcp",
      "env": { "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}" }
    }
  }
}
```

This keeps your key out of config files that may be synced, committed to dotfile repos, or shared in screenshots. Variable expansion is supported by Claude Desktop, Cursor, Cline, Windsurf, and Continue.dev; for clients that don't expand `${VAR}`, launch the client from a shell where the variable is already set.

That's it. The agent now has persistent memory and calls `memory_ingest` to store facts and `memory_recall` to retrieve them. Every MCP client on the same machine shares the same knowledge graph.

### Compatible Clients

Any MCP-compatible client works with Memento. Add the config block above to:

| Client | Config location |
|---|---|
| **Claude Desktop** | `claude_desktop_config.json` |
| **Claude Code** | `claude_code_config.json` or `--mcp-config` flag |
| **Cursor** | Settings or `~/.cursor/mcp.json` |
| **Cline** | MCP server settings |
| **Windsurf** | MCP server settings |
| **OpenClaw** | MCP server settings |
| **Codex CLI** | `.codex/config.yaml` MCP servers |
| **Gemini CLI** | `gemini mcp add memento -- memento-mcp` |
| **OpenCode** | `.opencode/config.json` MCP servers |
| **Goose** | `~/.config/goose/config.yaml` MCP servers |
| **Kilo Code** | MCP server settings |
| **Continue.dev** | MCP server settings |

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
Bitemporal Knowledge Graph (SQLite)
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

**90.8% overall accuracy on LongMemEval** with Claude Sonnet 4.6 (500 questions):

| Category | Accuracy |
|---|--:|
| Single-session (assistant) | 98.2% |
| Single-session (user) | 97.1% |
| Single-session (preference) | 93.3% |
| Temporal reasoning | 89.5% |
| Knowledge update | 88.5% |
| Multi-session | 86.5% |
| **Task-averaged** | **92.2%** |

### Any Model, Same Memory

Memento is model-agnostic. The same knowledge graph works across providers — only the answer-generation LLM changes. Results on a 50-question stratified sample:

| Model | Provider | Overall | Task-avg |
|---|---|--:|--:|
| **Claude Sonnet 4.6** ★ | Anthropic | **90.8%** | **92.2%** |
| Qwen 3 235B A22B | Together (Alibaba) | 94.0% | 94.2% |
| MiniMax M2.7 | Together (MiniMax) | 94.0% | 93.8% |
| Llama 3.3 70B | Together (Meta) | 88.0% | 87.5% |
| Gemma 4 31B | Together (Google) | 86.0% | 85.8% |
| DeepSeek V3.1 | Together (DeepSeek) | 84.0% | 83.3% |
| Kimi K2.5 | Together (Moonshot) | 70.0% | 70.0% |

★ Claude Sonnet result is from the full 500-question run. Open-source models were evaluated on a 50-question stratified sample covering all six categories. Same Memento graph, same retrieval pipeline, same GPT-4o judge — only the answer model changes.

Full methodology and reproduction steps: [BENCHMARKS.md](BENCHMARKS.md)

## Web Viewer

Browse the knowledge graph in your browser with a built-in web UI:

```bash
pip install memento-memory[web]
memento-web
```

Open http://localhost:8766. The viewer reads from the same `~/.memento/memento.db` that `memento-mcp` writes to — see your agent's memories update in real time.

![Memento graph viewer](docs/graph-viewer.png)

- **Entity list** — search and filter by type (person, organization, project, etc.)
- **Detail view** — properties, relationships, version history, confidence scores
- **Graph view** — interactive force-directed visualization with d3.js (zoom, drag, click to navigate)
- **Timeline view** — when facts were learned vs when they were true (bitemporal)

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
| `MEMENTO_EMBEDDING_PROVIDER` | `auto` | `auto`, `sentence-transformers`, `openai`, `gemini`, `ollama` |
| `ANTHROPIC_API_KEY` | — | Anthropic-specific key |
| `OPENAI_API_KEY` | — | OpenAI-specific key |
| `GOOGLE_API_KEY` | — | Gemini-specific key |

## License

MIT
