# NexusMemo

Local-first AI memory layer that gives LLMs persistent, structured memory.

> "We are building the memory operating system that turns AI from a temporary assistant into a long-term collaborator."

## Problem

LLMs are stateless. They forget previous sessions, lose architectural context, repeat mistakes, and require repeated explanations. NexusMemo fixes this by building a persistent knowledge graph from your conversations.

## How It Works

```
conversation → entity extraction → relationship graph → retrieval → prompt injection
```

When you tell NexusMemo something, it:
1. **Extracts** entities, relationships, and decisions via Claude (Client-side)
2. **Embeds** the text for semantic search
3. **Builds** a knowledge graph with NetworkX
4. **Stores** everything in a local SQLite database

When you query, it:
1. **Searches** semantically similar memories
2. **Expands** through the knowledge graph
3. **Reranks** by importance and relevance
4. **Compresses** into context ready for LLM injection

## Quick Start

### Install

```bash
pip install -e .
```

### Configure

```bash
cp .env.example .env
# Optional configuration, but everything works locally by default
```

### Use via CLI

```bash
# Add a memory
nexusmemo add "We replaced Airflow with Dagster because DAG maintenance became difficult"

# Search memories
nexusmemo search "What orchestration tool do we use?"

# Check status
nexusmemo status

# Get entity info
nexusmemo entity "Dagster"
```

### Use via API

```bash
# Start the server
nexusmemo serve

# Add memory
curl -X POST http://localhost:8765/memory/add \
  -H "Content-Type: application/json" \
  -d '{"text": "We migrated from MongoDB to PostgreSQL for better ACID compliance"}'

# Query
curl -X POST http://localhost:8765/memory/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What database do we use?"}'
```

### Use with Claude Code (MCP)

The easiest way to run the MCP server is via `uvx` directly from PyPI.

```bash
# Add as MCP server (will automatically download and run the latest version)
claude mcp add nexusmemo -- uvx nexusmemo@latest mcp
```

**Troubleshooting Path Issues:**
If Claude Code fails to find `uvx` (due to PATH environment issues), provide the absolute path to your `uvx` executable. You can find it by running `which uvx` in your terminal.

```bash
# Example with absolute path (replace with your actual path)
claude mcp add nexusmemo -- ~/.local/bin/uvx nexusmemo@latest mcp
```

Available MCP tools:
- `add_memory` — Store project context
- `search_memory` — Query past memories
- `get_project_context` — Get rich context about a topic
- `find_related` — Explore knowledge graph relationships
- `get_decisions` — Review past decisions
- `memory_status` — Check system stats

## Architecture

```
┌──────────────┐
│   Interfaces │  CLI / FastAPI / MCP Server
├──────────────┤
│     Core     │  NexusMemo orchestration class
├──────────────┤
│   Services   │  Extraction, Embedding, Graph, Importance
├──────────────┤
│   Storage    │  SQLite + NetworkX (in-memory)
└──────────────┘
```

## Tech Stack

- **Python 3.11+** with FastAPI
- **SQLite** for persistent storage
- **NetworkX** for in-memory knowledge graph
- **FastEmbed (ONNX)** for local semantic search (Zero API Keys)
- **MCP SDK** for native Claude Code integration

## Database

Four tables:
- `memories` — Raw text with embeddings
- `nodes` — Knowledge graph entities
- `edges` — Relationships between entities
- `decisions` — Project decisions with reasoning

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run API server in dev mode
nexusmemo serve
```

## License

MIT
