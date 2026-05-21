"""CLI entry point for NexusMemo."""

import json
import logging

import click
import uvicorn

from nexusmemo.config import get_settings


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose: bool):
    """NexusMemo — persistent, structured memory for AI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@cli.command()
@click.option("--host", default=None, help="Host to bind to")
@click.option("--port", default=None, type=int, help="Port to bind to")
def serve(host: str | None, port: int | None):
    """Start the NexusMemo HTTP API server."""
    settings = get_settings()
    uvicorn.run(
        "nexusmemo.api:app",
        host=host or settings.host,
        port=port or settings.port,
        reload=False,
    )


@cli.command()
def mcp():
    """Start the MCP server (stdio transport) for Claude Code / Cursor."""
    from nexusmemo.mcp_server import run_server

    run_server()


@cli.command()
@click.argument("text")
@click.option("--session", "-s", default=None, help="Session ID")
def add(text: str, session: str | None):
    """Add a memory from the command line."""
    from nexusmemo.core import NexusMemo

    engine = NexusMemo()
    result = engine.add_memory(
        text=text, 
        summary=text[:200],
        entities=[], 
        relations=[], 
        decisions=[], 
        session_id=session
    )
    click.echo(f"✓ Memory stored: {result.memory_id}")
    click.echo(f"  Entities: {result.entities_found}")
    click.echo(f"  Relations: {result.relations_found}")
    click.echo(f"  Decisions: {result.decisions_found}")
    click.echo(f"  Summary: {result.summary}")


@cli.command()
@click.argument("query")
@click.option("--limit", "-n", default=5, help="Max results")
@click.option("--context", "-c", is_flag=True, help="Show compressed context instead of results")
def search(query: str, limit: int, context: bool):
    """Search through stored memories."""
    from nexusmemo.core import NexusMemo

    engine = NexusMemo()
    result = engine.query(query=query, limit=limit)

    if context:
        click.echo(result.context)
        return

    if not result.results:
        click.echo("No relevant memories found.")
        return

    for i, r in enumerate(result.results, 1):
        click.echo(f"\n--- Result {i} (score: {r.relevance_score:.3f}) ---")
        click.echo(f"  {r.text[:200]}")
        if r.summary:
            click.echo(f"  Summary: {r.summary}")
        if r.related_entities:
            click.echo(f"  Entities: {', '.join(r.related_entities)}")


@cli.command()
def status():
    """Show NexusMemo status."""
    from nexusmemo.core import NexusMemo

    engine = NexusMemo()
    s = engine.get_status()
    click.echo("NexusMemo Status")
    click.echo(f"  Memories:  {s.total_memories}")
    click.echo(f"  Entities:  {s.total_entities}")
    click.echo(f"  Relations: {s.total_relations}")
    click.echo(f"  Decisions: {s.total_decisions}")
    click.echo(f"  DB size:   {s.database_size_mb} MB")
    click.echo(f"  DB path:   {get_settings().db_path}")


@cli.command()
@click.argument("name")
def entity(name: str):
    """Get information about a specific entity."""
    from nexusmemo.core import NexusMemo

    engine = NexusMemo()
    result = engine.get_entity(name)
    if not result:
        click.echo(f"Entity '{name}' not found.")
        return

    click.echo(f"\n{result['name']} ({result['type']})")
    if result.get("description"):
        click.echo(f"  {result['description']}")
    click.echo(f"  Importance: {result['importance']:.3f}")

    for rel in result.get("relations", []):
        arrow = "→" if rel["direction"] == "outgoing" else "←"
        click.echo(f"  {arrow} {rel['relation']} {rel['entity']}")


CLAUDE_MD_TEMPLATE = """
# NexusMemo (Persistent Memory) Usage Guidelines

As an AI assistant in this project, you are expected to actively and proactively use the `nexusmemo` MCP plugin to maintain long-term context.

1. **Recall Context Before Acting:** Before starting a complex task, making architectural changes, or fixing a deep bug, use the `search_memory` or `get_project_context` tools to retrieve historical decisions, known issues, and architectural rules.
2. **Proactively Save Important Decisions:** Whenever we make an architectural decision, add a new library, resolve a critical bug, or define a new business rule, immediately use the `add_memory` tool to save this information to the persistent memory without asking for my permission.
3. **Maintain a Rich Knowledge Graph:** When adding memories, write clear and descriptive sentences so the underlying system can successfully extract specific Entities and Relations to build a highly connected knowledge graph.
"""

@cli.command()
def init():
    """Initialize NexusMemo in the current project (creates/updates CLAUDE.md)."""
    import os
    
    claude_md_path = os.path.join(os.getcwd(), "CLAUDE.md")
    
    if os.path.exists(claude_md_path):
        with open(claude_md_path, "r") as f:
            content = f.read()
            
        if "NexusMemo" in content:
            click.echo("✓ NexusMemo guidelines already exist in CLAUDE.md")
            return
            
        with open(claude_md_path, "a") as f:
            f.write("\n" + CLAUDE_MD_TEMPLATE)
        click.echo("✓ Appended NexusMemo guidelines to existing CLAUDE.md")
    else:
        with open(claude_md_path, "w") as f:
            f.write(CLAUDE_MD_TEMPLATE.strip() + "\n")
        click.echo("✓ Created CLAUDE.md with NexusMemo guidelines")


if __name__ == "__main__":
    cli()
