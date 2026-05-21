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


@cli.command()
@click.option(
    "--target",
    "-t",
    default="claude",
    show_default=True,
    help=(
        "AI tool to configure. Options: claude, cursor, windsurf, copilot, agents, all. "
        "Use 'all' to write guidelines for every supported tool at once."
    ),
)
def init(target: str):
    """Initialize NexusMemo guidelines in the current project.

    Writes the appropriate config file(s) so the AI assistant knows to use
    NexusMemo MCP tools for persistent memory.

    \b
    Targets and their files:
      claude   → CLAUDE.md
      cursor   → .cursor/rules/nexusmemo.mdc
      windsurf → .windsurfrules
      copilot  → .github/copilot-instructions.md
      agents   → AGENTS.md   (Antigravity, Codex, other agents)
      all      → all of the above
    """
    from nexusmemo.guidelines import write_guidelines

    results = write_guidelines(target=target)
    for line in results:
        click.echo(line)


if __name__ == "__main__":
    cli()

