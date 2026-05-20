"""MCP server for Claude Code and other MCP-compatible clients."""

import logging

from mcp.server.fastmcp import FastMCP

from memoryos.core import MemoryOS

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "memoryos",
    description="AI memory layer — persistent, structured memory for LLMs",
)

# Lazy-initialized singleton
_engine: MemoryOS | None = None


def get_engine() -> MemoryOS:
    global _engine
    if _engine is None:
        _engine = MemoryOS()
    return _engine


@mcp.tool()
def add_memory(text: str) -> str:
    """Store a new memory. Use this to remember project decisions, architecture choices,
    migration history, bug fixes, deployment notes, or any important project context.

    Args:
        text: The information to remember. Be descriptive and include context.

    Returns:
        Summary of what was extracted and stored.
    """
    engine = get_engine()
    result = engine.add_memory(text)
    return (
        f"Memory stored (id: {result.memory_id}). "
        f"Extracted {result.entities_found} entities, "
        f"{result.relations_found} relations, "
        f"{result.decisions_found} decisions. "
        f"Summary: {result.summary}"
    )


@mcp.tool()
def search_memory(query: str, limit: int = 5) -> str:
    """Search through stored memories using semantic search and knowledge graph.
    Use this to recall past decisions, find related context, or check project history.

    Args:
        query: What you want to find. Use natural language.
        limit: Maximum number of results to return.

    Returns:
        Relevant memories and compressed context.
    """
    engine = get_engine()
    result = engine.query(query=query, limit=limit)

    if not result.results:
        return "No relevant memories found for this query."

    return result.context


@mcp.tool()
def get_project_context(topic: str) -> str:
    """Get comprehensive project context about a specific topic. This searches memories
    AND traverses the knowledge graph to find related entities and decisions.

    Args:
        topic: The topic to get context about (e.g., "database", "deployment", "authentication").

    Returns:
        Rich context including memories, entity relationships, and decisions.
    """
    engine = get_engine()

    # Query memories
    query_result = engine.query(query=topic, limit=5)
    context_parts = [query_result.context]

    # Get entity info if available
    entity = engine.get_entity(topic)
    if entity:
        relations = entity.get("relations", [])
        if relations:
            context_parts.append(f"\n### Entity: {entity['name']} ({entity['type']})")
            if entity.get("description"):
                context_parts.append(f"Description: {entity['description']}")
            for rel in relations[:10]:
                direction = "→" if rel["direction"] == "outgoing" else "←"
                context_parts.append(
                    f"  {direction} {rel['relation']} {rel['entity']}"
                )

    # Get relevant decisions
    decisions = engine.get_decisions(topic=topic)
    if decisions:
        context_parts.append("\n### Related Decisions")
        for dec in decisions[:5]:
            context_parts.append(f"- **{dec['what']}**")
            if dec.get("why"):
                context_parts.append(f"  Reason: {dec['why']}")
            if dec.get("alternatives"):
                context_parts.append(f"  Alternatives considered: {', '.join(dec['alternatives'])}")

    return "\n".join(context_parts)


@mcp.tool()
def find_related(entity_name: str) -> str:
    """Find all entities related to a given entity in the knowledge graph.

    Args:
        entity_name: Name of the entity to look up (e.g., "PostgreSQL", "Docker", "React").

    Returns:
        List of related entities with their relationships.
    """
    engine = get_engine()
    entity = engine.get_entity(entity_name)

    if not entity:
        return f"Entity '{entity_name}' not found in the knowledge graph."

    lines = [f"## {entity['name']} ({entity['type']})"]
    if entity.get("description"):
        lines.append(f"_{entity['description']}_\n")

    relations = entity.get("relations", [])
    if not relations:
        lines.append("No known relationships.")
        return "\n".join(lines)

    outgoing = [r for r in relations if r["direction"] == "outgoing"]
    incoming = [r for r in relations if r["direction"] == "incoming"]

    if outgoing:
        lines.append("**Outgoing:**")
        for r in outgoing:
            lines.append(f"  → {r['relation']} → {r['entity']}")

    if incoming:
        lines.append("**Incoming:**")
        for r in incoming:
            lines.append(f"  ← {r['relation']} ← {r['entity']}")

    return "\n".join(lines)


@mcp.tool()
def get_decisions(topic: str = "") -> str:
    """Get project decisions, optionally filtered by topic.

    Args:
        topic: Optional topic filter. Leave empty for all decisions.

    Returns:
        List of decisions with reasoning and alternatives.
    """
    engine = get_engine()
    decisions = engine.get_decisions(topic=topic if topic else None)

    if not decisions:
        return "No decisions recorded" + (f" for topic '{topic}'" if topic else "") + "."

    lines = ["## Project Decisions\n"]
    for dec in decisions:
        lines.append(f"### {dec['what']}")
        if dec.get("why"):
            lines.append(f"**Why:** {dec['why']}")
        if dec.get("alternatives"):
            alts = dec["alternatives"]
            if isinstance(alts, list) and alts:
                lines.append(f"**Alternatives:** {', '.join(alts)}")
        if dec.get("outcome"):
            lines.append(f"**Outcome:** {dec['outcome']}")
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
def memory_status() -> str:
    """Get MemoryOS status — total memories, entities, relations, and database size.

    Returns:
        System statistics.
    """
    engine = get_engine()
    status = engine.get_status()
    return (
        f"MemoryOS Status:\n"
        f"  Memories: {status.total_memories}\n"
        f"  Entities: {status.total_entities}\n"
        f"  Relations: {status.total_relations}\n"
        f"  Decisions: {status.total_decisions}\n"
        f"  Database: {status.database_size_mb} MB"
    )


def run_server():
    """Entry point for the MCP server."""
    logging.basicConfig(level=logging.INFO)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run_server()
