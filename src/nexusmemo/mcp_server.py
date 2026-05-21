"""MCP server for Claude Code and other MCP-compatible clients."""

import logging

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from nexusmemo.core import NexusMemo

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "nexusmemo",
)

# Lazy-initialized singleton
_engine: NexusMemo | None = None


def get_engine() -> NexusMemo:
    global _engine
    if _engine is None:
        _engine = NexusMemo()
    return _engine


from nexusmemo.schemas import Entity, Relation, Decision

@mcp.tool()
def add_memory(
    text: str,
    summary: str,
    entities: list[Entity] | None = Field(default=None, description="[REQUIRED] List of all meaningful tools, technologies, concepts, and people mentioned. You MUST provide this array. If none exist, send []. DO NOT OMIT THIS KEY."),
    relations: list[Relation] | None = Field(default=None, description="[REQUIRED] How the entities relate to each other. You MUST provide this array. If none exist, send []. DO NOT OMIT THIS KEY."),
    decisions: list[Decision] | None = Field(default=None, description="[REQUIRED] Any choices or architectural decisions made in the text. You MUST provide this array. If none exist, send []. DO NOT OMIT THIS KEY.")
) -> str:
    """Store a new memory. Use this to remember project decisions, architecture choices,
    migration history, bug fixes, deployment notes, or any important project context.

    As the AI assistant, you must extract ALL entities, relations, and decisions from the text and pass them to this tool.

    Args:
        text: The raw information or context to remember.
        summary: A one-line summary of the text.
        entities: List of all meaningful tools, technologies, concepts, and people mentioned.
        relations: How the entities relate to each other (e.g. USES, REPLACED_BY, DEPENDS_ON).
        decisions: Any choices or architectural decisions made in the text.
    """
    engine = get_engine()
    result = engine.add_memory(
        text=text, 
        summary=summary, 
        entities=entities, 
        relations=relations, 
        decisions=decisions
    )
    return (
        f"Memory stored (id: {result.memory_id}). "
        f"Saved {result.entities_found} entities, "
        f"{result.relations_found} relations, "
        f"{result.decisions_found} decisions."
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
    """Get NexusMemo status — total memories, entities, relations, and database size.

    Returns:
        System statistics.
    """
    engine = get_engine()
    status = engine.get_status()
    return (
        f"NexusMemo Status:\n"
        f"  Memories: {status.total_memories}\n"
        f"  Entities: {status.total_entities}\n"
        f"  Relations: {status.total_relations}\n"
        f"  Decisions: {status.total_decisions}\n"
        f"  Database: {status.database_size_mb} MB"
    )


@mcp.tool()
def initialize_memory_guidelines(
    target: str = "claude",
) -> str:
    """Initialize NexusMemo guidelines in the current project for the specified AI tool.

    Writes the appropriate configuration file so the AI assistant knows to use
    NexusMemo MCP tools for persistent memory.

    Args:
        target: Which AI tool to configure. Options:
            - "claude"   → CLAUDE.md               (Claude Code)
            - "cursor"   → .cursor/rules/nexusmemo.mdc  (Cursor)
            - "windsurf" → .windsurfrules          (Windsurf)
            - "copilot"  → .github/copilot-instructions.md (GitHub Copilot)
            - "agents"   → AGENTS.md               (Antigravity, Codex, other agents)
            - "all"      → writes all of the above

    Returns:
        Status messages for each file written.
    """
    from nexusmemo.guidelines import write_guidelines

    results = write_guidelines(target=target)
    return "\n".join(results)


def run_server():
    """Entry point for the MCP server."""
    logging.basicConfig(level=logging.INFO)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run_server()
