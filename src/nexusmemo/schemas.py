"""Pydantic domain models and API schemas."""

from datetime import datetime
from pydantic import BaseModel, Field


# ─── Domain Models ───────────────────────────────────────────────


class Entity(BaseModel):
    """An entity extracted from text."""

    name: str
    type: str = "concept"  # tool, concept, person, project, language, framework, etc.
    description: str = ""


class Relation(BaseModel):
    """A relationship between two entities."""

    source: str
    target: str
    relation: str  # USES, REPLACED_BY, DEPENDS_ON, CAUSED_BY, RELATED_TO, WORKED_WITH
    context: str = ""
    confidence: float = 1.0


class Decision(BaseModel):
    """A project decision with reasoning."""

    what: str
    why: str = ""
    alternatives: list[str] = Field(default_factory=list)


# ─── API Schemas ─────────────────────────────────────────────────


class AddMemoryRequest(BaseModel):
    """Request to add a new memory."""

    text: str
    session_id: str | None = None


class AddMemoryResponse(BaseModel):
    """Response after adding a memory."""

    memory_id: str
    entities_found: int
    relations_found: int
    decisions_found: int
    summary: str


class QueryRequest(BaseModel):
    """Request to query memories."""

    query: str
    limit: int = 5


class SearchResult(BaseModel):
    """A single search result."""

    memory_id: str
    text: str
    summary: str | None = None
    relevance_score: float
    related_entities: list[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    """Response to a memory query."""

    results: list[SearchResult]
    context: str  # compressed context ready for LLM injection
    total_found: int


class EntityInfo(BaseModel):
    """Detailed entity information."""

    id: str
    name: str
    type: str
    description: str
    importance_score: float
    connections: list[str] = Field(default_factory=list)


class GraphNeighbor(BaseModel):
    """A neighbor in the knowledge graph."""

    entity: str
    relation: str
    direction: str  # "outgoing" or "incoming"


class StatusResponse(BaseModel):
    """System status information."""

    total_memories: int
    total_entities: int
    total_relations: int
    total_decisions: int
    database_size_mb: float
