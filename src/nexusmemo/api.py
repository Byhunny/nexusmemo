"""FastAPI application for HTTP access to NexusMemo."""

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from nexusmemo.core import NexusMemo
from nexusmemo.schemas import (
    AddMemoryRequest,
    AddMemoryResponse,
    QueryRequest,
    QueryResponse,
    StatusResponse,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="NexusMemo",
    description="Local-first AI memory layer with persistent, structured memory",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-initialized singleton
_engine: NexusMemo | None = None


def get_engine() -> NexusMemo:
    global _engine
    if _engine is None:
        _engine = NexusMemo()
    return _engine


@app.on_event("startup")
def startup():
    get_engine()
    logger.info("NexusMemo API server started")


@app.get("/health")
def health():
    return {"status": "ok", "service": "nexusmemo"}


@app.post("/memory/add", response_model=AddMemoryResponse)
def add_memory(request: AddMemoryRequest):
    """Add a new memory to the system."""
    try:
        engine = get_engine()
        result = engine.add_memory(text=request.text, session_id=request.session_id)
        return result
    except Exception as e:
        logger.error("Failed to add memory: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/query", response_model=QueryResponse)
def query_memory(request: QueryRequest):
    """Query memories with hybrid semantic + graph retrieval."""
    try:
        engine = get_engine()
        result = engine.query(query=request.query, limit=request.limit)
        return result
    except Exception as e:
        logger.error("Query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/status", response_model=StatusResponse)
def get_status():
    """Get system status and statistics."""
    engine = get_engine()
    return engine.get_status()


@app.get("/entity/{name}")
def get_entity(name: str):
    """Get detailed information about a specific entity."""
    engine = get_engine()
    result = engine.get_entity(name)
    if not result:
        raise HTTPException(status_code=404, detail=f"Entity '{name}' not found")
    return result


@app.get("/decisions")
def get_decisions(topic: str | None = None):
    """Get stored decisions, optionally filtered by topic."""
    engine = get_engine()
    return engine.get_decisions(topic=topic)
