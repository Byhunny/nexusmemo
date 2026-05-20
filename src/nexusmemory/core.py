"""NexusMemory core — main orchestration class."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from nexusmemory.config import Settings, get_settings
from nexusmemory.database import Database, DecisionRow, EdgeRow, MemoryRow, NodeRow
from nexusmemory.embeddings import EmbeddingService
from nexusmemory.extraction import ExtractionService
from nexusmemory.graph import GraphManager
from nexusmemory.importance import ImportanceService
from nexusmemory.retrieval import RetrievalService
from nexusmemory.schemas import (
    AddMemoryResponse,
    ExtractionResult,
    QueryResponse,
    SearchResult,
    StatusResponse,
)

logger = logging.getLogger(__name__)


def _make_id() -> str:
    return uuid.uuid4().hex[:12]


class NexusMemory:
    """Main entry point — ties all services together."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.db = Database(self.settings.db_path)
        self.embeddings = EmbeddingService(self.settings)
        self.extraction = ExtractionService(self.settings)
        self.graph = GraphManager()
        self.importance = ImportanceService()
        self.retrieval = RetrievalService(
            self.settings, self.db, self.embeddings, self.graph
        )

        # Load existing graph into memory
        self.graph.load_from_db(self.db)
        logger.info("NexusMemory initialized — db: %s", self.settings.db_path)

    def add_memory(self, text: str, session_id: str | None = None) -> AddMemoryResponse:
        """Process and store a new memory.

        Pipeline: text → embed → extract → store → update graph
        """
        memory_id = _make_id()

        # 1. Generate embedding for the raw text
        try:
            embedding = self.embeddings.generate(text)
            embedding_bytes = self.embeddings.to_bytes(embedding)
        except Exception as e:
            logger.warning("Embedding generation failed, storing without: %s", e)
            embedding_bytes = None

        # 2. Extract entities, relations, decisions
        try:
            extraction = self.extraction.extract(text)
        except Exception as e:
            logger.warning("Extraction failed, storing raw only: %s", e)
            extraction = ExtractionResult(summary=text[:200])

        # 3. Store the memory
        token_count = self.importance.estimate_tokens(text)
        with self.db.session() as session:
            memory = MemoryRow(
                id=memory_id,
                raw_text=text,
                summary=extraction.summary,
                embedding=embedding_bytes,
                session_id=session_id,
                processed=True,
                token_count=token_count,
            )
            session.add(memory)

            # 4. Store entities as nodes
            for entity in extraction.entities:
                node_id = self._find_or_create_node(session, entity, extraction.importance)

            # 5. Store relations as edges
            for relation in extraction.relations:
                self._create_edge(session, relation, extraction)

            # 6. Store decisions
            for decision in extraction.decisions:
                dec = DecisionRow(
                    id=_make_id(),
                    what=decision.what,
                    why=decision.why,
                    alternatives_json=str(decision.alternatives),
                    memory_id=memory_id,
                )
                session.add(dec)

            session.commit()

        # 7. Update in-memory graph
        self.graph.load_from_db(self.db)

        return AddMemoryResponse(
            memory_id=memory_id,
            entities_found=len(extraction.entities),
            relations_found=len(extraction.relations),
            decisions_found=len(extraction.decisions),
            summary=extraction.summary,
        )

    def query(self, query: str, limit: int = 5) -> QueryResponse:
        """Query memories with hybrid retrieval."""
        results = self.retrieval.search(query, limit=limit)
        context = self.retrieval.build_context(results)

        search_results = [
            SearchResult(
                memory_id=r.memory_id,
                text=r.text,
                summary=r.summary,
                relevance_score=round(r.score, 4),
                related_entities=r.related_entities,
            )
            for r in results
        ]

        return QueryResponse(
            results=search_results,
            context=context,
            total_found=len(search_results),
        )

    def get_entity(self, name: str) -> dict | None:
        """Get detailed information about an entity."""
        node_id = self.graph.find_node_by_name(name)
        if not node_id:
            return None

        # Update access count
        with self.db.session() as session:
            node = session.get(NodeRow, node_id)
            if node:
                node.access_count += 1
                session.commit()

        data = self.graph.graph.nodes[node_id]
        relations = self.graph.get_all_relations_for_node(node_id)

        return {
            "id": node_id,
            "name": data.get("name", ""),
            "type": data.get("type", "concept"),
            "description": data.get("description", ""),
            "importance": data.get("importance", 0.5),
            "relations": relations,
        }

    def get_decisions(self, topic: str | None = None) -> list[dict]:
        """Get stored decisions, optionally filtered by topic."""
        with self.db.session() as session:
            query = session.query(DecisionRow)
            if topic:
                query = query.filter(
                    DecisionRow.what.ilike(f"%{topic}%")
                    | DecisionRow.why.ilike(f"%{topic}%")
                )
            decisions = query.order_by(DecisionRow.created_at.desc()).all()

            return [
                {
                    "id": d.id,
                    "what": d.what,
                    "why": d.why,
                    "alternatives": d.alternatives,
                    "outcome": d.outcome,
                    "created_at": d.created_at.isoformat() if d.created_at else None,
                }
                for d in decisions
            ]

    def get_status(self) -> StatusResponse:
        """Get system status and statistics."""
        db_size = 0.0
        if self.settings.db_path.exists():
            db_size = self.settings.db_path.stat().st_size / (1024 * 1024)

        with self.db.session() as session:
            return StatusResponse(
                total_memories=session.query(MemoryRow).count(),
                total_entities=session.query(NodeRow).count(),
                total_relations=session.query(EdgeRow).count(),
                total_decisions=session.query(DecisionRow).count(),
                database_size_mb=round(db_size, 2),
            )

    def _find_or_create_node(self, session, entity, importance: float) -> str:
        """Find existing node by name or create a new one."""
        existing = (
            session.query(NodeRow)
            .filter(NodeRow.name.ilike(entity.name))
            .first()
        )

        if existing:
            # Update existing node
            if entity.description and not existing.description:
                existing.description = entity.description
            existing.access_count += 1
            existing.updated_at = datetime.now(timezone.utc)
            existing.importance_score = self.importance.calculate(
                llm_score=importance,
                access_count=existing.access_count,
                connection_count=self.graph.graph.degree(existing.id)
                if self.graph.graph.has_node(existing.id)
                else 0,
                created_at=existing.created_at,
            )
            return existing.id

        node_id = _make_id()

        # Generate embedding for entity
        embed_text = f"{entity.name}: {entity.description}" if entity.description else entity.name
        try:
            emb = self.embeddings.generate(embed_text)
            emb_bytes = self.embeddings.to_bytes(emb)
        except Exception:
            emb_bytes = None

        node = NodeRow(
            id=node_id,
            type=entity.type,
            name=entity.name,
            description=entity.description,
            importance_score=self.importance.calculate(llm_score=importance),
            embedding=emb_bytes,
        )
        session.add(node)
        return node_id

    def _create_edge(self, session, relation, extraction: ExtractionResult) -> None:
        """Create an edge between two nodes."""
        # Find source and target nodes
        source_node = (
            session.query(NodeRow)
            .filter(NodeRow.name.ilike(relation.source))
            .first()
        )
        target_node = (
            session.query(NodeRow)
            .filter(NodeRow.name.ilike(relation.target))
            .first()
        )

        if not source_node or not target_node:
            return

        # Check for duplicate
        existing = (
            session.query(EdgeRow)
            .filter(
                EdgeRow.source_id == source_node.id,
                EdgeRow.target_id == target_node.id,
                EdgeRow.relation == relation.relation,
            )
            .first()
        )
        if existing:
            existing.weight = min(existing.weight + 0.1, 2.0)  # reinforce
            return

        edge = EdgeRow(
            id=_make_id(),
            source_id=source_node.id,
            target_id=target_node.id,
            relation=relation.relation,
            weight=1.0,
            confidence=relation.confidence,
            context=relation.context,
        )
        session.add(edge)
