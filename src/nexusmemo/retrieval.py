"""Retrieval pipeline — semantic search, graph expansion, reranking, compression."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from nexusmemo.config import Settings
from nexusmemo.database import Database, MemoryRow, NodeRow
from nexusmemo.embeddings import EmbeddingService
from nexusmemo.graph import GraphManager

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single result from the retrieval pipeline."""

    memory_id: str
    text: str
    summary: str | None
    score: float
    source: str = "semantic"  # semantic, graph, hybrid
    related_entities: list[str] = field(default_factory=list)


class RetrievalService:
    """Hybrid retrieval: semantic search + graph expansion + reranking."""

    def __init__(
        self,
        settings: Settings,
        db: Database,
        embedding_service: EmbeddingService,
        graph_manager: GraphManager,
    ) -> None:
        self._settings = settings
        self._db = db
        self._embeddings = embedding_service
        self._graph = graph_manager

    def search(self, query: str, limit: int | None = None) -> list[RetrievalResult]:
        """Execute the full retrieval pipeline."""
        limit = limit or self._settings.max_results

        # Step 1: Embed the query
        query_embedding = self._embeddings.generate(query)

        # Step 2: Semantic search over memories
        memory_results = self._semantic_search_memories(query_embedding, limit * 2)

        # Step 3: Semantic search over nodes
        node_matches = self._semantic_search_nodes(query_embedding, limit)

        # Step 4: Graph expansion — find neighbors of matching nodes
        expanded_entity_ids = set()
        for node_id, _ in node_matches:
            neighbors = self._graph.get_neighbors(
                node_id, depth=self._settings.graph_expansion_depth
            )
            for neighbor in neighbors:
                expanded_entity_ids.add(neighbor["id"])

        # Step 5: Enrich memory results with related entities
        for result in memory_results:
            result.related_entities = self._find_related_entities(result.text, node_matches)

        # Step 6: Rerank by combined score
        reranked = self._rerank(memory_results, node_matches)

        return reranked[:limit]

    def build_context(self, results: list[RetrievalResult], max_tokens: int = 2000) -> str:
        """Compress retrieval results into an injectable context string."""
        if not results:
            return "No relevant memories found."

        lines = ["## Relevant Project Memory\n"]
        token_count = 10  # header tokens

        for result in results:
            entry_lines = []
            summary = result.summary or result.text[:200]
            entry_lines.append(f"- **Memory**: {summary}")

            if result.related_entities:
                entities_str = ", ".join(result.related_entities[:5])
                entry_lines.append(f"  Related: {entities_str}")

            entry_text = "\n".join(entry_lines)
            entry_tokens = len(entry_text) // 4

            if token_count + entry_tokens > max_tokens:
                break

            lines.extend(entry_lines)
            token_count += entry_tokens

        # Add graph context for top entities
        top_entity_ids = []
        for result in results[:3]:
            for entity_name in result.related_entities[:2]:
                node_id = self._graph.find_node_by_name(entity_name)
                if node_id:
                    top_entity_ids.append(node_id)

        if top_entity_ids:
            graph_text = self._graph.get_subgraph_text(top_entity_ids)
            if graph_text and token_count + len(graph_text) // 4 < max_tokens:
                lines.append("\n### Knowledge Graph")
                lines.append(graph_text)

        return "\n".join(lines)

    def _semantic_search_memories(
        self, query_embedding: np.ndarray, limit: int
    ) -> list[RetrievalResult]:
        """Search memories by embedding similarity."""
        results = []

        with self._db.session() as session:
            memories = session.query(MemoryRow).filter(
                MemoryRow.embedding.isnot(None)
            ).all()

            for memory in memories:
                mem_embedding = self._embeddings.from_bytes(memory.embedding)
                score = self._embeddings.cosine_similarity(query_embedding, mem_embedding)

                if score >= self._settings.similarity_threshold:
                    results.append(
                        RetrievalResult(
                            memory_id=memory.id,
                            text=memory.raw_text,
                            summary=memory.summary,
                            score=score,
                            source="semantic",
                        )
                    )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def _semantic_search_nodes(
        self, query_embedding: np.ndarray, limit: int
    ) -> list[tuple[str, float]]:
        """Search nodes by embedding similarity. Returns (node_id, score) pairs."""
        scored = []

        with self._db.session() as session:
            nodes = session.query(NodeRow).filter(NodeRow.embedding.isnot(None)).all()

            for node in nodes:
                node_embedding = self._embeddings.from_bytes(node.embedding)
                score = self._embeddings.cosine_similarity(query_embedding, node_embedding)

                if score >= self._settings.similarity_threshold:
                    scored.append((node.id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def _find_related_entities(
        self, text: str, node_matches: list[tuple[str, float]]
    ) -> list[str]:
        """Find entity names related to a memory text."""
        related = []
        text_lower = text.lower()

        for node_id, _ in node_matches[:10]:
            if self._graph.graph.has_node(node_id):
                name = self._graph.graph.nodes[node_id].get("name", "")
                if name.lower() in text_lower:
                    related.append(name)

        return related

    def _rerank(
        self,
        memory_results: list[RetrievalResult],
        node_matches: list[tuple[str, float]],
    ) -> list[RetrievalResult]:
        """Rerank results by combining semantic score with graph importance."""
        # Build importance map from node matches
        importance_map = {}
        for node_id, score in node_matches:
            if self._graph.graph.has_node(node_id):
                name = self._graph.graph.nodes[node_id].get("name", "")
                importance = self._graph.graph.nodes[node_id].get("importance", 0.5)
                importance_map[name.lower()] = importance * score

        # Boost memory results that mention important entities
        for result in memory_results:
            boost = 0.0
            text_lower = result.text.lower()
            for name, imp_score in importance_map.items():
                if name in text_lower:
                    boost = max(boost, imp_score * 0.2)
            result.score = min(result.score + boost, 1.0)

        memory_results.sort(key=lambda r: r.score, reverse=True)
        return memory_results
