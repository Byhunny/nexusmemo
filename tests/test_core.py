"""Tests for NexusMemory core functionality."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nexusmemory.config import Settings
from nexusmemory.database import Base, Database, MemoryRow, NodeRow, EdgeRow, DecisionRow
from nexusmemory.embeddings import EmbeddingService
from nexusmemory.graph import GraphManager
from nexusmemory.importance import ImportanceService
from nexusmemory.schemas import ExtractionResult, Entity, Relation, Decision


# ─── Database Tests ──────────────────────────────────────────────


class TestDatabase:
    def test_create_tables(self, tmp_path):
        db = Database(tmp_path / "test.db")
        with db.session() as session:
            # tables should exist
            assert session.query(MemoryRow).count() == 0
            assert session.query(NodeRow).count() == 0
            assert session.query(EdgeRow).count() == 0
            assert session.query(DecisionRow).count() == 0

    def test_insert_and_query_memory(self, tmp_path):
        db = Database(tmp_path / "test.db")
        with db.session() as session:
            memory = MemoryRow(
                id="test-1",
                raw_text="We use PostgreSQL for the main database",
                summary="PostgreSQL used as main DB",
                processed=True,
            )
            session.add(memory)
            session.commit()

        with db.session() as session:
            result = session.get(MemoryRow, "test-1")
            assert result is not None
            assert result.raw_text == "We use PostgreSQL for the main database"
            assert result.processed is True

    def test_insert_node_and_edge(self, tmp_path):
        db = Database(tmp_path / "test.db")
        with db.session() as session:
            node1 = NodeRow(id="n1", type="tool", name="PostgreSQL")
            node2 = NodeRow(id="n2", type="tool", name="SQLite")
            edge = EdgeRow(
                id="e1",
                source_id="n1",
                target_id="n2",
                relation="REPLACED_BY",
            )
            session.add_all([node1, node2, edge])
            session.commit()

        with db.session() as session:
            assert session.query(NodeRow).count() == 2
            assert session.query(EdgeRow).count() == 1
            e = session.get(EdgeRow, "e1")
            assert e.relation == "REPLACED_BY"


# ─── Embedding Tests ─────────────────────────────────────────────


class TestEmbeddings:
    def test_cosine_similarity_identical(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert EmbeddingService.cosine_similarity(a, a) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert EmbeddingService.cosine_similarity(a, b) == pytest.approx(0.0)

    def test_cosine_similarity_opposite(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        assert EmbeddingService.cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_serialize_deserialize(self):
        original = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        as_bytes = EmbeddingService.to_bytes(original)
        restored = EmbeddingService.from_bytes(as_bytes, dimensions=4)
        np.testing.assert_array_almost_equal(original, restored)

    def test_zero_vector(self):
        a = np.zeros(3, dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert EmbeddingService.cosine_similarity(a, b) == 0.0


# ─── Graph Tests ─────────────────────────────────────────────────


class TestGraphManager:
    def test_add_and_find_node(self):
        gm = GraphManager()
        gm.add_node("n1", name="PostgreSQL", node_type="tool")
        assert gm.find_node_by_name("PostgreSQL") == "n1"
        assert gm.find_node_by_name("postgresql") == "n1"  # case insensitive
        assert gm.find_node_by_name("MySQL") is None

    def test_add_edge_and_get_relations(self):
        gm = GraphManager()
        gm.add_node("n1", name="Airflow", node_type="tool")
        gm.add_node("n2", name="Dagster", node_type="tool")
        gm.add_edge("n1", "n2", relation="REPLACED_BY")

        rels = gm.get_all_relations_for_node("n1")
        assert len(rels) == 1
        assert rels[0]["entity"] == "Dagster"
        assert rels[0]["relation"] == "REPLACED_BY"
        assert rels[0]["direction"] == "outgoing"

    def test_get_neighbors(self):
        gm = GraphManager()
        gm.add_node("n1", name="FastAPI", node_type="framework")
        gm.add_node("n2", name="Pydantic", node_type="library")
        gm.add_node("n3", name="SQLAlchemy", node_type="library")
        gm.add_edge("n1", "n2", relation="DEPENDS_ON")
        gm.add_edge("n1", "n3", relation="USES")

        neighbors = gm.get_neighbors("n1", depth=1)
        names = [n["name"] for n in neighbors]
        assert "Pydantic" in names
        assert "SQLAlchemy" in names

    def test_get_subgraph_text(self):
        gm = GraphManager()
        gm.add_node("n1", name="React", node_type="framework")
        gm.add_node("n2", name="TypeScript", node_type="language")
        gm.add_edge("n1", "n2", relation="BUILT_WITH")

        text = gm.get_subgraph_text(["n1"])
        assert "React" in text
        assert "BUILT_WITH" in text
        assert "TypeScript" in text

    def test_load_from_db(self, tmp_path):
        db = Database(tmp_path / "test.db")
        with db.session() as session:
            session.add(NodeRow(id="n1", type="tool", name="Docker"))
            session.add(NodeRow(id="n2", type="service", name="AWS"))
            session.add(
                EdgeRow(
                    id="e1",
                    source_id="n1",
                    target_id="n2",
                    relation="DEPLOYED_ON",
                )
            )
            session.commit()

        gm = GraphManager()
        gm.load_from_db(db)
        assert gm.graph.number_of_nodes() == 2
        assert gm.graph.number_of_edges() == 1
        assert gm.find_node_by_name("Docker") == "n1"


# ─── Importance Tests ────────────────────────────────────────────


class TestImportance:
    def test_basic_score(self):
        score = ImportanceService.calculate(llm_score=0.8)
        assert 0.0 <= score <= 1.0

    def test_high_access_increases_score(self):
        low = ImportanceService.calculate(access_count=0)
        high = ImportanceService.calculate(access_count=50)
        assert high > low

    def test_many_connections_increases_score(self):
        low = ImportanceService.calculate(connection_count=0)
        high = ImportanceService.calculate(connection_count=20)
        assert high > low

    def test_estimate_tokens(self):
        text = "hello world this is a test"
        tokens = ImportanceService.estimate_tokens(text)
        assert tokens > 0
        assert tokens < len(text)
