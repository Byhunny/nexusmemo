"""Database models and session management."""

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    LargeBinary,
    Real,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""

    pass


class MemoryRow(Base):
    """Raw memory storage — incoming text before processing."""

    __tablename__ = "memories"

    id = Column(Text, primary_key=True)
    raw_text = Column(Text, nullable=False)
    summary = Column(Text)
    embedding = Column(LargeBinary)
    session_id = Column(Text)
    processed = Column(Boolean, default=False)
    token_count = Column(Integer)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class NodeRow(Base):
    """Knowledge graph node — an entity extracted from memories."""

    __tablename__ = "nodes"

    id = Column(Text, primary_key=True)
    type = Column(Text, nullable=False)  # tool, concept, person, project, etc.
    name = Column(Text, nullable=False)
    description = Column(Text)
    importance_score = Column(Real, default=0.5)
    decay_score = Column(Real, default=1.0)
    access_count = Column(Integer, default=0)
    embedding = Column(LargeBinary)
    source = Column(Text, default="extraction")
    metadata_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    @property
    def metadata(self) -> dict:
        return json.loads(self.metadata_json or "{}")

    @metadata.setter
    def metadata(self, value: dict) -> None:
        self.metadata_json = json.dumps(value)


class EdgeRow(Base):
    """Knowledge graph edge — a relationship between two nodes."""

    __tablename__ = "edges"

    id = Column(Text, primary_key=True)
    source_id = Column(Text, nullable=False)
    target_id = Column(Text, nullable=False)
    relation = Column(Text, nullable=False)  # USES, REPLACED_BY, DEPENDS_ON, etc.
    weight = Column(Real, default=1.0)
    confidence = Column(Real, default=1.0)
    context = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class DecisionRow(Base):
    """Architectural or project decision with reasoning."""

    __tablename__ = "decisions"

    id = Column(Text, primary_key=True)
    what = Column(Text, nullable=False)
    why = Column(Text)
    alternatives_json = Column(Text, default="[]")
    outcome = Column(Text)
    memory_id = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    @property
    def alternatives(self) -> list[str]:
        return json.loads(self.alternatives_json or "[]")

    @alternatives.setter
    def alternatives(self, value: list[str]) -> None:
        self.alternatives_json = json.dumps(value)


class Database:
    """Database connection and session management."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self.engine)
        self._session_factory = sessionmaker(bind=self.engine)

    def session(self) -> Session:
        """Create a new database session."""
        return self._session_factory()
