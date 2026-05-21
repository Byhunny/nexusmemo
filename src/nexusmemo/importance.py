"""Importance scoring for memory nodes."""

from __future__ import annotations

import math
from datetime import datetime, timezone


class ImportanceService:
    """Calculate and update importance scores for entities."""

    # Weight factors for the scoring formula
    WEIGHT_RECENCY = 0.3
    WEIGHT_ACCESS = 0.2
    WEIGHT_CONNECTIONS = 0.2
    WEIGHT_LLM_SCORE = 0.2
    WEIGHT_DECAY = 0.1

    @staticmethod
    def calculate(
        llm_score: float = 0.5,
        access_count: int = 0,
        connection_count: int = 0,
        created_at: datetime | None = None,
        max_access: int = 100,
        max_connections: int = 50,
    ) -> float:
        """Calculate importance score for an entity.

        Formula:
            importance = (
                0.3 × recency +
                0.2 × normalized_access +
                0.2 × normalized_connections +
                0.2 × llm_score +
                0.1 × decay_factor
            )
        """
        # Recency: exponential decay based on age in days
        now = datetime.now(timezone.utc)
        if created_at:
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            age_days = (now - created_at).total_seconds() / 86400
        else:
            age_days = 0
        recency = math.exp(-0.05 * age_days)  # half-life ~14 days

        # Normalize access count (log scale)
        normalized_access = min(math.log1p(access_count) / math.log1p(max_access), 1.0)

        # Normalize connection count
        normalized_connections = min(connection_count / max_connections, 1.0)

        # Decay factor (starts at 1.0, decreases over time without access)
        decay = math.exp(-0.01 * age_days)

        score = (
            ImportanceService.WEIGHT_RECENCY * recency
            + ImportanceService.WEIGHT_ACCESS * normalized_access
            + ImportanceService.WEIGHT_CONNECTIONS * normalized_connections
            + ImportanceService.WEIGHT_LLM_SCORE * llm_score
            + ImportanceService.WEIGHT_DECAY * decay
        )

        return round(min(max(score, 0.0), 1.0), 4)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token count estimation (1 token ≈ 4 chars for English)."""
        return max(1, len(text) // 4)
