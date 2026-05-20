"""Embedding generation and similarity search."""

import numpy as np
from openai import OpenAI

from nexusmemo.config import Settings


class EmbeddingService:
    """Generate and compare text embeddings."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self._settings.openai_api_key)
        return self._client

    def generate(self, text: str) -> np.ndarray:
        """Generate embedding vector for a text string."""
        response = self.client.embeddings.create(
            model=self._settings.embedding_model,
            input=text,
            dimensions=self._settings.embedding_dimensions,
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def generate_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts in one API call."""
        if not texts:
            return []

        response = self.client.embeddings.create(
            model=self._settings.embedding_model,
            input=texts,
            dimensions=self._settings.embedding_dimensions,
        )
        return [
            np.array(item.embedding, dtype=np.float32)
            for item in sorted(response.data, key=lambda x: x.index)
        ]

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def to_bytes(embedding: np.ndarray) -> bytes:
        """Serialize embedding to bytes for SQLite storage."""
        return embedding.tobytes()

    @staticmethod
    def from_bytes(data: bytes, dimensions: int = 1536) -> np.ndarray:
        """Deserialize embedding from bytes."""
        return np.frombuffer(data, dtype=np.float32).copy()
