"""Embedding generation and similarity search."""

import numpy as np
from fastembed import TextEmbedding

from nexusmemo.config import Settings


class EmbeddingService:
    """Generate and compare text embeddings using local ONNX model."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: TextEmbedding | None = None

    @property
    def model(self) -> TextEmbedding:
        if self._model is None:
            # fastembed automatically downloads and caches the model on first use
            self._model = TextEmbedding(model_name=self._settings.embedding_model)
        return self._model

    def generate(self, text: str) -> np.ndarray:
        """Generate embedding vector for a text string."""
        # embed() returns a generator of embeddings
        embeddings = list(self.model.embed([text]))
        return np.array(embeddings[0], dtype=np.float32)

    def generate_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts in one pass."""
        if not texts:
            return []
        
        embeddings = list(self.model.embed(texts))
        return [np.array(emb, dtype=np.float32) for emb in embeddings]

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
    def from_bytes(data: bytes, dimensions: int = 384) -> np.ndarray:
        """Deserialize embedding from bytes."""
        # Optional: infer dimensions from length: len(data) // 4
        return np.frombuffer(data, dtype=np.float32).copy()

