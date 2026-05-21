"""NexusMemo configuration management."""

from pathlib import Path
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- Data ---
    data_dir: Path = Field(default_factory=lambda: Path.home() / ".nexusmemo")

    # --- Embeddings ---
    embedding_provider: str = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384

    # --- Retrieval ---
    max_results: int = 10
    similarity_threshold: float = 0.3
    graph_expansion_depth: int = 2

    # --- Server ---
    host: str = "127.0.0.1"
    port: int = 8765

    model_config = {
        "env_prefix": "NEXUSMEMO_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @property
    def db_path(self) -> Path:
        return self.data_dir / "memory.db"

    def ensure_dirs(self) -> None:
        """Create data directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    settings.ensure_dirs()
    return settings
