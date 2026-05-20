"""MemoryOS configuration management."""

from pathlib import Path
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- Data ---
    data_dir: Path = Field(default_factory=lambda: Path.home() / ".memoryos")

    # --- LLM ---
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    llm_provider: str = "openai"
    extraction_model: str = "gpt-4o-mini"

    # --- Embeddings ---
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # --- Retrieval ---
    max_results: int = 10
    similarity_threshold: float = 0.3
    graph_expansion_depth: int = 2

    # --- Server ---
    host: str = "127.0.0.1"
    port: int = 8765

    model_config = {
        "env_prefix": "MEMORYOS_",
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
