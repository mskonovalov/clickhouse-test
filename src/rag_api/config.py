"""Application configuration managed via pydantic-settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven configuration for the RAG API."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    clickhouse_host: str = "localhost"
    clickhouse_port: int = 8123
    clickhouse_db: str = "embeddings"
    clickhouse_user: str = "vectordbuser"
    clickhouse_password: str = "pwd_vectordb"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedder_max_workers: int = Field(default=4, ge=1)
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_max_workers: int = Field(default=2, ge=1)
    rerank_candidate_multiplier: int = Field(default=4, ge=1)
    rerank_min_candidates: int = Field(default=20, ge=1)


def get_settings() -> Settings:
    """Load application settings from environment variables and .env file."""
    return Settings()
