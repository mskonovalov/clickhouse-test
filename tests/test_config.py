"""Unit tests for application configuration (rag_api.config)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_api.config import Settings, get_settings


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify default values on Settings when no environment or .env overrides are set."""
    monkeypatch.chdir(tmp_path)
    for env_var in (
        "CLICKHOUSE_HOST",
        "CLICKHOUSE_PORT",
        "CLICKHOUSE_DB",
        "CLICKHOUSE_USER",
        "CLICKHOUSE_PASSWORD",
        "EMBEDDING_MODEL",
        "EMBEDDER_MAX_WORKERS",
        "RERANKER_ENABLED",
        "RERANKER_MODEL",
        "RERANKER_MAX_WORKERS",
        "RERANK_CANDIDATE_MULTIPLIER",
        "RERANK_MIN_CANDIDATES",
    ):
        monkeypatch.delenv(env_var, raising=False)

    settings = get_settings()

    assert settings.clickhouse_host == "localhost"
    assert settings.clickhouse_port == 8123
    assert settings.clickhouse_db == "embeddings"
    assert settings.clickhouse_user == "vectordbuser"
    assert settings.clickhouse_password == "pwd_vectordb"
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.embedder_max_workers == 4
    assert settings.reranker_enabled is True
    assert settings.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert settings.reranker_max_workers == 2
    assert settings.rerank_candidate_multiplier == 4
    assert settings.rerank_min_candidates == 20


def test_settings_env_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify environment variables override Settings defaults and validate positive integers."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLICKHOUSE_HOST", "clickhouse-prod")
    monkeypatch.setenv("CLICKHOUSE_PORT", "9000")
    monkeypatch.setenv("CLICKHOUSE_DB", "custom_db")
    monkeypatch.setenv("CLICKHOUSE_USER", "custom_user")
    monkeypatch.setenv("CLICKHOUSE_PASSWORD", "custom_pwd")
    monkeypatch.setenv("EMBEDDING_MODEL", "custom/model")
    monkeypatch.setenv("EMBEDDER_MAX_WORKERS", "8")
    monkeypatch.setenv("RERANKER_ENABLED", "false")
    monkeypatch.setenv("RERANKER_MODEL", "custom/reranker")
    monkeypatch.setenv("RERANKER_MAX_WORKERS", "4")
    monkeypatch.setenv("RERANK_CANDIDATE_MULTIPLIER", "6")
    monkeypatch.setenv("RERANK_MIN_CANDIDATES", "30")

    settings = Settings()

    assert settings.clickhouse_host == "clickhouse-prod"
    assert settings.clickhouse_port == 9000
    assert settings.clickhouse_db == "custom_db"
    assert settings.clickhouse_user == "custom_user"
    assert settings.clickhouse_password == "custom_pwd"
    assert settings.embedding_model == "custom/model"
    assert settings.embedder_max_workers == 8
    assert settings.reranker_enabled is False
    assert settings.reranker_model == "custom/reranker"
    assert settings.reranker_max_workers == 4
    assert settings.rerank_candidate_multiplier == 6
    assert settings.rerank_min_candidates == 30

    with pytest.raises(ValidationError):
        Settings(embedder_max_workers=0)
    with pytest.raises(ValidationError):
        Settings(reranker_max_workers=0)
    with pytest.raises(ValidationError):
        Settings(rerank_candidate_multiplier=0)
    with pytest.raises(ValidationError):
        Settings(rerank_min_candidates=0)
