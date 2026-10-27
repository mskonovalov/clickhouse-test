"""Unit tests for the RAG API."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_api.config import Settings, get_settings
from rag_api.main import app


def test_health_endpoint_returns_ok() -> None:
    """Verify GET /health returns 200 and {"status": "ok"}."""
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


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


def test_settings_env_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify environment variables override Settings defaults."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLICKHOUSE_HOST", "clickhouse-prod")
    monkeypatch.setenv("CLICKHOUSE_PORT", "9000")
    monkeypatch.setenv("CLICKHOUSE_DB", "custom_db")
    monkeypatch.setenv("CLICKHOUSE_USER", "custom_user")
    monkeypatch.setenv("CLICKHOUSE_PASSWORD", "custom_pwd")
    monkeypatch.setenv("EMBEDDING_MODEL", "custom/model")
    monkeypatch.setenv("EMBEDDER_MAX_WORKERS", "8")

    settings = Settings()

    assert settings.clickhouse_host == "clickhouse-prod"
    assert settings.clickhouse_port == 9000
    assert settings.clickhouse_db == "custom_db"
    assert settings.clickhouse_user == "custom_user"
    assert settings.clickhouse_password == "custom_pwd"
    assert settings.embedding_model == "custom/model"
    assert settings.embedder_max_workers == 8
