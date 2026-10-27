"""Unit tests for the RAG API."""

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag_api.config import Settings, get_settings
from rag_api.main import app, get_search_handler
from rag_api.schemas import DocumentResult


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


def test_search_with_default_stub_returns_200() -> None:
    """Verify POST /search with the default stub returns 200 and matching schema."""
    client = TestClient(app)
    response = client.post("/search", json={"query": "lightdash"})

    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert isinstance(data["results"], list)
    assert len(data["results"]) >= 1
    assert set(data["results"][0].keys()) == {"title", "body"}


def test_search_with_dependency_override_and_default_limit() -> None:
    """Verify POST /search uses default limit=5 and returns results from overridden dependency."""
    captured_calls: list[tuple[str, int]] = []

    async def fake_search_handler(query: str, limit: int) -> list[DocumentResult]:
        captured_calls.append((query, limit))
        return [
            DocumentResult(title=f"doc-{i}.md", body=f"Body {i} for {query}")
            for i in range(limit)
        ]

    app.dependency_overrides[get_search_handler] = lambda: fake_search_handler
    try:
        client = TestClient(app)
        response = client.post("/search", json={"query": "lightdash"})

        assert response.status_code == 200
        assert captured_calls == [("lightdash", 5)]
        data = response.json()
        assert len(data["results"]) == 5
        assert data["results"][0] == {"title": "doc-0.md", "body": "Body 0 for lightdash"}
    finally:
        app.dependency_overrides.clear()


def test_search_custom_limit_and_whitespace_stripping() -> None:
    """Verify POST /search strips whitespace on query and forwards custom limit."""
    captured_calls: list[tuple[str, int]] = []

    async def fake_search_handler(query: str, limit: int) -> list[DocumentResult]:
        captured_calls.append((query, limit))
        return [
            DocumentResult(title=f"doc-{i}.md", body=f"Body {i}")
            for i in range(limit)
        ]

    app.dependency_overrides[get_search_handler] = lambda: fake_search_handler
    try:
        client = TestClient(app)
        response = client.post("/search", json={"query": "  lightdash  ", "limit": 3})

        assert response.status_code == 200
        assert captured_calls == [("lightdash", 3)]
        assert len(response.json()["results"]) == 3
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "   "},
        {},
        {"query": "lightdash", "limit": 0},
        {"query": "lightdash", "limit": -1},
        {"query": "lightdash", "limit": 51},
        {"query": 123},
        {"query": ["lightdash"]},
        {"query": "lightdash", "limit": "not-an-int"},
        {"query": "lightdash", "limit": 3.5},
    ],
)
def test_search_validation_errors_return_422(payload: dict[str, Any]) -> None:
    """Verify invalid POST /search payloads are rejected with HTTP 422."""
    client = TestClient(app)
    response = client.post("/search", json=payload)

    assert response.status_code == 422
