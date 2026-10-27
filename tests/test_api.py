"""Unit tests for FastAPI routes and lifespan management (rag_api.main)."""

from typing import Any
from unittest.mock import AsyncMock

import pytest
from clickhouse_connect.driver.exceptions import OperationalError
from fastapi.testclient import TestClient

from rag_api.config import Settings
from rag_api.embedder import EXPECTED_EMBEDDING_DIM
from rag_api.main import app, get_app_settings, get_embedder, get_repository, get_reranker
from rag_api.repository import VectorRepository
from conftest import FakeAsyncClickHouseClient, FakeEmbedder, FakeReranker


def test_health_endpoint_returns_ok() -> None:
    """Verify GET /health returns 200 and {"status": "ok"} when repository ping succeeds."""
    fake_client = FakeAsyncClickHouseClient(ping_result=True)
    repo = VectorRepository(client=fake_client)
    app.dependency_overrides[get_repository] = lambda: repo

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_endpoint_returns_503_when_database_unavailable() -> None:
    """Verify GET /health returns 503 when repository.ping() returns False."""
    fake_client = FakeAsyncClickHouseClient(ping_result=False)
    repo = VectorRepository(client=fake_client)
    app.dependency_overrides[get_repository] = lambda: repo

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"detail": "Database unavailable"}


def test_search_with_dependency_override_and_default_limit() -> None:
    """Verify POST /search with default rerank=True over-fetches 20 candidates and reranks to top 5."""
    fake_client = FakeAsyncClickHouseClient(
        rows=[(f"doc-{i}.md", f"Body {i} for lightdash") for i in range(25)]
    )
    repo = VectorRepository(client=fake_client)
    fake_embedder = FakeEmbedder(vector=[0.25] * EXPECTED_EMBEDDING_DIM)
    fake_reranker = FakeReranker(reverse=True)

    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_embedder] = lambda: fake_embedder
    app.dependency_overrides[get_reranker] = lambda: fake_reranker

    client = TestClient(app)
    response = client.post("/search", json={"query": "lightdash"})

    assert response.status_code == 200
    assert fake_embedder.queries == ["lightdash"]
    data = response.json()
    assert len(data["results"]) == 5
    assert data["results"][0] == {"title": "doc-19.md", "body": "Body 19 for lightdash"}
    assert fake_client.queries[0][1] == {
        "query_embedding": [0.25] * EXPECTED_EMBEDDING_DIM,
        "limit": 20,
    }
    assert len(fake_reranker.calls) == 1
    query, candidates, top_n = fake_reranker.calls[0]
    assert query == "lightdash"
    assert len(candidates) == 20
    assert top_n == 5


def test_search_with_rerank_false_bypasses_reranker_and_queries_exact_limit() -> None:
    """Verify POST /search with rerank=False fetches request.limit directly and does not call reranker."""
    fake_client = FakeAsyncClickHouseClient(
        rows=[(f"doc-{i}.md", f"Body {i}") for i in range(10)]
    )
    repo = VectorRepository(client=fake_client)
    fake_embedder = FakeEmbedder(vector=[0.5] * EXPECTED_EMBEDDING_DIM)
    fake_reranker = FakeReranker(reverse=True)

    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_embedder] = lambda: fake_embedder
    app.dependency_overrides[get_reranker] = lambda: fake_reranker

    client = TestClient(app)
    response = client.post(
        "/search",
        json={"query": "  lightdash  ", "limit": 3, "rerank": False},
    )

    assert response.status_code == 200
    assert fake_embedder.queries == ["lightdash"]
    results = response.json()["results"]
    assert len(results) == 3
    assert results[0] == {"title": "doc-0.md", "body": "Body 0"}
    assert fake_client.queries[0][1] == {
        "query_embedding": [0.5] * EXPECTED_EMBEDDING_DIM,
        "limit": 3,
    }
    assert fake_reranker.calls == []


def test_search_custom_limit_and_whitespace_stripping() -> None:
    """Verify POST /search strips whitespace on query and computes candidate_limit from settings."""
    fake_client = FakeAsyncClickHouseClient(
        rows=[(f"doc-{i}.md", f"Body {i}") for i in range(50)]
    )
    repo = VectorRepository(client=fake_client)
    fake_embedder = FakeEmbedder(vector=[0.5] * EXPECTED_EMBEDDING_DIM)
    fake_reranker = FakeReranker()
    custom_settings = Settings(rerank_candidate_multiplier=3, rerank_min_candidates=10)

    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_embedder] = lambda: fake_embedder
    app.dependency_overrides[get_reranker] = lambda: fake_reranker
    app.dependency_overrides[get_app_settings] = lambda: custom_settings

    client = TestClient(app)
    response = client.post("/search", json={"query": "  lightdash  ", "limit": 8})

    assert response.status_code == 200
    assert fake_embedder.queries == ["lightdash"]
    assert len(response.json()["results"]) == 8
    assert fake_client.queries[0][1] == {
        "query_embedding": [0.5] * EXPECTED_EMBEDDING_DIM,
        "limit": 24,
    }
    assert len(fake_reranker.calls) == 1
    assert fake_reranker.calls[0][2] == 8


def test_search_rerank_min_candidates_floor_applies_for_small_limit() -> None:
    """Verify POST /search enforces rerank_min_candidates floor when limit * multiplier < min_candidates."""
    fake_client = FakeAsyncClickHouseClient(
        rows=[(f"doc-{i}.md", f"Body {i}") for i in range(25)]
    )
    repo = VectorRepository(client=fake_client)
    fake_embedder = FakeEmbedder(vector=[0.5] * EXPECTED_EMBEDDING_DIM)
    fake_reranker = FakeReranker(reverse=True)

    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_embedder] = lambda: fake_embedder
    app.dependency_overrides[get_reranker] = lambda: fake_reranker

    client = TestClient(app)
    response = client.post("/search", json={"query": "lightdash", "limit": 1})

    assert response.status_code == 200
    results = response.json()["results"]
    assert results == [{"title": "doc-19.md", "body": "Body 19"}]
    assert fake_client.queries[0][1] == {
        "query_embedding": [0.5] * EXPECTED_EMBEDDING_DIM,
        "limit": 20,
    }
    assert len(fake_reranker.calls) == 1
    query, candidates, top_n = fake_reranker.calls[0]
    assert query == "lightdash"
    assert len(candidates) == 20
    assert top_n == 1


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
        {"query": "lightdash", "rerank": "not-a-bool"},
        {"query": "lightdash", "rerank": "false"},
        {"query": "lightdash", "rerank": 0},
        {"query": "lightdash", "rerank": 1},
        {"query": "lightdash", "rerank": None},
    ],
)
def test_search_validation_errors_return_422(payload: dict[str, Any]) -> None:
    """Verify invalid POST /search payloads are rejected with HTTP 422."""
    fake_client = FakeAsyncClickHouseClient()
    repo = VectorRepository(client=fake_client)
    fake_embedder = FakeEmbedder()
    fake_reranker = FakeReranker()
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_embedder] = lambda: fake_embedder
    app.dependency_overrides[get_reranker] = lambda: fake_reranker

    client = TestClient(app)
    response = client.post("/search", json=payload)

    assert response.status_code == 422


def test_search_returns_503_when_embedder_not_initialized_on_app_state() -> None:
    """Verify POST /search returns 503 if app.state.embedder is missing."""
    fake_client = FakeAsyncClickHouseClient()
    repo = VectorRepository(client=fake_client)
    fake_reranker = FakeReranker()
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_reranker] = lambda: fake_reranker
    if hasattr(app.state, "embedder"):
        delattr(app.state, "embedder")

    client = TestClient(app)
    response = client.post("/search", json={"query": "lightdash"})

    assert response.status_code == 503
    assert response.json() == {"detail": "Embedder unavailable"}


def test_search_returns_503_when_reranker_not_initialized_on_app_state() -> None:
    """Verify POST /search returns 503 if app.state.reranker is missing when rerank=True, but succeeds when rerank=False."""
    fake_client = FakeAsyncClickHouseClient(
        rows=[("lightdash-and-clickhouse.md", "How to connect Lightdash to ClickHouse")]
    )
    repo = VectorRepository(client=fake_client)
    fake_embedder = FakeEmbedder()
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_embedder] = lambda: fake_embedder
    if hasattr(app.state, "reranker"):
        delattr(app.state, "reranker")

    client = TestClient(app)
    response = client.post("/search", json={"query": "lightdash"})

    assert response.status_code == 503
    assert response.json() == {"detail": "Reranker unavailable"}
    assert fake_embedder.queries == []
    assert fake_client.queries == []

    fallback_response = client.post(
        "/search",
        json={"query": "lightdash", "rerank": False},
    )
    assert fallback_response.status_code == 200
    assert fallback_response.json() == {
        "results": [
            {
                "title": "lightdash-and-clickhouse.md",
                "body": "How to connect Lightdash to ClickHouse",
            }
        ]
    }


def test_search_returns_503_when_clickhouse_query_fails_with_operational_error() -> None:
    """Verify POST /search returns 503 when client.query raises OperationalError mid-run."""
    fake_client = FakeAsyncClickHouseClient(
        query_error=OperationalError("Connection reset by peer")
    )
    repo = VectorRepository(client=fake_client)
    fake_embedder = FakeEmbedder(vector=[0.1] * EXPECTED_EMBEDDING_DIM)
    fake_reranker = FakeReranker()
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_embedder] = lambda: fake_embedder
    app.dependency_overrides[get_reranker] = lambda: fake_reranker

    client = TestClient(app)
    response = client.post("/search", json={"query": "lightdash"})

    assert response.status_code == 503
    assert response.json() == {"detail": "Database unavailable"}


def test_lifespan_starts_and_serves_503_when_clickhouse_is_down_at_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify app starts cleanly when ClickHouse is unreachable at boot and returns 503 on /health and /search."""
    mock_get_async_client = AsyncMock(side_effect=ConnectionError("ClickHouse down"))
    monkeypatch.setattr(
        "rag_api.repository.clickhouse_connect.get_async_client",
        mock_get_async_client,
    )
    fake_embedder = FakeEmbedder()
    fake_reranker = FakeReranker()
    monkeypatch.setattr(
        "rag_api.main.SentenceTransformerEmbedder",
        lambda **_kwargs: fake_embedder,
    )
    monkeypatch.setattr(
        "rag_api.main.CrossEncoderReranker",
        lambda **_kwargs: fake_reranker,
    )

    with TestClient(app) as client:
        health_response = client.get("/health")
        assert health_response.status_code == 503
        assert health_response.json() == {"detail": "Database unavailable"}

        search_response = client.post("/search", json={"query": "lightdash"})
        assert search_response.status_code == 503
        assert search_response.json() == {"detail": "Database unavailable"}

    assert fake_embedder.closed is True
    assert fake_reranker.closed is True
