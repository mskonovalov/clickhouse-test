"""Unit tests for the RAG API."""

import asyncio
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from rag_api.config import Settings, get_settings
from rag_api.embedder import SentenceTransformerEmbedder
from rag_api.main import app, get_embedder, get_repository
from rag_api.repository import VECTOR_SEARCH_SQL, VectorRepository
from rag_api.schemas import DocumentResult

EMBEDDING_DIM = 384


class FakeEncodedVector:
    """Array-like encoded vector double supporting .tolist()."""

    def __init__(self, values: Sequence[float]) -> None:
        self._values = list(values)

    def tolist(self) -> Sequence[float]:
        """Return the underlying float values."""
        return self._values


class FakeSentenceTransformerModel:
    """In-memory SentenceTransformer model double for unit testing SentenceTransformerEmbedder."""

    def __init__(self, vector: Sequence[float] | None = None) -> None:
        self.vector = list(vector) if vector is not None else [0.1] * EMBEDDING_DIM
        self.calls: list[tuple[str, bool, str]] = []

    def encode(
        self,
        sentences: str,
        *,
        convert_to_numpy: bool = True,
    ) -> FakeEncodedVector:
        """Record call arguments and thread name and return a fake vector."""
        self.calls.append((sentences, convert_to_numpy, threading.current_thread().name))
        return FakeEncodedVector(self.vector)


class FakeEmbedder:
    """In-memory async Embedder double for FastAPI route unit tests."""

    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector if vector is not None else [0.1] * EMBEDDING_DIM
        self.queries: list[str] = []
        self.closed = False

    async def embed_query(self, text: str) -> list[float]:
        """Record query text and return the configured embedding vector."""
        self.queries.append(text)
        return self.vector

    def close(self) -> None:
        """Mark the fake embedder as closed."""
        self.closed = True


class FakeQueryResult:
    """In-memory ClickHouse query result double."""

    def __init__(self, rows: list[tuple[str, str]]) -> None:
        self.result_rows = rows


class FakeAsyncClickHouseClient:
    """In-memory async ClickHouse client double for unit testing VectorRepository."""

    def __init__(
        self,
        *,
        ping_result: bool = True,
        ping_error: Exception | None = None,
        rows: list[tuple[str, str]] | None = None,
    ) -> None:
        self.ping_result = ping_result
        self.ping_error = ping_error
        self.rows = rows if rows is not None else [("stub.md", "Stub document body")]
        self.queries: list[tuple[str, dict[str, Any] | None]] = []
        self.closed = False

    async def ping(self) -> bool:
        """Return configured ping result or raise configured error."""
        if self.ping_error is not None:
            raise self.ping_error
        return self.ping_result

    async def query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> FakeQueryResult:
        """Record query and parameters and return configured rows."""
        self.queries.append((query, parameters))
        limit = len(self.rows)
        if parameters and "limit" in parameters:
            limit = int(parameters["limit"])
        return FakeQueryResult(self.rows[:limit])

    async def close(self) -> None:
        """Mark the client as closed."""
        self.closed = True


def test_health_endpoint_returns_ok() -> None:
    """Verify GET /health returns 200 and {"status": "ok"} when repository ping succeeds."""
    fake_client = FakeAsyncClickHouseClient(ping_result=True)
    repo = VectorRepository(client=fake_client)
    app.dependency_overrides[get_repository] = lambda: repo
    try:
        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
    finally:
        app.dependency_overrides.clear()


def test_health_endpoint_returns_503_when_database_unavailable() -> None:
    """Verify GET /health returns 503 when repository.ping() returns False."""
    fake_client = FakeAsyncClickHouseClient(ping_result=False)
    repo = VectorRepository(client=fake_client)
    app.dependency_overrides[get_repository] = lambda: repo
    try:
        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 503
        assert response.json() == {"detail": "Database unavailable"}
    finally:
        app.dependency_overrides.clear()


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


def test_search_with_fake_embedder_and_fake_repository_returns_200() -> None:
    """Verify POST /search embeds query via Embedder and queries VectorRepository."""
    fake_client = FakeAsyncClickHouseClient(
        rows=[("lightdash-and-clickhouse.md", "How to connect Lightdash to ClickHouse")]
    )
    repo = VectorRepository(client=fake_client)
    fake_embedder = FakeEmbedder(vector=[0.1] * EMBEDDING_DIM)
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_embedder] = lambda: fake_embedder
    try:
        client = TestClient(app)
        response = client.post("/search", json={"query": "lightdash"})

        assert response.status_code == 200
        data = response.json()
        assert data == {
            "results": [
                {
                    "title": "lightdash-and-clickhouse.md",
                    "body": "How to connect Lightdash to ClickHouse",
                }
            ]
        }
        assert fake_embedder.queries == ["lightdash"]
        assert len(fake_client.queries) == 1
        _, params = fake_client.queries[0]
        assert params is not None
        assert params["limit"] == 5
        assert params["query_embedding"] == [0.1] * EMBEDDING_DIM
    finally:
        app.dependency_overrides.clear()


def test_search_with_dependency_override_and_default_limit() -> None:
    """Verify POST /search uses default limit=5 and forwards embedding to repository."""
    fake_client = FakeAsyncClickHouseClient(
        rows=[(f"doc-{i}.md", f"Body {i} for lightdash") for i in range(10)]
    )
    repo = VectorRepository(client=fake_client)
    fake_embedder = FakeEmbedder(vector=[0.25] * EMBEDDING_DIM)

    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_embedder] = lambda: fake_embedder
    try:
        client = TestClient(app)
        response = client.post("/search", json={"query": "lightdash"})

        assert response.status_code == 200
        assert fake_embedder.queries == ["lightdash"]
        data = response.json()
        assert len(data["results"]) == 5
        assert data["results"][0] == {"title": "doc-0.md", "body": "Body 0 for lightdash"}
        assert fake_client.queries[0][1] == {
            "query_embedding": [0.25] * EMBEDDING_DIM,
            "limit": 5,
        }
    finally:
        app.dependency_overrides.clear()


def test_search_custom_limit_and_whitespace_stripping() -> None:
    """Verify POST /search strips whitespace on query and forwards custom limit."""
    fake_client = FakeAsyncClickHouseClient(
        rows=[(f"doc-{i}.md", f"Body {i}") for i in range(5)]
    )
    repo = VectorRepository(client=fake_client)
    fake_embedder = FakeEmbedder(vector=[0.5] * EMBEDDING_DIM)

    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_embedder] = lambda: fake_embedder
    try:
        client = TestClient(app)
        response = client.post("/search", json={"query": "  lightdash  ", "limit": 3})

        assert response.status_code == 200
        assert fake_embedder.queries == ["lightdash"]
        assert len(response.json()["results"]) == 3
        assert fake_client.queries[0][1] == {
            "query_embedding": [0.5] * EMBEDDING_DIM,
            "limit": 3,
        }
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
    fake_client = FakeAsyncClickHouseClient()
    repo = VectorRepository(client=fake_client)
    fake_embedder = FakeEmbedder()
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_embedder] = lambda: fake_embedder
    try:
        client = TestClient(app)
        response = client.post("/search", json=payload)

        assert response.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_search_returns_503_when_embedder_not_initialized_on_app_state() -> None:
    """Verify POST /search returns 503 if app.state.embedder is missing."""
    fake_client = FakeAsyncClickHouseClient()
    repo = VectorRepository(client=fake_client)
    app.dependency_overrides[get_repository] = lambda: repo
    if hasattr(app.state, "embedder"):
        delattr(app.state, "embedder")
    try:
        client = TestClient(app)
        response = client.post("/search", json={"query": "lightdash"})

        assert response.status_code == 503
        assert response.json() == {"detail": "Embedder unavailable"}
    finally:
        app.dependency_overrides.clear()


def test_sentence_transformer_embedder_runs_on_worker_thread_and_closes() -> None:
    """Verify SentenceTransformerEmbedder offloads .encode() to its dedicated thread pool."""
    fake_model = FakeSentenceTransformerModel(vector=[0.25, 0.5, 0.75])
    embedder = SentenceTransformerEmbedder(max_workers=2, model=fake_model)
    try:
        result = asyncio.run(embedder.embed_query("  Can I use Lightdash?  "))

        assert result == [0.25, 0.5, 0.75]
        assert len(fake_model.calls) == 1
        sentence, convert_to_numpy, thread_name = fake_model.calls[0]
        assert sentence == "Can I use Lightdash?"
        assert convert_to_numpy is True
        assert thread_name.startswith("embedder")
    finally:
        embedder.close()

    with pytest.raises(RuntimeError):
        asyncio.run(embedder.embed_query("after close"))


def test_sentence_transformer_embedder_validates_inputs() -> None:
    """Verify SentenceTransformerEmbedder validates max_workers >= 1 and non-empty text."""
    fake_model = FakeSentenceTransformerModel()

    with pytest.raises(ValueError, match="max_workers must be at least 1"):
        SentenceTransformerEmbedder(max_workers=0, model=fake_model)

    embedder = SentenceTransformerEmbedder(max_workers=1, model=fake_model)
    try:
        with pytest.raises(ValueError, match="text must not be empty"):
            asyncio.run(embedder.embed_query("   "))
    finally:
        embedder.close()


def test_vector_repository_create_connects_lazily_and_disables_autogenerate_session_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify VectorRepository.create defers connection until first use and sets autogenerate_session_id=False."""
    fake_client = FakeAsyncClickHouseClient()
    mock_get_async_client = AsyncMock(return_value=fake_client)
    monkeypatch.setattr("rag_api.repository.clickhouse_connect.get_async_client", mock_get_async_client)

    settings = Settings(
        clickhouse_host="ch-host",
        clickhouse_port=8123,
        clickhouse_user="ch-user",
        clickhouse_password="ch-password",
        clickhouse_db="ch-db",
    )
    repo = asyncio.run(VectorRepository.create(settings))

    assert isinstance(repo, VectorRepository)
    mock_get_async_client.assert_not_awaited()

    assert asyncio.run(repo.ping()) is True
    mock_get_async_client.assert_awaited_once_with(
        host="ch-host",
        port=8123,
        username="ch-user",
        password="ch-password",
        database="ch-db",
        autogenerate_session_id=False,
    )


def test_lifespan_starts_and_serves_503_when_clickhouse_is_down_at_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify app starts cleanly when ClickHouse is unreachable at boot and returns 503 on /health and /search."""
    mock_get_async_client = AsyncMock(side_effect=ConnectionError("ClickHouse down"))
    monkeypatch.setattr("rag_api.repository.clickhouse_connect.get_async_client", mock_get_async_client)
    fake_embedder = FakeEmbedder()
    monkeypatch.setattr(
        "rag_api.main.SentenceTransformerEmbedder",
        lambda **_kwargs: fake_embedder,
    )

    with TestClient(app) as client:
        health_response = client.get("/health")
        assert health_response.status_code == 503
        assert health_response.json() == {"detail": "Database unavailable"}

        search_response = client.post("/search", json={"query": "lightdash"})
        assert search_response.status_code == 503
        assert search_response.json() == {"detail": "Database unavailable"}

    assert fake_embedder.closed is True


def test_vector_repository_ping_handles_success_failure_and_exceptions() -> None:
    """Verify VectorRepository.ping returns True/False and catches connection errors."""
    ok_repo = VectorRepository(client=FakeAsyncClickHouseClient(ping_result=True))
    false_repo = VectorRepository(client=FakeAsyncClickHouseClient(ping_result=False))
    error_repo = VectorRepository(
        client=FakeAsyncClickHouseClient(ping_error=RuntimeError("Connection refused"))
    )

    assert asyncio.run(ok_repo.ping()) is True
    assert asyncio.run(false_repo.ping()) is False
    assert asyncio.run(error_repo.ping()) is False


def test_vector_repository_search_by_vector_executes_parameterized_cosine_query() -> None:
    """Verify VectorRepository.search_by_vector binds parameters and maps rows to DocumentResult."""
    fake_client = FakeAsyncClickHouseClient(
        rows=[
            ('  {"source_file": "doc-1.md"}', "First chunk body"),
            ("doc-2.md", "Second chunk body"),
        ]
    )
    repo = VectorRepository(client=fake_client)
    query_vector = [0.1] * EMBEDDING_DIM

    results = asyncio.run(repo.search_by_vector(query_vector, limit=2))

    assert results == [
        DocumentResult(title="doc-1.md", body="First chunk body"),
        DocumentResult(title="doc-2.md", body="Second chunk body"),
    ]
    assert len(fake_client.queries) == 1
    sql, params = fake_client.queries[0]
    assert sql == VECTOR_SEARCH_SQL
    assert "cosineDistance(embeddings, {query_embedding:Array(Float32)}) ASC" in sql
    assert "LIMIT {limit:UInt32}" in sql
    assert params == {"query_embedding": query_vector, "limit": 2}


def test_vector_repository_search_by_vector_validates_inputs() -> None:
    """Verify VectorRepository.search_by_vector rejects empty embeddings and invalid limits."""
    repo = VectorRepository(client=FakeAsyncClickHouseClient())

    with pytest.raises(ValueError, match="embedding must not be empty"):
        asyncio.run(repo.search_by_vector([], limit=5))

    with pytest.raises(ValueError, match="limit must be at least 1"):
        asyncio.run(repo.search_by_vector([0.1] * EMBEDDING_DIM, limit=0))


def test_vector_repository_close_closes_underlying_client() -> None:
    """Verify VectorRepository.close delegates to the underlying async client."""
    fake_client = FakeAsyncClickHouseClient()
    repo = VectorRepository(client=fake_client)

    asyncio.run(repo.close())

    assert fake_client.closed is True
