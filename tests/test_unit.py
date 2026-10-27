"""Unit tests for the RAG API."""

import asyncio
import math
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from clickhouse_connect.driver.exceptions import DatabaseError, OperationalError
from fastapi.testclient import TestClient
from pydantic import ValidationError

from rag_api.config import Settings, get_settings
from rag_api.embedder import EXPECTED_EMBEDDING_DIM, SentenceTransformerEmbedder
from rag_api.main import app, get_app_settings, get_embedder, get_repository, get_reranker
from rag_api.repository import VECTOR_SEARCH_SQL, VectorRepository
from rag_api.reranker import CrossEncoderReranker
from rag_api.schemas import DocumentResult


class FakeEncodedVector:
    """Array-like encoded vector double supporting .tolist()."""

    def __init__(self, values: Sequence[float]) -> None:
        self._values = list(values)

    def tolist(self) -> Sequence[float]:
        """Return the underlying float values."""
        return self._values


class FakeSentenceTransformerModel:
    """In-memory SentenceTransformer model double for unit testing SentenceTransformerEmbedder."""

    def __init__(
        self,
        vector: Sequence[float] | None = None,
        *,
        reported_dim: int | None = None,
    ) -> None:
        self.vector = (
            list(vector) if vector is not None else [0.1] * EXPECTED_EMBEDDING_DIM
        )
        self._reported_dim = reported_dim
        self.calls: list[tuple[str, bool, str]] = []

    def get_embedding_dimension(self) -> int | None:
        """Return the configured model embedding dimension if set."""
        return self._reported_dim

    def encode(
        self,
        sentences: str,
        *,
        convert_to_numpy: bool = True,
    ) -> FakeEncodedVector:
        """Record call arguments and thread name and return a fake vector."""
        self.calls.append((sentences, convert_to_numpy, threading.current_thread().name))
        return FakeEncodedVector(self.vector)


class FakeCrossEncoderModel:
    """In-memory CrossEncoder model double for unit testing CrossEncoderReranker."""

    def __init__(
        self,
        scores: Sequence[float] | None = None,
        *,
        as_array: bool = True,
    ) -> None:
        self.scores = list(scores) if scores is not None else None
        self.as_array = as_array
        self.calls: list[tuple[list[tuple[str, str]], str]] = []

    def predict(
        self,
        sentences: list[tuple[str, str]],
    ) -> FakeEncodedVector | list[float]:
        """Record sentence pairs and worker thread name and return configured scores."""
        self.calls.append((list(sentences), threading.current_thread().name))
        values = (
            self.scores
            if self.scores is not None
            else [float(len(sentences) - idx) for idx in range(len(sentences))]
        )
        return FakeEncodedVector(values) if self.as_array else list(values)


class FakeEmbedder:
    """In-memory async Embedder double for FastAPI route unit tests."""

    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector if vector is not None else [0.1] * EXPECTED_EMBEDDING_DIM
        self.queries: list[str] = []
        self.closed = False

    async def embed_query(self, text: str) -> list[float]:
        """Record query text and return the configured embedding vector."""
        self.queries.append(text)
        return self.vector

    def close(self) -> None:
        """Mark the fake embedder as closed."""
        self.closed = True


class FakeReranker:
    """In-memory async Reranker double for FastAPI route unit tests."""

    def __init__(self, *, reverse: bool = False) -> None:
        self.reverse = reverse
        self.calls: list[tuple[str, list[DocumentResult], int]] = []
        self.closed = False

    async def rerank(
        self,
        query: str,
        documents: Sequence[DocumentResult],
        top_n: int,
    ) -> list[DocumentResult]:
        """Record call arguments and return top_n documents (optionally reversed)."""
        docs = list(documents)
        self.calls.append((query, docs, top_n))
        ordered = list(reversed(docs)) if self.reverse else docs
        return ordered[:top_n]

    def close(self) -> None:
        """Mark the fake reranker as closed."""
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
        query_error: Exception | None = None,
        rows: list[tuple[str, str]] | None = None,
    ) -> None:
        self.ping_result = ping_result
        self.ping_error = ping_error
        self.query_error = query_error
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
        """Record query and parameters and return configured rows or raise query_error."""
        self.queries.append((query, parameters))
        if self.query_error is not None:
            raise self.query_error
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
    assert settings.reranker_model == "custom/reranker"
    assert settings.reranker_max_workers == 4
    assert settings.rerank_candidate_multiplier == 6
    assert settings.rerank_min_candidates == 30

    with pytest.raises(ValidationError):
        Settings(reranker_max_workers=0)
    with pytest.raises(ValidationError):
        Settings(rerank_candidate_multiplier=0)
    with pytest.raises(ValidationError):
        Settings(rerank_min_candidates=0)


def test_search_with_fake_embedder_and_fake_repository_returns_200() -> None:
    """Verify POST /search embeds query via Embedder, over-fetches from VectorRepository, and reranks."""
    fake_client = FakeAsyncClickHouseClient(
        rows=[("lightdash-and-clickhouse.md", "How to connect Lightdash to ClickHouse")]
    )
    repo = VectorRepository(client=fake_client)
    fake_embedder = FakeEmbedder(vector=[0.1] * EXPECTED_EMBEDDING_DIM)
    fake_reranker = FakeReranker()
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_embedder] = lambda: fake_embedder
    app.dependency_overrides[get_reranker] = lambda: fake_reranker
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
        assert params["limit"] == 20
        assert params["query_embedding"] == [0.1] * EXPECTED_EMBEDDING_DIM
        assert len(fake_reranker.calls) == 1
        assert fake_reranker.calls[0][0] == "lightdash"
        assert fake_reranker.calls[0][2] == 5
    finally:
        app.dependency_overrides.clear()


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
    try:
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
    finally:
        app.dependency_overrides.clear()


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
    try:
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
    finally:
        app.dependency_overrides.clear()


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
    try:
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
    finally:
        app.dependency_overrides.clear()


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
    try:
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
    fake_reranker = FakeReranker()
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_reranker] = lambda: fake_reranker
    if hasattr(app.state, "embedder"):
        delattr(app.state, "embedder")
    try:
        client = TestClient(app)
        response = client.post("/search", json={"query": "lightdash"})

        assert response.status_code == 503
        assert response.json() == {"detail": "Embedder unavailable"}
    finally:
        app.dependency_overrides.clear()


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
    try:
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
    finally:
        app.dependency_overrides.clear()


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
    try:
        client = TestClient(app)
        response = client.post("/search", json={"query": "lightdash"})

        assert response.status_code == 503
        assert response.json() == {"detail": "Database unavailable"}
    finally:
        app.dependency_overrides.clear()


def test_vector_repository_propagates_non_connectivity_database_errors() -> None:
    """Verify non-connectivity DatabaseError from ClickHouse is not masked as DatabaseUnavailableError."""
    fake_client = FakeAsyncClickHouseClient(
        query_error=DatabaseError("Code: 43. DB::Exception: Illegal types")
    )
    repo = VectorRepository(client=fake_client)

    with pytest.raises(DatabaseError, match="Illegal types"):
        asyncio.run(repo.search_by_vector([0.1] * EXPECTED_EMBEDDING_DIM, limit=5))


def test_sentence_transformer_embedder_runs_on_worker_thread_and_closes() -> None:
    """Verify SentenceTransformerEmbedder offloads .encode() to its dedicated thread pool."""
    expected_vector = [0.25] * EXPECTED_EMBEDDING_DIM
    fake_model = FakeSentenceTransformerModel(
        vector=expected_vector,
        reported_dim=EXPECTED_EMBEDDING_DIM,
    )
    embedder = SentenceTransformerEmbedder(max_workers=2, model=fake_model)
    try:
        result = asyncio.run(embedder.embed_query("  Can I use Lightdash?  "))

        assert result == expected_vector
        assert len(fake_model.calls) == 1
        sentence, convert_to_numpy, thread_name = fake_model.calls[0]
        assert sentence == "Can I use Lightdash?"
        assert convert_to_numpy is True
        assert thread_name.startswith("embedder")
    finally:
        embedder.close()

    with pytest.raises(RuntimeError):
        asyncio.run(embedder.embed_query("after close"))


def test_sentence_transformer_embedder_validates_inputs_and_dimensions() -> None:
    """Verify SentenceTransformerEmbedder validates max_workers, expected_dim, model dim, and output dim."""
    fake_model = FakeSentenceTransformerModel()

    with pytest.raises(ValueError, match="max_workers must be at least 1"):
        SentenceTransformerEmbedder(max_workers=0, model=fake_model)

    with pytest.raises(ValueError, match="expected_dim must be at least 1"):
        SentenceTransformerEmbedder(expected_dim=0, model=fake_model)

    wrong_dim_model = FakeSentenceTransformerModel(reported_dim=768)
    with pytest.raises(ValueError, match="produces 768-dimensional embeddings, expected 384"):
        SentenceTransformerEmbedder(model=wrong_dim_model)

    wrong_output_model = FakeSentenceTransformerModel(vector=[0.1, 0.2, 0.3])
    embedder = SentenceTransformerEmbedder(max_workers=1, model=wrong_output_model)
    try:
        with pytest.raises(ValueError, match="text must not be empty"):
            asyncio.run(embedder.embed_query("   "))

        with pytest.raises(ValueError, match="Expected 384-dimensional embedding, got 3"):
            asyncio.run(embedder.embed_query("lightdash"))
    finally:
        embedder.close()


def test_cross_encoder_reranker_runs_on_worker_thread_sorts_deterministically_and_closes() -> None:
    """Verify CrossEncoderReranker offloads .predict() to a reranker thread, sorts by score with tie-breaking, and slices to top_n."""
    docs = [
        DocumentResult(title="doc-0.md", body="First stage-1 chunk"),
        DocumentResult(title="doc-1.md", body="Second stage-1 chunk"),
        DocumentResult(title="doc-2.md", body="Third stage-1 chunk"),
        DocumentResult(title="doc-3.md", body="Fourth stage-1 chunk"),
    ]
    fake_model = FakeCrossEncoderModel(scores=[0.2, 0.9, 0.9, 1.5])
    reranker = CrossEncoderReranker(max_workers=2, model=fake_model)
    try:
        ranked = asyncio.run(reranker.rerank("  lightdash query  ", docs, top_n=3))

        assert ranked == [docs[3], docs[1], docs[2]]
        assert len(fake_model.calls) == 1
        pairs, thread_name = fake_model.calls[0]
        assert pairs == [
            ("lightdash query", "First stage-1 chunk"),
            ("lightdash query", "Second stage-1 chunk"),
            ("lightdash query", "Third stage-1 chunk"),
            ("lightdash query", "Fourth stage-1 chunk"),
        ]
        assert thread_name.startswith("reranker")
    finally:
        reranker.close()

    with pytest.raises(RuntimeError):
        asyncio.run(reranker.rerank("after close", docs, top_n=2))


def test_cross_encoder_reranker_short_circuits_empty_and_single_document() -> None:
    """Verify CrossEncoderReranker returns early for 0 or 1 candidate documents without calling .predict()."""
    fake_model = FakeCrossEncoderModel(scores=[1.0], as_array=False)
    reranker = CrossEncoderReranker(max_workers=1, model=fake_model)
    try:
        single_doc = DocumentResult(title="only.md", body="Single candidate")

        assert asyncio.run(reranker.rerank("lightdash", [], top_n=5)) == []
        assert asyncio.run(reranker.rerank("lightdash", [single_doc], top_n=5)) == [single_doc]
        assert fake_model.calls == []
    finally:
        reranker.close()


def test_cross_encoder_reranker_validates_inputs_and_scores() -> None:
    """Verify CrossEncoderReranker validates max_workers, query, top_n, score length, and score finiteness."""
    fake_model = FakeCrossEncoderModel()

    with pytest.raises(ValueError, match="max_workers must be at least 1"):
        CrossEncoderReranker(max_workers=0, model=fake_model)

    docs = [
        DocumentResult(title="a.md", body="Body A"),
        DocumentResult(title="b.md", body="Body B"),
    ]

    mismatch_model = FakeCrossEncoderModel(scores=[0.5], as_array=False)
    reranker = CrossEncoderReranker(max_workers=1, model=mismatch_model)
    try:
        with pytest.raises(ValueError, match="query must not be empty"):
            asyncio.run(reranker.rerank("   ", docs, top_n=1))

        with pytest.raises(ValueError, match="top_n must be at least 1"):
            asyncio.run(reranker.rerank("lightdash", docs, top_n=0))

        with pytest.raises(ValueError, match="Expected 2 reranker scores, got 1"):
            asyncio.run(reranker.rerank("lightdash", docs, top_n=2))
    finally:
        reranker.close()

    for bad_score in (math.nan, math.inf, -math.inf):
        non_finite_model = FakeCrossEncoderModel(scores=[1.0, bad_score])
        bad_reranker = CrossEncoderReranker(max_workers=1, model=non_finite_model)
        try:
            with pytest.raises(ValueError, match="Reranker produced non-finite score"):
                asyncio.run(bad_reranker.rerank("lightdash", docs, top_n=2))
        finally:
            bad_reranker.close()


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
    query_vector = [0.1] * EXPECTED_EMBEDDING_DIM

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
    """Verify VectorRepository.search_by_vector rejects empty/wrong-dimension embeddings and invalid limits."""
    repo = VectorRepository(client=FakeAsyncClickHouseClient())

    with pytest.raises(ValueError, match="embedding must not be empty"):
        asyncio.run(repo.search_by_vector([], limit=5))

    with pytest.raises(ValueError, match="embedding must have length 384, got 3"):
        asyncio.run(repo.search_by_vector([0.1, 0.2, 0.3], limit=5))

    with pytest.raises(ValueError, match="limit must be at least 1"):
        asyncio.run(repo.search_by_vector([0.1] * EXPECTED_EMBEDDING_DIM, limit=0))


def test_vector_repository_close_closes_underlying_client() -> None:
    """Verify VectorRepository.close delegates to the underlying async client."""
    fake_client = FakeAsyncClickHouseClient()
    repo = VectorRepository(client=fake_client)

    asyncio.run(repo.close())

    assert fake_client.closed is True
