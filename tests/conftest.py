"""Shared test doubles and fixtures for the RAG API test suite."""

import threading
from collections.abc import Iterator, Sequence
from typing import Any

import pytest

from rag_api.embedder import EXPECTED_EMBEDDING_DIM
from rag_api.main import app
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
        self.vector = list(vector) if vector is not None else [0.1] * EXPECTED_EMBEDDING_DIM
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
            self.scores if self.scores is not None else [float(len(sentences) - idx) for idx in range(len(sentences))]
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


@pytest.fixture(autouse=True)
def clear_dependency_overrides() -> Iterator[None]:
    """Ensure FastAPI dependency overrides are cleared after every test."""
    yield
    app.dependency_overrides.clear()
