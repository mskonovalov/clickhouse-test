"""Unit tests for the async ClickHouse VectorRepository (rag_api.repository)."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from clickhouse_connect.driver.exceptions import DatabaseError
from conftest import FakeAsyncClickHouseClient

from rag_api.config import Settings
from rag_api.embedder import EXPECTED_EMBEDDING_DIM
from rag_api.repository import VECTOR_SEARCH_SQL, VectorRepository
from rag_api.schemas import DocumentResult


def test_vector_repository_create_connects_lazily_and_disables_autogenerate_session_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify VectorRepository.create defers connection until first use and sets autogenerate_session_id=False."""
    fake_client = FakeAsyncClickHouseClient()
    mock_get_async_client = AsyncMock(return_value=fake_client)
    monkeypatch.setattr(
        "rag_api.repository.clickhouse_connect.get_async_client",
        mock_get_async_client,
    )

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


def test_vector_repository_ping_handles_success_failure_and_exceptions() -> None:
    """Verify VectorRepository.ping returns True/False and catches connection errors."""
    ok_repo = VectorRepository(client=FakeAsyncClickHouseClient(ping_result=True))
    false_repo = VectorRepository(client=FakeAsyncClickHouseClient(ping_result=False))
    error_repo = VectorRepository(client=FakeAsyncClickHouseClient(ping_error=RuntimeError("Connection refused")))

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


def test_vector_repository_propagates_non_connectivity_database_errors() -> None:
    """Verify non-connectivity DatabaseError from ClickHouse is not masked as DatabaseUnavailableError."""
    fake_client = FakeAsyncClickHouseClient(query_error=DatabaseError("Code: 43. DB::Exception: Illegal types"))
    repo = VectorRepository(client=fake_client)

    with pytest.raises(DatabaseError, match="Illegal types"):
        asyncio.run(repo.search_by_vector([0.1] * EXPECTED_EMBEDDING_DIM, limit=5))


def test_vector_repository_close_closes_underlying_client() -> None:
    """Verify VectorRepository.close delegates to the underlying async client."""
    fake_client = FakeAsyncClickHouseClient()
    repo = VectorRepository(client=fake_client)

    asyncio.run(repo.close())

    assert fake_client.closed is True
