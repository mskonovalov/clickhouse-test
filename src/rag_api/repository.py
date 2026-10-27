"""Async ClickHouse repository for vector similarity search."""

import asyncio
import json
from collections.abc import Sequence
from typing import Any, Protocol, Self

import clickhouse_connect
from clickhouse_connect.driver.exceptions import OperationalError

from rag_api.config import Settings
from rag_api.embedder import EXPECTED_EMBEDDING_DIM
from rag_api.schemas import DocumentResult

VECTOR_SEARCH_SQL = """
SELECT metadata AS title, document AS body
FROM vectors
ORDER BY cosineDistance(embeddings, {query_embedding:Array(Float32)}) ASC
LIMIT {limit:UInt32}
""".strip()


def _extract_title(raw_metadata: Any) -> str:
    """Extract the document title from a ClickHouse metadata value.

    Supports both JSON strings such as '{"source_file": "lightdash-and-clickhouse.md"}'
    and plain filename strings.
    """
    text = str(raw_metadata)
    if text.lstrip().startswith("{"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and isinstance(parsed.get("source_file"), str):
                return parsed["source_file"]
        except json.JSONDecodeError:
            pass
    return text


class DatabaseUnavailableError(RuntimeError):
    """Raised when ClickHouse cannot be reached or a query fails due to connectivity."""


class QueryResultProtocol(Protocol):
    """Protocol for ClickHouse query result objects."""

    @property
    def result_rows(self) -> Sequence[Sequence[Any]]:
        """Rows returned by the ClickHouse query."""
        ...


class AsyncClickHouseClient(Protocol):
    """Protocol for the async ClickHouse client used by VectorRepository."""

    async def ping(self) -> bool:
        """Check connectivity to ClickHouse."""
        ...

    async def query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> QueryResultProtocol:
        """Execute a parameterized SELECT query."""
        ...

    async def close(self) -> None:
        """Close the underlying client connection."""
        ...


class VectorRepository:
    """Async repository for querying document embeddings in ClickHouse."""

    def __init__(
        self,
        client: AsyncClickHouseClient | None = None,
        *,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the repository with either an injected client or settings for lazy connection."""
        if client is None and settings is None:
            raise ValueError("Either client or settings must be provided")
        self._client = client
        self._settings = settings
        self._lock = asyncio.Lock()

    @classmethod
    async def create(cls, settings: Settings) -> Self:
        """Create a VectorRepository configured for lazy async ClickHouse connection.

        Connection establishment is deferred until the first ping() or search_by_vector()
        call so application startup succeeds even if ClickHouse is temporarily unreachable,
        allowing GET /health to serve HTTP 503 during startup outages and recover automatically.
        When connecting, autogenerate_session_id=False is always set so concurrent async
        requests do not share a single ClickHouse session_id and fail with SESSION_IS_LOCKED.
        """
        return cls(settings=settings)

    async def _get_client(self) -> AsyncClickHouseClient:
        """Return the active async ClickHouse client, creating it lazily under a lock if needed."""
        if self._client is not None:
            return self._client

        async with self._lock:
            if self._client is not None:
                return self._client
            if self._settings is None:
                raise DatabaseUnavailableError("ClickHouse settings are not configured")
            try:
                self._client = await clickhouse_connect.get_async_client(
                    host=self._settings.clickhouse_host,
                    port=self._settings.clickhouse_port,
                    username=self._settings.clickhouse_user,
                    password=self._settings.clickhouse_password,
                    database=self._settings.clickhouse_db,
                    autogenerate_session_id=False,
                )
            except Exception as exc:
                raise DatabaseUnavailableError("ClickHouse is unavailable") from exc
            return self._client

    async def ping(self) -> bool:
        """Verify ClickHouse connectivity without raising driver exceptions."""
        try:
            client = await self._get_client()
            return bool(await client.ping())
        except Exception:
            return False

    async def search_by_vector(
        self,
        embedding: list[float],
        limit: int,
    ) -> list[DocumentResult]:
        """Find the nearest document chunks by cosine distance.

        Args:
            embedding: Query embedding vector (must be 384-dimensional).
            limit: Maximum number of results to return.

        Returns:
            Ordered list of DocumentResult items from closest to farthest.

        Raises:
            ValueError: If embedding is empty, does not have length 384, or limit < 1.
            DatabaseUnavailableError: If ClickHouse cannot be reached or the query fails due to
                connectivity/transport errors.
        """
        if not embedding:
            raise ValueError("embedding must not be empty")
        if len(embedding) != EXPECTED_EMBEDDING_DIM:
            raise ValueError(f"embedding must have length {EXPECTED_EMBEDDING_DIM}, got {len(embedding)}")
        if limit < 1:
            raise ValueError("limit must be at least 1")

        client = await self._get_client()
        try:
            result = await client.query(
                VECTOR_SEARCH_SQL,
                parameters={"query_embedding": embedding, "limit": limit},
            )
        except (OperationalError, OSError, ConnectionError, TimeoutError) as exc:
            raise DatabaseUnavailableError("ClickHouse query failed due to connectivity") from exc

        return [DocumentResult(title=_extract_title(row[0]), body=str(row[1])) for row in result.result_rows]

    async def close(self) -> None:
        """Close the underlying async ClickHouse client if one was created."""
        async with self._lock:
            client = self._client
            self._client = None
        if client is not None:
            await client.close()
