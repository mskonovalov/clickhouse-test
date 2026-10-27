"""Integration tests against the live ClickHouse container."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from rag_api.config import get_settings
from rag_api.main import EMBEDDING_DIM, app
from rag_api.repository import VectorRepository


@pytest.mark.integration
def test_vector_repository_live_ping_and_search() -> None:
    """Verify VectorRepository connects to live ClickHouse and returns cosineDistance results."""

    async def _run() -> None:
        settings = get_settings()
        repo = await VectorRepository.create(settings)
        try:
            assert await repo.ping() is True

            results = await repo.search_by_vector([0.1] * EMBEDDING_DIM, limit=3)
            assert len(results) == 3
            for item in results:
                assert item.title.strip() != ""
                assert item.body.strip() != ""
        finally:
            await repo.close()

    asyncio.run(_run())


@pytest.mark.integration
def test_fastapi_app_live_health_and_search() -> None:
    """Verify FastAPI /health and /search endpoints against live ClickHouse via lifespan."""
    with TestClient(app) as client:
        health_response = client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json() == {"status": "ok"}

        search_response = client.post("/search", json={"query": "lightdash", "limit": 3})
        assert search_response.status_code == 200
        payload = search_response.json()
        assert "results" in payload
        assert len(payload["results"]) == 3
        for item in payload["results"]:
            assert set(item.keys()) == {"title", "body"}
            assert item["title"].strip() != ""
            assert item["body"].strip() != ""
