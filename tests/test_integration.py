"""Integration tests against the live ClickHouse container and real embedding model."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from rag_api.config import get_settings
from rag_api.main import app
from rag_api.repository import VectorRepository

EMBEDDING_DIM = 384


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
def test_end_to_end_rag_retrieval_with_real_embedder_and_clickhouse() -> None:
    """Verify end-to-end semantic retrieval with SentenceTransformerEmbedder and live ClickHouse."""
    with TestClient(app) as client:
        health_response = client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json() == {"status": "ok"}

        search_response = client.post(
            "/search",
            json={"query": "Can I create a visualization with Lightdash?", "limit": 5},
        )
        assert search_response.status_code == 200
        payload = search_response.json()
        assert "results" in payload
        results = payload["results"]
        assert len(results) == 5
        for item in results:
            assert set(item.keys()) == {"title", "body"}
            assert item["title"].strip() != ""
            assert item["body"].strip() != ""

        titles = [item["title"] for item in results]
        assert "lightdash-and-clickhouse.md" in titles

        limit_two_response = client.post(
            "/search",
            json={"query": "Can I create a visualization with Lightdash?", "limit": 2},
        )
        assert limit_two_response.status_code == 200
        assert len(limit_two_response.json()["results"]) == 2
