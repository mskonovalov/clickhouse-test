"""Integration tests against the live ClickHouse container and real embedding and reranking models."""

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
def test_end_to_end_rag_retrieval_with_real_embedder_reranker_and_clickhouse() -> None:
    """Verify end-to-end semantic retrieval and cross-encoder reranking against live ClickHouse."""
    query = "Can I create a visualization with Lightdash?"

    with TestClient(app) as client:
        health_response = client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json() == {"status": "ok"}

        reranked_response = client.post(
            "/search",
            json={"query": query, "limit": 5, "rerank": True},
        )
        assert reranked_response.status_code == 200
        reranked_payload = reranked_response.json()
        assert "results" in reranked_payload
        reranked_results = reranked_payload["results"]
        assert len(reranked_results) == 5
        for item in reranked_results:
            assert set(item.keys()) == {"title", "body"}
            assert item["title"].strip() != ""
            assert item["body"].strip() != ""

        assert reranked_results[0]["title"] == "lightdash-and-clickhouse.md"
        assert not reranked_results[0]["body"].lstrip().startswith("sidebar_label:")

        prose_indices = [
            idx for idx, item in enumerate(reranked_results) if not item["body"].lstrip().startswith("sidebar_label:")
        ]
        frontmatter_indices = [
            idx for idx, item in enumerate(reranked_results) if item["body"].lstrip().startswith("sidebar_label:")
        ]
        assert prose_indices
        if frontmatter_indices:
            assert prose_indices[0] < frontmatter_indices[0]

        stage_one_response = client.post(
            "/search",
            json={"query": query, "limit": 5, "rerank": False},
        )
        assert stage_one_response.status_code == 200
        stage_one_results = stage_one_response.json()["results"]
        assert len(stage_one_results) == 5
        assert "lightdash-and-clickhouse.md" in [item["title"] for item in stage_one_results]

        limit_two_response = client.post(
            "/search",
            json={"query": query, "limit": 2},
        )
        assert limit_two_response.status_code == 200
        assert len(limit_two_response.json()["results"]) == 2
