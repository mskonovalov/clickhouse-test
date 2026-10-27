"""Unit tests for CrossEncoderReranker (rag_api.reranker)."""

import asyncio
import math

import pytest
from conftest import FakeCrossEncoderModel

from rag_api.reranker import CrossEncoderReranker
from rag_api.schemas import DocumentResult


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
