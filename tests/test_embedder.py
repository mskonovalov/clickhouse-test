"""Unit tests for SentenceTransformerEmbedder (rag_api.embedder)."""

import asyncio

import pytest
from conftest import FakeSentenceTransformerModel

from rag_api.embedder import EXPECTED_EMBEDDING_DIM, SentenceTransformerEmbedder


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
