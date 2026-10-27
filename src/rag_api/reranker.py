"""Async cross-encoder reranking interface and local CrossEncoder implementation."""

import asyncio
import math
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

from sentence_transformers import CrossEncoder

from rag_api.schemas import DocumentResult


class CrossEncoderModelProtocol(Protocol):
    """Structural protocol for CrossEncoder-compatible scoring models."""

    def predict(self, sentences: list[tuple[str, str]]) -> Any:
        """Score a batch of (query, document) text pairs."""
        ...


class Reranker(Protocol):
    """Protocol for async second-stage document rerankers."""

    async def rerank(
        self,
        query: str,
        documents: Sequence[DocumentResult],
        top_n: int,
    ) -> list[DocumentResult]:
        """Rerank candidate documents by relevance to query and return the top_n."""
        ...

    def close(self) -> None:
        """Release any underlying resources held by the reranker."""
        ...


class CrossEncoderReranker:
    """Local CrossEncoder reranker offloading CPU inference to a dedicated ThreadPoolExecutor."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_workers: int = 2,
        *,
        model: CrossEncoderModelProtocol | None = None,
    ) -> None:
        """Initialize the local CrossEncoder model and dedicated thread pool.

        Args:
            model_name: Hugging Face cross-encoder model identifier to load when model is not injected.
            max_workers: Number of worker threads in the dedicated executor (must be >= 1).
            model: Optional pre-built model instance for unit testing without loading weights.

        Raises:
            ValueError: If max_workers < 1.
        """
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")

        self._model_name = model_name
        self._model: CrossEncoderModelProtocol = model if model is not None else CrossEncoder(model_name)
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="reranker",
        )

    def _rerank_sync(
        self,
        query: str,
        documents: Sequence[DocumentResult],
        top_n: int,
    ) -> list[DocumentResult]:
        """Synchronously score and sort candidate documents on a worker thread."""
        pairs = [(query, doc.body) for doc in documents]
        raw_scores = self._model.predict(pairs)
        score_values = raw_scores.tolist() if hasattr(raw_scores, "tolist") else raw_scores
        scores = [float(score) for score in score_values]
        if len(scores) != len(documents):
            raise ValueError(f"Expected {len(documents)} reranker scores, got {len(scores)}")
        if not all(math.isfinite(score) for score in scores):
            raise ValueError("Reranker produced non-finite score")

        ranked = sorted(
            enumerate(documents),
            key=lambda item: (-scores[item[0]], item[0]),
        )
        return [doc for _, doc in ranked[:top_n]]

    async def rerank(
        self,
        query: str,
        documents: Sequence[DocumentResult],
        top_n: int,
    ) -> list[DocumentResult]:
        """Asynchronously rerank candidate documents using the dedicated thread pool.

        Args:
            query: Non-empty search query string.
            documents: Candidate documents from first-stage vector retrieval.
            top_n: Maximum number of reranked documents to return (must be >= 1).

        Returns:
            Top top_n documents ordered by descending cross-encoder relevance score.

        Raises:
            ValueError: If query is empty/whitespace-only, top_n < 1, or model scores are invalid.
        """
        stripped = query.strip()
        if not stripped:
            raise ValueError("query must not be empty")
        if top_n < 1:
            raise ValueError("top_n must be at least 1")
        if not documents:
            return []
        if len(documents) == 1:
            return list(documents[:top_n])

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._rerank_sync,
            stripped,
            list(documents),
            top_n,
        )

    def close(self) -> None:
        """Shut down the dedicated ThreadPoolExecutor and wait for pending work."""
        self._executor.shutdown(wait=True)
