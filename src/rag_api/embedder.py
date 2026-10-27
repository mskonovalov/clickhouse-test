"""Async embedding interface and local SentenceTransformer implementation."""

import asyncio
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from sentence_transformers import SentenceTransformer


class EncodedVectorProtocol(Protocol):
    """Protocol for an array-like embedding output supporting .tolist()."""

    def tolist(self) -> Sequence[float]:
        """Convert the encoded vector to a Python sequence of floats."""
        ...


class SentenceTransformerModelProtocol(Protocol):
    """Structural protocol for SentenceTransformer-compatible encoding models."""

    def encode(
        self,
        sentences: str,
        *,
        convert_to_numpy: bool = True,
    ) -> EncodedVectorProtocol:
        """Encode a query string into a vector representation."""
        ...


class Embedder(Protocol):
    """Protocol for async query embedding providers."""

    async def embed_query(self, text: str) -> list[float]:
        """Generate an embedding vector for the given query text."""
        ...

    def close(self) -> None:
        """Release any underlying resources held by the embedder."""
        ...


class SentenceTransformerEmbedder:
    """Local SentenceTransformer embedder offloading CPU inference to a dedicated ThreadPoolExecutor."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_workers: int = 4,
        *,
        model: SentenceTransformerModelProtocol | None = None,
    ) -> None:
        """Initialize the local SentenceTransformer model and dedicated thread pool.

        Args:
            model_name: Hugging Face model identifier to load when model is not injected.
            max_workers: Number of worker threads in the dedicated executor (must be >= 1).
            model: Optional pre-built model instance for unit testing without loading weights.

        Raises:
            ValueError: If max_workers is less than 1.
        """
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")

        self._model_name = model_name
        self._model: SentenceTransformerModelProtocol = (
            model if model is not None else SentenceTransformer(model_name)
        )
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="embedder",
        )

    def _encode_sync(self, text: str) -> list[float]:
        """Synchronously encode text on a worker thread and return a list of floats."""
        encoded = self._model.encode(text, convert_to_numpy=True)
        return [float(value) for value in encoded.tolist()]

    async def embed_query(self, text: str) -> list[float]:
        """Asynchronously encode a query string using the dedicated thread pool.

        Args:
            text: Non-empty query string to embed.

        Returns:
            Embedding vector as a list of floats.

        Raises:
            ValueError: If text is empty or whitespace-only.
        """
        stripped = text.strip()
        if not stripped:
            raise ValueError("text must not be empty")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._encode_sync, stripped)

    def close(self) -> None:
        """Shut down the dedicated ThreadPoolExecutor and wait for pending work."""
        self._executor.shutdown(wait=True)
