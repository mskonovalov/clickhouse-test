"""FastAPI application entrypoint."""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request

from rag_api.config import get_settings
from rag_api.repository import DatabaseUnavailableError, VectorRepository
from rag_api.schemas import HealthResponse, SearchRequest, SearchResponse

EMBEDDING_DIM = 384

type QueryEmbedder = Callable[[str], Awaitable[list[float]]]


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Initialize and clean up application-level resources."""
    settings = get_settings()
    repository = await VectorRepository.create(settings)
    application.state.repository = repository
    try:
        yield
    finally:
        await repository.close()


app = FastAPI(title="ClickHouse RAG Retrieval API", lifespan=lifespan)


def get_repository(request: Request) -> VectorRepository:
    """Retrieve the VectorRepository stored on application state."""
    repository: VectorRepository | None = getattr(request.app.state, "repository", None)
    if repository is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return repository


async def _stub_embed_query(_query: str) -> list[float]:
    """Temporary 384-dimensional non-zero embedding stub for Increment 4."""
    return [0.1] * EMBEDDING_DIM


def get_query_embedder() -> QueryEmbedder:
    """Return the active async query embedding callable."""
    return _stub_embed_query


RepositoryDep = Annotated[VectorRepository, Depends(get_repository)]
QueryEmbedderDep = Annotated[QueryEmbedder, Depends(get_query_embedder)]


@app.get("/health", response_model=HealthResponse)
async def health(repository: RepositoryDep) -> HealthResponse:
    """Check service and ClickHouse connectivity health."""
    if not await repository.ping():
        raise HTTPException(status_code=503, detail="Database unavailable")
    return HealthResponse(status="ok")


@app.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    repository: RepositoryDep,
    embed_query: QueryEmbedderDep,
) -> SearchResponse:
    """Search documents matching the query."""
    embedding = await embed_query(request.query)
    try:
        results = await repository.search_by_vector(embedding, request.limit)
    except DatabaseUnavailableError as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc
    return SearchResponse(results=results)
