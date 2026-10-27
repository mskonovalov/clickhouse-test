"""FastAPI application entrypoint."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request

from rag_api.config import get_settings
from rag_api.embedder import Embedder, SentenceTransformerEmbedder
from rag_api.repository import DatabaseUnavailableError, VectorRepository
from rag_api.schemas import HealthResponse, SearchRequest, SearchResponse


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Initialize and clean up application-level resources."""
    settings = get_settings()
    repository: VectorRepository | None = None
    embedder: Embedder | None = None
    try:
        repository = await VectorRepository.create(settings)
        embedder = SentenceTransformerEmbedder(
            model_name=settings.embedding_model,
            max_workers=settings.embedder_max_workers,
        )
        application.state.repository = repository
        application.state.embedder = embedder
        yield
    finally:
        try:
            if embedder is not None:
                embedder.close()
        finally:
            if repository is not None:
                await repository.close()


app = FastAPI(title="ClickHouse RAG Retrieval API", lifespan=lifespan)


def get_repository(request: Request) -> VectorRepository:
    """Retrieve the VectorRepository stored on application state."""
    repository: VectorRepository | None = getattr(request.app.state, "repository", None)
    if repository is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return repository


def get_embedder(request: Request) -> Embedder:
    """Retrieve the Embedder stored on application state."""
    embedder: Embedder | None = getattr(request.app.state, "embedder", None)
    if embedder is None:
        raise HTTPException(status_code=503, detail="Embedder unavailable")
    return embedder


RepositoryDep = Annotated[VectorRepository, Depends(get_repository)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]


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
    embedder: EmbedderDep,
) -> SearchResponse:
    """Search documents matching the query."""
    embedding = await embedder.embed_query(request.query)
    try:
        results = await repository.search_by_vector(embedding, request.limit)
    except DatabaseUnavailableError as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc
    return SearchResponse(results=results)
