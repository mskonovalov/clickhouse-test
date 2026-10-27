"""FastAPI application entrypoint."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request

from rag_api.config import Settings, get_settings
from rag_api.embedder import Embedder, SentenceTransformerEmbedder
from rag_api.repository import DatabaseUnavailableError, VectorRepository
from rag_api.reranker import CrossEncoderReranker, Reranker
from rag_api.schemas import HealthResponse, SearchRequest, SearchResponse


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Initialize and clean up application-level resources."""
    settings = get_settings()
    repository: VectorRepository | None = None
    embedder: Embedder | None = None
    reranker: Reranker | None = None
    try:
        repository = await VectorRepository.create(settings)
        embedder = SentenceTransformerEmbedder(
            model_name=settings.embedding_model,
            max_workers=settings.embedder_max_workers,
        )
        reranker = CrossEncoderReranker(
            model_name=settings.reranker_model,
            max_workers=settings.reranker_max_workers,
        )
        application.state.settings = settings
        application.state.repository = repository
        application.state.embedder = embedder
        application.state.reranker = reranker
        yield
    finally:
        try:
            if reranker is not None:
                reranker.close()
        finally:
            try:
                if embedder is not None:
                    embedder.close()
            finally:
                if repository is not None:
                    await repository.close()


app = FastAPI(title="ClickHouse RAG Retrieval API", lifespan=lifespan)


async def get_app_settings(request: Request) -> Settings:
    """Retrieve the Settings stored on application state or load defaults."""
    settings: Settings | None = getattr(request.app.state, "settings", None)
    return settings if settings is not None else get_settings()


async def get_repository(request: Request) -> VectorRepository:
    """Retrieve the VectorRepository stored on application state."""
    repository: VectorRepository | None = getattr(request.app.state, "repository", None)
    if repository is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return repository


async def get_embedder(request: Request) -> Embedder:
    """Retrieve the Embedder stored on application state."""
    embedder: Embedder | None = getattr(request.app.state, "embedder", None)
    if embedder is None:
        raise HTTPException(status_code=503, detail="Embedder unavailable")
    return embedder


async def get_reranker(request: Request) -> Reranker | None:
    """Retrieve the optional Reranker stored on application state."""
    return getattr(request.app.state, "reranker", None)


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
RepositoryDep = Annotated[VectorRepository, Depends(get_repository)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
RerankerDep = Annotated[Reranker | None, Depends(get_reranker)]


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
    reranker: RerankerDep,
    settings: SettingsDep,
) -> SearchResponse:
    """Search documents matching the query with optional cross-encoder reranking."""
    if request.rerank and reranker is None:
        raise HTTPException(status_code=503, detail="Reranker unavailable")

    embedding = await embedder.embed_query(request.query)
    candidate_limit = (
        max(
            request.limit * settings.rerank_candidate_multiplier,
            settings.rerank_min_candidates,
        )
        if request.rerank
        else request.limit
    )
    try:
        candidates = await repository.search_by_vector(embedding, candidate_limit)
    except DatabaseUnavailableError as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc

    if request.rerank:
        assert reranker is not None
        results = await reranker.rerank(
            request.query,
            candidates,
            top_n=request.limit,
        )
    else:
        results = candidates[: request.limit]
    return SearchResponse(results=results)
