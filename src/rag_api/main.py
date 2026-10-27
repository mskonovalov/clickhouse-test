"""FastAPI application entrypoint."""

from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import Depends, FastAPI

from rag_api.config import Settings, get_settings
from rag_api.schemas import DocumentResult, HealthResponse, SearchRequest, SearchResponse

app = FastAPI(title="ClickHouse RAG Retrieval API")

SettingsDep = Annotated[Settings, Depends(get_settings)]
type SearchHandler = Callable[[str, int], Awaitable[list[DocumentResult]]]


async def _stub_search_handler(query: str, limit: int) -> list[DocumentResult]:
    """Temporary async search stub for Increment 3."""
    return [DocumentResult(title="stub.md", body=f"Stub result for {query}")][:limit]


def get_search_handler() -> SearchHandler:
    """Return the active search handler callable."""
    return _stub_search_handler


SearchHandlerDep = Annotated[SearchHandler, Depends(get_search_handler)]


@app.get("/health", response_model=HealthResponse)
async def health(_settings: SettingsDep) -> HealthResponse:
    """Check service health."""
    return HealthResponse(status="ok")


@app.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    search_handler: SearchHandlerDep,
) -> SearchResponse:
    """Search documents matching the query."""
    results = await search_handler(request.query, request.limit)
    return SearchResponse(results=results)
