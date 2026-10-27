"""FastAPI application entrypoint."""

from typing import Annotated

from fastapi import Depends, FastAPI

from rag_api.config import Settings, get_settings

app = FastAPI(title="ClickHouse RAG Retrieval API")

SettingsDep = Annotated[Settings, Depends(get_settings)]


@app.get("/health")
async def health(_settings: SettingsDep) -> dict[str, str]:
    """Check service health."""
    return {"status": "ok"}
