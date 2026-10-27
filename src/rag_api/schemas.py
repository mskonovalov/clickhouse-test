"""Pydantic request and response schemas for the RAG API."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    """Request payload for vector similarity search."""

    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=50)

    @field_validator("query", mode="after")
    @classmethod
    def validate_query_not_whitespace(cls, value: str) -> str:
        """Strip leading/trailing whitespace and reject whitespace-only queries."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must not be empty or whitespace")
        return stripped


class DocumentResult(BaseModel):
    """A single document chunk returned from vector search."""

    title: str
    body: str


class SearchResponse(BaseModel):
    """Response payload for vector similarity search."""

    results: list[DocumentResult]


class HealthResponse(BaseModel):
    """Response payload for the service health endpoint."""

    status: Literal["ok"]
