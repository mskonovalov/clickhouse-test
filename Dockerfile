# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim-bookworm AS runtime

RUN groupadd --system --gid 10001 appuser && \
    useradd --system --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin appuser && \
    mkdir -p /cache/huggingface && \
    chown -R appuser:appuser /cache/huggingface

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

ENV HF_HOME=/cache/huggingface \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER appuser

EXPOSE 8000

CMD ["uvicorn", "rag_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
