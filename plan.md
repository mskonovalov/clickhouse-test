# Implementation Plan: ClickHouse RAG Retrieval API

## 1. Summary of Decisions

| Area | Decision |
| :--- | :--- |
| **Framework** | FastAPI + Uvicorn (`async def` routes and dependencies end-to-end) |
| **Config** | `pydantic-settings` (`BaseSettings`) |
| **Schemas** | Pydantic `BaseModel` |
| **Embedder Interface** | Async `Embedder` `Protocol` (`async def embed_query(self, text: str) -> list[float]`) so it can be swapped for an external network embedding API without changing callers |
| **Local Embedder** | `SentenceTransformerEmbedder` (`sentence-transformers/all-MiniLM-L6-v2`, CPU-only PyTorch wheels on Linux) offloading CPU-bound `.encode()` to a dedicated `ThreadPoolExecutor` via `loop.run_in_executor` |
| **Reranker Interface & Local Reranker** | Async `Reranker` `Protocol` (`async def rerank(self, query: str, documents: Sequence[DocumentResult], top_n: int) -> list[DocumentResult]`) + `CrossEncoderReranker` (`cross-encoder/ms-marco-MiniLM-L-6-v2`) with a dedicated `ThreadPoolExecutor`, toggleable globally via `RERANKER_ENABLED` and per-request via `rerank: bool = True` |
| **Model & Data Persistence** | Named Docker volumes for both ClickHouse (`clickhouse-data:/var/lib/clickhouse`) and Hugging Face model cache (`huggingface-cache:/cache/huggingface`) |
| **Vector Search** | Async `VectorRepository` using `clickhouse_connect.get_async_client` (`autogenerate_session_id=False`) and `cosineDistance(embeddings, {query_embedding:Array(Float32)}) ASC LIMIT {limit:UInt32}` |
| **Search Endpoint** | `POST /search` accepting `{"query": "...", "limit": 5, "rerank": true}` (`limit` optional, default `5`, range `1..50`; `rerank` optional `StrictBool`, default `true`) |
| **Search Response** | `{"results": [{"title": "...", "body": "..."}]}` (raw chunks as-is; note deduplication by `title` in README / Next steps) |
| **Health Endpoint** | `GET /health` verifying ClickHouse connectivity |
| **Layout & Runtime** | `src/rag_api/` package layout; runnable both locally via `uv` and as a multi-stage (`builder` -> `python:3.12-slim-bookworm` `runtime`), non-root (`appuser`) containerized `api` service in `docker-compose.yml` |
| **Tests & Linting** | Per-module unit tests (`tests/test_api.py`, `test_config.py`, `test_embedder.py`, `test_repository.py`, `test_reranker.py` + `conftest.py`), live integration tests (`tests/test_integration.py`), and `ruff` linting/formatting |
| **Extensions** | Cross-encoder reranking (Increment 6) |

---

## 2. Target Project Structure

```text
clickhouse/
├── clickhouse/
│   └── docker-entrypoint-initdb.d/
│       ├── 01-init-tables.sql
│       └── 02-load-data.sql
├── src/
│   └── rag_api/
│       ├── __init__.py
│       ├── config.py          # Env-driven settings via pydantic-settings
│       ├── schemas.py         # Pydantic models: SearchRequest, DocumentResult, SearchResponse, HealthResponse
│       ├── repository.py      # Async ClickHouse query layer using cosineDistance
│       ├── embedder.py        # Async Embedder Protocol + SentenceTransformerEmbedder with dedicated ThreadPoolExecutor
│       ├── reranker.py        # Async Reranker Protocol + CrossEncoderReranker with dedicated ThreadPoolExecutor
│       └── main.py            # FastAPI app, lifespan resource management, async /health and /search routes
├── tests/
│   ├── conftest.py            # Shared test doubles and fixtures
│   ├── test_api.py            # FastAPI route and lifespan unit tests
│   ├── test_config.py         # Settings defaults, env overrides, and validation unit tests
│   ├── test_embedder.py       # SentenceTransformerEmbedder unit tests
│   ├── test_repository.py     # VectorRepository unit tests
│   ├── test_reranker.py       # CrossEncoderReranker unit tests
│   └── test_integration.py    # End-to-end tests against running ClickHouse + real embedding & reranking models
├── Dockerfile                 # Multi-stage non-root container image for the FastAPI service
├── docker-compose.yml         # ClickHouse (with healthcheck & data volume) + API service (with HF cache volume)
├── pyproject.toml             # Dependencies, CPU PyTorch source, pytest & ruff config
├── uv.lock
├── test_connection.py         # Existing smoke script (kept intact)
└── README.md                  # Updated with run/test instructions, architecture notes, and "Next steps" (incl. chunk deduplication)
```

---

## 3. Incremental Delivery Plan

Every step produces a runnable, testable increment before moving on to the next.

### Increment 1: FastAPI Skeleton & Local Test Harness
**Goal:** Get the Python package structure, minimal async FastAPI app, and `pytest` runner working locally via `uv`.

- **Changes:**
  - Update `pyproject.toml` with a `[build-system]` (`hatchling`) so `src/rag_api` is installed as a package, core dependencies (`fastapi`, `uvicorn[standard]`, `pydantic-settings`, `clickhouse-connect`), and dev dependencies (`pytest`, `httpx`).
  - Create `src/rag_api/__init__.py` and `src/rag_api/config.py` (`Settings` class using `BaseSettings` with defaults for ClickHouse host/port/db/user/password, `embedding_model="sentence-transformers/all-MiniLM-L6-v2"`, and `embedder_max_workers=4`).
  - Create `src/rag_api/main.py` with a minimal FastAPI app exposing `async def health()` at `GET /health` returning `{"status": "ok"}`.
    - *Technical safeguard (Import-time isolation):* Do not instantiate heavy resources at module import time; wire dependencies via FastAPI `Depends(...)` so importing `rag_api.main` in unit tests stays instantaneous and network-free.
  - Create initial unit tests exercising `GET /health` via FastAPI `TestClient`.
- **Verification:**
  - `uv sync` installs dependencies cleanly.
  - `uv run pytest` passes.
  - `uv run uvicorn rag_api.main:app` starts locally and `curl http://localhost:8000/health` returns `200 {"status": "ok"}`.

---

### Increment 2: Docker & Docker Compose Integration (with Persistent Volumes)
**Goal:** Containerize the FastAPI service with persistent volumes for both the ClickHouse dataset and Hugging Face cache, and run it alongside ClickHouse in Docker Compose.

- **Changes:**
  - Create a multi-stage `Dockerfile` for the API service:
    - `builder` stage (`ghcr.io/astral-sh/uv:python3.12-bookworm-slim`) that installs dependencies and `rag_api` (`--no-editable`) from `uv.lock` into `/app/.venv`.
    - `runtime` stage (`python:3.12-slim-bookworm` without `uv`) that creates a non-root `appuser` (UID/GID `10001`), pre-creates `/cache/huggingface` owned by `appuser:appuser`, copies `/app/.venv` from `builder`, runs as `USER appuser`, and starts `uvicorn rag_api.main:app --host 0.0.0.0 --port 8000`.
  - Update `docker-compose.yml`:
    - *Technical safeguard (ClickHouse Alpine healthcheck & first-boot Hugging Face Parquet download):* `clickhouse/clickhouse-server:24.3-alpine` does not have `curl` installed, and `02-load-data.sql` downloads a Parquet file from Hugging Face on first startup (which can take 15–60s). Configure the `clickhouse` healthcheck using `clickhouse-client` (`SELECT count() FROM embeddings.vectors`) with `start_period: 60s` and `CLICKHOUSE_ALWAYS_RUN_INITDB_SCRIPTS: 1` (paired with an empty-table guard in `02-load-data.sql`) so Docker waits until the initial dataset is loaded into the persistent `clickhouse-data:/var/lib/clickhouse` volume.
    - *Technical safeguard (Persistent Hugging Face model cache):* Add a named Docker volume `huggingface-cache` mounted at `/cache/huggingface` on the `api` container (with `HF_HOME: /cache/huggingface`) so Hugging Face downloads are writable by `appuser` and persist across container restarts and image rebuilds instead of re-downloading.
    - Add the `api` service on port `8000:8000` with `CLICKHOUSE_HOST: clickhouse` and `depends_on: clickhouse: condition: service_healthy`.
- **Verification:**
  - `docker compose up --build -d` builds and starts both `clickhouse` and `api` containers cleanly.
  - `curl http://localhost:8000/health` against the containerized API returns `200 {"status": "ok"}`.

---

### Increment 3: `POST /search` API Contract & Schemas
**Goal:** Define and verify the HTTP contract and input/output validation for `POST /search` using Pydantic models.

- **Changes:**
  - Create `src/rag_api/schemas.py`:
    - `SearchRequest`: `query: str = Field(min_length=1)` with a `@field_validator("query")` that strips leading/trailing whitespace and rejects whitespace-only strings, plus `limit: int = Field(default=5, ge=1, le=50)`.
    - `DocumentResult`: `title: str`, `body: str`.
    - `SearchResponse`: `results: list[DocumentResult]`.
    - `HealthResponse`: `status: Literal["ok"]`.
  - Update `src/rag_api/main.py`:
    - Add `async def search(request: SearchRequest)` at `POST /search` returning `SearchResponse` (wired via FastAPI `Depends` with a temporary stub search dependency for this increment).
  - Add route and schema validation unit tests:
    - Test `POST /search` returns `200` and matches `{"results": [{"title": "...", "body": "..."}]}`.
    - Test default `limit=5` when `limit` is omitted.
    - Test validation errors (`422`) for empty/whitespace `query`, missing `query`, `limit < 1`, and `limit > 50`.
- **Verification:**
  - `uv run pytest` passes all route and schema validation tests.
  - `curl -X POST http://localhost:8000/search -H "Content-Type: application/json" -d '{"query": "lightdash"}'` returns `200` with valid JSON; invalid payloads return `422`.

---

### Increment 4: Async ClickHouse Repository (`VectorRepository`)
**Goal:** Connect the API to ClickHouse asynchronously for real health checks and vector similarity queries using `cosineDistance`.

- **Changes:**
  - Create `src/rag_api/repository.py` with async `VectorRepository`:
    - *Technical safeguard (Concurrency & ClickHouse session locking):* Create the client lazily via `await clickhouse_connect.get_async_client(..., autogenerate_session_id=False)` under an `asyncio.Lock`. Disabling `autogenerate_session_id` prevents concurrent `async def` requests from sharing a single ClickHouse `session_id` and failing with `SESSION_IS_LOCKED`.
    - `async def ping() -> bool`: verifies ClickHouse connectivity without blocking the event loop.
    - `async def search_by_vector(embedding: list[float], limit: int) -> list[DocumentResult]`: executes parameterized query:
      ```sql
      SELECT metadata AS title, document AS body
      FROM vectors
      ORDER BY cosineDistance(embeddings, {query_embedding:Array(Float32)}) ASC
      LIMIT {limit:UInt32}
      ```
    - `async def close() -> None`: closes the async client cleanly on app shutdown.
  - Update `src/rag_api/main.py`:
    - Initialize and close `VectorRepository` inside FastAPI's `lifespan` context manager.
    - Wire `GET /health` to `await repository.ping()` (returns `200` when healthy, `503` when ClickHouse is unreachable).
    - Wire `POST /search` to call `await repository.search_by_vector(...)`.
    - *Technical safeguard (384-dim non-zero vector constraint):* Because `vectors.embeddings` stores 384-float vectors (`all-MiniLM-L6-v2`), ClickHouse's `cosineDistance` throws a size mismatch error if the query vector length != 384 and returns `NaN` if all values are `0.0`.
  - Add unit tests for `VectorRepository` (`tests/test_repository.py`) and live ClickHouse integration tests (`tests/test_integration.py`).
- **Verification:**
  - `uv run pytest` passes both unit tests and ClickHouse repository integration tests.
  - `curl http://localhost:8000/health` and `curl -X POST http://localhost:8000/search ...` confirm live database queries.

---

### Increment 5: Async Embedder (`sentence-transformers` + `ThreadPoolExecutor`) & End-to-End Retrieval
**Goal:** Implement the swappable async `Embedder` interface and non-blocking local `SentenceTransformerEmbedder`, and verify full end-to-end RAG retrieval.

- **Changes:**
  - Update `pyproject.toml` and `uv.lock` to add `sentence-transformers`:
    - *Technical safeguard (Cross-platform CPU PyTorch wheels):* Configure `[[tool.uv.index]]` for `https://download.pytorch.org/whl/cpu` and scope `[tool.uv.sources]` for `torch` with `marker = "sys_platform == 'linux'"`. This ensures Linux Docker builds download the ~180 MB CPU-only PyTorch wheel instead of ~2.5 GB of CUDA packages while macOS continues using native wheels cleanly.
  - Create `src/rag_api/embedder.py`:
    - *Technical safeguard (Swappable async interface for external LLM/embedding services):* Define `class Embedder(Protocol)` with `async def embed_query(self, text: str) -> list[float]`.
    - *Technical safeguard (Non-blocking CPU inference via dedicated `ThreadPoolExecutor`):* Implement `SentenceTransformerEmbedder` validating 384-dim output and offloading `.encode()` to a dedicated `ThreadPoolExecutor(max_workers=settings.embedder_max_workers, thread_name_prefix="embedder")`.
  - Update `src/rag_api/main.py` to initialize and close `SentenceTransformerEmbedder` in `lifespan`.
  - Add embedder unit tests (`tests/test_embedder.py`) and end-to-end integration tests (`tests/test_integration.py`).
- **Verification:**
  - `uv run pytest` passes the entire unit and integration test suite.

---

### Increment 6: Cross-Encoder Reranking (`CrossEncoderReranker`), Per-Module Test Suite & Ruff Linting
**Goal:** Add second-stage cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to promote substantive documentation chunks over noisy YAML frontmatter, organize unit tests into per-module files, and enforce code style with `ruff`.

- **Changes:**
  - Update `src/rag_api/config.py` with reranker settings:
    - `reranker_enabled: bool = True`
    - `reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"`
    - `reranker_max_workers: int = Field(default=2, ge=1)`
    - `rerank_candidate_multiplier: int = Field(default=4, ge=1)`
    - `rerank_min_candidates: int = Field(default=20, ge=1)`
  - Update `src/rag_api/schemas.py` to add `rerank: StrictBool = Field(default=True, ...)` to `SearchRequest`.
  - Create `src/rag_api/reranker.py`:
    - Define `class Reranker(Protocol)` (`async def rerank(self, query: str, documents: Sequence[DocumentResult], top_n: int) -> list[DocumentResult]` and `def close(self) -> None`).
    - Implement `CrossEncoderReranker` offloading synchronous `CrossEncoder.predict()` to a dedicated `ThreadPoolExecutor(thread_name_prefix="reranker")`, validating score count and finiteness (`math.isfinite`), and breaking score ties deterministically by Stage-1 rank order.
  - Update `src/rag_api/main.py`:
    - Initialize `CrossEncoderReranker` in `lifespan` when `settings.reranker_enabled` is `True`.
    - When `request.rerank and settings.reranker_enabled`, over-fetch `max(request.limit * settings.rerank_candidate_multiplier, settings.rerank_min_candidates)` candidates from ClickHouse and rerank them to `top_n=request.limit`; otherwise fetch `request.limit` directly from ClickHouse.
  - Split unit tests into `tests/conftest.py`, `tests/test_api.py`, `tests/test_config.py`, `tests/test_embedder.py`, `tests/test_repository.py`, and `tests/test_reranker.py`, and add Ruff configuration to `pyproject.toml`.
- **Verification:**
  - `uv run pytest -v` passes all unit and integration tests.
  - `uv run ruff check .` and `uv run ruff format --check .` pass cleanly.
