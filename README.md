# ClickHouse RAG Retrieval API

A FastAPI service that provides the retrieval component of a Retrieval-Augmented Generation (RAG) pipeline backed by ClickHouse vector search (`cosineDistance`) and `sentence-transformers/all-MiniLM-L6-v2`.

---

## Quickstart & Running the Service

### Option 1: Run Everything with Docker Compose (Recommended)

Starts both ClickHouse (automatically initializing the `embeddings.vectors` table from Hugging Face on first boot) and the containerized FastAPI service on port `8000`:

```bash
docker compose up --build -d
```

 Both ClickHouse data (`clickhouse-data`) and the Hugging Face model weights (`huggingface-cache`) are persisted in named Docker volumes so subsequent restarts and rebuilds are instantaneous.

### Option 2: Run the API Locally with `uv`

1. Start the ClickHouse container:
   ```bash
   docker compose up -d clickhouse
   ```
2. Install dependencies and start the FastAPI server locally:
   ```bash
   uv sync
   uv run uvicorn rag_api.main:app --reload --host 0.0.0.0 --port 8000
   ```

---

## API Usage & Examples

Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

### Health Check (`GET /health`)

Verifies API readiness and live ClickHouse connectivity:

```bash
curl -s http://localhost:8000/health
```

```json
{"status": "ok"}
```

### Semantic Document Search (`POST /search`)

Encodes the query string into a 384-dimensional vector using `sentence-transformers/all-MiniLM-L6-v2` and queries ClickHouse ordered by `cosineDistance(embeddings, query_embedding) ASC`.

- `query` (`string`, required): Non-empty search query (leading/trailing whitespace is stripped).
- `limit` (`int`, optional, default `5`, range `1..50`): Maximum number of matching chunks to return.

```bash
curl -s -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Can I create a visualization with Lightdash?", "limit": 3}'
```

Example response:

```json
{
  "results": [
    {
      "title": "lightdash-and-clickhouse.md",
      "body": "The \nExplore\n page is made up of five main areas:\n\n\n\n\nDimensions and Metrics\n — all fields available on the selected table ..."
    },
    {
      "title": "lightdash-and-clickhouse.md",
      "body": "sidebar_label: 'Lightdash'\nsidebar_position: 131\nslug: /integrations/lightdash\nkeywords: ['clickhouse', ..."
    },
    {
      "title": "index.md",
      "body": "sidebar_label: 'Overview'\nsidebar_position: 1\nkeywords: ['ClickHouse', 'connect', 'Luzmo', 'Explo', 'Fabi.ai', 'Tableau',..."
    }
  ]
}
```

---

## Running Tests

- **Run all tests (unit + live integration tests):**
  ```bash
  uv run pytest -v
  ```
- **Run fast, network-free unit tests only:**
  ```bash
  uv run pytest -m "not integration" -v
  ```
- **Run live ClickHouse + embedding model integration tests only:**
  ```bash
  uv run pytest -m integration -v
  ```

---

## Architecture & Design Decisions

1. **End-to-End `async` FastAPI Architecture (`src/rag_api/main.py`)**
   - Routes and dependencies are `async def` end-to-end, managed via FastAPI's `lifespan` context manager so module imports remain fast and side-effect-free for unit testing.
2. **Swappable Async `Embedder` Protocol & Dedicated `ThreadPoolExecutor` (`src/rag_api/embedder.py`)**
   - `Embedder` is defined as an async `typing.Protocol` (`async def embed_query(self, text: str) -> list[float]`). Swapping local CPU inference for an external embedding service (e.g., OpenAI, Cohere, or an internal Text Embeddings Inference server via `httpx.AsyncClient`) requires zero changes to route handlers or callers.
   - `SentenceTransformerEmbedder` offloads synchronous, CPU-bound `SentenceTransformer.encode()` calls to a dedicated `ThreadPoolExecutor` via `loop.run_in_executor(...)` (PyTorch releases the GIL during C++ tensor operations), isolating model inference from FastAPI's default threadpool and keeping the `asyncio` event loop responsive.
3. **Async ClickHouse `VectorRepository` (`src/rag_api/repository.py`)**
   - Uses `clickhouse_connect.get_async_client(..., autogenerate_session_id=False)` with lazy connection initialization under an `asyncio.Lock`.
   - Disabling `autogenerate_session_id` prevents concurrent `async` requests from sharing a single ClickHouse `session_id` and failing with `SESSION_IS_LOCKED`.
   - Deferring client creation until the first request allows the application to boot cleanly even if ClickHouse is temporarily unreachable at startup, serving `503 Service Unavailable` on `/health` and `/search` and recovering automatically once ClickHouse comes online.
4. **Multi-Stage Non-Root Docker Build & CPU-Only PyTorch on Linux (`Dockerfile`, `pyproject.toml`, `docker-compose.yml`)**
   - `pyproject.toml` pins `torch` on Linux to the official PyTorch CPU wheel index (`https://download.pytorch.org/whl/cpu`) via environment markers while keeping standard PyPI wheels on macOS, avoiding ~2.5 GB of unnecessary CUDA packages in the Linux container image.
   - Multi-stage `Dockerfile` builds the virtual environment in `ghcr.io/astral-sh/uv:python3.12-bookworm-slim` and copies only `/app/.venv` into a minimal `python:3.12-slim-bookworm` runtime image running as non-root `appuser` (UID/GID `10001`).

---

## Next Steps

- **Document-Level Deduplication / Grouping by `title`**: Because `embeddings.vectors` stores multiple chunks per Markdown file (`metadata` is a JSON object such as `{"source_file": "lightdash-and-clickhouse.md"}` from which `source_file` is extracted as `title`, and `document` holds the chunk text), matching chunks are currently returned as-is so multiple chunks from the same document can appear in the top-K results. Depending on downstream prompt construction needs, we can deduplicate by the extracted `source_file` in SQL (`LIMIT 1 BY JSONExtractString(metadata, 'source_file')`) or group adjacent chunks per document in Python.
- **Vector Similarity Index (`vector_similarity`)**: For larger datasets beyond exhaustive `cosineDistance` scans, add an HNSW vector similarity index on `vectors.embeddings` in ClickHouse.
- **Hybrid Search & Reranking**: Combine ClickHouse full-text / BM25 keyword matching with vector cosine distance (Reciprocal Rank Fusion) followed by a cross-encoder reranker.
- **Query Embedding Cache & Observability**: Add an LRU/Redis cache for repeated query embeddings and OpenTelemetry traces/metrics around embedding latency and ClickHouse query duration.

---

## Original Assignment Context

<details>
<summary>Click to expand original assignment instructions</summary>

### Work assignment

Timebox: Aim for 3–6 hours. If you run out of time, include a "Next steps" section.

We are looking for a Senior Software Engineer to drive the development of AI/ML-powered features in ClickHouse Cloud and database integrations. Given the current project, let's create a retrieval component of a RAG pipeline. This project is used as a conversation piece for the next stages of the interview process.

Implement the following feature in Python

- Create an API that takes a query string and returns a list of document titles and document bodies in JSON.

### ClickHouse Vector Database

A Docker Compose configuration is provided to run ClickHouse with vector search capabilities. The schema of the table `vectors` is defined in [clickhouse/docker-entrypoint-initdb.d/01-init-tables.sql](clickhouse/docker-entrypoint-initdb.d/01-init-tables.sql).

The model used to create the embeddings is: `sentence-transformers/all-MiniLM-L6-v2`.

**Connection details:**
```
HOST: localhost:8123
CLICKHOUSE_DB: embeddings
CLICKHOUSE_USER: vectordbuser
CLICKHOUSE_PASSWORD: pwd_vectordb
```

**Run the test_connection.py script**

```bash
uv sync
uv run test_connection.py
```

</details>
