# ClickHouse RAG Retrieval API

A FastAPI service that provides a two-stage Retrieval-Augmented Generation (RAG) retrieval pipeline backed by ClickHouse vector search (`cosineDistance` with `sentence-transformers/all-MiniLM-L6-v2`) and second-stage cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`).

---

## Quickstart & Running the Service

### Option 1: Run Everything with Docker Compose (Recommended)

Starts both ClickHouse (automatically initializing the `embeddings.vectors` table from Hugging Face on first boot) and the containerized FastAPI service on port `8000`:

```bash
docker compose up --build -d
```

Both ClickHouse data (`clickhouse-data`) and the Hugging Face model weights (`huggingface-cache`) are persisted in named Docker volumes so subsequent restarts and rebuilds are instantaneous. Initialization is idempotent (`CLICKHOUSE_ALWAYS_RUN_INITDB_SCRIPTS: 1` with an empty-table guard in `02-load-data.sql`), and you can reset all volumes from scratch at any time with `docker compose down -v`.

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

Executes a two-stage retrieval pipeline:
1. **Stage 1 — Candidate Generation (Bi-Encoder + ClickHouse `cosineDistance`)**: Encodes `query` into a 384-dimensional vector using `sentence-transformers/all-MiniLM-L6-v2` and fetches `max(limit * rerank_candidate_multiplier, rerank_min_candidates)` (default `max(limit * 4, 20)`) candidate chunks from ClickHouse ordered by `cosineDistance(embeddings, query_embedding) ASC` (or `limit` directly when `rerank` is `false`).
2. **Stage 2 — Cross-Encoder Reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`)**: Scores each `(query, chunk_body)` pair jointly with full cross-attention and returns the top `limit` chunks ordered by descending relevance score. This promotes substantive documentation prose ahead of noisy YAML frontmatter (`sidebar_label: ...`) chunks that may have high bi-encoder cosine similarity.

Request fields:
- `query` (`string`, required): Non-empty search query (leading/trailing whitespace is stripped).
- `limit` (`int`, optional, default `5`, range `1..50`): Maximum number of matching chunks to return.
- `rerank` (`bool`, optional, default `true`): Whether to apply second-stage cross-encoder reranking.

```bash
curl -s -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Can I create a visualization with Lightdash?", "limit": 3, "rerank": true}'
```

Example response (`rerank: true`):

```json
{
  "results": [
    {
      "title": "lightdash-and-clickhouse.md",
      "body": "Connecting Lightdash to ClickHouse\u200b\n\n\nLightdash is an open-source BI tool that transforms your dbt project into a full-stack BI platform..."
    },
    {
      "title": "lightdash-and-clickhouse.md",
      "body": "4. Explore and visualize your data in Lightdash\u200b\n\n\nOnce you've deployed your dbt models and connected Lightdash to ClickHouse..."
    },
    {
      "title": "lightdash-and-clickhouse.md",
      "body": "The \nExplore\n page is made up of five main areas:\n\n\n\n\nDimensions and Metrics\n — all fields available on the selected table ..."
    }
  ]
}
```

To bypass second-stage reranking and return raw Stage-1 `cosineDistance` order:

```bash
curl -s -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Can I create a visualization with Lightdash?", "limit": 3, "rerank": false}'
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
- **Run live ClickHouse + embedding & reranking model integration tests only:**
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
3. **Two-Stage Retrieval with Async `Reranker` Protocol & `CrossEncoderReranker` (`src/rag_api/reranker.py`)**
   - **Bi-Encoder vs. Cross-Encoder Trade-off**: Bi-encoders (`all-MiniLM-L6-v2`) encode queries and documents independently into fixed vectors, enabling fast sub-linear or vectorized `cosineDistance` candidate generation in ClickHouse, but compressing whole chunks into a single pooling vector can rank keyword-dense YAML frontmatter (`sidebar_label: 'Lightdash'`) nearly as high as actual documentation prose. Cross-encoders (`cross-encoder/ms-marco-MiniLM-L-6-v2`) process `(query, document)` pairs jointly with full self-attention across all tokens, producing substantially higher precision on top-N candidates at the cost of \(O(N)\) forward passes.
   - **Over-Fetch + Rerank Pipeline**: When `rerank=True` (default), `POST /search` over-fetches `max(limit * rerank_candidate_multiplier, rerank_min_candidates)` (default `max(limit * 4, 20)`) candidates from ClickHouse and reranks them to `top_n=limit` on a dedicated `"reranker"` `ThreadPoolExecutor`, breaking score ties deterministically by Stage-1 rank order and validating score finiteness (`math.isfinite`) before sorting.
4. **Async ClickHouse `VectorRepository` (`src/rag_api/repository.py`)**
   - Uses `clickhouse_connect.get_async_client(..., autogenerate_session_id=False)` with lazy connection initialization under an `asyncio.Lock`.
   - Disabling `autogenerate_session_id` prevents concurrent `async` requests from sharing a single ClickHouse `session_id` and failing with `SESSION_IS_LOCKED`.
   - Deferring client creation until the first request allows the application to boot cleanly even if ClickHouse is temporarily unreachable at startup, serving `503 Service Unavailable` on `/health` and `/search` and recovering automatically once ClickHouse comes online.
5. **Multi-Stage Non-Root Docker Build & CPU-Only PyTorch on Linux (`Dockerfile`, `pyproject.toml`, `docker-compose.yml`)**
   - `pyproject.toml` pins `torch` on Linux to the official PyTorch CPU wheel index (`https://download.pytorch.org/whl/cpu`) via environment markers while keeping standard PyPI wheels on macOS, avoiding ~2.5 GB of unnecessary CUDA packages in the Linux container image.
   - Multi-stage `Dockerfile` builds the virtual environment in `ghcr.io/astral-sh/uv:python3.12-bookworm-slim` and copies only `/app/.venv` into a minimal `python:3.12-slim-bookworm` runtime image running as non-root `appuser` (UID/GID `10001`).

---

## Next Steps

- **Document-Level Deduplication / Grouping by `title`**: Because `embeddings.vectors` stores multiple chunks per Markdown file (`metadata` is a JSON object such as `{"source_file": "lightdash-and-clickhouse.md"}` from which `source_file` is extracted as `title`, and `document` holds the chunk text), matching chunks are currently returned as-is so multiple chunks from the same document can appear in the top-K results. Depending on downstream prompt construction needs, we can deduplicate by the extracted `source_file` in SQL (`LIMIT 1 BY JSONExtractString(metadata, 'source_file')`) or group adjacent chunks per document in Python.
- **Vector Similarity Index (`vector_similarity`)**: For larger datasets beyond exhaustive `cosineDistance` scans, add an HNSW vector similarity index on `vectors.embeddings` in ClickHouse.
- **Hybrid Search (BM25 + Vector RRF)**: Combine ClickHouse full-text / BM25 keyword matching with vector cosine distance via Reciprocal Rank Fusion before feeding candidates into the cross-encoder reranker.
- **Query Embedding Cache & Observability**: Add an LRU/Redis cache for repeated query embeddings and OpenTelemetry traces/metrics around embedding latency, ClickHouse query duration, and cross-encoder reranking latency.

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
