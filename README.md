## Work assignment

Timebox: Aim for 3–6 hours. If you run out of time, include a "Next steps" section.

We are looking for a Senior Software Engineer to drive the development of AI/ML-powered features in ClickHouse Cloud and database integrations. Given the current project, let's create a retrieval component of a RAG pipeline. This project is used as a conversation piece for the next stages of the interview process.

Implement the following feature in Python

- Create an API that takes a query string and returns a list of document titles and document bodies in JSON.

The following extensions are optional and will not receive more weight than a complete, well-tested core implementation. Choose at most one if time permits.

- Reranking
- Hybrid keyword/vector search
- Filtering
- Caching
- Observability
- LLM answer generation
- Reindexing with another model

Note:

We are not reviewing the quality of the output of the embeddings model - just the integration work. If you use a 3rd party with API key, please provide instructions for the reviewer(s) to obtain their own key - commercial models are fine. If you have any questions please reach out to the hiring manager or recruiter. Feel free to use an agentic coding tool, but be prepared to explain any code you submit.

## Setup

- Requires git
- Requires Python 3.12+
- Requires uv 0.12.15+
- Requires a vector database, here is the available vector database that we created for the assignment: [vector database available](#clickhouse-vector-database) 
- Requires Docker

## Example to conceptually understand our ask

One of the goals of the RAG pipeline and API is to reply correctly to the questions of our users. To increase the chance of replying correctly, we want to provide the right context by selecting the relevant text(s) in our docs.

As an example:
*Input* user's query string:
```
Can I create a visualization with Lightdash?
```
*API output* (would need to be JSON output, you can modify how this output looks like and disregard the `title` and `body` values below)
It is just an example and **by no means your solution needs to match this input-output pair**:

```
"body": "The \nExplore\n page is made up of five main areas:\n\n\n\n\nDimensions and Metrics\n — all fields available on the selected table ...",
"title": "lightdash-and-clickhouse.md"

"body": "sidebar_label: 'Lightdash'\nsidebar_position: 131\nslug: /integrations/lightdash\nkeywords: ['clickhouse', ...",
"title": "lightdash-and-clickhouse.md"

"body": "sidebar_label: 'Overview'\nsidebar_position: 1\nkeywords: ['ClickHouse', 'connect', 'Luzmo', 'Explo', 'Fabi.ai', 'Tableau',...",
"title": "index.md"
```

## ClickHouse Vector Database

A Docker Compose configuration is provided to run ClickHouse with vector search capabilities. The schema of the table `vectors` is defined in [clickhouse/docker-entrypoint-initdb.d/01-init-tables.sql](clickhouse/docker-entrypoint-initdb.d/01-init-tables.sql).

The model used to create the embeddings is: `sentence-transformers/all-MiniLM-L6-v2`.

Feel free to use a different embedding model and reindex the documents

**Start the database:**

```bash
docker compose up -d
```

**Connection details:**
```
HOST: localhost:8123
CLICKHOUSE_DB: embeddings
CLICKHOUSE_USER: vectordbuser
CLICKHOUSE_PASSWORD: pwd_vectordb
```

**Run the test_connection.py script**

After you create and run the Docker container, you can test that all works by running:

```bash
uv sync
uv run test_connection.py
```
