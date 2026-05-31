# Build an Autonomous Customer Support RAG Platform

This repository provides a FastAPI-based RAG platform with pgvector, BM25 retrieval, and evaluator tooling.

## CI & Smoke Tests

- CI docs: see `docs/CI.md` for workflow details and secrets configuration.
- Local smoke test (uses Docker Compose):

```bash
make smoke
```

This runs `docker compose up --build`, executes the smoke test in `scripts/compose_smoke_test.py`, then tears the stack down.

If you run CI via GitHub Actions, set the secret `OPENAI_API_KEY` in your repository settings for full integration tests; otherwise the workflows will warn and run with deterministic fallbacks.
# IntelliSupport

IntelliSupport is a production-style autonomous customer support platform for Nexora, a fictional B2B project management SaaS company. The system ingests knowledge base articles, chunks and embeds them, retrieves evidence with a hybrid dense+sparse search stack, and generates grounded answers for customer questions. The implementation is built from first principles with raw SQL, pgvector, BM25, FastAPI, Pydantic v2, and OpenAI-compatible components.

The focus of this project is reliability over abstraction. Retrieval is implemented directly against PostgreSQL using pgvector cosine distance and an in-memory BM25 index. Response generation is grounded by prompt construction that includes explicit chunk citations, and the evaluation layer measures faithfulness and relevance so the pipeline can be benchmarked instead of judged by intuition. The repository also includes a feedback loop, health endpoint, containerization, and an automated benchmark runner.

## Architecture

```text
Customer Query
    |
    v
FastAPI /query
    |
    +--> Intent Classifier (OpenAI JSON mode or heuristic fallback)
    |
    +--> Embedder (text-embedding-3-small or deterministic fallback)
    |
    +--> Hybrid Retriever
    |        |-- pgvector cosine search
    |        |-- BM25 keyword search
    |        `-- Jaccard reranking
    |
    +--> Prompt Builder
    |
    +--> Response Generator (OpenAI or grounded fallback)
    |
    +--> Store query + response in PostgreSQL
    |
    v
Client Response
    |
    v
/evaluate/{response_id}
    |
    +--> Faithfulness Judge
    +--> Relevance Judge
    +--> Persist scores in PostgreSQL
```

## Setup

1. Create a virtual environment and install dependencies.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Create a `.env` file from `.env.example` and set your values.

```bash
copy .env.example .env
```

3. Start PostgreSQL with pgvector.

```bash
docker compose up -d db
```

To run the full Dockerized stack, build and start both services:

```bash
docker compose up --build
```

The API container runs the database migration on startup and waits for Postgres health before serving traffic.

4. Apply the migration and seed the knowledge base.

The benchmark script applies `database/migrations/001_initial.sql`, truncates any prior data, seeds the 10 Nexora documents, chunks them, embeds them, and prints benchmark metrics.

```bash
python scripts/run_benchmark.py
```

Docker startup performs the same database preparation automatically through `docker-entrypoint.sh`, which runs the migration and seed scripts before launching Uvicorn.

## Running the API

Start the FastAPI app with Uvicorn:

```bash
uvicorn api.main:app --reload
```

Health check:

```bash
GET /health
```

Main endpoints:

```text
POST /query
POST /evaluate/{response_id}
POST /feedback
GET /feedback/summary/{response_id}
GET /health
```

## Running Tests

Run the full test suite:

```bash
pytest -q tests
```

## Evaluation Results

Measured with the included benchmark runner on the seeded Nexora corpus:

| Metric | Your Score | Threshold |
|---|---:|---:|
| Retrieval Hit Rate | 1.00 | >= 0.60 |
| Intent Accuracy | 0.875 | >= 0.75 |
| Avg Faithfulness | 0.80 | >= 0.60 |
| Avg Relevance | 0.7625 | >= 0.60 |

## Design Decisions

1. Raw SQL was used everywhere instead of an ORM so the data model, pgvector usage, and evaluation writes stay explicit and easy to audit.
2. The retrieval layer separates dense search, BM25 search, and hybrid merge logic so each component can be tested and tuned independently.
3. The generation path uses strict prompts plus a grounded fallback response path so the API remains usable even when OpenAI credentials are unavailable.
4. The benchmark runner seeds the full corpus before evaluation, which makes retrieval and generation metrics reproducible on a fresh database.

## Notes

- All required tables live under the `intellisupport` PostgreSQL schema.
- The project uses `text-embedding-3-small` for embeddings and `gpt-4o-mini` for generation and judge tasks when an API key is provided.
- When `OPENAI_API_KEY` is not set, the pipeline falls back to deterministic local behavior so development and testing can still proceed.
