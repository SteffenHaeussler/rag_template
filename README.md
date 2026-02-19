# RAG Service

A FastAPI-based Retrieval-Augmented Generation (RAG) microservice template.

## Architecture

```
                ┌────────────────────────────────────┐
                │           FastAPI Backend           │
                │              :8000                  │
                │                                     │
                │  ┌─────────────┐  ┌──────────────┐ │
                │  │  Retrieval  │  │  Generation  │ │
                │  │  Service    │  │  Service     │ │
                │  │             │  │              │ │
                │  │ bi-encoder  │  │  Gemini API  │ │
                │  │  (ONNX)     │  │  (litellm)   │ │
                │  │             │  │              │ │
                │  │cross-encoder│  └──────────────┘ │
                │  │  (ONNX)     │                   │
                │  └──────┬──────┘                   │
                └─────────┼───────────────────────────┘
                          │
                    ┌─────▼─────┐
                    │  Qdrant   │
                    │  :6333    │
                    └───────────┘
```

**2 services:**

| Service | Description | Port |
|---------|-------------|------|
| **Backend** | FastAPI — embedding, reranking, retrieval, generation, RAG | 8000 |
| **Qdrant** | Vector database (official image) | 6333 / 6334 |

## Quick Start

### Prerequisites

- Docker and Docker Compose
- A [Gemini API key](https://aistudio.google.com/apikey) (free tier works)
- `uv` for local development

### Setup

```bash
# 1. Clone the repo
git clone <repo-url> && cd rag_template/backend

# 2. Prepare ONNX models
uv run python scripts/onnx_conversion.py

# 3. Configure environment
cp dev.env .env
# Edit .env and set GEMINI_API_KEY=your-key-here

# 4. Start services
docker compose up --build -d

# 5. Ingest documents
uv run python scripts/ingest_documents.py
```

### Running locally (without Docker)

```bash
cd backend
make dev
# OR
export FASTAPI_ENV="dev"
./run_app.sh
```

### Stop

```bash
docker compose down
```

## API

All endpoints are prefixed with `/v1`. Interactive docs: `http://localhost:8000/docs`

### Health

```
GET  /health           # Core health check
GET  /v1/health        # Versioned health check (returns version + timestamp)
```

### Collections

```
POST   /v1/collections/                    # Create a collection
DELETE /v1/collections/{collection_name}   # Delete a collection
```

### Datapoints

```
POST   /v1/collections/{name}/datapoints/           # Insert single datapoint
POST   /v1/collections/{name}/datapoints/bulk       # Bulk insert (batch embedding, 10-50x faster)
GET    /v1/collections/{name}/datapoints/{id}       # Get datapoint
GET    /v1/collections/{name}/datapoints/{id}/embedding  # Get datapoint embedding
PUT    /v1/collections/{name}/datapoints/{id}       # Update datapoint
DELETE /v1/collections/{name}/datapoints/{id}       # Delete datapoint
```

### Embedding, Ranking & Search

```
POST /v1/embedding/                        # Generate embedding for text
POST /v1/ranking/                          # Rerank texts by relevance
POST /v1/collections/{name}/search/       # Vector search in a collection
```

### RAG

```
POST /v1/query/   # Retrieve context: embed + search + rerank (no generation)
POST /v1/chat/    # Generate answer from provided context (no retrieval)
POST /v1/rag/     # Full RAG pipeline: retrieve + generate
```

#### Example: Full RAG

```bash
curl -X POST http://localhost:8000/v1/rag/ \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is RAG?",
    "collection_name": "my_docs",
    "n_retrieval": 20,
    "n_ranking": 5
  }'
```

Response:

```json
{
  "answer": "RAG stands for Retrieval-Augmented Generation..."
}
```

## Project Structure

```
rag_template/
├── README.md
├── data/                         # Sample documents for ingestion
└── backend/
    ├── compose.yaml              # Docker Compose (backend + qdrant)
    ├── dev.env                   # Development environment config
    ├── Makefile                  # Common commands (dev, test, lint, ...)
    ├── pyproject.toml
    ├── models/
    │   ├── bi_encoder/           # ONNX bi-encoder (embeddings)
    │   └── cross_encoder/        # ONNX cross-encoder (reranking)
    ├── scripts/
    │   ├── onnx_conversion.py    # Convert sentence-transformers → ONNX
    │   ├── onnx_test.py          # Verify ONNX models
    │   └── ingest_documents.py   # Ingest documents from data/ into Qdrant
    ├── notebooks/
    │   ├── api_requests.ipynb    # Example API usage
    │   ├── data_ingestion.ipynb  # Interactive ingestion exploration
    │   └── qdrant_crud.ipynb     # Qdrant CRUD examples
    └── src/app/
        ├── main.py               # App entrypoint, lifespan, middleware
        ├── config.py             # Pydantic settings (env-driven)
        ├── core/                 # Core router (health check)
        ├── v1/
        │   ├── router.py         # All v1 API routes
        │   └── schema.py         # Pydantic request/response models
        ├── services/
        │   ├── retrieval.py      # Embedding, reranking, vector search
        │   └── generation.py     # LLM answer generation
        ├── prompts/
        │   └── generation.yaml   # Prompt templates
        ├── ingestion.py          # Document ingestion logic
        ├── exceptions.py         # Custom exception classes
        └── handlers.py           # Global exception handlers
```

## Configuration

All configuration is via environment variables (loaded from `dev.env` / `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | **Required.** Google Gemini API key |
| `bi_encoder` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model for embeddings |
| `bi_encoder_path` | `models/bi_encoder` | Path to ONNX bi-encoder |
| `cross_encoder` | `cross-encoder/ms-marco-MiniLM-L12-v2` | HuggingFace model for reranking |
| `cross_encoder_path` | `models/cross_encoder` | Path to ONNX cross-encoder |
| `generation_model` | `gemini/gemini-2.0-flash` | LiteLLM model string for generation |
| `temperature` | `0.0` | Generation temperature |
| `kb_host` | `localhost` | Qdrant hostname |
| `kb_port` | `6333` | Qdrant port |
| `kb_name` | `temp` | Default Qdrant collection name |
| `kb_limit` | `20` | Default number of retrieved results |
| `kb_batch_size` | `100` | Batch size for bulk upsert to Qdrant |
| `prompt_path` | `prompts/generation.yaml` | Path to prompt templates file |
| `prompt_key` | `prompt` | Which prompt template to use |
| `prompt_language` | `en` | Prompt language |

## Running Tests

```bash
cd backend

# Unit tests
make test

# End-to-end tests (requires services running)
make test-e2e

# E2E tests in Docker (spins up containers automatically)
make test-docker
```

## Model Preparation

Before first run, convert sentence-transformers models to ONNX:

```bash
cd backend
uv run python scripts/onnx_conversion.py
uv run python scripts/onnx_test.py   # verify
```

## Design Decisions

- **ONNX models** for embeddings and reranking — local inference, no API key needed, fast CPU inference.
- **Two-stage retrieval** — bi-encoder for fast approximate search, cross-encoder for precise reranking.
- **LiteLLM** for generation — swap the generation model by changing `generation_model` in config (supports OpenAI, Anthropic, Gemini, etc.).
- **Flexible ingestion** — ingest via the REST API (`/v1/collections/{name}/datapoints/bulk`) or the ingestion script.
- **All configuration via environment variables** — follows the twelve-factor app methodology.
