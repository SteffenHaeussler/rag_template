# RAG Service

A full-stack Retrieval-Augmented Generation (RAG) service with a microservice architecture.

## Architecture

```
                ┌────────────┐
                │  Streamlit  │  (frontend)
                │   :8501     │
                └──────┬─────┘
                       │ HTTP
                ┌──────▼─────┐
                │   FastAPI   │  (backend)
                │   :8000     │
                └──┬──────┬──┘
                   │      │
          Qdrant   │      │  Gemini API
          query    │      │  (generation)
                   │      │
                ┌──▼──┐   │
                │Qdrant│   │
                │:6333 │   │
                └──▲──┘   │
                   │
          embed +  │
          store    │
        ┌──────────┘
        │
  ┌─────┴──────┐
  │  Ingestion  │  (one-shot job)
  │  container  │
  └─────────────┘
```

**4 services**, 3 long-running + 1 one-shot:

| Service | Description | Port |
|---------|-------------|------|
| **Qdrant** | Vector database (official image) | 6333 |
| **Backend** | FastAPI — RAG queries (retrieve + generate) | 8000 |
| **Frontend** | Streamlit chat UI | 8501 |
| **Ingestion** | One-shot job: reads `data/`, chunks, embeds, stores in Qdrant | — |

## Quick Start

### Prerequisites

- Docker and Docker Compose
- A [Gemini API key](https://aistudio.google.com/apikey) (free tier works)

### Setup

```bash
# 1. Clone the repo
git clone <repo-url> && cd rag_template

# 2. Create .env with your Gemini API key
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your-key-here

# 3. (Optional) Add your own documents to data/

# 4. Build and run everything
./scripts/run.sh
```

### Usage

- **Chat UI**: http://localhost:8501
- **API**: http://localhost:8000
- **Health check**: http://localhost:8000/health

### Stop

```bash
docker compose down
```

## API

### `POST /query`

```json
{
  "question": "What is RAG?",
  "top_k": 3
}
```

Response:

```json
{
  "answer": "RAG stands for Retrieval-Augmented Generation...",
  "sources": [
    {
      "text": "Retrieval-Augmented Generation (RAG) is an AI framework...",
      "filename": "sample1.md",
      "score": 0.92
    }
  ]
}
```

### `GET /health`

```json
{
  "status": "ok",
  "qdrant": "connected"
}
```

## Project Structure

```
rag_template/
├── docker-compose.yml
├── .env.example
├── scripts/run.sh
├── data/                     # Documents to ingest
├── ingestion/                # Ingestion pipeline service
│   ├── src/ingestion/
│   │   ├── main.py           # Entry point: load → chunk → embed → store
│   │   ├── loader.py         # File reader (.txt, .md)
│   │   ├── chunker.py        # Text chunking with overlap
│   │   └── config.py         # Environment config
│   └── tests/
├── backend/                  # FastAPI backend service
│   ├── src/api/
│   │   ├── main.py           # App + lifespan
│   │   ├── routes.py         # POST /query, GET /health
│   │   ├── rag.py            # Retrieve + generate
│   │   ├── schemas.py        # Pydantic models
│   │   └── config.py         # Environment config
│   └── tests/
└── frontend/                 # Streamlit frontend
    └── src/frontend/app.py
```

## Configuration

All configuration is via environment variables (`.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | **Required.** Google Gemini API key |
| `QDRANT_HOST` | `qdrant` | Qdrant hostname |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `COLLECTION_NAME` | `documents` | Qdrant collection name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers embedding model |
| `GENERATION_MODEL` | `gemini-2.0-flash` | Gemini generation model |

## Running Tests

```bash
# Ingestion tests
cd ingestion && pip install -e ".[dev]" && pytest

# Backend tests
cd backend && pip install -e ".[dev]" && pytest
```

## Design Decisions

- **Sentence-transformers** for embeddings (`all-MiniLM-L6-v2`) — local inference, no API key needed for embeddings, consistent model across ingestion and serving.
- **Fixed-size chunking with overlap** — simple, effective, ~500 tokens per chunk with 50-token overlap.
- **Ingestion as a one-shot container** — uses Docker Compose profiles, runs only when explicitly invoked.
- **All configuration via environment variables** — follows the twelve-factor app methodology.
