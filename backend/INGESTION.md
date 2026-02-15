# Data Ingestion Pipeline

This document describes how to ingest markdown documents into Qdrant for the RAG system.

## Overview

The ingestion pipeline:
1. Loads markdown files from the data directory
2. Chunks the text into manageable pieces (default: 500 characters with 50 character overlap)
3. Calls the API to generate embeddings and insert into Qdrant
4. Stores chunks with metadata (filename, filepath, chunk index)

## Prerequisites

**IMPORTANT**: The API server must be running before using the ingestion tools!

1. Start the services:
   ```bash
   cd backend
   make dev  # or: docker compose up
   ```

2. Verify the API is accessible:
   ```bash
   curl http://localhost:8000/v1/health
   ```

3. Have markdown files ready in the `../data/` directory

## Method 1: Interactive Notebook

Use the Jupyter notebook for interactive exploration and testing:

```bash
cd backend
jupyter notebook notebooks/data_ingestion.ipynb
```

The notebook allows you to:
- Step through the process interactively
- Inspect chunks and embeddings
- Test retrieval immediately after ingestion
- Adjust parameters on the fly

## Method 2: Production Script

Use the Python script for automated/production ingestion.

**IMPORTANT**: Make sure the API is running first (`make dev`)

### Basic Usage

```bash
cd backend
python scripts/ingest_documents.py
```

This will connect to the API at `http://localhost:8000` and ingest documents.

### Advanced Usage

```bash
# Custom data directory
python scripts/ingest_documents.py --data-dir /path/to/markdown/files

# Custom collection name
python scripts/ingest_documents.py --collection-name my_collection

# Custom chunking parameters (in characters)
python scripts/ingest_documents.py --chunk-size 1000 --chunk-overlap 100

# Custom API URL (if running on different host/port)
python scripts/ingest_documents.py --api-url http://api.example.com:8000

# Skip collection creation (if collection already exists)
python scripts/ingest_documents.py --skip-create-collection

# Full example with all options
python scripts/ingest_documents.py \
  --data-dir ../data \
  --api-url http://localhost:8000 \
  --collection-name documents \
  --chunk-size 500 \
  --chunk-overlap 50 \
  --batch-size 50
```

### Command Line Options

- `--data-dir`: Directory containing markdown files (default: `../data`)
- `--api-url`: API base URL (default: `http://localhost:8000`)
- `--collection-name`: Collection name (default: `temp`)
- `--chunk-size`: Chunk size in characters (default: `500`)
- `--chunk-overlap`: Chunk overlap in characters (default: `50`)
- `--batch-size`: Batch size for ingestion (default: `50`)
- `--skip-create-collection`: Skip collection creation (useful if collection exists)

## Metadata Structure

Each ingested chunk includes the following metadata:

```python
{
    "text": "The actual text content of the chunk",
    "filename": "sample1.md",
    "filepath": "/full/path/to/sample1.md",
    "chunk_index": 0,  # Index of this chunk within the document
    "total_chunks": 5   # Total number of chunks for this document
}
```

## Chunking Strategy

The default chunking strategy uses:
- **Paragraph-aware chunking**: Splits text at paragraph boundaries when possible
- **Character-based sizing**: Targets ~500 characters per chunk
- **Overlap**: 50 characters overlap between consecutive chunks to preserve context

This ensures:
- Semantic coherence (respects paragraph boundaries)
- No information loss at chunk boundaries
- Reasonable chunk sizes for the embedding model

## Verifying Ingestion

After ingestion, verify the data:

```python
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)
collection_info = client.get_collection("temp")
print(f"Points in collection: {collection_info.points_count}")
```

Or use the notebook's verification cells.

## Troubleshooting

### Collection already exists error

If you get an error that the collection already exists:

```bash
# Option 1: Use skip flag
python scripts/ingest_documents.py --skip-create-collection

# Option 2: Delete and recreate (caution: data loss!)
# Use the notebook or Qdrant API to delete the collection first
```

### API connection refused

Ensure the API server is running:

```bash
# Start the API
make dev

# Or via Docker Compose
docker compose up -d

# Check API is accessible
curl http://localhost:8000/v1/health
```

If the API starts but fails to connect to Qdrant, check that Qdrant is running and accessible.

## Next Steps

After ingestion:
1. Test retrieval via the API: `GET /v1/search`
2. Test RAG via: `POST /v1/chat/rag`
3. Monitor Qdrant at: http://localhost:6333/dashboard
