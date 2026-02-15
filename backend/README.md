# fastapi_skeleton

Simple fastapi skeleton for a stateless microservice (application for main models, optimization, ...)

## Architecture

The project follows a layered architecture to ensure separation of concerns:

- **v1 (Routes)**: Handles HTTP requests, input validation using Pydantic schemas, and response serialization. This layer is responsible for the API interface.
- **Services**: Contains the core business logic and orchestrates interactions with external systems such as the Qdrant vector database and LLM providers. This layer is independent of the HTTP transport details.


## Model Preparation

Before running the application, you need to prepare the models by converting them to ONNX format.

1. Ensure your environment variables are set correctly in `dev.env` (or your active environment file). Specifically check `bi_encoder` and `cross_encoder`.
2. Run the conversion script:

```bash
uv run python scripts/onnx_conversion.py
```

3. Verify the models are working correctly:

```bash
uv run python scripts/onnx_test.py
```

## Data Ingestion

After model preparation and before using the RAG endpoints, you need to ingest your documents into Qdrant.

### Quick Start

**IMPORTANT**: The API must be running before ingestion!

```bash
# 1. Start the API (if not already running)
make dev

# 2. In a new terminal, run the ingestion script
cd backend
uv run python scripts/ingest_documents.py
```

This will:
- Load all markdown files from `../data/`
- Chunk them into ~500 character pieces with 50-character overlap
- Call the API to generate embeddings and insert into Qdrant
- Store metadata (filename, filepath, chunk index)

### Advanced Usage

See [INGESTION.md](./INGESTION.md) for:
- Interactive notebook for exploration
- Custom chunking parameters
- API-based ingestion details
- Metadata structure
- Troubleshooting guide

## Running service manually

To run the service manually in debug mode ensure you have `uv` installed.

You can run the service in debug mode using the Makefile:

```bash
make dev
```

Or manually:

```bash
export FASTAPI_ENV="DEV"
./run_app.sh
```

## Running service in Docker

To build the Docker image:

`docker build -t "fastapi-api:latest" .`

To run the Docker image:

```
docker run -p 8000:8000 -ti fastapi-api:latest
```

## Local querying

To check that the service is alive, run:

`curl -X GET "http://localhost:8000/health" -H  "accept: application/json"`

## API Documentation

The user interface for the API is defined in `http://localhost:8000/docs` endpoint.

## Testing

To run the tests:

```bash
make test
```

Or directly with uv:

`uv run pytest`

## End-to-End Testing

End-to-end tests verify the entire application flow, including the database and other services.

To run local e2e tests (requires services to be running):

```bash
make test-e2e
```

To run e2e tests in a Docker environment (spins up necessary containers):

```bash
make test-docker
```
