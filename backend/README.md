# fastapi_skeleton

Simple fastapi skeleton for a stateless microservice (application for main models, optimization, ...)

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
