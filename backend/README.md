# fastapi_skeleton

Simple fastapi skeleton for a stateless microservice (application for main models, optimization, ...)

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
