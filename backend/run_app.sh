#!/bin/sh

if [ "$FASTAPI_ENV" = "PROD" ]; then
	uv run --env-file dev.env uvicorn src.app.main:app --port 8000 --workers 2 --log-level "error"
elif [ "$FASTAPI_ENV" = "TEST" ]; then
	uv run pytest --cov-report html --cov=app tests
else
	 uv run --env-file dev.env uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level "debug"
fi
