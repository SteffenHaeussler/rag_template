.PHONY: dev prod test clean # Declare phony targets

export COMPOSE_DOCKER_CLI_BUILD=1
export DOCKER_BUILDKIT=1

all: down build up test

dev:
	@echo "Setting FASTAPI_ENV to DEV and running application..."
	FASTAPI_ENV=DEV ./run_app.sh
prod:
	@echo "Setting FASTAPI_ENV to PROD and running application..."
	FASTAPI_ENV=PROD ./run_app.sh
run:
	@echo "Setting FASTAPI_ENV to TEST and running tests..."
	FASTAPI_ENV=TEST ./run_app.sh

test:
	uv run pytest --ignore=tests/e2e

coverage:
	uv run pytest --cov=src


build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down --remove-orphans

notebook:
	uv run jupyter lab --ip=0.0.0.0 --port=8888 --notebook-dir=./notebooks

nb: notebook

test-e2e:
	uv run pytest tests/e2e

test-docker:
	docker compose -f compose.yaml --profile test up tests --build --abort-on-container-exit --exit-code-from tests

