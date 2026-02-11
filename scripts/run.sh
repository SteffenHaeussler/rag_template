#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Check for .env file
if [ ! -f .env ]; then
    echo "ERROR: .env file not found."
    echo "Copy .env.example to .env and set your GEMINI_API_KEY:"
    echo "  cp .env.example .env"
    exit 1
fi

echo "==> Building all images..."
docker compose build

echo "==> Starting Qdrant, Backend, and Frontend..."
docker compose up -d qdrant backend frontend

echo "==> Waiting for Qdrant to be healthy..."
timeout=60
elapsed=0
until docker compose exec qdrant bash -c "exec 3<>/dev/tcp/localhost/6333" 2>/dev/null; do
    if [ $elapsed -ge $timeout ]; then
        echo "ERROR: Qdrant did not become healthy within ${timeout}s"
        exit 1
    fi
    sleep 2
    elapsed=$((elapsed + 2))
    echo "  Waiting... (${elapsed}s)"
done
echo "  Qdrant is ready."

echo "==> Running ingestion pipeline..."
docker compose run --rm ingestion

echo ""
echo "==> All services are running!"
echo "  Frontend:  http://localhost:8501"
echo "  Backend:   http://localhost:8000"
echo "  Health:    http://localhost:8000/health"
echo ""
echo "To stop all services: docker compose down"
