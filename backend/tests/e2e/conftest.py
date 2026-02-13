import os
import pytest
import httpx
from qdrant_client import QdrantClient

@pytest.fixture(scope="session")
def backend_url():
    """Base URL for the backend API."""
    return os.getenv("BACKEND_URL", "http://localhost:8000")

@pytest.fixture(scope="session")
def qdrant_host():
    """Hostname for Qdrant service."""
    return os.getenv("QDRANT_HOST", "localhost")

@pytest.fixture(scope="session")
def qdrant_client(qdrant_host):
    """Qdrant client instance."""
    return QdrantClient(host=qdrant_host, port=6333)

@pytest.fixture(scope="session")
def http_client(backend_url):
    """HTTP client for backend requests."""
    with httpx.Client(base_url=backend_url, timeout=10.0) as client:
        yield client
