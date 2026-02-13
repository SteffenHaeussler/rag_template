def test_backend_health(http_client):
    """Verify backend health endpoint is reachable."""
    response = http_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "timestamp" in data
    assert "version" in data

def test_qdrant_connectivity(qdrant_client):
    """Verify Qdrant is reachable."""
    collections = qdrant_client.get_collections()
    assert collections is not None
