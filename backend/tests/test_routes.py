from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient


def _make_app():
    """Create a test app with mocked clients."""
    import api.main as main_mod

    mock_gemini = MagicMock()
    mock_qdrant = MagicMock()
    mock_embedder = MagicMock()
    main_mod._app_state["gemini"] = mock_gemini
    main_mod._app_state["qdrant"] = mock_qdrant
    main_mod._app_state["embedder"] = mock_embedder

    return main_mod.app, mock_gemini, mock_qdrant, mock_embedder


def test_health_connected():
    app, _, mock_qdrant, _ = _make_app()
    mock_qdrant.get_collections.return_value = []

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["qdrant"] == "connected"


def test_health_disconnected():
    app, _, mock_qdrant, _ = _make_app()
    mock_qdrant.get_collections.side_effect = Exception("connection refused")

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["qdrant"] == "disconnected"


def test_query_success():
    app, mock_gemini, mock_qdrant, mock_embedder = _make_app()

    # Mock embedding — sentence-transformers returns a numpy array
    mock_embedder.encode.return_value = np.array([0.1] * 384)

    # Mock Qdrant search
    mock_point = MagicMock()
    mock_point.payload = {
        "text": "Sample context text",
        "filename": "sample.md",
        "chunk_index": 0,
    }
    mock_point.score = 0.95
    mock_search_result = MagicMock()
    mock_search_result.points = [mock_point]
    mock_qdrant.query_points.return_value = mock_search_result

    # Mock generation
    mock_gen_result = MagicMock()
    mock_gen_result.text = "This is the answer."
    mock_gemini.models.generate_content.return_value = mock_gen_result

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/query", json={"question": "What is this about?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "This is the answer."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["filename"] == "sample.md"
