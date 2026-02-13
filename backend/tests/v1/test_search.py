from fastapi import status
from unittest.mock import MagicMock


def test_embedding(client, mock_models):
    response = client.post("/v1/embedding/", json={"text": "hello world"})

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["text"] == "hello world"
    assert len(data["embedding"]) == 384

    mock_models["bi_encoder"].assert_called()


def test_ranking(client, mock_models):
    payload = {
        "question": "query",
        "texts": ["relevant", "irrelevant"]
    }
    response = client.post("/v1/ranking/", json=payload)

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data["results"]) == 2
    # Check if results are sorted by score (mock returns [0.9, 0.1])
    assert data["results"][0]["score"] > data["results"][1]["score"]

    mock_models["cross_encoder"].assert_called()


def test_search(client, mock_qdrant):
    mock_point = MagicMock()
    mock_point.id = "uuid-1"
    mock_point.score = 0.95
    mock_point.payload = {"text": "match", "meta": "data"}

    mock_search_result = MagicMock()
    mock_search_result.points = [mock_point]
    mock_qdrant.query_points.return_value = mock_search_result

    payload = {
        "embedding": [0.1] * 384,
        "n_retrieval": 10
    }
    response = client.post("/v1/collections/test_collection/search/", json=payload)

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["text"] == "match"


def test_full_rag_pipeline(client, mock_qdrant, mock_models):
    # Mock retrieval
    mock_point = MagicMock()
    mock_point.id = "uuid-1"
    mock_point.payload = {"text": "match"}
    mock_search_result = MagicMock()
    mock_search_result.points = [mock_point]
    mock_qdrant.query_points.return_value = mock_search_result

    # Mock reranking is handled by mock_models fixture returning logits

    payload = {
        "question": "test query",
        "collection_name": "test_collection"
    }
    response = client.post("/v1/query/", json=payload)

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["text"] == "match"
