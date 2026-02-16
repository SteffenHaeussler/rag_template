from fastapi import status
from unittest.mock import MagicMock
from qdrant_client.models import Distance, VectorParams


def test_create_collection_success(client, mock_qdrant):
    mock_qdrant.collection_exists.return_value = False

    payload = {
        "name": "test_collection",
        "dimension": 384,
        "distance_metric": "cosine"
    }
    response = client.post("/v1/collections/", json=payload)

    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["name"] == "test_collection"
    assert data["status"] == "created"

    mock_qdrant.create_collection.assert_called_once()
    call_args = mock_qdrant.create_collection.call_args
    assert call_args.kwargs["collection_name"] == "test_collection"
    assert call_args.kwargs["vectors_config"].size == 384
    assert call_args.kwargs["vectors_config"].distance == Distance.COSINE


def test_create_collection_already_exists(client, mock_qdrant):
    mock_qdrant.collection_exists.return_value = True

    payload = {
        "name": "test_collection",
        "dimension": 384,
        "distance_metric": "cosine"
    }
    response = client.post("/v1/collections/", json=payload)

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Collection already exists" in response.json()["detail"]


def test_create_collection_invalid_metric(client, mock_qdrant):
    mock_qdrant.collection_exists.return_value = False

    payload = {
        "name": "test_collection",
        "dimension": 384,
        "distance_metric": "invalid_metric"
    }
    response = client.post("/v1/collections/", json=payload)

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Unknown distance metric" in response.json()["detail"]


def test_delete_collection_success(client, mock_qdrant):
    mock_qdrant.collection_exists.return_value = True

    response = client.delete("/v1/collections/test_collection")

    assert response.status_code == status.HTTP_204_NO_CONTENT
    mock_qdrant.delete_collection.assert_called_with("test_collection")


def test_delete_collection_not_found(client, mock_qdrant):
    mock_qdrant.collection_exists.return_value = False

    response = client.delete("/v1/collections/unknown_collection")

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "Collection not found" in response.json()["detail"]
