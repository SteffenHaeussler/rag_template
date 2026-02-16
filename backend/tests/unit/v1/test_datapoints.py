from fastapi import status
from unittest.mock import MagicMock
from qdrant_client.models import PointStruct


def test_insert_datapoint_success(client, mock_qdrant):
    mock_qdrant.collection_exists.return_value = True

    payload = {
        "text": "sample text",
        "metadata": {"source": "test"}
    }
    response = client.post("/v1/collections/test_collection/datapoints/", json=payload)

    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["status"] == "inserted"
    assert "id" in data

    # Verify upsert call
    mock_qdrant.upsert.assert_called_once()
    call_args = mock_qdrant.upsert.call_args
    assert call_args.kwargs["collection_name"] == "test_collection"
    points = call_args.kwargs["points"]
    assert len(points) == 1
    assert points[0].payload["text"] == "sample text"
    assert points[0].payload["source"] == "test"


def test_insert_bulk_datapoints(client, mock_qdrant, mock_models):
    """Test bulk insert with batch embedding."""
    mock_qdrant.collection_exists.return_value = True

    # Mock get_collection for dimension validation
    mock_collection_info = MagicMock()
    mock_collection_info.config.params.vectors.size = 384
    mock_qdrant.get_collection.return_value = mock_collection_info

    # Mock batch embedding to return 2 embeddings
    import numpy as np
    mock_output = MagicMock()
    mock_output.last_hidden_state = np.array([
        [[0.1] * 384, [0.1] * 384],  # text1
        [[0.2] * 384, [0.2] * 384],  # text2
    ])
    mock_models["bi_tokenizer"].return_value = {"input_ids": []}
    mock_models["bi_encoder"].return_value = mock_output

    payload = [
        {"text": "text1", "metadata": {"id": 1}},
        {"text": "text2", "metadata": {"id": 2}}
    ]
    response = client.post("/v1/collections/test_collection/datapoints/bulk", json=payload)

    assert response.status_code == status.HTTP_201_CREATED
    assert response.json()["inserted_count"] == 2

    mock_qdrant.upsert.assert_called()


def test_get_datapoint_success(client, mock_qdrant):
    mock_point = MagicMock()
    mock_point.id = "uuid-1"
    mock_point.payload = {"text": "found me", "meta": "data"}
    mock_qdrant.retrieve.return_value = [mock_point]

    response = client.get("/v1/collections/test_collection/datapoints/12345")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["text"] == "found me"
    assert data["metadata"]["meta"] == "data"


def test_get_datapoint_not_found(client, mock_qdrant):
    mock_qdrant.retrieve.return_value = []

    response = client.get("/v1/collections/test_collection/datapoints/12345")

    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_get_datapoint_embedding_success(client, mock_qdrant):
    mock_point = MagicMock()
    mock_point.id = "uuid-1"
    mock_point.vector = [0.1] * 384
    mock_qdrant.retrieve.return_value = [mock_point]

    response = client.get("/v1/collections/test_collection/datapoints/12345/embedding")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "embedding" in data
    assert len(data["embedding"]) == 384


def test_update_datapoint_success(client, mock_qdrant):
    mock_point = MagicMock()
    mock_point.id = "uuid-1"
    mock_point.vector = [0.1] * 384
    mock_point.payload = {"text": "original", "other": "keep"}
    mock_qdrant.retrieve.return_value = [mock_point]

    payload = {
        "text": "updated text",
        "metadata": {"new": "meta"}
    }
    response = client.put("/v1/collections/test_collection/datapoints/12345", json=payload)

    assert response.status_code == status.HTTP_200_OK

    # Verify upsert with merged payload and new vector
    mock_qdrant.upsert.assert_called()
    call_args = mock_qdrant.upsert.call_args
    points = call_args.kwargs["points"]
    assert points[0].payload["text"] == "updated text"
    assert points[0].payload["other"] == "keep"
    assert points[0].payload["new"] == "meta"


def test_delete_datapoint(client, mock_qdrant):
    response = client.delete("/v1/collections/test_collection/datapoints/12345")

    assert response.status_code == status.HTTP_204_NO_CONTENT
    mock_qdrant.delete.assert_called_once()
