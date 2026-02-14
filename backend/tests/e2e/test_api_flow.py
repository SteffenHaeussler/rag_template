import pytest
from httpx import Client

def test_rag_lifecycle(http_client: Client):
    """
    Test the full RAG lifecycle:
    1. Create Collection
    2. Insert DataPoints
    3. Retrieval & Search
    4. Cleanup
    """
    COLLECTION_NAME = "e2e_test_collection"

    # 1. Create Collection
    payload = {
        "name": COLLECTION_NAME,
        "dimension": 384,
        "distance_metric": "cosine"
    }
    response = http_client.post("/v1/collections/", json=payload)
    # 201 Created, 200 OK, or 400 if exists
    assert response.status_code in [200, 201, 400]

    # 2. Insert DataPoint
    datapoint = {
        "id": 1,
        "text": "Jupyter notebooks are great for interactive testing.",
        "metadata": {"source": "manual_test", "author": "dev"}
    }
    response = http_client.post(
        f"/v1/collections/{COLLECTION_NAME}/datapoints/",
        json=datapoint
    )
    assert response.status_code == 201
    assert response.json()["status"] == "inserted"

    # 3. Insert Bulk
    bulk_payload = [
        {
            "id": 2,
            "text": "Deep Learning requires a lot of GPU power.",
            "metadata": {"category": "hardware"}
        },
        {
            "id": 3,
            "text": "Python is the primary language for AI.",
            "metadata": {"category": "language"}
        }
    ]
    response = http_client.post(
        f"/v1/collections/{COLLECTION_NAME}/datapoints/bulk",
        json=bulk_payload
    )
    assert response.status_code == 201
    assert response.json()["inserted_count"] == 2

    # 4. Get DataPoint
    response = http_client.get(f"/v1/collections/{COLLECTION_NAME}/datapoints/2")
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "Deep Learning requires a lot of GPU power."

    # 5. Get Embedding
    embed_payload = {"text": "What does Wärmedämmung mean?"}
    response = http_client.post("/v1/embedding/", json=embed_payload)
    assert response.status_code == 200
    query_vector = response.json()["embedding"]
    assert len(query_vector) == 384

    # 6. Search
    search_payload = {
        "embedding": query_vector,
        "n_items": 2,
    }
    response = http_client.post(f"/v1/collections/{COLLECTION_NAME}/search/", json=search_payload)
    assert response.status_code == 200
    results = response.json().get("results", [])
    assert len(results) > 0

    # 7. Ranking
    ranking_payload = {
        "question": "What does Wärmedämmung mean?",
        "texts": [
            "Python is the primary language for AI.",
            "Jupyter notebooks are great for interactive testing.",
        ]
    }
    response = http_client.post("/v1/ranking/", json=ranking_payload)
    assert response.status_code == 200
    assert len(response.json()["results"]) == 2

    # 8. Full Query
    query_request = {
        "question": "What does Wärmedämmung mean?",
        "n_retrieval": 10,
        "n_ranking": 3,
        "collection_name": COLLECTION_NAME
    }
    response = http_client.post("/v1/query/", json=query_request)
    assert response.status_code == 200
    assert "results" in response.json()

    # 9. Chat / Generation
    chat_payload = {
        "question": "What is the primary language for AI?",
        "temperature": 0.1
    }
    # Note: This might fail if LLM API key is not configured in the running backend
    response = http_client.post("/v1/chat/", json=chat_payload)

    # We allow 500 if detailed error suggests missing API key, but ideally 200
    if response.status_code == 200:
        assert "answer" in response.json()
        assert len(response.json()["answer"]) > 0
    else:
        # Optional: Print warning if it fails due to missing key, but don't fail test hard
        # API might return 500 or 400 depending on implementation.
        # For strict E2E, we might want to assert 200, assuming env is set up.
        # Given user prompt "can you also write me some e2e tests", I will enforce success
        # but add a SKIP mechanism or assume the user has keys.
        # However, checking the logs suggests the user wants "services to e2e tests".
        # Let's assert 200.
        assert response.status_code == 200

    # 10. Cleanup - Delete Collection
    response = http_client.delete(f"/v1/collections/{COLLECTION_NAME}")
    assert response.status_code == 204
