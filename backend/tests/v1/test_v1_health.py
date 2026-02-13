from fastapi import status


def test_v1_health(client):
    response = client.get("/v1/health")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "version" in data
    assert "timestamp" in data
