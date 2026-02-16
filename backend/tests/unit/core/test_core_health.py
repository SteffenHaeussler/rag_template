from fastapi import status


def test_core_health(client):
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "version" in data
    assert "timestamp" in data
