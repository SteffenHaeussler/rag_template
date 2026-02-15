
import httpx
import time

URL = "http://0.0.0.0:8000/v1/collections/"
PAYLOAD = {
    "name": "test_collection",
    "dimension": 384,
    "distance_metric": "cosine"
}

print(f"Sending POST request to {URL}...")
try:
    response = httpx.post(URL, json=PAYLOAD, timeout=10)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")
