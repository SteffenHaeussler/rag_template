from fastapi import status
from unittest.mock import AsyncMock, MagicMock
import pytest

from src.app.dependencies import get_generation_service, get_retrieval_service


@pytest.fixture(autouse=True)
def clear_overrides(test_app):
    yield
    test_app.dependency_overrides.clear()


# ==========================================
# Tests for /v1/chat/ endpoint
# ==========================================

def test_chat_success(client, test_app):
    """Test chat endpoint with valid question and context."""
    mock_service = MagicMock()
    mock_service.generate_answer = AsyncMock(return_value="Python is the primary language for AI.")
    test_app.dependency_overrides[get_generation_service] = lambda: mock_service

    payload = {
        "question": "What is the primary language for AI?",
        "context": [
            "Python is widely used for AI and machine learning.",
            "TensorFlow and PyTorch are Python frameworks."
        ]
    }
    response = client.post("/v1/chat/", json=payload)

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "answer" in data
    assert data["answer"] == "Python is the primary language for AI."

    mock_service.generate_answer.assert_called_once()
    call_args = mock_service.generate_answer.call_args
    assert call_args.kwargs["question"] == payload["question"]
    assert call_args.kwargs["context"] == payload["context"]


def test_chat_with_optional_params(client, test_app):
    """Test chat endpoint with optional parameters."""
    mock_service = MagicMock()
    mock_service.generate_answer = AsyncMock(return_value="Test answer")
    test_app.dependency_overrides[get_generation_service] = lambda: mock_service

    payload = {
        "question": "Test question?",
        "context": ["Context 1", "Context 2"],
        "temperature": 0.7,
        "prompt_key": "custom_prompt",
        "prompt_language": "en"
    }
    response = client.post("/v1/chat/", json=payload)

    assert response.status_code == status.HTTP_200_OK

    call_args = mock_service.generate_answer.call_args
    assert call_args.kwargs["temperature"] == 0.7
    assert call_args.kwargs["prompt_key"] == "custom_prompt"
    assert call_args.kwargs["prompt_language"] == "en"


def test_chat_missing_question(client):
    """Test chat endpoint with missing question field."""
    payload = {
        "context": ["Some context"]
    }
    response = client.post("/v1/chat/", json=payload)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_chat_missing_context(client):
    """Test chat endpoint with missing context field."""
    payload = {
        "question": "What is AI?"
    }
    response = client.post("/v1/chat/", json=payload)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_chat_empty_context(client):
    """Test chat endpoint with empty context array is rejected (min_length=1)."""
    payload = {
        "question": "What is AI?",
        "context": []
    }
    response = client.post("/v1/chat/", json=payload)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


# ==========================================
# Tests for /v1/rag/ endpoint
# ==========================================

def test_rag_success(client, test_app, mock_qdrant):
    """Test RAG endpoint with valid question and collection."""
    mock_ret = MagicMock()
    mock_result = MagicMock()
    mock_result.text = "Python is the primary language for AI."
    mock_ret.retrieve_context = AsyncMock(return_value=[mock_result])
    test_app.dependency_overrides[get_retrieval_service] = lambda: mock_ret

    mock_gen = MagicMock()
    mock_gen.generate_answer = AsyncMock(return_value="Based on the context, Python is the primary language.")
    test_app.dependency_overrides[get_generation_service] = lambda: mock_gen

    mock_qdrant.collection_exists.return_value = True

    payload = {
        "question": "What is the primary language for AI?",
        "collection_name": "test_collection"
    }
    response = client.post("/v1/rag/", json=payload)

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["answer"] == "Based on the context, Python is the primary language."

    mock_ret.retrieve_context.assert_called_once()
    retrieve_call = mock_ret.retrieve_context.call_args
    assert retrieve_call.kwargs["question"] == payload["question"]
    assert retrieve_call.kwargs["collection_name"] == payload["collection_name"]

    mock_gen.generate_answer.assert_called_once()
    gen_call = mock_gen.generate_answer.call_args
    assert gen_call.kwargs["context"] == ["Python is the primary language for AI."]


def test_rag_with_optional_params(client, test_app, mock_qdrant):
    """Test RAG endpoint with optional retrieval parameters."""
    mock_ret = MagicMock()
    mock_ret.retrieve_context = AsyncMock(return_value=[])
    test_app.dependency_overrides[get_retrieval_service] = lambda: mock_ret

    mock_gen = MagicMock()
    mock_gen.generate_answer = AsyncMock(return_value="Answer")
    test_app.dependency_overrides[get_generation_service] = lambda: mock_gen

    mock_qdrant.collection_exists.return_value = True

    payload = {
        "question": "Test?",
        "collection_name": "test_collection",
        "n_retrieval": 10,
        "n_ranking": 3,
        "temperature": 0.5
    }
    response = client.post("/v1/rag/", json=payload)

    assert response.status_code == status.HTTP_200_OK

    retrieve_call = mock_ret.retrieve_context.call_args
    assert retrieve_call.kwargs["n_retrieval"] == 10
    assert retrieve_call.kwargs["n_ranking"] == 3

    gen_call = mock_gen.generate_answer.call_args
    assert gen_call.kwargs["temperature"] == 0.5


def test_rag_missing_collection_name(client):
    """Test RAG endpoint with missing collection_name."""
    payload = {
        "question": "What is AI?"
    }
    response = client.post("/v1/rag/", json=payload)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_rag_missing_question(client):
    """Test RAG endpoint with missing question."""
    payload = {
        "collection_name": "test_collection"
    }
    response = client.post("/v1/rag/", json=payload)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_rag_collection_not_found(client, test_app, mock_qdrant):
    """Test RAG endpoint when collection doesn't exist."""
    from fastapi import HTTPException

    mock_ret = MagicMock()
    mock_ret.retrieve_context = AsyncMock(side_effect=HTTPException(
        status_code=404,
        detail="Collection 'nonexistent' not found"
    ))
    test_app.dependency_overrides[get_retrieval_service] = lambda: mock_ret

    payload = {
        "question": "Test?",
        "collection_name": "nonexistent"
    }
    response = client.post("/v1/rag/", json=payload)

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "not found" in response.json()["detail"].lower()
