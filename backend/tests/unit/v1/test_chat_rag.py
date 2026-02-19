from fastapi import status
from unittest.mock import MagicMock, patch
import pytest


# ==========================================
# Tests for /v1/chat/ endpoint
# ==========================================

def test_chat_success(client):
    """Test chat endpoint with valid question and context."""
    with patch('src.app.v1.router.GenerationService') as mock_gen_service:
        mock_service_instance = MagicMock()
        mock_service_instance.generate_answer.return_value = "Python is the primary language for AI."
        mock_gen_service.return_value = mock_service_instance

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

        # Verify GenerationService was called correctly
        mock_service_instance.generate_answer.assert_called_once()
        call_args = mock_service_instance.generate_answer.call_args
        assert call_args.kwargs["question"] == payload["question"]
        assert call_args.kwargs["context"] == payload["context"]


def test_chat_with_optional_params(client):
    """Test chat endpoint with optional parameters."""
    with patch('src.app.v1.router.GenerationService') as mock_gen_service:
        mock_service_instance = MagicMock()
        mock_service_instance.generate_answer.return_value = "Test answer"
        mock_gen_service.return_value = mock_service_instance

        payload = {
            "question": "Test question?",
            "context": ["Context 1", "Context 2"],
            "temperature": 0.7,
            "prompt_key": "custom_prompt",
            "prompt_language": "en"
        }
        response = client.post("/v1/chat/", json=payload)

        assert response.status_code == status.HTTP_200_OK

        # Verify optional params were passed
        call_args = mock_service_instance.generate_answer.call_args
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

def test_rag_success(client, mock_qdrant):
    """Test RAG endpoint with valid question and collection."""
    with patch('src.app.v1.router.RetrievalService') as mock_ret_service, \
         patch('src.app.v1.router.GenerationService') as mock_gen_service:

        # Mock RetrievalService
        mock_ret_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "Python is the primary language for AI."
        mock_ret_instance.retrieve_context.return_value = [mock_result]
        mock_ret_service.return_value = mock_ret_instance

        # Mock GenerationService
        mock_gen_instance = MagicMock()
        mock_gen_instance.generate_answer.return_value = "Based on the context, Python is the primary language."
        mock_gen_service.return_value = mock_gen_instance

        # Mock Qdrant
        mock_qdrant.collection_exists.return_value = True

        payload = {
            "question": "What is the primary language for AI?",
            "collection_name": "test_collection"
        }
        response = client.post("/v1/rag/", json=payload)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "answer" in data
        assert data["answer"] == "Based on the context, Python is the primary language."

        # Verify retrieve_context was called
        mock_ret_instance.retrieve_context.assert_called_once()
        retrieve_call = mock_ret_instance.retrieve_context.call_args
        assert retrieve_call.kwargs["question"] == payload["question"]
        assert retrieve_call.kwargs["collection_name"] == payload["collection_name"]

        # Verify generate_answer was called with retrieved context
        mock_gen_instance.generate_answer.assert_called_once()
        gen_call = mock_gen_instance.generate_answer.call_args
        assert gen_call.kwargs["context"] == ["Python is the primary language for AI."]


def test_rag_with_optional_params(client, mock_qdrant):
    """Test RAG endpoint with optional retrieval parameters."""
    with patch('src.app.v1.router.RetrievalService') as mock_ret_service, \
         patch('src.app.v1.router.GenerationService') as mock_gen_service:

        mock_ret_instance = MagicMock()
        mock_ret_instance.retrieve_context.return_value = []
        mock_ret_service.return_value = mock_ret_instance

        mock_gen_instance = MagicMock()
        mock_gen_instance.generate_answer.return_value = "Answer"
        mock_gen_service.return_value = mock_gen_instance

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

        # Verify optional params were passed to retrieve_context
        retrieve_call = mock_ret_instance.retrieve_context.call_args
        assert retrieve_call.kwargs["n_retrieval"] == 10
        assert retrieve_call.kwargs["n_ranking"] == 3

        # Verify temperature was passed to generate_answer
        gen_call = mock_gen_instance.generate_answer.call_args
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


def test_rag_collection_not_found(client, mock_qdrant):
    """Test RAG endpoint when collection doesn't exist."""
    with patch('src.app.v1.router.RetrievalService') as mock_ret_service:
        mock_ret_instance = MagicMock()
        from fastapi import HTTPException
        mock_ret_instance.retrieve_context.side_effect = HTTPException(
            status_code=404,
            detail="Collection 'nonexistent' not found"
        )
        mock_ret_service.return_value = mock_ret_instance

        payload = {
            "question": "Test?",
            "collection_name": "nonexistent"
        }
        response = client.post("/v1/rag/", json=payload)

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"].lower()
