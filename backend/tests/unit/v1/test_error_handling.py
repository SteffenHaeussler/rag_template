"""Tests for router error handling and HTTP error responses."""

import pytest
from fastapi import status
from unittest.mock import AsyncMock, MagicMock

from src.app.dependencies import get_retrieval_service, get_generation_service
from src.app.exceptions import (
    EmbeddingError,
    RerankingError,
    GenerationError,
    VectorDBError,
    ConfigurationError,
)


@pytest.fixture(autouse=True)
def clear_overrides(test_app):
    yield
    test_app.dependency_overrides.clear()


class TestEmbeddingEndpointErrors:
    """Test /v1/embedding/ endpoint error handling."""

    def test_embedding_service_error(self, client, test_app, mock_qdrant):
        """Test that embedding errors return 500."""
        mock_service = MagicMock()
        mock_service._embed_text = AsyncMock(side_effect=EmbeddingError("Model inference failed"))
        test_app.dependency_overrides[get_retrieval_service] = lambda: mock_service

        response = client.post("/v1/embedding/", json={"text": "test text"})

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Failed to generate embedding" in response.json()["detail"]


class TestRankingEndpointErrors:
    """Test /v1/ranking/ endpoint error handling."""

    def test_ranking_service_error(self, client, test_app, mock_qdrant):
        """Test that reranking errors return 500."""
        mock_service = MagicMock()
        mock_service.rerank = AsyncMock(side_effect=RerankingError("Reranking model failed"))
        test_app.dependency_overrides[get_retrieval_service] = lambda: mock_service

        response = client.post("/v1/ranking/", json={
            "question": "test question",
            "texts": ["text1", "text2"]
        })

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Failed to rerank" in response.json()["detail"]


class TestSearchEndpointErrors:
    """Test /v1/collections/{name}/search/ endpoint error handling."""

    def test_search_vectordb_error(self, client, test_app, mock_qdrant):
        """Test that vector DB errors return 500."""
        mock_qdrant.collection_exists.return_value = True
        mock_qdrant.get_collection.return_value = MagicMock(
            config=MagicMock(params=MagicMock(vectors=MagicMock(size=384)))
        )

        mock_service = MagicMock()
        mock_service.search = AsyncMock(side_effect=VectorDBError("Qdrant connection failed"))
        test_app.dependency_overrides[get_retrieval_service] = lambda: mock_service

        response = client.post("/v1/collections/test/search/", json={
            "embedding": [0.1] * 384
        })

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Search operation failed" in response.json()["detail"]


class TestQueryEndpointErrors:
    """Test /v1/query/ endpoint error handling."""

    def test_query_embedding_error(self, client, test_app, mock_qdrant):
        """Test that embedding errors in query return 500."""
        mock_service = MagicMock()
        mock_service.retrieve_context = AsyncMock(side_effect=EmbeddingError("Embedding failed"))
        test_app.dependency_overrides[get_retrieval_service] = lambda: mock_service

        response = client.post("/v1/query/", json={
            "question": "test question",
            "collection_name": "test"
        })

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Failed to generate embedding" in response.json()["detail"]

    def test_query_vectordb_error(self, client, test_app, mock_qdrant):
        """Test that vector DB errors in query return 500."""
        mock_service = MagicMock()
        mock_service.retrieve_context = AsyncMock(side_effect=VectorDBError("Search failed"))
        test_app.dependency_overrides[get_retrieval_service] = lambda: mock_service

        response = client.post("/v1/query/", json={
            "question": "test question",
            "collection_name": "test"
        })

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Search operation failed" in response.json()["detail"]

    def test_query_reranking_error(self, client, test_app, mock_qdrant):
        """Test that reranking errors in query return 500."""
        mock_service = MagicMock()
        mock_service.retrieve_context = AsyncMock(side_effect=RerankingError("Rerank failed"))
        test_app.dependency_overrides[get_retrieval_service] = lambda: mock_service

        response = client.post("/v1/query/", json={
            "question": "test question",
            "collection_name": "test"
        })

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Failed to rerank results" in response.json()["detail"]


class TestChatEndpointErrors:
    """Test /v1/chat/ endpoint error handling."""

    def test_chat_configuration_error(self, client, test_app):
        """Test that configuration errors return 400 (client error — bad prompt key/language)."""
        mock_service = MagicMock()
        mock_service.generate_answer = AsyncMock(side_effect=ConfigurationError("Missing prompt"))
        test_app.dependency_overrides[get_generation_service] = lambda: mock_service

        response = client.post("/v1/chat/", json={
            "question": "test question",
            "context": ["context1"]
        })

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Configuration error" in response.json()["detail"]

    def test_chat_generation_error(self, client, test_app):
        """Test that generation errors return 500."""
        mock_service = MagicMock()
        mock_service.generate_answer = AsyncMock(side_effect=GenerationError("LLM API failed"))
        test_app.dependency_overrides[get_generation_service] = lambda: mock_service

        response = client.post("/v1/chat/", json={
            "question": "test question",
            "context": ["context1"]
        })

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Failed to generate answer" in response.json()["detail"]


class TestRAGEndpointErrors:
    """Test /v1/rag/ endpoint error handling."""

    def test_rag_retrieval_error(self, client, test_app, mock_qdrant):
        """Test that retrieval errors in RAG return 500."""
        mock_ret = MagicMock()
        mock_ret.retrieve_context = AsyncMock(side_effect=EmbeddingError("Embedding failed"))
        test_app.dependency_overrides[get_retrieval_service] = lambda: mock_ret

        response = client.post("/v1/rag/", json={
            "question": "test question",
            "collection_name": "test"
        })

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Failed to generate embedding" in response.json()["detail"]

    def test_rag_generation_error(self, client, test_app, mock_qdrant):
        """Test that generation errors in RAG return 500."""
        from src.app.v1.schema import SearchResult

        mock_ret = MagicMock()
        mock_ret.retrieve_context = AsyncMock(return_value=[
            SearchResult(text="context", score=0.9, metadata={})
        ])
        test_app.dependency_overrides[get_retrieval_service] = lambda: mock_ret

        mock_gen = MagicMock()
        mock_gen.generate_answer = AsyncMock(side_effect=GenerationError("LLM failed"))
        test_app.dependency_overrides[get_generation_service] = lambda: mock_gen

        response = client.post("/v1/rag/", json={
            "question": "test question",
            "collection_name": "test"
        })

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Failed to generate answer" in response.json()["detail"]

    def test_rag_configuration_error(self, client, test_app, mock_qdrant):
        """Test that configuration errors in RAG return 400."""
        from src.app.v1.schema import SearchResult

        mock_ret = MagicMock()
        mock_ret.retrieve_context = AsyncMock(return_value=[
            SearchResult(text="context", score=0.9, metadata={})
        ])
        test_app.dependency_overrides[get_retrieval_service] = lambda: mock_ret

        mock_gen = MagicMock()
        mock_gen.generate_answer = AsyncMock(side_effect=ConfigurationError("Missing config"))
        test_app.dependency_overrides[get_generation_service] = lambda: mock_gen

        response = client.post("/v1/rag/", json={
            "question": "test question",
            "collection_name": "test"
        })

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Configuration error" in response.json()["detail"]


class TestDatapointInsertErrors:
    """Test datapoint insert error handling."""

    def test_insert_datapoint_embedding_error(self, client, test_app, mock_qdrant):
        """Test that embedding errors during insert return 500."""
        mock_qdrant.collection_exists.return_value = True
        mock_qdrant.get_collection.return_value = MagicMock(
            config=MagicMock(params=MagicMock(vectors=MagicMock(size=384)))
        )

        mock_service = MagicMock()
        mock_service._embed_text = AsyncMock(side_effect=EmbeddingError("Model failed"))
        test_app.dependency_overrides[get_retrieval_service] = lambda: mock_service

        response = client.post("/v1/collections/test/datapoints/", json={
            "text": "test text"
        })

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Failed to generate embedding" in response.json()["detail"]

    def test_bulk_insert_embedding_error(self, client, test_app, mock_qdrant):
        """Test that embedding errors during bulk insert return 500."""
        mock_qdrant.collection_exists.return_value = True
        mock_qdrant.get_collection.return_value = MagicMock(
            config=MagicMock(params=MagicMock(vectors=MagicMock(size=384)))
        )

        mock_service = MagicMock()
        mock_service._embed_texts_batch = AsyncMock(side_effect=EmbeddingError("Batch embedding failed"))
        test_app.dependency_overrides[get_retrieval_service] = lambda: mock_service

        response = client.post("/v1/collections/test/datapoints/bulk", json=[
            {"text": "text1"},
            {"text": "text2"}
        ])

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        detail = response.json()["detail"]
        assert "Batch embedding failed" in detail

    def test_update_datapoint_embedding_error(self, client, test_app, mock_qdrant):
        """Test that embedding errors during update return 500."""
        mock_point = MagicMock()
        mock_point.id = "123"
        mock_point.vector = [0.1] * 384
        mock_point.payload = {"text": "old text"}
        mock_qdrant.retrieve.return_value = [mock_point]

        mock_service = MagicMock()
        mock_service._embed_text = AsyncMock(side_effect=EmbeddingError("Model failed"))
        test_app.dependency_overrides[get_retrieval_service] = lambda: mock_service

        response = client.put("/v1/collections/test/datapoints/123", json={
            "text": "new text"
        })

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Failed to generate embedding" in response.json()["detail"]


class TestErrorMessageQuality:
    """Test that error messages are informative."""

    def test_error_preserves_original_message(self, client, test_app, mock_qdrant):
        """Test that original error messages are preserved in response."""
        mock_service = MagicMock()
        mock_service._embed_text = AsyncMock(side_effect=EmbeddingError("Specific model error: Out of memory"))
        test_app.dependency_overrides[get_retrieval_service] = lambda: mock_service

        response = client.post("/v1/embedding/", json={"text": "test"})

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        detail = response.json()["detail"]
        assert "Specific model error" in detail  # Original message preserved

    def test_error_message_is_user_friendly(self, client, test_app):
        """Test that error messages don't expose internal details unnecessarily."""
        mock_service = MagicMock()
        mock_service.generate_answer = AsyncMock(side_effect=GenerationError("API rate limit exceeded"))
        test_app.dependency_overrides[get_generation_service] = lambda: mock_service

        response = client.post("/v1/chat/", json={
            "question": "test",
            "context": ["context"]
        })

        detail = response.json()["detail"]
        assert "Failed to generate answer" in detail
        assert "rate limit" in detail.lower()
