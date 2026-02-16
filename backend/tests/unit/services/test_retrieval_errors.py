"""Tests for RetrievalService error handling."""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from fastapi import HTTPException

from src.app.services.retrieval import RetrievalService
from src.app.exceptions import EmbeddingError, RerankingError, VectorDBError


@pytest.fixture
def mock_request():
    """Create mock request with config and models."""
    request = MagicMock()
    request.app.state.models = {
        "bi_tokenizer": MagicMock(),
        "bi_encoder": MagicMock(),
        "cross_tokenizer": MagicMock(),
        "cross_encoder": MagicMock(),
    }
    request.app.state.config.kb_name = "test-collection"
    request.app.state.config.kb_limit = 10
    request.app.state.qdrant = MagicMock()
    return request


class TestEmbedText:
    """Test _embed_text method error handling."""

    def test_embed_empty_text(self, mock_request):
        """Test embedding empty text raises error."""
        service = RetrievalService(mock_request)

        with pytest.raises(EmbeddingError, match="Cannot embed empty text"):
            service._embed_text("")

    def test_embed_whitespace_only_text(self, mock_request):
        """Test embedding whitespace-only text raises error."""
        service = RetrievalService(mock_request)

        with pytest.raises(EmbeddingError, match="Cannot embed empty text"):
            service._embed_text("   \n  \t  ")

    def test_embed_with_missing_tokenizer(self, mock_request):
        """Test embedding with missing tokenizer."""
        mock_request.app.state.models = {}  # Empty models dict
        service = RetrievalService(mock_request)

        with pytest.raises(EmbeddingError, match="Model configuration error"):
            service._embed_text("test text")

    def test_embed_with_tokenizer_error(self, mock_request):
        """Test embedding when tokenizer fails."""
        service = RetrievalService(mock_request)
        mock_request.app.state.models["bi_tokenizer"].side_effect = Exception("Tokenizer failed")

        with pytest.raises(EmbeddingError, match="Failed to generate embedding"):
            service._embed_text("test text")

    def test_embed_with_model_error(self, mock_request):
        """Test embedding when model inference fails."""
        service = RetrievalService(mock_request)

        # Mock tokenizer success but model failure
        mock_request.app.state.models["bi_tokenizer"].return_value = {"input_ids": []}
        mock_request.app.state.models["bi_encoder"].side_effect = RuntimeError("Model inference failed")

        with pytest.raises(EmbeddingError, match="Failed to generate embedding"):
            service._embed_text("test text")

    def test_embed_with_missing_hidden_state(self, mock_request):
        """Test embedding when model output is invalid."""
        service = RetrievalService(mock_request)

        # Mock model output without last_hidden_state
        mock_output = MagicMock(spec=[])  # No last_hidden_state attribute
        mock_request.app.state.models["bi_tokenizer"].return_value = {"input_ids": []}
        mock_request.app.state.models["bi_encoder"].return_value = mock_output

        with pytest.raises(EmbeddingError, match="missing 'last_hidden_state'"):
            service._embed_text("test text")

    def test_embed_with_empty_embedding(self, mock_request):
        """Test embedding when result is empty."""
        service = RetrievalService(mock_request)

        # Mock empty embedding (which causes numpy to return a scalar)
        mock_output = MagicMock()
        mock_output.last_hidden_state = np.array([[]])  # Empty
        mock_request.app.state.config.models["bi_tokenizer"].return_value = {"input_ids": []}
        mock_request.app.state.config.models["bi_encoder"].return_value = mock_output

        # Empty array causes numpy mean to return scalar, which fails when we try to convert to list
        with pytest.raises(EmbeddingError, match="Failed to generate embedding"):
            service._embed_text("test text")

    def test_embed_successful(self, mock_request):
        """Test successful embedding generation."""
        service = RetrievalService(mock_request)

        # Mock successful embedding - needs proper shape for mean pooling
        mock_output = MagicMock()
        # Shape: [batch_size, sequence_length, hidden_size]
        # Mean over axis=1 gives [batch_size, hidden_size]
        mock_output.last_hidden_state = np.array([[[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]])
        mock_request.app.state.models["bi_tokenizer"].return_value = {"input_ids": []}
        mock_request.app.state.models["bi_encoder"].return_value = mock_output

        embedding = service._embed_text("test text")

        assert len(embedding) == 3
        # Mean over sequence dimension (axis=1) gives us the pooled embedding
        assert embedding == [0.1, 0.2, 0.3]


class TestSearch:
    """Test search method error handling."""

    def test_search_with_qdrant_error(self, mock_request):
        """Test search when Qdrant fails."""
        service = RetrievalService(mock_request)
        mock_request.app.state.qdrant.query_points.side_effect = Exception("Qdrant connection failed")

        with pytest.raises(VectorDBError, match="Vector search failed"):
            service.search("test-collection", [0.1, 0.2], 10)

    def test_search_successful(self, mock_request):
        """Test successful search."""
        service = RetrievalService(mock_request)

        mock_result = MagicMock()
        mock_result.points = [MagicMock(), MagicMock()]
        mock_request.app.state.qdrant.query_points.return_value = mock_result

        points = service.search("test-collection", [0.1, 0.2], 10)

        assert len(points) == 2
        mock_request.app.state.qdrant.query_points.assert_called_once_with(
            collection_name="test-collection",
            query=[0.1, 0.2],
            limit=10
        )

    @patch('time.sleep')  # Speed up test by mocking sleep
    def test_search_retries_on_failure(self, mock_sleep, mock_request):
        """Test that search retries on transient failures."""
        service = RetrievalService(mock_request)

        mock_result = MagicMock()
        mock_result.points = [MagicMock()]

        # Fail once, then succeed
        mock_request.app.state.qdrant.query_points.side_effect = [
            Exception("Connection timeout"),
            mock_result
        ]

        points = service.search("test-collection", [0.1, 0.2], 10)

        assert len(points) == 1
        assert mock_request.app.state.qdrant.query_points.call_count == 2


class TestRerank:
    """Test rerank method error handling."""

    def test_rerank_empty_candidates(self, mock_request):
        """Test reranking with empty candidates."""
        service = RetrievalService(mock_request)

        results = service.rerank("question", [], 5)

        assert results == []

    def test_rerank_with_empty_question(self, mock_request):
        """Test reranking with empty question."""
        service = RetrievalService(mock_request)

        candidates = [{"text": "doc1"}, {"text": "doc2"}]
        results = service.rerank("", candidates, 2)

        # Should return unranked results with score 0
        assert len(results) == 2
        assert all(r.score == 0.0 for r in results)

    def test_rerank_with_missing_model(self, mock_request):
        """Test reranking with missing model."""
        mock_request.app.state.models = {}  # Empty models dict
        service = RetrievalService(mock_request)

        with pytest.raises(RerankingError, match="Model configuration error"):
            service.rerank("question", [{"text": "doc"}], 1)

    def test_rerank_with_model_error(self, mock_request):
        """Test reranking when model fails."""
        service = RetrievalService(mock_request)

        mock_request.app.state.models["cross_tokenizer"].return_value = {"input_ids": []}
        mock_request.app.state.models["cross_encoder"].side_effect = RuntimeError("Model failed")

        with pytest.raises(RerankingError, match="Failed to rerank"):
            service.rerank("question", [{"text": "doc"}], 1)

    def test_rerank_with_missing_logits(self, mock_request):
        """Test reranking when model output is invalid."""
        service = RetrievalService(mock_request)

        # Mock model output without logits
        mock_output = MagicMock(spec=[])
        mock_request.app.state.models["cross_tokenizer"].return_value = {"input_ids": []}
        mock_request.app.state.models["cross_encoder"].return_value = mock_output

        with pytest.raises(RerankingError, match="missing 'logits'"):
            service.rerank("question", [{"text": "doc"}], 1)

    def test_rerank_with_score_count_mismatch(self, mock_request):
        """Test reranking when score count doesn't match candidates."""
        service = RetrievalService(mock_request)

        # Mock model returning wrong number of scores
        mock_output = MagicMock()
        mock_output.logits = MagicMock()
        mock_output.logits.reshape.return_value.tolist.return_value = [0.5]  # 1 score for 2 candidates

        mock_request.app.state.models["cross_tokenizer"].return_value = {"input_ids": []}
        mock_request.app.state.models["cross_encoder"].return_value = mock_output

        with pytest.raises(RerankingError, match="Score count mismatch"):
            service.rerank("question", [{"text": "doc1"}, {"text": "doc2"}], 2)

    def test_rerank_successful(self, mock_request):
        """Test successful reranking."""
        service = RetrievalService(mock_request)

        # Mock successful reranking
        mock_output = MagicMock()
        mock_output.logits = MagicMock()
        mock_output.logits.reshape.return_value.tolist.return_value = [0.9, 0.3, 0.7]

        mock_request.app.state.models["cross_tokenizer"].return_value = {"input_ids": []}
        mock_request.app.state.models["cross_encoder"].return_value = mock_output

        candidates = [
            {"text": "doc1"},
            {"text": "doc2"},
            {"text": "doc3"}
        ]

        results = service.rerank("question", candidates, 2)

        assert len(results) == 2
        assert results[0].text == "doc1"  # Highest score (0.9)
        assert results[0].score == 0.9
        assert results[1].text == "doc3"  # Second highest (0.7)
        assert results[1].score == 0.7


class TestRetrieveContext:
    """Test retrieve_context method error handling."""

    def test_retrieve_with_nonexistent_collection(self, mock_request):
        """Test retrieving from nonexistent collection."""
        service = RetrievalService(mock_request)
        mock_request.app.state.qdrant.collection_exists.return_value = False

        with pytest.raises(HTTPException) as exc_info:
            service.retrieve_context("question", collection_name="missing")

        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail

    def test_retrieve_with_collection_check_error(self, mock_request):
        """Test retrieving when collection check fails."""
        service = RetrievalService(mock_request)
        mock_request.app.state.qdrant.collection_exists.side_effect = Exception("DB error")

        with pytest.raises(VectorDBError, match="Failed to access collection"):
            service.retrieve_context("question")

    def test_retrieve_with_embedding_error(self, mock_request):
        """Test retrieving when embedding fails."""
        service = RetrievalService(mock_request)
        mock_request.app.state.qdrant.collection_exists.return_value = True

        # Make _embed_text fail
        service._embed_text = MagicMock(side_effect=EmbeddingError("Embedding failed"))

        with pytest.raises(EmbeddingError):
            service.retrieve_context("question")

    def test_retrieve_with_search_error(self, mock_request):
        """Test retrieving when search fails."""
        service = RetrievalService(mock_request)
        mock_request.app.state.qdrant.collection_exists.return_value = True

        service._embed_text = MagicMock(return_value=[0.1, 0.2])
        service.search = MagicMock(side_effect=VectorDBError("Search failed"))

        with pytest.raises(VectorDBError):
            service.retrieve_context("question")

    def test_retrieve_with_no_results(self, mock_request):
        """Test retrieving when no results found."""
        service = RetrievalService(mock_request)
        mock_request.app.state.qdrant.collection_exists.return_value = True

        service._embed_text = MagicMock(return_value=[0.1, 0.2])
        service.search = MagicMock(return_value=[])  # No results

        results = service.retrieve_context("question")

        assert results == []

    def test_retrieve_with_reranking_error(self, mock_request):
        """Test retrieving when reranking fails."""
        service = RetrievalService(mock_request)
        mock_request.app.state.qdrant.collection_exists.return_value = True

        service._embed_text = MagicMock(return_value=[0.1, 0.2])

        mock_point = MagicMock()
        mock_point.payload = {"text": "doc"}
        service.search = MagicMock(return_value=[mock_point])
        service.rerank = MagicMock(side_effect=RerankingError("Rerank failed"))

        with pytest.raises(RerankingError):
            service.retrieve_context("question")

    def test_retrieve_successful(self, mock_request):
        """Test successful context retrieval."""
        service = RetrievalService(mock_request)
        mock_request.app.state.qdrant.collection_exists.return_value = True

        service._embed_text = MagicMock(return_value=[0.1, 0.2])

        mock_point = MagicMock()
        mock_point.payload = {"text": "doc1"}
        service.search = MagicMock(return_value=[mock_point])

        from src.app.v1.schema import SearchResult
        service.rerank = MagicMock(return_value=[
            SearchResult(text="doc1", score=0.9, metadata={})
        ])

        results = service.retrieve_context("question", n_retrieval=5, n_ranking=3)

        assert len(results) == 1
        assert results[0].text == "doc1"
        service._embed_text.assert_called_once_with("question")
        service.search.assert_called_once()
        service.rerank.assert_called_once()
