"""Tests for RetrievalService error handling."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
from fastapi import HTTPException

from src.app.services.retrieval import RetrievalService
from src.app.exceptions import EmbeddingError, RerankingError, VectorDBError


@pytest.fixture
def mock_qdrant():
    mock = MagicMock()
    mock.collection_exists = AsyncMock(return_value=True)
    mock.query_points = AsyncMock()
    return mock


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.kb_name = "test-collection"
    config.kb_limit = 10
    return config


@pytest.fixture
def mock_models():
    return {
        "bi_tokenizer": MagicMock(),
        "bi_encoder": MagicMock(),
        "cross_tokenizer": MagicMock(),
        "cross_encoder": MagicMock(),
    }


@pytest.fixture
def service(mock_qdrant, mock_config, mock_models):
    return RetrievalService(qdrant=mock_qdrant, config=mock_config, models=mock_models)


class TestEmbedText:
    """Test _embed_text method error handling."""

    async def test_embed_empty_text(self, service):
        """Test embedding empty text raises error."""
        with pytest.raises(EmbeddingError, match="Cannot embed empty text"):
            await service._embed_text("")

    async def test_embed_whitespace_only_text(self, service):
        """Test embedding whitespace-only text raises error."""
        with pytest.raises(EmbeddingError, match="Cannot embed empty text"):
            await service._embed_text("   \n  \t  ")

    async def test_embed_with_missing_tokenizer(self, mock_qdrant, mock_config):
        """Test embedding with missing tokenizer."""
        service = RetrievalService(qdrant=mock_qdrant, config=mock_config, models={})

        with pytest.raises(EmbeddingError, match="Model configuration error"):
            await service._embed_text("test text")

    async def test_embed_with_tokenizer_error(self, service, mock_models):
        """Test embedding when tokenizer fails."""
        mock_models["bi_tokenizer"].side_effect = Exception("Tokenizer failed")

        with pytest.raises(EmbeddingError, match="Failed to generate embedding"):
            await service._embed_text("test text")

    async def test_embed_with_model_error(self, service, mock_models):
        """Test embedding when model inference fails."""
        mock_models["bi_tokenizer"].return_value = {"input_ids": []}
        mock_models["bi_encoder"].side_effect = RuntimeError("Model inference failed")

        with pytest.raises(EmbeddingError, match="Failed to generate embedding"):
            await service._embed_text("test text")

    async def test_embed_with_missing_hidden_state(self, service, mock_models):
        """Test embedding when model output is invalid."""
        mock_output = MagicMock(spec=[])  # No last_hidden_state attribute
        mock_models["bi_tokenizer"].return_value = {"input_ids": []}
        mock_models["bi_encoder"].return_value = mock_output

        with pytest.raises(EmbeddingError, match="missing 'last_hidden_state'"):
            await service._embed_text("test text")

    async def test_embed_with_empty_embedding(self, service, mock_models):
        """Test embedding when result is empty."""
        mock_output = MagicMock()
        mock_output.last_hidden_state = np.array([[[]]])  # shape (1,1,0) — zero-dim hidden state
        mock_models["bi_tokenizer"].return_value = {"input_ids": []}
        mock_models["bi_encoder"].return_value = mock_output

        with pytest.raises(EmbeddingError):
            await service._embed_text("test text")

    async def test_embed_successful(self, service, mock_models):
        """Test successful embedding generation."""
        mock_output = MagicMock()
        # Shape: [batch_size, sequence_length, hidden_size]
        # Mean over axis=1 gives [batch_size, hidden_size]
        mock_output.last_hidden_state = np.array([[[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]])
        mock_models["bi_tokenizer"].return_value = {"input_ids": []}
        mock_models["bi_encoder"].return_value = mock_output

        embedding = await service._embed_text("test text")

        assert len(embedding) == 3
        # Mean over sequence dimension (axis=1) gives us the pooled embedding
        assert embedding == [0.1, 0.2, 0.3]


class TestSearch:
    """Test search method error handling."""

    async def test_search_with_qdrant_error(self, service, mock_qdrant):
        """Test search when Qdrant fails."""
        mock_qdrant.query_points.side_effect = Exception("Qdrant connection failed")

        with pytest.raises(VectorDBError, match="Vector search failed"):
            await service.search("test-collection", [0.1, 0.2], 10)

    async def test_search_successful(self, service, mock_qdrant):
        """Test successful search."""
        mock_result = MagicMock()
        mock_result.points = [MagicMock(), MagicMock()]
        mock_qdrant.query_points.return_value = mock_result

        points = await service.search("test-collection", [0.1, 0.2], 10)

        assert len(points) == 2
        mock_qdrant.query_points.assert_called_once_with(
            collection_name="test-collection",
            query=[0.1, 0.2],
            limit=10
        )

    @patch('src.app.retry.asyncio.sleep', new_callable=AsyncMock)
    async def test_search_retries_on_failure(self, mock_sleep, service, mock_qdrant):
        """Test that search retries on transient failures."""
        mock_result = MagicMock()
        mock_result.points = [MagicMock()]

        # Fail once, then succeed
        mock_qdrant.query_points.side_effect = [
            Exception("Connection timeout"),
            mock_result
        ]

        points = await service.search("test-collection", [0.1, 0.2], 10)

        assert len(points) == 1
        assert mock_qdrant.query_points.call_count == 2


class TestRerank:
    """Test rerank method error handling."""

    async def test_rerank_empty_candidates(self, service):
        """Test reranking with empty candidates."""
        results = await service.rerank("question", [], 5)

        assert results == []

    async def test_rerank_with_empty_question(self, service):
        """Test reranking with empty question."""
        candidates = [{"text": "doc1"}, {"text": "doc2"}]
        results = await service.rerank("", candidates, 2)

        # Should return unranked results with score 0
        assert len(results) == 2
        assert all(r.score == 0.0 for r in results)

    async def test_rerank_with_missing_model(self, mock_qdrant, mock_config):
        """Test reranking with missing model."""
        service = RetrievalService(qdrant=mock_qdrant, config=mock_config, models={})

        with pytest.raises(RerankingError, match="Model configuration error"):
            await service.rerank("question", [{"text": "doc"}], 1)

    async def test_rerank_with_model_error(self, service, mock_models):
        """Test reranking when model fails."""
        mock_models["cross_tokenizer"].return_value = {"input_ids": []}
        mock_models["cross_encoder"].side_effect = RuntimeError("Model failed")

        with pytest.raises(RerankingError, match="Failed to rerank"):
            await service.rerank("question", [{"text": "doc"}], 1)

    async def test_rerank_with_missing_logits(self, service, mock_models):
        """Test reranking when model output is invalid."""
        mock_output = MagicMock(spec=[])
        mock_models["cross_tokenizer"].return_value = {"input_ids": []}
        mock_models["cross_encoder"].return_value = mock_output

        with pytest.raises(RerankingError, match="missing 'logits'"):
            await service.rerank("question", [{"text": "doc"}], 1)

    async def test_rerank_with_score_count_mismatch(self, service, mock_models):
        """Test reranking when score count doesn't match candidates."""
        mock_output = MagicMock()
        mock_output.logits = MagicMock()
        mock_output.logits.reshape.return_value.tolist.return_value = [0.5]  # 1 score for 2 candidates

        mock_models["cross_tokenizer"].return_value = {"input_ids": []}
        mock_models["cross_encoder"].return_value = mock_output

        with pytest.raises(RerankingError, match="Score count mismatch"):
            await service.rerank("question", [{"text": "doc1"}, {"text": "doc2"}], 2)

    async def test_rerank_successful(self, service, mock_models):
        """Test successful reranking."""
        mock_output = MagicMock()
        mock_output.logits = MagicMock()
        mock_output.logits.reshape.return_value.tolist.return_value = [0.9, 0.3, 0.7]

        mock_models["cross_tokenizer"].return_value = {"input_ids": []}
        mock_models["cross_encoder"].return_value = mock_output

        candidates = [
            {"text": "doc1"},
            {"text": "doc2"},
            {"text": "doc3"}
        ]

        results = await service.rerank("question", candidates, 2)

        assert len(results) == 2
        assert results[0].text == "doc1"  # Highest logit (0.9) → highest sigmoid
        assert results[1].text == "doc3"  # Second highest logit (0.7) → second sigmoid
        assert 0.0 < results[0].score < 1.0
        assert 0.0 < results[1].score < 1.0
        assert results[0].score > results[1].score


class TestRetrieveContext:
    """Test retrieve_context method error handling."""

    async def test_retrieve_with_nonexistent_collection(self, service, mock_qdrant):
        """Test retrieving from nonexistent collection."""
        mock_qdrant.collection_exists.return_value = False

        with pytest.raises(HTTPException) as exc_info:
            await service.retrieve_context("question", collection_name="missing")

        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail

    async def test_retrieve_with_collection_check_error(self, service, mock_qdrant):
        """Test retrieving when collection check fails."""
        mock_qdrant.collection_exists.side_effect = Exception("DB error")

        with pytest.raises(VectorDBError, match="Failed to access collection"):
            await service.retrieve_context("question")

    async def test_retrieve_with_embedding_error(self, service, mock_qdrant):
        """Test retrieving when embedding fails."""
        mock_qdrant.collection_exists.return_value = True

        service._embed_text = AsyncMock(side_effect=EmbeddingError("Embedding failed"))

        with pytest.raises(EmbeddingError):
            await service.retrieve_context("question")

    async def test_retrieve_with_search_error(self, service, mock_qdrant):
        """Test retrieving when search fails."""
        mock_qdrant.collection_exists.return_value = True

        service._embed_text = AsyncMock(return_value=[0.1, 0.2])
        service.search = AsyncMock(side_effect=VectorDBError("Search failed"))

        with pytest.raises(VectorDBError):
            await service.retrieve_context("question")

    async def test_retrieve_with_no_results(self, service, mock_qdrant):
        """Test retrieving when no results found."""
        mock_qdrant.collection_exists.return_value = True

        service._embed_text = AsyncMock(return_value=[0.1, 0.2])
        service.search = AsyncMock(return_value=[])  # No results

        results = await service.retrieve_context("question")

        assert results == []

    async def test_retrieve_with_reranking_error(self, service, mock_qdrant):
        """Test retrieving when reranking fails."""
        mock_qdrant.collection_exists.return_value = True

        service._embed_text = AsyncMock(return_value=[0.1, 0.2])

        mock_point = MagicMock()
        mock_point.payload = {"text": "doc"}
        service.search = AsyncMock(return_value=[mock_point])
        service.rerank = AsyncMock(side_effect=RerankingError("Rerank failed"))

        with pytest.raises(RerankingError):
            await service.retrieve_context("question")

    async def test_retrieve_successful(self, service, mock_qdrant):
        """Test successful context retrieval."""
        mock_qdrant.collection_exists.return_value = True

        service._embed_text = AsyncMock(return_value=[0.1, 0.2])

        mock_point = MagicMock()
        mock_point.payload = {"text": "doc1"}
        service.search = AsyncMock(return_value=[mock_point])

        from src.app.v1.schema import SearchResult
        service.rerank = AsyncMock(return_value=[
            SearchResult(text="doc1", score=0.9, metadata={})
        ])

        results = await service.retrieve_context("question", n_retrieval=5, n_ranking=3)

        assert len(results) == 1
        assert results[0].text == "doc1"
        service._embed_text.assert_called_once_with("question")
        service.search.assert_called_once()
        service.rerank.assert_called_once()
