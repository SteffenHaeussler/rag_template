"""Tests for batch embedding functionality."""

import pytest
from unittest.mock import MagicMock
import numpy as np

from src.app.services.retrieval import RetrievalService
from src.app.exceptions import EmbeddingError


@pytest.fixture
def mock_request():
    """Create mock request with models."""
    request = MagicMock()
    request.app.state.models = {
        "bi_tokenizer": MagicMock(),
        "bi_encoder": MagicMock(),
    }
    request.app.state.config.kb_name = "test-collection"
    request.app.state.config.kb_limit = 10
    request.app.state.qdrant = MagicMock()
    return request


class TestBatchEmbedding:
    """Test batch embedding functionality."""

    def test_batch_embed_single_text(self, mock_request):
        """Test batch embedding with a single text."""
        service = RetrievalService(mock_request)

        # Mock successful embedding
        mock_output = MagicMock()
        mock_output.last_hidden_state = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]])
        mock_request.app.state.models["bi_tokenizer"].return_value = {"input_ids": []}
        mock_request.app.state.models["bi_encoder"].return_value = mock_output

        embeddings = service._embed_texts_batch(["test text"])

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 3
        # Mean of [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]] over axis 1
        # Use approximate comparison for floating point
        np.testing.assert_array_almost_equal(embeddings[0], [0.25, 0.35, 0.45], decimal=5)

    def test_batch_embed_multiple_texts(self, mock_request):
        """Test batch embedding with multiple texts."""
        service = RetrievalService(mock_request)

        # Mock successful batch embedding
        mock_output = MagicMock()
        # 3 texts, each with 2 tokens, 3-dimensional embeddings
        mock_output.last_hidden_state = np.array([
            [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]],  # text 1
            [[0.4, 0.5, 0.6], [0.4, 0.5, 0.6]],  # text 2
            [[0.7, 0.8, 0.9], [0.7, 0.8, 0.9]],  # text 3
        ])
        mock_request.app.state.models["bi_tokenizer"].return_value = {"input_ids": []}
        mock_request.app.state.models["bi_encoder"].return_value = mock_output

        texts = ["text1", "text2", "text3"]
        embeddings = service._embed_texts_batch(texts)

        assert len(embeddings) == 3
        assert all(len(emb) == 3 for emb in embeddings)
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]
        assert embeddings[2] == [0.7, 0.8, 0.9]

    def test_batch_embed_large_batch(self, mock_request):
        """Test batch embedding with a large batch (100 texts)."""
        service = RetrievalService(mock_request)

        # Mock successful batch embedding for 100 texts
        batch_size = 100
        embedding_dim = 384
        mock_output = MagicMock()
        # Create random embeddings for 100 texts
        mock_output.last_hidden_state = np.random.rand(batch_size, 2, embedding_dim)
        mock_request.app.state.models["bi_tokenizer"].return_value = {"input_ids": []}
        mock_request.app.state.models["bi_encoder"].return_value = mock_output

        texts = [f"text {i}" for i in range(batch_size)]
        embeddings = service._embed_texts_batch(texts)

        assert len(embeddings) == batch_size
        assert all(len(emb) == embedding_dim for emb in embeddings)

    def test_batch_embed_empty_list(self, mock_request):
        """Test batch embedding with empty list."""
        service = RetrievalService(mock_request)

        embeddings = service._embed_texts_batch([])

        assert embeddings == []

    def test_batch_embed_with_empty_text(self, mock_request):
        """Test batch embedding when one text is empty."""
        service = RetrievalService(mock_request)

        # Need to set up the mock to return proper embeddings for valid texts
        mock_output = MagicMock()
        # Only 2 valid texts (indices 0 and 2), so model returns 2 embeddings
        mock_output.last_hidden_state = np.array([
            [[0.1, 0.2], [0.1, 0.2]],
            [[0.3, 0.4], [0.3, 0.4]],
        ])
        mock_request.app.state.models["bi_tokenizer"].return_value = {"input_ids": []}
        mock_request.app.state.models["bi_encoder"].return_value = mock_output

        with pytest.raises(EmbeddingError, match="Text at index 1"):
            service._embed_texts_batch(["valid text", "", "another text"])

    def test_batch_embed_all_empty_texts(self, mock_request):
        """Test batch embedding when all texts are empty."""
        service = RetrievalService(mock_request)

        with pytest.raises(EmbeddingError, match="all texts are empty"):
            service._embed_texts_batch(["", "   ", "\n\t"])

    def test_batch_embed_with_whitespace_only(self, mock_request):
        """Test batch embedding with whitespace-only text."""
        service = RetrievalService(mock_request)

        # Mock for the 2 valid texts
        mock_output = MagicMock()
        mock_output.last_hidden_state = np.array([
            [[0.1, 0.2], [0.1, 0.2]],
            [[0.3, 0.4], [0.3, 0.4]],
        ])
        mock_request.app.state.models["bi_tokenizer"].return_value = {"input_ids": []}
        mock_request.app.state.models["bi_encoder"].return_value = mock_output

        with pytest.raises(EmbeddingError, match="Text at index 1"):
            service._embed_texts_batch(["valid", "   ", "text"])

    def test_batch_embed_model_error(self, mock_request):
        """Test batch embedding when model fails."""
        service = RetrievalService(mock_request)

        mock_request.app.state.models["bi_tokenizer"].return_value = {"input_ids": []}
        mock_request.app.state.models["bi_encoder"].side_effect = RuntimeError("Model failed")

        with pytest.raises(EmbeddingError, match="Failed to generate batch embeddings"):
            service._embed_texts_batch(["text1", "text2"])

    def test_batch_embed_missing_hidden_state(self, mock_request):
        """Test batch embedding when model output is invalid."""
        service = RetrievalService(mock_request)

        mock_output = MagicMock(spec=[])  # No last_hidden_state
        mock_request.app.state.models["bi_tokenizer"].return_value = {"input_ids": []}
        mock_request.app.state.models["bi_encoder"].return_value = mock_output

        with pytest.raises(EmbeddingError, match="missing 'last_hidden_state'"):
            service._embed_texts_batch(["text1", "text2"])

    def test_batch_embed_count_mismatch(self, mock_request):
        """Test batch embedding when embedding count doesn't match input count."""
        service = RetrievalService(mock_request)

        # Mock model returning wrong number of embeddings
        mock_output = MagicMock()
        # 2 embeddings for 3 texts
        mock_output.last_hidden_state = np.array([
            [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]],
            [[0.4, 0.5, 0.6], [0.4, 0.5, 0.6]],
        ])
        mock_request.app.state.models["bi_tokenizer"].return_value = {"input_ids": []}
        mock_request.app.state.models["bi_encoder"].return_value = mock_output

        with pytest.raises(EmbeddingError, match="Embedding count mismatch"):
            service._embed_texts_batch(["text1", "text2", "text3"])

    def test_batch_embed_preserves_order(self, mock_request):
        """Test that batch embedding preserves input order."""
        service = RetrievalService(mock_request)

        # Create embeddings with distinct values to verify order
        mock_output = MagicMock()
        mock_output.last_hidden_state = np.array([
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],  # text A -> [1, 0, 0]
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],  # text B -> [0, 1, 0]
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],  # text C -> [0, 0, 1]
        ])
        mock_request.app.state.models["bi_tokenizer"].return_value = {"input_ids": []}
        mock_request.app.state.models["bi_encoder"].return_value = mock_output

        texts = ["textA", "textB", "textC"]
        embeddings = service._embed_texts_batch(texts)

        # Verify order is preserved
        assert embeddings[0] == [1.0, 0.0, 0.0]
        assert embeddings[1] == [0.0, 1.0, 0.0]
        assert embeddings[2] == [0.0, 0.0, 1.0]

    def test_batch_embed_tokenizer_receives_batch(self, mock_request):
        """Test that tokenizer receives all texts in one call."""
        service = RetrievalService(mock_request)

        mock_output = MagicMock()
        mock_output.last_hidden_state = np.array([
            [[0.1, 0.2], [0.1, 0.2]],
            [[0.3, 0.4], [0.3, 0.4]],
            [[0.5, 0.6], [0.5, 0.6]],
        ])
        mock_tokenizer = mock_request.app.state.models["bi_tokenizer"]
        mock_tokenizer.return_value = {"input_ids": []}
        mock_request.app.state.models["bi_encoder"].return_value = mock_output

        texts = ["text1", "text2", "text3"]
        service._embed_texts_batch(texts)

        # Verify tokenizer was called once with all texts
        mock_tokenizer.assert_called_once()
        call_args = mock_tokenizer.call_args[0]
        assert call_args[0] == texts  # First arg should be the list of texts


class TestBatchEmbeddingPerformance:
    """Performance-related tests for batch embedding."""

    def test_batch_vs_single_call_count(self, mock_request):
        """Verify batch method calls model once vs N times for single method."""
        service = RetrievalService(mock_request)

        # Mock for batch embedding
        mock_output = MagicMock()
        mock_output.last_hidden_state = np.array([
            [[0.1, 0.2], [0.1, 0.2]],
            [[0.3, 0.4], [0.3, 0.4]],
            [[0.5, 0.6], [0.5, 0.6]],
        ])
        mock_request.app.state.models["bi_tokenizer"].return_value = {"input_ids": []}
        mock_encoder = mock_request.app.state.models["bi_encoder"]
        mock_encoder.return_value = mock_output

        # Batch embed 3 texts
        texts = ["text1", "text2", "text3"]
        service._embed_texts_batch(texts)

        # Should only call model once for all 3 texts
        assert mock_encoder.call_count == 1
