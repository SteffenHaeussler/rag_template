
import pytest
from unittest.mock import MagicMock
from fastapi import Request, HTTPException
import numpy as np

from src.app.services.retrieval import RetrievalService
from src.app.v1.schema import SearchResult

@pytest.fixture
def mock_request():
    request = MagicMock(spec=Request)
    request.app.state.models = {
        "bi_tokenizer": MagicMock(),
        "bi_encoder": MagicMock(),
        "cross_tokenizer": MagicMock(),
        "cross_encoder": MagicMock(),
    }
    request.app.state.config.llm_api_key = "test-key"
    request.app.state.config.generation_model = "test-model"
    request.app.state.config.kb_name = "test-collection"
    request.app.state.config.kb_limit = 5
    request.app.state.qdrant = MagicMock()
    return request

class TestRetrievalService:
    def test_embed_text(self, mock_request):
        service = RetrievalService(mock_request)

        # Mock encoder output
        class MockModelOutput:
            last_hidden_state = np.array([[[1.0, 1.0], [1.0, 1.0]]]) # batch 1, seq 2, dim 2

        mock_request.app.state.models["bi_encoder"].return_value = MockModelOutput()

        embedding = service._embed_text("test")
        assert len(embedding) == 2
        assert embedding == [1.0, 1.0]

    def test_search(self, mock_request):
        service = RetrievalService(mock_request)

        mock_points = [
            MagicMock(payload={"text": "doc1"}, score=0.9),
            MagicMock(payload={"text": "doc2"}, score=0.8)
        ]
        mock_request.app.state.qdrant.query_points.return_value.points = mock_points

        results = service.search("test-collection", [0.1, 0.2], 5)

        assert len(results) == 2
        assert results[0].payload["text"] == "doc1"
        mock_request.app.state.qdrant.query_points.assert_called_once_with(
            collection_name="test-collection",
            query=[0.1, 0.2],
            limit=5
        )

    def test_rerank(self, mock_request):
        service = RetrievalService(mock_request)

        mock_tokenizer = mock_request.app.state.models["cross_tokenizer"]
        mock_model = mock_request.app.state.models["cross_encoder"]

        # Mock model output
        mock_output = MagicMock()
        # 3 candidates -> 3 scores
        mock_output.logits.reshape.return_value.tolist.return_value = [0.1, 0.9, 0.5]
        mock_model.return_value = mock_output

        candidates = [
            {"text": "low score"},
            {"text": "high score"},
            {"text": "medium score"}
        ]

        results = service.rerank("question", candidates, 2)

        assert len(results) == 2
        assert results[0].text == "high score"   # highest logit (0.9) → highest sigmoid
        assert results[1].text == "medium score"  # second logit (0.5)
        # Scores are sigmoid-transformed logits, so in (0, 1)
        assert 0.0 < results[0].score < 1.0
        assert 0.0 < results[1].score < 1.0
        assert results[0].score > results[1].score

    def test_retrieve_context(self, mock_request):
        service = RetrievalService(mock_request)

        # Mock internal methods
        service._embed_text = MagicMock(return_value=[0.1, 0.2])
        service.search = MagicMock(return_value=[
            MagicMock(payload={"text": "doc1"}, score=0.9),
        ])
        service.rerank = MagicMock(return_value=[
            SearchResult(text="doc1", score=0.95, metadata={})
        ])

        # Test default collection retrieval
        results = service.retrieve_context("question")

        assert len(results) == 1
        assert results[0].text == "doc1"
        service.qdrant.collection_exists.assert_called_with("test-collection")
        service._embed_text.assert_called_once()
        service.search.assert_called_once()
        service.rerank.assert_called_once()

    def test_retrieve_context_collection_not_found(self, mock_request):
        service = RetrievalService(mock_request)
        service.qdrant.collection_exists.return_value = False

        with pytest.raises(HTTPException) as excinfo:
            service.retrieve_context("question", collection_name="missing-collection")

        assert excinfo.value.status_code == 404
        assert "missing-collection" in excinfo.value.detail
