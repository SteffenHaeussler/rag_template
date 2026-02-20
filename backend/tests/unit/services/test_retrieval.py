import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi import HTTPException
import numpy as np

from src.app.services.retrieval import RetrievalService
from src.app.v1.schema import SearchResult


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
    config.kb_limit = 5
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


class TestRetrievalService:
    async def test_embed_text(self, service, mock_models):
        class MockModelOutput:
            last_hidden_state = np.array([[[1.0, 1.0], [1.0, 1.0]]]) # batch 1, seq 2, dim 2

        mock_models["bi_encoder"].return_value = MockModelOutput()

        embedding = await service._embed_text("test")
        assert len(embedding) == 2
        assert embedding == [1.0, 1.0]

    async def test_search(self, service, mock_qdrant):
        mock_points = [
            MagicMock(payload={"text": "doc1"}, score=0.9),
            MagicMock(payload={"text": "doc2"}, score=0.8)
        ]
        mock_qdrant.query_points.return_value = MagicMock(points=mock_points)

        results = await service.search("test-collection", [0.1, 0.2], 5)

        assert len(results) == 2
        assert results[0].payload["text"] == "doc1"
        mock_qdrant.query_points.assert_called_once_with(
            collection_name="test-collection",
            query=[0.1, 0.2],
            limit=5
        )

    async def test_rerank(self, service, mock_models):
        mock_output = MagicMock()
        # 3 candidates -> 3 scores
        mock_output.logits.reshape.return_value.tolist.return_value = [0.1, 0.9, 0.5]
        mock_models["cross_encoder"].return_value = mock_output

        candidates = [
            {"text": "low score"},
            {"text": "high score"},
            {"text": "medium score"}
        ]

        results = await service.rerank("question", candidates, 2)

        assert len(results) == 2
        assert results[0].text == "high score"   # highest logit (0.9) → highest sigmoid
        assert results[1].text == "medium score"  # second logit (0.5)
        # Scores are sigmoid-transformed logits, so in (0, 1)
        assert 0.0 < results[0].score < 1.0
        assert 0.0 < results[1].score < 1.0
        assert results[0].score > results[1].score

    async def test_retrieve_context(self, service, mock_qdrant):
        # Mock internal methods
        service._embed_text = AsyncMock(return_value=[0.1, 0.2])
        service.search = AsyncMock(return_value=[
            MagicMock(payload={"text": "doc1"}, score=0.9),
        ])
        service.rerank = AsyncMock(return_value=[
            SearchResult(text="doc1", score=0.95, metadata={})
        ])

        # Test default collection retrieval
        results = await service.retrieve_context("question")

        assert len(results) == 1
        assert results[0].text == "doc1"
        mock_qdrant.collection_exists.assert_called_with("test-collection")
        service._embed_text.assert_called_once()
        service.search.assert_called_once()
        service.rerank.assert_called_once()

    async def test_retrieve_context_collection_not_found(self, service, mock_qdrant):
        mock_qdrant.collection_exists.return_value = False

        with pytest.raises(HTTPException) as excinfo:
            await service.retrieve_context("question", collection_name="missing-collection")

        assert excinfo.value.status_code == 404
        assert "missing-collection" in excinfo.value.detail
