import pytest
from unittest.mock import MagicMock, patch
from fastapi import Request
from src.app.services.retrieval import RetrievalService
from src.app.services.generation import GenerationService
from src.app.v1.schema import ChatRequest, ChatResponse, SearchResult

@pytest.fixture
def mock_request():
    request = MagicMock(spec=Request)
    request.app.state.config.models = {
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
        mock_output = MagicMock()
        mock_output.last_hidden_state = [[1.0, 2.0], [3.0, 4.0]] # shape (1, 2, 2) roughly
        # actual shape handling in code: np.mean(outputs.last_hidden_state, axis=1).tolist()[0]
        # if input is (1, seq_len, hidden_size) -> mean(axis=1) -> (1, hidden_size)

        import numpy as np
        # Improve mock to return numpy array structure expected
        class MockModelOutput:
            last_hidden_state = np.array([[[1.0, 1.0], [1.0, 1.0]]]) # batch 1, seq 2, dim 2

        mock_request.app.state.config.models["bi_encoder"].return_value = MockModelOutput()

        embedding = service._embed_text("test")
        assert len(embedding) == 2
        assert embedding == [1.0, 1.0]

    def test_retrieve_context(self, mock_request):
        service = RetrievalService(mock_request)

        # Mock deps
        service._embed_text = MagicMock(return_value=[0.1, 0.2])
        service.search = MagicMock(return_value=[
            MagicMock(payload={"text": "doc1"}, score=0.9),
            MagicMock(payload={"text": "doc2"}, score=0.8)
        ])
        service.rerank = MagicMock(return_value=[
            SearchResult(text="doc1", score=0.95, metadata={}),
             SearchResult(text="doc2", score=0.85, metadata={})
        ])

        results = service.retrieve_context("question")

        assert len(results) == 2
        assert results[0].text == "doc1"
        service._embed_text.assert_called_once()
        service.search.assert_called_once()
        service.rerank.assert_called_once()

class TestGenerationService:
    @patch("src.app.services.generation.completion")
    def test_generate_answer(self, mock_completion, mock_request):
        service = GenerationService(mock_request)

        # Mock litellm response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Generated Answer"))]
        mock_completion.return_value = mock_response

        answer = service.generate_answer("question", ["context1", "context2"])

        assert answer == "Generated Answer"
        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["model"] == "test-model"
        assert "context1" in call_kwargs["messages"][0]["content"]
