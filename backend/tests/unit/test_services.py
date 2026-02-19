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



class TestGenerationService:
    @patch("src.app.services.generation.completion")
    def test_generate_answer(self, mock_completion, mock_request):
        mock_request.app.state.prompts = {
            "prompt": {
                "en": "{% for c in context %}{{ c }} {% endfor %}{{ question }}",
                "de": "{% for c in context %}{{ c }} {% endfor %}{{ question }} DE"
            }
        }
        mock_request.app.state.config.prompt_key = "prompt"
        mock_request.app.state.config.prompt_language = "en"
        mock_request.app.state.config.temperature = 0.0

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
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["temperature"] == 0.0
        # Check if prompts are formatted correctly (default en)
        assert "context1" in call_kwargs["messages"][0]["content"]

    @patch("src.app.services.generation.completion")
    def test_generate_answer_dynamic(self, mock_completion, mock_request):
        mock_request.app.state.prompts = {
            "prompt": {
                "en": "{% for c in context %}{{ c }} {% endfor %}{{ question }}",
                "de": "{% for c in context %}{{ c }} {% endfor %}{{ question }} DE"
            }
        }
        mock_request.app.state.config.prompt_key = "prompt"
        mock_request.app.state.config.prompt_language = "en"
        mock_request.app.state.config.temperature = 0.0

        service = GenerationService(mock_request)

        # Mock litellm response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Generated Answer DE"))]
        mock_completion.return_value = mock_response

        # Override with DE and temp 0.7
        answer = service.generate_answer("question", ["context1"], prompt_language="de", temperature=0.7)

        assert answer == "Generated Answer DE"
        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["api_key"] == "test-key"
        # Check if DE prompt was used
        assert "context1 question DE" in call_kwargs["messages"][0]["content"]
        assert call_kwargs["temperature"] == 0.7
