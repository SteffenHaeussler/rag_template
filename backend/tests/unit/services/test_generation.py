
import pytest
from unittest.mock import MagicMock, patch
from fastapi import Request
from pathlib import Path

from src.app.services.generation import GenerationService

@pytest.fixture
def mock_request():
    request = MagicMock(spec=Request)
    request.app.state.config.generation_model = "test-model"
    request.app.state.config.prompt_key = "default_key"
    request.app.state.config.prompt_language = "en"
    request.app.state.config.temperature = 0.7
    request.app.state.prompts = {"default_key": {"en": "Question: {{ question }} Context: {{ context }}"}}
    return request

class TestGenerationService:
    def test_init(self, mock_request):
        service = GenerationService(mock_request)

        assert service.model == "test-model"
        assert service.prompts == mock_request.app.state.prompts

    @patch("src.app.services.generation.completion")
    def test_generate_answer(self, mock_completion, mock_request):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Generated Answer"))]
        mock_completion.return_value = mock_response

        service = GenerationService(mock_request)

        answer = service.generate_answer("test question", ["context1"])

        assert answer == "Generated Answer"

        # Check that completion was called (may be called multiple times due to retry)
        assert mock_completion.called
        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["timeout"] == 30.0

    @patch("src.app.services.generation.completion")
    def test_generate_answer_custom_params(self, mock_completion, mock_request):
        mock_request.app.state.prompts = {"custom_key": {"fr": "Question: {{ question }}"}}

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Oui"))]
        mock_completion.return_value = mock_response

        service = GenerationService(mock_request)

        answer = service.generate_answer(
            "test question",
            ["context"],
            prompt_key="custom_key",
            prompt_language="fr",
            temperature=0.5
        )

        assert answer == "Oui"

        # Check that completion was called with correct params
        assert mock_completion.called
        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["timeout"] == 30.0

    def test_generate_answer_key_error(self, mock_request):
        mock_request.app.state.prompts = {}

        service = GenerationService(mock_request)

        from src.app.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError) as excinfo:
            service.generate_answer("q", [], prompt_key="missing")

        assert "Prompt not found" in str(excinfo.value)

    def test_render_prompt(self, mock_request):
        service = GenerationService(mock_request)

        template = "Hello {{ name }}!"
        rendered = service._render_prompt(template, name="World")
        assert rendered == "Hello World!"
