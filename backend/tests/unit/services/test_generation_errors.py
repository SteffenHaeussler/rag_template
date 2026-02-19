"""Tests for GenerationService error handling."""

import pytest
from unittest.mock import MagicMock, patch

from src.app.services.generation import GenerationService
from src.app.exceptions import GenerationError, ConfigurationError


@pytest.fixture
def valid_prompts():
    """Valid prompt configuration."""
    return {
        "default": {
            "en": "Answer the question: {{ question }}\nContext: {{ context }}"
        }
    }


@pytest.fixture
def mock_request(valid_prompts):
    """Create mock request with config."""
    request = MagicMock()
    request.app.state.config.generation_model = "test-model"
    request.app.state.config.prompt_key = "default"
    request.app.state.config.prompt_language = "en"
    request.app.state.config.temperature = 0.0
    request.app.state.prompts = valid_prompts
    return request


class TestGenerationServiceInit:
    """Test GenerationService initialization."""

    def test_successful_init(self, mock_request, valid_prompts):
        """Test successful service initialization."""
        service = GenerationService(mock_request)

        assert service.model == "test-model"
        assert service.prompts == valid_prompts

    def test_init_with_missing_prompt_file(self, mock_request):
        """Test initialization succeeds (prompt loading moved to startup)."""
        service = GenerationService(mock_request)
        assert service.prompts is not None

    def test_init_with_invalid_yaml(self, mock_request):
        """Test initialization succeeds when prompts are pre-loaded from app state."""
        service = GenerationService(mock_request)
        assert service.model == "test-model"


class TestCallLLM:
    """Test _call_llm method."""

    @patch('src.app.services.generation.completion')
    def test_successful_llm_call(self, mock_completion, mock_request):
        """Test successful LLM call."""
        service = GenerationService(mock_request)

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test answer"
        mock_completion.return_value = mock_response

        result = service._call_llm("Test prompt", 0.5)

        assert result == "Test answer"
        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["timeout"] == 30.0

    @patch('src.app.services.generation.completion')
    def test_llm_call_with_empty_response(self, mock_completion, mock_request):
        """Test LLM call with empty response."""
        service = GenerationService(mock_request)

        # Mock empty response
        mock_response = MagicMock()
        mock_response.choices = []
        mock_completion.return_value = mock_response

        with pytest.raises(GenerationError, match="empty response"):
            service._call_llm("Test prompt", 0.5)

    @patch('src.app.services.generation.completion')
    def test_llm_call_with_empty_content(self, mock_completion, mock_request):
        """Test LLM call with empty content."""
        service = GenerationService(mock_request)

        # Mock response with empty content
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_completion.return_value = mock_response

        with pytest.raises(GenerationError, match="empty content"):
            service._call_llm("Test prompt", 0.5)

    @patch('src.app.services.generation.completion')
    def test_llm_call_with_api_error(self, mock_completion, mock_request):
        """Test that non-transient API errors propagate as-is from _call_llm.
        Wrapping into GenerationError happens in generate_answer."""
        service = GenerationService(mock_request)

        mock_completion.side_effect = Exception("API error: Invalid API key")

        with pytest.raises(Exception, match="API error"):
            service._call_llm("Test prompt", 0.5)

    @patch('src.app.services.generation.completion')
    @patch('src.app.services.generation.is_transient_error')
    def test_llm_call_retries_transient_errors(self, mock_is_transient, mock_completion, mock_request):
        """Test that transient errors are retried."""
        service = GenerationService(mock_request)

        # Mock transient error followed by success
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Success"

        mock_completion.side_effect = [
            Exception("Rate limit exceeded"),
            mock_response
        ]
        mock_is_transient.return_value = True

        result = service._call_llm("Test prompt", 0.5)

        assert result == "Success"
        assert mock_completion.call_count == 2  # 1 failure + 1 success


class TestGenerateAnswer:
    """Test generate_answer method."""

    @patch('src.app.services.generation.completion')
    def test_successful_generation(self, mock_completion, mock_request):
        """Test successful answer generation."""
        service = GenerationService(mock_request)

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Generated answer"
        mock_completion.return_value = mock_response

        result = service.generate_answer(
            question="What is AI?",
            context=["AI is artificial intelligence"]
        )

        assert result == "Generated answer"

    @patch('src.app.services.generation.completion')
    def test_generation_with_missing_prompt_key(self, mock_completion, mock_request):
        """Test generation with missing prompt key."""
        service = GenerationService(mock_request)

        with pytest.raises(ConfigurationError, match="Prompt not found"):
            service.generate_answer(
                question="Test",
                context=["Context"],
                prompt_key="nonexistent"
            )

    @patch('src.app.services.generation.completion')
    def test_generation_with_missing_language(self, mock_completion, mock_request):
        """Test generation with missing language."""
        service = GenerationService(mock_request)

        with pytest.raises(ConfigurationError, match="Prompt not found"):
            service.generate_answer(
                question="Test",
                context=["Context"],
                prompt_language="es"
            )

    @patch('src.app.services.generation.completion')
    def test_generation_with_template_error(self, mock_completion, mock_request):
        """Test generation with template rendering error."""
        mock_request.app.state.prompts = {
            "default": {
                "en": "{{ question|invalid_filter }}"  # Invalid Jinja2 filter
            }
        }
        service = GenerationService(mock_request)

        with pytest.raises(GenerationError, match="Failed to render prompt"):
            service.generate_answer(
                question="Test",
                context=["Context"]
            )

    @patch('src.app.services.generation.completion')
    def test_generation_with_custom_temperature(self, mock_completion, mock_request):
        """Test generation with custom temperature."""
        service = GenerationService(mock_request)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Answer"
        mock_completion.return_value = mock_response

        service.generate_answer(
            question="Test",
            context=["Context"],
            temperature=0.9
        )

        # Check that custom temperature was used
        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["temperature"] == 0.9

    @patch('src.app.services.generation.completion')
    def test_generation_propagates_llm_error(self, mock_completion, mock_request):
        """Test that LLM errors are propagated."""
        service = GenerationService(mock_request)

        mock_completion.side_effect = Exception("LLM API error")

        with pytest.raises(GenerationError):
            service.generate_answer(
                question="Test",
                context=["Context"]
            )
