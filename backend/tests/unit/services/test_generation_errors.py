"""Tests for GenerationService error handling."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

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
def mock_config():
    """Create mock config."""
    config = MagicMock()
    config.generation_model = "test-model"
    config.prompt_key = "default"
    config.prompt_language = "en"
    config.temperature = 0.0
    config.llm_api_key = "test-key"
    return config


@pytest.fixture
def service(mock_config, valid_prompts):
    return GenerationService(config=mock_config, prompts=valid_prompts)


class TestGenerationServiceInit:
    """Test GenerationService initialization."""

    def test_successful_init(self, mock_config, valid_prompts):
        """Test successful service initialization."""
        service = GenerationService(config=mock_config, prompts=valid_prompts)

        assert service.model == "test-model"
        assert service.prompts == valid_prompts

    def test_init_with_missing_prompt_file(self, mock_config):
        """Test initialization succeeds (prompt loading moved to startup)."""
        service = GenerationService(config=mock_config, prompts={})
        assert service.prompts == {}

    def test_init_with_invalid_yaml(self, mock_config):
        """Test initialization succeeds when prompts are pre-loaded from app state."""
        service = GenerationService(config=mock_config, prompts={})
        assert service.model == "test-model"


class TestCallLLM:
    """Test _call_llm method."""

    @patch('src.app.services.generation.acompletion', new_callable=AsyncMock)
    async def test_successful_llm_call(self, mock_acompletion, service):
        """Test successful LLM call."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test answer"
        mock_acompletion.return_value = mock_response

        result = await service._call_llm("Test prompt", 0.5)

        assert result == "Test answer"
        mock_acompletion.assert_called_once()
        call_kwargs = mock_acompletion.call_args.kwargs
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["timeout"] == 30.0

    @patch('src.app.services.generation.acompletion', new_callable=AsyncMock)
    async def test_llm_call_with_empty_response(self, mock_acompletion, service):
        """Test LLM call with empty response."""
        mock_response = MagicMock()
        mock_response.choices = []
        mock_acompletion.return_value = mock_response

        with pytest.raises(GenerationError, match="empty response"):
            await service._call_llm("Test prompt", 0.5)

    @patch('src.app.services.generation.acompletion', new_callable=AsyncMock)
    async def test_llm_call_with_empty_content(self, mock_acompletion, service):
        """Test LLM call with empty content."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_acompletion.return_value = mock_response

        with pytest.raises(GenerationError, match="empty content"):
            await service._call_llm("Test prompt", 0.5)

    @patch('src.app.services.generation.acompletion', new_callable=AsyncMock)
    async def test_llm_call_with_api_error(self, mock_acompletion, service):
        """Test that non-transient API errors propagate as-is from _call_llm.
        Wrapping into GenerationError happens in generate_answer."""
        mock_acompletion.side_effect = Exception("API error: Invalid API key")

        with pytest.raises(Exception, match="API error"):
            await service._call_llm("Test prompt", 0.5)

    @patch('src.app.services.generation.acompletion', new_callable=AsyncMock)
    @patch('src.app.services.generation.is_transient_error')
    @patch('src.app.retry.asyncio.sleep', new_callable=AsyncMock)
    async def test_llm_call_retries_transient_errors(self, mock_sleep, mock_is_transient, mock_acompletion, service):
        """Test that transient errors are retried."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Success"

        mock_acompletion.side_effect = [
            Exception("Rate limit exceeded"),
            mock_response
        ]
        mock_is_transient.return_value = True

        result = await service._call_llm("Test prompt", 0.5)

        assert result == "Success"
        assert mock_acompletion.call_count == 2  # 1 failure + 1 success


class TestGenerateAnswer:
    """Test generate_answer method."""

    @patch('src.app.services.generation.acompletion', new_callable=AsyncMock)
    async def test_successful_generation(self, mock_acompletion, service):
        """Test successful answer generation."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Generated answer"
        mock_acompletion.return_value = mock_response

        result = await service.generate_answer(
            question="What is AI?",
            context=["AI is artificial intelligence"]
        )

        assert result == "Generated answer"

    @patch('src.app.services.generation.acompletion', new_callable=AsyncMock)
    async def test_generation_with_missing_prompt_key(self, mock_acompletion, service):
        """Test generation with missing prompt key."""
        with pytest.raises(ConfigurationError, match="Prompt not found"):
            await service.generate_answer(
                question="Test",
                context=["Context"],
                prompt_key="nonexistent"
            )

    @patch('src.app.services.generation.acompletion', new_callable=AsyncMock)
    async def test_generation_with_missing_language(self, mock_acompletion, service):
        """Test generation with missing language."""
        with pytest.raises(ConfigurationError, match="Prompt not found"):
            await service.generate_answer(
                question="Test",
                context=["Context"],
                prompt_language="es"
            )

    @patch('src.app.services.generation.acompletion', new_callable=AsyncMock)
    async def test_generation_with_template_error(self, mock_acompletion, mock_config):
        """Test generation with template rendering error."""
        prompts = {
            "default": {
                "en": "{{ question|invalid_filter }}"  # Invalid Jinja2 filter
            }
        }
        service = GenerationService(config=mock_config, prompts=prompts)

        with pytest.raises(GenerationError, match="Failed to render prompt"):
            await service.generate_answer(
                question="Test",
                context=["Context"]
            )

    @patch('src.app.services.generation.acompletion', new_callable=AsyncMock)
    async def test_generation_with_custom_temperature(self, mock_acompletion, service):
        """Test generation with custom temperature."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Answer"
        mock_acompletion.return_value = mock_response

        await service.generate_answer(
            question="Test",
            context=["Context"],
            temperature=0.9
        )

        call_kwargs = mock_acompletion.call_args.kwargs
        assert call_kwargs["temperature"] == 0.9

    @patch('src.app.services.generation.acompletion', new_callable=AsyncMock)
    async def test_generation_propagates_llm_error(self, mock_acompletion, service):
        """Test that LLM errors are propagated."""
        mock_acompletion.side_effect = Exception("LLM API error")

        with pytest.raises(GenerationError):
            await service.generate_answer(
                question="Test",
                context=["Context"]
            )
