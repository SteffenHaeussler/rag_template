import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from src.app.services.generation import GenerationService
from src.app.v1.schema import ChatRequest, ChatResponse, SearchResult


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.llm_api_key = "test-key"
    config.generation_model = "test-model"
    config.kb_name = "test-collection"
    config.kb_limit = 5
    config.prompt_key = "prompt"
    config.prompt_language = "en"
    config.temperature = 0.0
    return config


@pytest.fixture
def mock_prompts():
    return {
        "prompt": {
            "en": "{% for c in context %}{{ c }} {% endfor %}{{ question }}",
            "de": "{% for c in context %}{{ c }} {% endfor %}{{ question }} DE"
        }
    }


class TestGenerationService:
    @patch("src.app.services.generation.acompletion", new_callable=AsyncMock)
    async def test_generate_answer(self, mock_acompletion, mock_config, mock_prompts):
        service = GenerationService(config=mock_config, prompts=mock_prompts)

        # Mock litellm response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Generated Answer"))]
        mock_acompletion.return_value = mock_response

        answer = await service.generate_answer("question", ["context1", "context2"])

        assert answer == "Generated Answer"
        mock_acompletion.assert_called_once()
        call_kwargs = mock_acompletion.call_args[1]
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["temperature"] == 0.0
        # Check if prompts are formatted correctly (default en)
        assert "context1" in call_kwargs["messages"][0]["content"]

    @patch("src.app.services.generation.acompletion", new_callable=AsyncMock)
    async def test_generate_answer_dynamic(self, mock_acompletion, mock_config, mock_prompts):
        service = GenerationService(config=mock_config, prompts=mock_prompts)

        # Mock litellm response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Generated Answer DE"))]
        mock_acompletion.return_value = mock_response

        # Override with DE and temp 0.7
        answer = await service.generate_answer("question", ["context1"], prompt_language="de", temperature=0.7)

        assert answer == "Generated Answer DE"
        mock_acompletion.assert_called_once()
        call_kwargs = mock_acompletion.call_args[1]
        assert call_kwargs["api_key"] == "test-key"
        # Check if DE prompt was used
        assert "context1 question DE" in call_kwargs["messages"][0]["content"]
        assert call_kwargs["temperature"] == 0.7
