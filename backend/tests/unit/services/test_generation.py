import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from src.app.services.generation import GenerationService


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.generation_model = "test-model"
    config.prompt_key = "default_key"
    config.prompt_language = "en"
    config.temperature = 0.7
    config.llm_api_key = "test-key"
    return config


@pytest.fixture
def mock_prompts():
    return {"default_key": {"en": "Question: {{ question }} Context: {{ context }}"}}


@pytest.fixture
def service(mock_config, mock_prompts):
    return GenerationService(config=mock_config, prompts=mock_prompts)


class TestGenerationService:
    def test_init(self, service, mock_prompts):
        assert service.model == "test-model"
        assert service.prompts == mock_prompts

    @patch("src.app.services.generation.acompletion", new_callable=AsyncMock)
    async def test_generate_answer(self, mock_acompletion, service):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Generated Answer"))]
        mock_acompletion.return_value = mock_response

        answer = await service.generate_answer("test question", ["context1"])

        assert answer == "Generated Answer"

        assert mock_acompletion.called
        call_kwargs = mock_acompletion.call_args.kwargs
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["timeout"] == 30.0

    @patch("src.app.services.generation.acompletion", new_callable=AsyncMock)
    async def test_generate_answer_custom_params(self, mock_acompletion, mock_config):
        prompts = {"custom_key": {"fr": "Question: {{ question }}"}}
        service = GenerationService(config=mock_config, prompts=prompts)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Oui"))]
        mock_acompletion.return_value = mock_response

        answer = await service.generate_answer(
            "test question",
            ["context"],
            prompt_key="custom_key",
            prompt_language="fr",
            temperature=0.5
        )

        assert answer == "Oui"

        assert mock_acompletion.called
        call_kwargs = mock_acompletion.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["timeout"] == 30.0

    async def test_generate_answer_key_error(self, mock_config):
        service = GenerationService(config=mock_config, prompts={})

        from src.app.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError) as excinfo:
            await service.generate_answer("q", [], prompt_key="missing")

        assert "Prompt not found" in str(excinfo.value)

    def test_render_prompt(self, service):
        rendered = service._render_prompt("Hello {{ name }}!", name="World")
        assert rendered == "Hello World!"
