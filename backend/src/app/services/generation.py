from typing import Any, Dict, List, Optional
from litellm import acompletion
from loguru import logger
from src.app.config import Config
from src.app.exceptions import GenerationError, ConfigurationError
from src.app.retry import async_retry_with_backoff, is_transient_error

from jinja2.sandbox import SandboxedEnvironment

class GenerationService:
    def __init__(self, config: Config, prompts: Dict[str, Any]):
        self.config = config
        self.model = config.generation_model
        self.prompts = prompts

    @async_retry_with_backoff(max_retries=2, initial_delay=1.0, exceptions=(Exception,), retryable=is_transient_error)
    async def _call_llm(self, prompt: str, temperature: float) -> str:
        """
        Call LLM API with retry logic for transient failures.

        Raw exceptions propagate so the decorator's retryable check sees the
        original exception type and message, not a wrapped GenerationError.

        Args:
            prompt: The formatted prompt
            temperature: Sampling temperature

        Returns:
            Generated text response

        Raises:
            GenerationError: If the API returns an empty response or content
            Exception: Raw API exceptions propagate for the retry decorator
        """
        response = await acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            timeout=30.0,  # 30 second timeout
            api_key=self.config.llm_api_key,
        )

        if not response or not response.choices:
            raise GenerationError("LLM returned empty response")

        content = response.choices[0].message.content
        if not content:
            raise GenerationError("LLM returned empty content")

        return content

    async def generate_answer(self, question: str, context: List[str], prompt_key: Optional[str] = None, prompt_language: Optional[str] = None, temperature: Optional[float] = None) -> str:
        """
        Generates an answer using the configured LLM based on the provided context.

        Args:
            question: User question
            context: List of context strings
            prompt_key: Optional prompt template key
            prompt_language: Optional language code
            temperature: Optional sampling temperature

        Returns:
            Generated answer text

        Raises:
            ConfigurationError: If prompt configuration is invalid
            GenerationError: If LLM generation fails
        """
        # Determine key and language
        key = prompt_key or self.config.prompt_key
        language = prompt_language or self.config.prompt_language
        temp = temperature if temperature is not None else self.config.temperature

        try:
            prompt_template_str = self.prompts[key][language]
        except KeyError:
            raise ConfigurationError(
                f"Prompt not found for key='{key}' and language='{language}'"
            )

        try:
            prompt = self._render_prompt(prompt_template_str, context=context, question=question)
        except Exception as e:
            logger.error(f"Prompt rendering failed: {e}")
            raise GenerationError(f"Failed to render prompt template", original_error=e)

        try:
            return await self._call_llm(prompt, temp)
        except GenerationError:
            raise
        except Exception as e:
            raise GenerationError(f"LLM generation failed: {e}", original_error=e)

    def _render_prompt(self, template_str: str, **kwargs) -> str:
        env = SandboxedEnvironment()
        template = env.from_string(template_str)
        return template.render(**kwargs)
