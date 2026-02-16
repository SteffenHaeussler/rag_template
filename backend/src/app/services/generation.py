from typing import List, Optional, Dict, Any
from fastapi import Request
from litellm import completion
from loguru import logger
from src.app.v1.schema import ChatResponse
from src.app.exceptions import GenerationError, ConfigurationError
from src.app.retry import retry_with_backoff, is_transient_error

import yaml
import jinja2
from pathlib import Path

class GenerationService:
    def __init__(self, request: Request):
        self.config = request.app.state.config
        self.model = self.config.generation_model

        # Load prompt
        prompt_path = Path(self.config.BASEDIR) / self.config.prompt_path
        self.prompts = self._load_yaml(prompt_path)

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML file with error handling."""
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise ConfigurationError(f"Prompt file not found: {path}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in prompt file: {path}", original_error=e)

    @retry_with_backoff(max_retries=2, initial_delay=1.0, exceptions=(Exception,))
    def _call_llm(self, prompt: str, temperature: float) -> str:
        """
        Call LLM API with retry logic for transient failures.

        Args:
            prompt: The formatted prompt
            temperature: Sampling temperature

        Returns:
            Generated text response

        Raises:
            GenerationError: If LLM call fails after retries
        """
        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                timeout=30.0,  # 30 second timeout
            )

            if not response or not response.choices:
                raise GenerationError("LLM returned empty response")

            content = response.choices[0].message.content
            if not content:
                raise GenerationError("LLM returned empty content")

            return content

        except Exception as e:
            error_msg = str(e)
            logger.error(f"LLM generation failed: {error_msg}")

            # Check if error is transient (for retry logic)
            if not is_transient_error(e):
                # Non-transient errors shouldn't be retried
                raise GenerationError(
                    f"LLM generation failed: {error_msg}",
                    original_error=e
                )
            # Transient errors will be retried by decorator
            raise

    def generate_answer(self, question: str, context: List[str], prompt_key: Optional[str] = None, prompt_language: Optional[str] = None, temperature: Optional[float] = None) -> str:
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

        return self._call_llm(prompt, temp)

    def _render_prompt(self, template_str: str, **kwargs) -> str:
        template = jinja2.Template(template_str)
        return template.render(**kwargs)
