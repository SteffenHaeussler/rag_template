from typing import List, Optional, Dict, Any
from fastapi import Request
from litellm import completion
from src.app.v1.schema import ChatResponse

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
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def generate_answer(self, question: str, context: List[str], prompt_key: Optional[str] = None, prompt_language: Optional[str] = None, temperature: Optional[float] = None) -> str:
        """
        Generates an answer using the configured LLM based on the provided context.
        """
        # Determine key and language
        key = prompt_key or self.config.prompt_key
        language = prompt_language or self.config.prompt_language
        temp = temperature if temperature is not None else self.config.temperature

        try:
            prompt_template_str = self.prompts[key][language]
        except KeyError:
            raise ValueError(f"Prompt not found for key='{key}' and language='{language}'")

        prompt = self._render_prompt(prompt_template_str, context=context, question=question)

        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp
        )

        return response.choices[0].message.content

    def _render_prompt(self, template_str: str, **kwargs) -> str:
        template = jinja2.Template(template_str)
        return template.render(**kwargs)
