import importlib.metadata
import os
from pathlib import Path
from typing import Dict

from pydantic import Field, constr, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    _env_file: str = os.getenv("ENV_FILE", ".env")

    # workaround for storing db items
    models: Dict = {}

    FASTAPI_ENV: constr(to_upper=True) = Field(default="DEV")
    BASEDIR: str = str(Path(__file__).resolve().parent)
    ROOTDIR: str = str(Path(__file__).resolve().parents[2])
    VERSION: str = importlib.metadata.version("backend")

    DEBUG: bool = False

    bi_encoder: str
    bi_encoder_path: str
    cross_encoder: str
    cross_encoder_path: str

    llm_api_key: str = Field(validation_alias=AliasChoices("GEMINI_API_KEY"))
    generation_model: str
    temperature: float = 0.0

    prompt_path: str
    prompt_key: str
    prompt_language: str = "en"

    kb_host: str
    kb_port: int = 6333
    kb_name: str
    kb_limit: int = 20
    kb_batch_size: int = 100

    model_config = SettingsConfigDict(
        env_file=_env_file,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @property
    def api_mode(self) -> str:
        return dict(self).get("FASTAPI_ENV")

    def model_post_init(self, __context):
        # Set the API key as an environment variable for provider-specific lookup
        os.environ["GEMINI_API_KEY"] = self.llm_api_key
        # print(f"DEBUG CONFIG: kb_host={self.kb_host}, api_mode={self.api_mode}")
        # print(f"DEBUG ENV: KB_HOST={os.getenv('KB_HOST')}, kb_host={os.getenv('kb_host')}")
        # print(f"DEBUG: Config initialized. llm_api_key (first 5 chars): {self.llm_api_key[:5] if self.llm_api_key else 'None'}...")

