import importlib.metadata
import os
from pathlib import Path
from typing import Dict

from pydantic import Field, constr, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
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
        env_file=(f"{os.getenv('FASTAPI_ENV', 'dev')}.env",),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @property
    def api_mode(self) -> str:
        return dict(self).get("FASTAPI_ENV")

