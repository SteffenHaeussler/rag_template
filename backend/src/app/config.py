import importlib.metadata
from pathlib import Path
from typing import Dict

from pydantic import Field, constr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    _env_file: str = "dev.env"

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

    llm_api_key: str
    generation_model: str

    kb_host: str
    kb_port: int = 6333
    kb_name: str
    kb_limit: int = 20
    kb_batch_size: int = 100

    model_config = SettingsConfigDict(env_file=_env_file, env_file_encoding="utf-8", extra="ignore")

    @property
    def api_mode(self) -> str:
        return dict(self).get("FASTAPI_ENV")

