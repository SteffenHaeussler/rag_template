import importlib.metadata
from pathlib import Path
from typing import Dict

from pydantic import Field, constr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    _env_file: str = "dev.env"

    # workaround for storing db items
    models: Dict = {}

    FASTAPI_ENV: constr(to_upper=True) = Field(default="DEV", env="FASTAPI_ENV")
    BASEDIR: str = str(Path(__file__).resolve().parent)
    ROOTDIR: str = str(Path(__file__).resolve().parents[2])
    VERSION: str = importlib.metadata.version("sim_ir")

    DEBUG: bool = False

    bi_encoder_path: str = Field(env="bi_encoder_path")
    cross_encoder_path: str = Field(env="cross_encoder_path")

    llm_api_key: str = Field(env="llm_api_key")
    generation_model: str = Field(env="generation_model")

    kb_host: str = Field(env="kb_host")
    kb_port: int = Field(env="kb_port", default=6333)
    kb_name: str = Field(env="kb_name")
    kb_limit: int = Field(env="kb_limit", default=20)

    model_config = SettingsConfigDict(env_file=_env_file, env_file_encoding="utf-8")

    @property
    def api_mode(self) -> str:
        return dict(self).get("FASTAPI_ENV")

