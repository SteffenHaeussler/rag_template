"""FastAPI dependency providers for services and shared resources."""

from typing import Any, Dict

from fastapi import Depends, Request
from qdrant_client import QdrantClient

from src.app.config import Config
from src.app.services.generation import GenerationService
from src.app.services.retrieval import RetrievalService


def get_config(request: Request) -> Config:
    return request.app.state.config


def get_qdrant(request: Request) -> QdrantClient:
    return request.app.state.qdrant


def get_models(request: Request) -> Dict[str, Any]:
    return request.app.state.models


def get_prompts(request: Request) -> Dict[str, Any]:
    return request.app.state.prompts


def get_retrieval_service(
    qdrant: QdrantClient = Depends(get_qdrant),
    config: Config = Depends(get_config),
    models: Dict[str, Any] = Depends(get_models),
) -> RetrievalService:
    return RetrievalService(qdrant=qdrant, config=config, models=models)


def get_generation_service(
    config: Config = Depends(get_config),
    prompts: Dict[str, Any] = Depends(get_prompts),
) -> GenerationService:
    return GenerationService(config=config, prompts=prompts)
