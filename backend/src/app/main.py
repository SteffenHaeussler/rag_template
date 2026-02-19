#!/usr/bin/env python
import yaml
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from loguru import logger
from qdrant_client import QdrantClient

from src.app.config import Config
from src.app.core import router as core_router
from src.app.logging import setup_logger
from src.app.middleware import RequestTimer, add_request_id
from src.app.utils import load_models
from src.app.v1 import router as v1_router
from src.app.exceptions import (
    EmbeddingError,
    RerankingError,
    GenerationError,
    VectorDBError,
    ConfigurationError,
    ValidationError,
)
from src.app.handlers import (
    embedding_error_handler,
    reranking_error_handler,
    generation_error_handler,
    vectordb_error_handler,
    configuration_error_handler,
    validation_error_handler,
)


def get_application(config: Config) -> FastAPI:
    """Create the FastAPI app with lifespan management."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        # Startup: initialize clients and load models
        application.state.qdrant = QdrantClient(
            host=config.kb_host, port=config.kb_port
        )
        logger.info("Qdrant client initialized")

        try:
            # Load models into app state
            application.state.models = load_models(
                config.ROOTDIR,
                config.bi_encoder_path,
                config.cross_encoder_path,
            )
            logger.info("Models loaded into app state")

            prompt_path = Path(config.BASEDIR) / config.prompt_path
            with open(prompt_path) as f:
                application.state.prompts = yaml.safe_load(f)
            logger.info("Prompts loaded into app state")

        except Exception as e:
            logger.error(f"Startup failed: {e}")
            application.state.qdrant.close()
            raise

        yield

        # Shutdown: clean up
        application.state.qdrant.close()
        logger.info("Clients shut down")

    request_timer = RequestTimer()
    application = FastAPI(lifespan=lifespan)

    application.state.config = config

    # Register global exception handlers
    application.add_exception_handler(EmbeddingError, embedding_error_handler)
    application.add_exception_handler(RerankingError, reranking_error_handler)
    application.add_exception_handler(GenerationError, generation_error_handler)
    application.add_exception_handler(VectorDBError, vectordb_error_handler)
    application.add_exception_handler(ConfigurationError, configuration_error_handler)
    application.add_exception_handler(ValidationError, validation_error_handler)

    application.middleware("http")(request_timer)
    application.middleware("http")(add_request_id)

    application.include_router(core_router.core, tags=["core"])

    application.include_router(v1_router.v1, prefix="/v1", tags=["v1"])

    logger.info(f"API running in {config.api_mode} mode")
    return application


config = Config()

setup_logger(config.api_mode)
app = get_application(config)
