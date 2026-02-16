#!/usr/bin/env python
from contextlib import asynccontextmanager
from os import getenv
from typing import Dict

from fastapi import FastAPI
from loguru import logger
from qdrant_client import QdrantClient

from src.app.config import Config
from src.app.core import router as core_router
from src.app.logging import setup_logger
from src.app.middleware import RequestTimer, add_request_id
from src.app.utils import load_models
from src.app.v1 import router as v1_router


def get_application(config: Config) -> FastAPI:
    """Create the FastAPI app with lifespan management."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        # Startup: initialize clients and load models
        application.state.qdrant = QdrantClient(
            host=config.kb_host, port=config.kb_port
        )
        logger.info("Qdrant client initialized")

        # Load models into app state
        application.state.models = load_models(
            config.ROOTDIR,
            config.bi_encoder_path,
            config.cross_encoder_path,
        )
        logger.info("Models loaded into app state")

        yield

        # Shutdown: clean up
        application.state.qdrant.close()
        logger.info("Clients shut down")

    request_timer = RequestTimer()
    application = FastAPI(lifespan=lifespan)

    application.state.config = config

    application.middleware("http")(request_timer)
    application.middleware("http")(add_request_id)

    application.include_router(core_router.core, tags=["core"])

    application.include_router(v1_router.v1, prefix="/v1", tags=["v1"])

    logger.info(f"API running in {config.api_mode} mode")
    return application


Config._env_file = (f"{getenv('FASTAPI_ENV', 'dev')}.env",)
config = Config()

setup_logger(config.api_mode)
app = get_application(config)
