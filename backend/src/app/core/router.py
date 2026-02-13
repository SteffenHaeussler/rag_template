from time import time

from fastapi import APIRouter, Request
from loguru import logger

from src.app.core.schema import HealthCheckResponse

core = APIRouter()


@core.api_route("/health", methods=["GET", "POST"], response_model=HealthCheckResponse)
def health(request: Request) -> HealthCheckResponse:
    logger.debug(f"Method: {request.method} on {request.url.path}")
    return {"version": request.app.state.config.VERSION, "timestamp": time()}
