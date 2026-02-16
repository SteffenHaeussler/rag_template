import time
import uuid
from typing import Callable, Awaitable

from fastapi import Request, Response
from loguru import logger

from src.app.context import ctx_request_id


class RequestTimer:
    """Middleware to time request processing."""

    async def __call__(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        """Time the request and add X-Process-Time header."""
        logger.debug(f"Method: {request.method} on {request.url.path}")
        start_time = time.time()

        response = await call_next(request)

        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)

        logger.info(f"Processing this request took {process_time} seconds")

        return response


async def add_request_id(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """Add unique request ID to context and response headers."""
    ctx_request_id.set(uuid.uuid4().hex)
    response = await call_next(request)

    response.headers["x-request-id"] = ctx_request_id.get()
    return response
