"""Global exception handlers for the RAG application."""

from fastapi import Request
from fastapi.responses import JSONResponse
from loguru import logger

from src.app.exceptions import (
    EmbeddingError,
    RerankingError,
    GenerationError,
    VectorDBError,
    ConfigurationError,
    ValidationError,
)


async def embedding_error_handler(request: Request, exc: EmbeddingError) -> JSONResponse:
    """Handle embedding generation failures."""
    logger.error(f"Embedding failed: {exc.message}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Failed to generate embedding: {exc.message}"},
    )


async def reranking_error_handler(request: Request, exc: RerankingError) -> JSONResponse:
    """Handle reranking operation failures."""
    logger.error(f"Reranking failed: {exc.message}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Failed to rerank results: {exc.message}"},
    )


async def generation_error_handler(request: Request, exc: GenerationError) -> JSONResponse:
    """Handle LLM text generation failures."""
    logger.error(f"Generation failed: {exc.message}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Failed to generate answer: {exc.message}"},
    )


async def vectordb_error_handler(request: Request, exc: VectorDBError) -> JSONResponse:
    """Handle vector database operation failures."""
    logger.error(f"Vector DB operation failed: {exc.message}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Search operation failed: {exc.message}"},
    )


async def configuration_error_handler(request: Request, exc: ConfigurationError) -> JSONResponse:
    """Handle configuration errors."""
    logger.error(f"Configuration error: {exc.message}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Configuration error: {exc.message}"},
    )


async def validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
    """Handle validation errors."""
    logger.error(f"Validation error: {exc.message}")
    return JSONResponse(
        status_code=400,
        content={"detail": f"Validation error: {exc.message}"},
    )
