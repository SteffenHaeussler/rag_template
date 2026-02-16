"""Custom exception classes for the RAG application."""


class RAGException(Exception):
    """Base exception for all RAG application errors."""

    def __init__(self, message: str, original_error: Exception | None = None):
        self.message = message
        self.original_error = original_error
        super().__init__(self.message)


class EmbeddingError(RAGException):
    """Raised when embedding generation fails."""

    pass


class RerankingError(RAGException):
    """Raised when reranking operation fails."""

    pass


class GenerationError(RAGException):
    """Raised when LLM text generation fails."""

    pass


class VectorDBError(RAGException):
    """Raised when vector database operations fail."""

    pass


class ConfigurationError(RAGException):
    """Raised when configuration is invalid or missing."""

    pass


class ValidationError(RAGException):
    """Raised when input validation fails."""

    pass
