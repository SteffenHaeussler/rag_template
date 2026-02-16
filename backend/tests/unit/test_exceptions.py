"""Tests for custom exception classes."""

import pytest
from src.app.exceptions import (
    RAGException,
    EmbeddingError,
    RerankingError,
    GenerationError,
    VectorDBError,
    ConfigurationError,
    ValidationError,
)


class TestRAGException:
    """Test base RAGException class."""

    def test_basic_exception(self):
        """Test creating exception with just message."""
        exc = RAGException("Something went wrong")
        assert str(exc) == "Something went wrong"
        assert exc.message == "Something went wrong"
        assert exc.original_error is None

    def test_exception_with_original_error(self):
        """Test creating exception with original error."""
        original = ValueError("Original error")
        exc = RAGException("Wrapped error", original_error=original)

        assert str(exc) == "Wrapped error"
        assert exc.message == "Wrapped error"
        assert exc.original_error is original
        assert isinstance(exc.original_error, ValueError)

    def test_exception_inheritance(self):
        """Test that RAGException inherits from Exception."""
        exc = RAGException("Test")
        assert isinstance(exc, Exception)


class TestEmbeddingError:
    """Test EmbeddingError class."""

    def test_embedding_error(self):
        """Test embedding error creation."""
        exc = EmbeddingError("Failed to generate embedding")
        assert str(exc) == "Failed to generate embedding"
        assert isinstance(exc, RAGException)

    def test_embedding_error_with_original(self):
        """Test embedding error with original exception."""
        original = RuntimeError("Model failed")
        exc = EmbeddingError("Embedding failed", original_error=original)

        assert exc.message == "Embedding failed"
        assert exc.original_error is original


class TestRerankingError:
    """Test RerankingError class."""

    def test_reranking_error(self):
        """Test reranking error creation."""
        exc = RerankingError("Reranking failed")
        assert str(exc) == "Reranking failed"
        assert isinstance(exc, RAGException)


class TestGenerationError:
    """Test GenerationError class."""

    def test_generation_error(self):
        """Test generation error creation."""
        exc = GenerationError("LLM call failed")
        assert str(exc) == "LLM call failed"
        assert isinstance(exc, RAGException)

    def test_generation_error_with_api_error(self):
        """Test generation error wrapping API error."""
        api_error = Exception("Rate limit exceeded")
        exc = GenerationError("Generation failed", original_error=api_error)

        assert "Generation failed" in str(exc)
        assert exc.original_error is api_error


class TestVectorDBError:
    """Test VectorDBError class."""

    def test_vectordb_error(self):
        """Test vector DB error creation."""
        exc = VectorDBError("Qdrant connection failed")
        assert str(exc) == "Qdrant connection failed"
        assert isinstance(exc, RAGException)


class TestConfigurationError:
    """Test ConfigurationError class."""

    def test_configuration_error(self):
        """Test configuration error creation."""
        exc = ConfigurationError("Missing API key")
        assert str(exc) == "Missing API key"
        assert isinstance(exc, RAGException)

    def test_configuration_error_with_file_error(self):
        """Test configuration error wrapping file error."""
        file_error = FileNotFoundError("config.yaml not found")
        exc = ConfigurationError("Config load failed", original_error=file_error)

        assert exc.message == "Config load failed"
        assert isinstance(exc.original_error, FileNotFoundError)


class TestValidationError:
    """Test ValidationError class."""

    def test_validation_error(self):
        """Test validation error creation."""
        exc = ValidationError("Invalid input")
        assert str(exc) == "Invalid input"
        assert isinstance(exc, RAGException)


class TestExceptionChaining:
    """Test exception chaining and propagation."""

    def test_multiple_wrapped_exceptions(self):
        """Test wrapping exceptions multiple times."""
        level1 = ValueError("Base error")
        level2 = EmbeddingError("Embedding failed", original_error=level1)
        level3 = VectorDBError("DB operation failed", original_error=level2)

        assert level3.message == "DB operation failed"
        assert isinstance(level3.original_error, EmbeddingError)
        assert isinstance(level3.original_error.original_error, ValueError)

    def test_exception_can_be_raised_and_caught(self):
        """Test that custom exceptions can be raised and caught."""
        with pytest.raises(EmbeddingError) as exc_info:
            raise EmbeddingError("Test error")

        assert "Test error" in str(exc_info.value)

    def test_catching_base_exception(self):
        """Test that derived exceptions can be caught as RAGException."""
        with pytest.raises(RAGException):
            raise EmbeddingError("Test")

        with pytest.raises(RAGException):
            raise GenerationError("Test")
