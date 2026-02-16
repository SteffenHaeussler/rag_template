"""Tests for retry logic and utilities."""

import pytest
from unittest.mock import MagicMock, patch
import time
from src.app.retry import retry_with_backoff, is_transient_error


class TestRetryWithBackoff:
    """Test retry_with_backoff decorator."""

    def test_successful_call_no_retry(self):
        """Test that successful calls don't retry."""
        mock_func = MagicMock(return_value="success")
        decorated = retry_with_backoff(max_retries=3)(mock_func)

        result = decorated()

        assert result == "success"
        assert mock_func.call_count == 1

    def test_retry_on_failure(self):
        """Test that function retries on failure."""
        mock_func = MagicMock(side_effect=[
            Exception("Fail 1"),
            Exception("Fail 2"),
            "success"
        ])
        mock_func.__name__ = "mock_func"  # Add __name__ for logging
        decorated = retry_with_backoff(max_retries=3, initial_delay=0.01)(mock_func)

        result = decorated()

        assert result == "success"
        assert mock_func.call_count == 3

    def test_max_retries_exceeded(self):
        """Test that function raises after max retries."""
        mock_func = MagicMock(side_effect=Exception("Always fails"))
        mock_func.__name__ = "mock_func"
        decorated = retry_with_backoff(max_retries=2, initial_delay=0.01)(mock_func)

        with pytest.raises(Exception):  # Just check exception is raised
            decorated()

        assert mock_func.call_count == 3  # 1 initial + 2 retries

    def test_exponential_backoff(self):
        """Test that backoff delay increases exponentially."""
        call_times = []

        def failing_func():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise Exception("Fail")
            return "success"

        decorated = retry_with_backoff(
            max_retries=3,
            initial_delay=0.1,
            backoff_factor=2.0
        )(failing_func)

        result = decorated()

        assert result == "success"
        assert len(call_times) == 3

        # Check that delays are approximately correct (with tolerance)
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]

        assert 0.08 < delay1 < 0.15  # ~0.1s
        assert 0.18 < delay2 < 0.25  # ~0.2s

    def test_specific_exception_types(self):
        """Test that only specified exception types are retried."""
        mock_func = MagicMock(side_effect=ValueError("Value error"))
        decorated = retry_with_backoff(
            max_retries=2,
            initial_delay=0.01,
            exceptions=(RuntimeError,)  # Only retry RuntimeError
        )(mock_func)

        # ValueError should not be retried
        with pytest.raises(ValueError):
            decorated()

        assert mock_func.call_count == 1  # No retries

    def test_retry_with_args_and_kwargs(self):
        """Test that function arguments are preserved during retries."""
        mock_func = MagicMock(side_effect=[
            Exception("Fail"),
            "success"
        ])
        mock_func.__name__ = "mock_func"
        decorated = retry_with_backoff(max_retries=2, initial_delay=0.01)(mock_func)

        result = decorated("arg1", "arg2", kwarg1="value1")

        assert result == "success"
        assert mock_func.call_count == 2

        # Check that args were passed correctly
        mock_func.assert_called_with("arg1", "arg2", kwarg1="value1")

    @patch('src.app.retry.logger')
    def test_logging_on_retry(self, mock_logger):
        """Test that retries are logged."""
        mock_func = MagicMock(side_effect=[
            Exception("Fail 1"),
            "success"
        ])
        mock_func.__name__ = "test_function"
        decorated = retry_with_backoff(max_retries=2, initial_delay=0.01)(mock_func)

        decorated()

        # Should have logged the retry warning
        assert mock_logger.warning.called
        warning_call = mock_logger.warning.call_args[0][0]
        assert "test_function" in warning_call
        assert "Retrying" in warning_call

    @patch('src.app.retry.logger')
    def test_logging_on_final_failure(self, mock_logger):
        """Test that final failure is logged as error."""
        mock_func = MagicMock(side_effect=Exception("Always fails"))
        mock_func.__name__ = "test_function"
        decorated = retry_with_backoff(max_retries=1, initial_delay=0.01)(mock_func)

        with pytest.raises(Exception):
            decorated()

        # Should have logged the final error
        assert mock_logger.error.called
        error_call = mock_logger.error.call_args[0][0]
        assert "test_function" in error_call
        assert "attempts failed" in error_call


class TestIsTransientError:
    """Test is_transient_error helper function."""

    def test_timeout_error_is_transient(self):
        """Test that timeout errors are detected as transient."""
        error = Exception("Request timeout after 30s")
        assert is_transient_error(error) is True

    def test_connection_error_is_transient(self):
        """Test that connection errors are detected as transient."""
        error = Exception("Connection refused")
        assert is_transient_error(error) is True

    def test_rate_limit_error_is_transient(self):
        """Test that rate limit errors are detected as transient."""
        error = Exception("Rate limit exceeded")
        assert is_transient_error(error) is True

    def test_throttle_error_is_transient(self):
        """Test that throttle errors are detected as transient."""
        error = Exception("Request throttled")
        assert is_transient_error(error) is True

    def test_503_error_is_transient(self):
        """Test that 503 errors are detected as transient."""
        error = Exception("Service unavailable: 503")
        assert is_transient_error(error) is True

    def test_502_error_is_transient(self):
        """Test that 502 errors are detected as transient."""
        error = Exception("Bad gateway: 502")
        assert is_transient_error(error) is True

    def test_504_error_is_transient(self):
        """Test that 504 errors are detected as transient."""
        error = Exception("Gateway timeout: 504")
        assert is_transient_error(error) is True

    def test_too_many_requests_is_transient(self):
        """Test that 'too many requests' is detected as transient."""
        error = Exception("429 Too many requests")
        assert is_transient_error(error) is True

    def test_temporary_error_is_transient(self):
        """Test that temporary errors are detected as transient."""
        error = Exception("Temporary failure")
        assert is_transient_error(error) is True

    def test_unavailable_error_is_transient(self):
        """Test that unavailable errors are detected as transient."""
        error = Exception("Service temporarily unavailable")
        assert is_transient_error(error) is True

    def test_validation_error_is_not_transient(self):
        """Test that validation errors are not transient."""
        error = Exception("Invalid input format")
        assert is_transient_error(error) is False

    def test_auth_error_is_not_transient(self):
        """Test that auth errors are not transient."""
        error = Exception("Authentication failed")
        assert is_transient_error(error) is False

    def test_not_found_error_is_not_transient(self):
        """Test that not found errors are not transient."""
        error = Exception("Resource not found")
        assert is_transient_error(error) is False

    def test_case_insensitive_matching(self):
        """Test that error detection is case-insensitive."""
        error1 = Exception("TIMEOUT occurred")
        error2 = Exception("Connection REFUSED")

        assert is_transient_error(error1) is True
        assert is_transient_error(error2) is True
