"""Retry utilities with exponential backoff."""

import time
from functools import wraps
from typing import Callable, Type, TypeVar
from loguru import logger

T = TypeVar("T")


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    retryable: Callable[[Exception], bool] | None = None,
):
    """
    Retry decorator with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        backoff_factor: Multiplier for delay between retries
        exceptions: Tuple of exception types to catch and retry
        retryable: Optional callable; if provided and returns False for an exception,
            the exception is re-raised immediately without retrying

    Returns:
        Decorated function that retries on failure
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if retryable is not None and not retryable(e):
                        raise
                    last_exception = e

                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {str(e)}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}: {str(e)}"
                        )

            # If we get here, all retries failed
            raise last_exception

        return wrapper

    return decorator


def is_transient_error(error: Exception) -> bool:
    """
    Determine if an error is transient and should be retried.

    Args:
        error: Exception to check

    Returns:
        True if error is transient, False otherwise
    """
    error_message = str(error).lower()

    # Common transient error patterns — use specific substrings to avoid
    # matching non-retryable errors like "invalid connection string"
    transient_patterns = [
        "timeout",
        "connection refused",
        "connection reset",
        "connection timed out",
        "connection error",
        "rate limit",
        "throttle",
        "too many requests",
        "503",
        "504",
        "502",
        "temporary",
        "unavailable",
    ]

    return any(pattern in error_message for pattern in transient_patterns)
