"""
Retry utilities for handling API errors with exponential backoff.
"""

import time
import logging
from functools import wraps
from typing import Callable, Optional, Tuple, Type, Union

logger = logging.getLogger(__name__)


def retry_with_exponential_backoff(
    max_retries: int = 5,
    initial_sleep: float = 2.0,
    max_sleep: float = 60.0,
    exponential_base: float = 2.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    retryable_error_codes: Tuple[int, ...] = (429, 503, 500, 502, 504),
) -> Callable:
    """
    Decorator to retry a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_sleep: Initial sleep duration in seconds
        max_sleep: Maximum sleep duration in seconds
        exponential_base: Base for exponential backoff calculation
        retryable_exceptions: Tuple of exception types to retry on
        retryable_error_codes: Tuple of HTTP error codes to retry on

    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    error_str = str(e)

                    # Check if this is a retryable error
                    is_retryable = False

                    # Check for specific error codes in the exception message
                    for code in retryable_error_codes:
                        if str(code) in error_str:
                            is_retryable = True
                            break

                    # Also check for common API error patterns
                    if any(pattern in error_str.lower() for pattern in [
                        "resource exhausted",
                        "rate limit",
                        "too many requests",
                        "timeout",
                        "service unavailable",
                        "quota exceeded"
                    ]):
                        is_retryable = True

                    if not is_retryable or attempt == max_retries:
                        # Not retryable or last attempt
                        logger.error(f"Failed after {attempt + 1} attempts: {e}")
                        raise

                    # Calculate sleep time with exponential backoff
                    sleep_time = min(
                        initial_sleep * (exponential_base ** attempt),
                        max_sleep
                    )

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed with {type(e).__name__}: {e}. "
                        f"Retrying in {sleep_time:.1f} seconds..."
                    )

                    time.sleep(sleep_time)

            # This shouldn't be reached, but just in case
            if last_exception:
                raise last_exception

        return wrapper
    return decorator


def is_retryable_error(error: Exception) -> bool:
    """
    Check if an error is retryable based on common patterns.

    Args:
        error: The exception to check

    Returns:
        True if the error is retryable, False otherwise
    """
    error_str = str(error).lower()

    retryable_patterns = [
        "429",
        "503",
        "500",
        "502",
        "504",
        "resource exhausted",
        "rate limit",
        "too many requests",
        "timeout",
        "service unavailable",
        "quota exceeded",
        "temporarily unavailable",
        "connection reset",
        "connection refused"
    ]

    return any(pattern in error_str for pattern in retryable_patterns)


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_retries: int = 5,
        initial_sleep: float = 2.0,
        max_sleep: float = 60.0,
        exponential_base: float = 2.0
    ):
        self.max_retries = max_retries
        self.initial_sleep = initial_sleep
        self.max_sleep = max_sleep
        self.exponential_base = exponential_base

    @classmethod
    def from_dict(cls, config: dict) -> "RetryConfig":
        """Create RetryConfig from dictionary."""
        return cls(
            max_retries=config.get("max_retries", 5),
            initial_sleep=config.get("initial_sleep", 2.0),
            max_sleep=config.get("max_sleep", 60.0),
            exponential_base=config.get("exponential_base", 2.0)
        )

    def to_decorator_kwargs(self) -> dict:
        """Convert to kwargs for the retry decorator."""
        return {
            "max_retries": self.max_retries,
            "initial_sleep": self.initial_sleep,
            "max_sleep": self.max_sleep,
            "exponential_base": self.exponential_base
        }