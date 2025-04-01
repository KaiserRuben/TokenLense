from typing import TypeVar, Callable, Any, Sequence
from functools import reduce, partial
from returns.result import Result, Success, Failure
from returns.maybe import Maybe, Some, Nothing
from toolz import compose as toolz_compose
import logging
from datetime import datetime

# Type variables for generic functions
A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')


def pipe(value: A, *functions: Sequence[Callable]) -> Any:
    """
    Pipes a value through a sequence of functions.

    Args:
        value: Initial value
        *functions: Functions to apply sequentially

    Returns:
        Result of applying all functions in sequence
    """
    return reduce(lambda acc, f: f(acc), functions, value)


def compose(*functions: Sequence[Callable]) -> Callable:
    """
    Composes multiple functions right to left.

    Args:
        *functions: Functions to compose

    Returns:
        Composed function
    """
    return toolz_compose(*functions)


def safe_operation(operation: Callable[..., B]) -> Callable[..., Result[B, Exception]]:
    """
    Wraps an operation in a Result type for safe execution.

    Args:
        operation: Function to wrap

    Returns:
        Wrapped function returning Result
    """

    def wrapped(*args: Any, **kwargs: Any) -> Result[B, Exception]:
        try:
            return Success(operation(*args, **kwargs))
        except Exception as e:
            logging.error(f"Operation failed: {str(e)}")
            return Failure(e)

    return wrapped


def ensure_not_none(value: A | None) -> Maybe[A]:
    """
    Converts a potentially None value to Maybe type.

    Args:
        value: Value to check

    Returns:
        Maybe containing the value or Nothing
    """
    return Some(value) if value is not None else Nothing


def log_operation(logger: logging.Logger) -> Callable[[Callable[..., B]], Callable[..., B]]:
    """
    Decorator to log function calls with timing information.

    Args:
        logger: Logger instance to use

    Returns:
        Decorated function
    """

    def decorator(func: Callable[..., B]) -> Callable[..., B]:
        def wrapped(*args: Any, **kwargs: Any) -> B:
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"{func.__name__} completed in {duration:.2f}s")
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(f"{func.__name__} failed after {duration:.2f}s: {str(e)}")
                raise

        return wrapped

    return decorator


def retry_operation(
        max_attempts: int = 3,
        delay_seconds: float = 1.0
) -> Callable[[Callable[..., B]], Callable[..., Result[B, Exception]]]:
    """
    Decorator to retry failed operations with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        delay_seconds: Initial delay between retries (doubles each attempt)

    Returns:
        Decorated function returning Result
    """
    from time import sleep

    def decorator(func: Callable[..., B]) -> Callable[..., Result[B, Exception]]:
        def wrapped(*args: Any, **kwargs: Any) -> Result[B, Exception]:
            last_exception: Exception | None = None

            for attempt in range(max_attempts):
                try:
                    result = func(*args, **kwargs)
                    return Success(result)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        sleep(delay_seconds * (2 ** attempt))

            assert last_exception is not None  # for type checker
            return Failure(last_exception)

        return wrapped

    return decorator


def memoize(func: Callable[..., B]) -> Callable[..., B]:
    """
    Memoizes a function's results.

    Args:
        func: Function to memoize

    Returns:
        Memoized function
    """
    cache: dict = {}

    def memoized(*args: Any, **kwargs: Any) -> B:
        key = (args, frozenset(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return memoized