"""Logging configuration and utilities for Apple Health Analyzer.

Provides structured logging with performance monitoring and environment-aware configuration.
"""

import functools
import sys
import time
from collections.abc import Callable
from types import FrameType
from typing import Any

from loguru import logger

from src.config import get_config


def setup_logging() -> None:
    """Setup logging configuration based on current config."""
    config = get_config()

    # Remove default handler
    logger.remove()

    # Common format
    base_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # Console handler
    if config.is_development:
        # Development: colored, detailed format
        logger.add(
            sys.stdout,
            format=base_format,
            level=config.log_level,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )
    else:
        # Production: simple format
        logger.add(
            sys.stdout,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            level=config.log_level,
            colorize=False,
        )

    # File handler (if configured)
    if config.log_file:
        log_file_path = config.log_file
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
            level=config.log_level,
            rotation="10 MB",
            retention="1 week",
            encoding="utf-8",
        )

        logger.info(f"Logging to file: {log_file_path}")

    # Set loguru as the standard library logger
    import logging
    class InterceptHandler(logging.Handler):
        def __init__(self):
            super().__init__()

        def emit(self, record: logging.LogRecord) -> None:
            # Get corresponding Loguru level if it exists
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where originated the logged message
            frame = logging.currentframe()
            depth = 2
            while frame and frame.f_code.co_name == 'emit':
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

    # Replace standard library handlers
    logging.basicConfig(handlers=[InterceptHandler()], level=0)

def get_logger(name: str) -> Any:
    """Get a logger instance with the specified name."""
    return logger.bind(name=name)

def performance_logger(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to log function performance metrics.

    Usage:
        @performance_logger
        def my_function():
            pass
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        config = get_config()
        start_time = time.time()
        start_memory = _get_memory_usage() if config.debug else 0

        logger.debug(f"Starting {func.__name__}")

        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            end_memory = _get_memory_usage() if config.debug else 0

            duration = end_time - start_time
            memory_delta = end_memory - start_memory if config.debug else 0

            if config.is_development:
                memory_info = f" (+{memory_delta:.1f}MB)" if config.debug and memory_delta > 0 else ""
                logger.info(f"Completed {func.__name__} in {duration:.3f}s{memory_info}")
            else:
                logger.debug(f"Completed {func.__name__} in {duration:.3f}s")

            return result

        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            logger.error(f"Failed {func.__name__} after {duration:.3f}s: {e}")
            raise

    return wrapper

def _get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        # psutil not available
        return 0.0

class ProgressLogger:
    """Context manager for logging progress of long-running operations."""

    def __init__(self, operation: str, total: int | None = None, log_interval: int = 1000):
        self.operation = operation
        self.total = total
        self.log_interval = log_interval
        self.start_time = None
        self.processed = 0
        self.logger = get_logger(__name__)

    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting {self.operation}")
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: FrameType | None) -> bool | None:
        end_time = time.time()
        duration = end_time - (self.start_time or 0)

        if exc_type is None:
            if self.total:
                self.logger.info(f"Completed {self.operation}: {self.processed}/{self.total} items in {duration:.1f}s")
            else:
                self.logger.info(f"Completed {self.operation}: {self.processed} items in {duration:.1f}s")
        else:
            self.logger.error(f"Failed {self.operation} after {duration:.1f}s: {exc_val}")
            # Re-raise the exception so calling code can handle it
            return False

    def update(self, count: int = 1) -> None:
        """Update progress counter."""
        self.processed += count

        if self.processed % self.log_interval == 0:
            elapsed = time.time() - (self.start_time or 0)
            rate = self.processed / elapsed if elapsed > 0 else 0

            if self.total:
                percent = (self.processed / self.total) * 100
                eta = (self.total - self.processed) / rate if rate > 0 else 0
                self.logger.info(f"{self.operation}: {self.processed}/{self.total} ({percent:.1f}%) at {rate:.0f} items/s, ETA: {eta:.0f}s")
            else:
                self.logger.info(f"{self.operation}: {self.processed} items at {rate:.0f} items/s")

# Initialize logging on import
setup_logging()
