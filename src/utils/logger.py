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
from rich.console import Console

from src.config import get_config
from src.i18n import Translator, resolve_locale

# Global console instance for progress display
console = Console()


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

    translator = Translator(resolve_locale())
    logger.info(translator.t("log.logger.file_logging", path=log_file_path))

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
      while frame and frame.f_code.co_name == "emit":
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
    translator = Translator(resolve_locale())
    start_time = time.time()
    start_memory = _get_memory_usage() if config.debug else 0

    logger.debug(translator.t("log.performance.start", function=func.__name__))

    try:
      result = func(*args, **kwargs)
      end_time = time.time()
      end_memory = _get_memory_usage() if config.debug else 0

      duration = end_time - start_time
      memory_delta = end_memory - start_memory if config.debug else 0

      if config.is_development:
        memory_info = (
          translator.t("log.performance.memory", memory=memory_delta)
          if config.debug and memory_delta > 0
          else ""
        )
        logger.info(
          translator.t(
            "log.performance.completed",
            function=func.__name__,
            duration=duration,
            memory=memory_info,
          )
        )
      else:
        logger.debug(
          translator.t(
            "log.performance.completed",
            function=func.__name__,
            duration=duration,
            memory="",
          )
        )

      return result

    except Exception as e:
      end_time = time.time()
      duration = end_time - start_time
      logger.error(
        translator.t(
          "log.performance.failed",
          function=func.__name__,
          duration=duration,
          error=e,
        )
      )
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
  except Exception:
    # Any other error (e.g., psutil not working)
    return 0.0


class ProgressLogger:
  """Context manager for logging progress of long-running operations."""

  def __init__(
    self, operation: str, total: int | None = None, log_interval: int = 1000
  ):
    self.operation = operation
    self.total = total
    self.log_interval = log_interval
    self.start_time = None
    self.processed = 0
    self.logger = get_logger(__name__)

  def __enter__(self):
    self.start_time = time.time()
    translator = Translator(resolve_locale())
    self.logger.info(translator.t("log.progress.start", operation=self.operation))
    return self

  def __exit__(
    self,
    exc_type: type[BaseException] | None,
    exc_val: BaseException | None,
    exc_tb: FrameType | None,
  ) -> bool | None:
    end_time = time.time()
    duration = end_time - (self.start_time or 0)

    if exc_type is None:
      if self.total:
        translator = Translator(resolve_locale())
        self.logger.info(
          translator.t(
            "log.progress.completed_total",
            operation=self.operation,
            processed=self.processed,
            total=self.total,
            duration=duration,
          )
        )
      else:
        translator = Translator(resolve_locale())
        self.logger.info(
          translator.t(
            "log.progress.completed",
            operation=self.operation,
            processed=self.processed,
            duration=duration,
          )
        )
    else:
      translator = Translator(resolve_locale())
      self.logger.error(
        translator.t(
          "log.progress.failed",
          operation=self.operation,
          duration=duration,
          error=exc_val,
        )
      )
      # Re-raise the exception so calling code can handle it
      return False

  def update(self, count: int = 1) -> None:
    """Update progress counter."""
    self.processed += count

    # Skip logging if log_interval is 0 or negative
    if self.log_interval <= 0:
      return

    if self.processed % self.log_interval == 0:
      elapsed = time.time() - (self.start_time or 0)
      rate = self.processed / elapsed if elapsed > 0 else 0
      translator = Translator(resolve_locale())

      if self.total:
        percent = (self.processed / self.total) * 100
        eta = (self.total - self.processed) / rate if rate > 0 else 0
        self.logger.info(
          translator.t(
            "log.progress.update_total",
            operation=self.operation,
            processed=self.processed,
            total=self.total,
            percent=percent,
            rate=rate,
            eta=eta,
          )
        )
      else:
        self.logger.info(
          translator.t(
            "log.progress.update",
            operation=self.operation,
            processed=self.processed,
            rate=rate,
          )
        )


class UnifiedProgress:
  """Unified progress display system combining console status and detailed logging.

  Features:
  - Rich console progress bars with ETA
  - Detailed logging for background monitoring
  - Multi-step operation support
  - Memory-efficient for large datasets
  """

  def __init__(
    self,
    operation: str,
    total: int | None = None,
    show_progress: bool = True,
    log_interval: int = 1000,
    console_update_interval: float = 0.1,
    quiet: bool = False,
  ):
    """
    Initialize unified progress display.

    Args:
        operation: Name of the operation being performed
        total: Total number of items to process (None for unknown)
        show_progress: Whether to show console progress bar
        log_interval: How often to log progress (in items)
        console_update_interval: How often to update console progress (in seconds)
    """
    self.operation = operation
    self.total = total
    self.show_progress = show_progress
    self.log_interval = log_interval
    self.console_update_interval = console_update_interval
    self.quiet = quiet

    self.start_time = None
    self.processed = 0
    self.current_step = ""
    self.step_start_time = None

    self.logger = get_logger(__name__)

    # Import here to avoid circular imports
    from rich.progress import (
      BarColumn,
      Progress,
      SpinnerColumn,
      TaskProgressColumn,
      TextColumn,
      TimeRemainingColumn,
    )

    if show_progress:
      if self.total is None:
        # For unknown total, use spinner without progress bar
        self.progress = Progress(
          SpinnerColumn(),
          TextColumn("[bold blue]{task.description}"),
          TextColumn("[dim]{task.fields[details]}"),
          console=console,
          refresh_per_second=10,
        )
      else:
        # For known total, use full progress bar
        self.progress = Progress(
          SpinnerColumn(),
          TextColumn("[bold blue]{task.description}"),
          BarColumn(),
          TaskProgressColumn(),
          TextColumn("•"),
          TimeRemainingColumn(),
          TextColumn("[dim]{task.fields[details]}"),
          console=console,
          refresh_per_second=10,
        )
    else:
      self.progress = None

    self.task_id = None

  def __enter__(self):
    self.start_time = time.time()
    if not self.quiet:
      translator = Translator(resolve_locale())
      self.logger.info(translator.t("log.progress.start", operation=self.operation))

    if self.progress:
      self.progress.__enter__()
      self.task_id = self.progress.add_task(
        self.operation, total=self.total, details="Initializing..."
      )

    return self

  def __exit__(
    self,
    exc_type: type[BaseException] | None,
    exc_val: BaseException | None,
    exc_tb: FrameType | None,
  ) -> bool | None:
    end_time = time.time()
    duration = end_time - (self.start_time or 0)

    if exc_type is None:
      if not self.quiet:
        translator = Translator(resolve_locale())
        if self.total:
          self.logger.info(
            translator.t(
              "log.progress.completed_total",
              operation=self.operation,
              processed=self.processed,
              total=self.total,
              duration=duration,
            )
          )
        else:
          self.logger.info(
            translator.t(
              "log.progress.completed",
              operation=self.operation,
              processed=self.processed,
              duration=duration,
            )
          )

      if self.progress and self.task_id is not None:
        self.progress.update(
          self.task_id,
          completed=self.total or self.processed,
          details="✓ Completed",
        )
        self.progress.__exit__(None, None, None)

    else:
      if not self.quiet:
        translator = Translator(resolve_locale())
        self.logger.error(
          translator.t(
            "log.progress.failed",
            operation=self.operation,
            duration=duration,
            error=exc_val,
          )
        )

      if self.progress and self.task_id is not None:
        self.progress.update(self.task_id, details=f"❌ Failed: {exc_val}")
        self.progress.__exit__(None, None, None)

      # Re-raise the exception so calling code can handle it
      return False

  def update(
    self, count: int = 1, details: str | None = None, step: str | None = None
  ) -> None:
    """Update progress counter and display information."""
    self.processed += count

    # Update step information
    if step:
      self.current_step = step
      self.step_start_time = time.time()
      if not self.quiet:
        translator = Translator(resolve_locale())
        self.logger.info(translator.t("log.progress.step", step=step))

    # Update console progress
    if self.progress and self.task_id is not None:
      display_details = (
        details
        or self.current_step
        or Translator(resolve_locale()).t("log.progress.processing")
      )
      self.progress.update(
        self.task_id, completed=self.processed, details=display_details
      )

    # Log progress at intervals
    if (
      not self.quiet
      and self.log_interval > 0
      and self.processed % self.log_interval == 0
    ):
      elapsed = time.time() - (self.start_time or 0)
      rate = self.processed / elapsed if elapsed > 0 else 0
      translator = Translator(resolve_locale())

      if self.total:
        percent = (self.processed / self.total) * 100
        eta = (self.total - self.processed) / rate if rate > 0 else 0
        self.logger.info(
          translator.t(
            "log.progress.update_total",
            operation=self.operation,
            processed=self.processed,
            total=self.total,
            percent=percent,
            rate=rate,
            eta=eta,
          )
        )
      else:
        self.logger.info(
          translator.t(
            "log.progress.update",
            operation=self.operation,
            processed=self.processed,
            rate=rate,
          )
        )

  def set_step(self, step: str, details: str | None = None) -> None:
    """Set current operation step."""
    self.current_step = step
    self.step_start_time = time.time()
    if not self.quiet:
      translator = Translator(resolve_locale())
      self.logger.info(translator.t("log.progress.step", step=step))

    if self.progress and self.task_id is not None:
      translator = Translator(resolve_locale())
      display_details = details or translator.t("log.progress.step", step=step)
      self.progress.update(self.task_id, details=display_details)

  def get_stats(self) -> dict[str, Any]:
    """Get current progress statistics."""
    elapsed = time.time() - (self.start_time or 0)
    rate = self.processed / elapsed if elapsed > 0 else 0

    stats = {
      "operation": self.operation,
      "processed": self.processed,
      "total": self.total,
      "elapsed_seconds": elapsed,
      "rate_per_second": rate,
      "current_step": self.current_step,
    }

    if self.total:
      stats["percent_complete"] = (self.processed / self.total) * 100
      stats["eta_seconds"] = (self.total - self.processed) / rate if rate > 0 else 0

    return stats


# Initialize logging on import
setup_logging()
