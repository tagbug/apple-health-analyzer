"""Coverage for logger utilities and progress helpers."""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.utils import logger as logger_module


def test_setup_logging_development(monkeypatch):
  """Setup logging should configure handlers in development."""
  config = SimpleNamespace(
    is_development=True,
    log_level="INFO",
    log_file=None,
    debug=False,
  )
  monkeypatch.setattr(logger_module, "get_config", lambda: config)
  mock_logger = MagicMock()
  monkeypatch.setattr(logger_module, "logger", mock_logger)

  logger_module.setup_logging()

  assert mock_logger.remove.called
  assert mock_logger.add.called


def test_setup_logging_production_with_file(monkeypatch, tmp_path):
  """Setup logging should create file handler in production."""
  log_file = tmp_path / "app.log"
  config = SimpleNamespace(
    is_development=False,
    log_level="INFO",
    log_file=log_file,
    debug=False,
  )
  monkeypatch.setattr(logger_module, "get_config", lambda: config)
  mock_logger = MagicMock()
  monkeypatch.setattr(logger_module, "logger", mock_logger)

  logger_module.setup_logging()

  assert mock_logger.add.called

  if log_file.exists():
    log_file.unlink()


def test_get_logger_returns_bound(monkeypatch):
  """get_logger should bind name to logger."""
  mock_logger = MagicMock()
  mock_logger.bind.return_value = "bound"
  monkeypatch.setattr(logger_module, "logger", mock_logger)

  assert logger_module.get_logger("x") == "bound"


def test_performance_logger_success(monkeypatch):
  """Decorator should log completion on success."""
  config = SimpleNamespace(is_development=True, debug=False)
  monkeypatch.setattr(logger_module, "get_config", lambda: config)
  mock_logger = MagicMock()
  monkeypatch.setattr(logger_module, "logger", mock_logger)

  @logger_module.performance_logger
  def add(a, b):
    return a + b

  assert add(1, 2) == 3
  assert mock_logger.info.called


def test_performance_logger_failure(monkeypatch):
  """Decorator should log error on failure."""
  config = SimpleNamespace(is_development=True, debug=False)
  monkeypatch.setattr(logger_module, "get_config", lambda: config)
  mock_logger = MagicMock()
  monkeypatch.setattr(logger_module, "logger", mock_logger)

  @logger_module.performance_logger
  def fail():
    raise RuntimeError("boom")

  try:
    fail()
  except RuntimeError:
    pass

  assert mock_logger.error.called


def test_get_memory_usage_fallback(monkeypatch):
  """Memory usage should return 0 when psutil missing."""
  import sys

  monkeypatch.setitem(sys.modules, "psutil", None)

  assert logger_module._get_memory_usage() == 0.0


def test_progress_logger_flow():
  """ProgressLogger should update and log without errors."""
  progress = logger_module.ProgressLogger("Test", total=2, log_interval=1)
  progress.logger = MagicMock()

  with progress:
    progress.update()
    progress.update()

  assert progress.processed == 2


def test_progress_logger_failure():
  """ProgressLogger should log on failure."""
  progress = logger_module.ProgressLogger("Test", total=None, log_interval=1)
  progress.logger = MagicMock()

  try:
    with progress:
      raise ValueError("boom")
  except ValueError:
    pass

  assert progress.logger.error.called


def test_unified_progress_basic(monkeypatch):
  """UnifiedProgress should update progress without errors."""
  progress = logger_module.UnifiedProgress("Test", total=2, show_progress=False)
  progress.logger = MagicMock()

  with progress:
    progress.update(details="step")
    progress.update(details="done")

  assert progress.processed == 2


def test_unified_progress_failure(monkeypatch):
  """UnifiedProgress should log errors on failure."""
  progress = logger_module.UnifiedProgress("Test", total=None, show_progress=False)
  progress.logger = MagicMock()

  try:
    with progress:
      raise ValueError("boom")
  except ValueError:
    pass

  assert progress.logger.error.called
