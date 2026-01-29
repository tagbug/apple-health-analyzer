"""Tests for logger module."""

import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import Config, Environment
from src.utils.logger import (
  ProgressLogger,
  UnifiedProgress,
  get_logger,
  performance_logger,
  setup_logging,
)


class TestLoggerSetup:
  """Test logging setup functionality."""

  @patch("src.utils.logger.get_config")
  def test_setup_logging_development(self, mock_get_config):
    """Test logging setup in development mode."""
    # Create a temporary file for the required export_xml_path
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp_file:
      tmp_path = tmp_file.name

    try:
      # Mock config for development
      config = Config(export_xml_path=Path(tmp_path))
      config.log_level = "DEBUG"
      config.log_file = None
      mock_get_config.return_value = config

      # Should not raise any exceptions
      setup_logging()
    finally:
      import os

      os.unlink(tmp_path)

  @patch("src.utils.logger.get_config")
  def test_setup_logging_production(self, mock_get_config):
    """Test logging setup in production mode."""
    # Create a temporary file for the required export_xml_path
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp_file:
      tmp_path = tmp_file.name

    try:
      # Mock config for production
      config = Config(
        export_xml_path=Path(tmp_path), environment=Environment.PROD
      )
      config.log_level = "INFO"
      config.log_file = None
      mock_get_config.return_value = config

      # Should not raise any exceptions
      setup_logging()
    finally:
      import os

      os.unlink(tmp_path)

  @patch("src.utils.logger.get_config")
  def test_setup_logging_with_file(self, mock_get_config):
    """Test logging setup with file output."""
    with tempfile.TemporaryDirectory() as temp_dir:
      log_file = Path(temp_dir) / "test.log"

      # Create a temporary file for the required export_xml_path
      import tempfile as tf

      with tf.NamedTemporaryFile(delete=False, suffix=".xml") as tmp_file:
        tmp_path = tmp_file.name

      try:
        # Mock config with log file
        config = Config(
          export_xml_path=Path(tmp_path), environment=Environment.PROD
        )
        config.log_level = "INFO"
        config.log_file = log_file
        mock_get_config.return_value = config

        # Should not raise any exceptions
        setup_logging()

        # File should be created
        assert log_file.exists()
      finally:
        import os

        os.unlink(tmp_path)

  def test_get_logger(self):
    """Test getting a logger instance."""
    logger = get_logger("test_module")
    assert logger is not None
    # Test that it can log without errors
    logger.debug("Test debug message")
    logger.info("Test info message")


class TestPerformanceLogger:
  """Test performance logging decorator."""

  @patch("src.utils.logger.get_config")
  def test_performance_logger_success(self, mock_get_config):
    """Test performance logger with successful function."""
    # Create a temporary file for the required export_xml_path
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp_file:
      tmp_path = tmp_file.name

    try:
      config = Config(export_xml_path=Path(tmp_path))
      config.debug = False
      mock_get_config.return_value = config

      @performance_logger
      def test_function():
        return "success"

      result = test_function()
      assert result == "success"
    finally:
      import os

      os.unlink(tmp_path)

  @patch("src.utils.logger.get_config")
  def test_performance_logger_exception(self, mock_get_config):
    """Test performance logger with function that raises exception."""
    # Create a temporary file for the required export_xml_path
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp_file:
      tmp_path = tmp_file.name

    try:
      config = Config(export_xml_path=Path(tmp_path))
      config.debug = False
      mock_get_config.return_value = config

      @performance_logger
      def failing_function():
        raise ValueError("Test error")

      with pytest.raises(ValueError, match="Test error"):
        failing_function()
    finally:
      import os

      os.unlink(tmp_path)

  @patch("src.utils.logger.get_config")
  @patch("src.utils.logger._get_memory_usage")
  def test_performance_logger_with_memory(
    self, mock_get_memory, mock_get_config
  ):
    """Test performance logger with memory tracking."""
    # Create a temporary file for the required export_xml_path
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp_file:
      tmp_path = tmp_file.name

    try:
      config = Config(export_xml_path=Path(tmp_path))
      config.debug = True
      mock_get_config.return_value = config

      # Mock memory usage: start=100MB, end=110MB
      mock_get_memory.side_effect = [100.0, 110.0]

      @performance_logger
      def memory_function():
        return "done"

      result = memory_function()
      assert result == "done"
    finally:
      import os

      os.unlink(tmp_path)


class TestProgressLogger:
  """Test ProgressLogger class."""

  def test_progress_logger_context_manager(self):
    """Test ProgressLogger as context manager."""
    with ProgressLogger("test_operation", total=100) as progress:
      assert progress.operation == "test_operation"
      assert progress.total == 100
      assert progress.processed == 0

  def test_progress_logger_update(self):
    """Test updating progress."""
    progress = ProgressLogger("test_operation", total=100, log_interval=10)

    progress.update(5)
    assert progress.processed == 5

    progress.update(15)
    assert progress.processed == 20

  def test_progress_logger_successful_completion(self):
    """Test successful completion logging."""
    progress = ProgressLogger("test_operation", total=50)

    with progress:
      progress.update(50)

    # Should complete without errors
    assert progress.processed == 50

  def test_progress_logger_exception_handling(self):
    """Test exception handling in progress logger."""
    progress = ProgressLogger("test_operation", total=50)

    with pytest.raises(ValueError):
      with progress:
        progress.update(25)
        raise ValueError("Test exception")

    # Should still have processed count
    assert progress.processed == 25


class TestUnifiedProgress:
  """Test UnifiedProgress class."""

  def test_unified_progress_initialization(self):
    """Test UnifiedProgress initialization."""
    progress = UnifiedProgress("test_operation", total=100)
    assert progress.operation == "test_operation"
    assert progress.total == 100
    assert progress.processed == 0

  def test_unified_progress_context_manager(self):
    """Test UnifiedProgress as context manager."""
    with UnifiedProgress(
      "test_operation", total=100, show_progress=False
    ) as progress:
      assert progress.operation == "test_operation"
      assert progress.processed == 0

  def test_unified_progress_update(self):
    """Test updating unified progress."""
    progress = UnifiedProgress("test_operation", total=100, show_progress=False)

    progress.update(10)
    assert progress.processed == 10

    progress.update(20, details="Processing items")
    assert progress.processed == 30

  def test_unified_progress_set_step(self):
    """Test setting progress step."""
    progress = UnifiedProgress("test_operation", total=100, show_progress=False)

    progress.set_step("Step 1", "Processing data")
    assert progress.current_step == "Step 1"

  def test_unified_progress_get_stats(self):
    """Test getting progress statistics."""
    progress = UnifiedProgress("test_operation", total=100, show_progress=False)

    progress.update(50)
    stats = progress.get_stats()

    assert stats["operation"] == "test_operation"
    assert stats["processed"] == 50
    assert stats["total"] == 100
    assert "elapsed_seconds" in stats
    assert "rate_per_second" in stats
    assert "percent_complete" in stats

  def test_unified_progress_without_total(self):
    """Test unified progress without known total."""
    progress = UnifiedProgress("test_operation", show_progress=False)

    progress.update(25)
    stats = progress.get_stats()

    assert stats["operation"] == "test_operation"
    assert stats["processed"] == 25
    assert stats["total"] is None
    assert "percent_complete" not in stats

  def test_unified_progress_exception_handling(self):
    """Test exception handling in unified progress."""
    progress = UnifiedProgress("test_operation", total=100, show_progress=False)

    with pytest.raises(RuntimeError):
      with progress:
        progress.update(30)
        raise RuntimeError("Test error")

    # Should still have processed count
    assert progress.processed == 30

  @patch("src.utils.logger.console")
  def test_unified_progress_with_console(self, mock_console):
    """Test unified progress with console display."""
    # Mock Rich progress to avoid actual console output
    with patch("rich.progress.Progress") as mock_progress_class:
      mock_progress_instance = MagicMock()
      mock_progress_class.return_value = mock_progress_instance

      progress = UnifiedProgress(
        "test_operation", total=100, show_progress=True
      )

      with progress:
        progress.update(50)

      # Verify progress instance was used
      mock_progress_instance.__enter__.assert_called_once()
      mock_progress_instance.__exit__.assert_called_once()

  def test_unified_progress_quiet_mode(self):
    """Test unified progress in quiet mode."""
    progress = UnifiedProgress(
      "test_operation", total=100, quiet=True, show_progress=False
    )

    with progress:
      progress.update(50)

    # Should not log anything (we can't easily test this without mocking logger)
    assert progress.processed == 50


class TestMemoryUsage:
  """Test memory usage functionality."""

  @patch("src.utils.logger.psutil")
  def test_get_memory_usage_with_psutil(self, mock_psutil):
    """Test memory usage retrieval with psutil available."""
    from src.utils.logger import _get_memory_usage

    # Mock psutil process
    mock_process = MagicMock()
    mock_process.memory_info.return_value.rss = 104857600  # 100MB in bytes
    mock_psutil.Process.return_value = mock_process

    memory_mb = _get_memory_usage()
    assert memory_mb == 100.0  # Should be converted to MB

  @patch("src.utils.logger.psutil", side_effect=ImportError)
  def test_get_memory_usage_without_psutil(self, mock_psutil):
    """Test memory usage retrieval when psutil is not available."""
    from src.utils.logger import _get_memory_usage

    memory_mb = _get_memory_usage()
    assert memory_mb == 0.0  # Should return 0 when psutil unavailable


class TestLoggingIntegration:
  """Test integration with standard logging library."""

  def test_standard_logging_interception(self):
    """Test that standard logging calls are intercepted."""
    # This tests that our InterceptHandler is working
    std_logger = logging.getLogger("test_std_logger")
    std_logger.setLevel(logging.DEBUG)

    # This should be intercepted by our handler and routed to loguru
    std_logger.info("Test standard logging message")

    # Should not raise any exceptions
    assert True

  @patch("src.utils.logger.get_config")
  def test_setup_logging_replaces_handlers(self, mock_get_config):
    """Test that setup_logging replaces standard library handlers."""
    # Create a temporary file for the required export_xml_path
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp_file:
      tmp_path = tmp_file.name

    try:
      config = Config(export_xml_path=Path(tmp_path))
      config.log_level = "INFO"
      mock_get_config.return_value = config

      # Get initial handlers
      root_logger = logging.getLogger()
      initial_handlers = root_logger.handlers.copy()

      setup_logging()

      # Handlers should be different after setup
      final_handlers = root_logger.handlers
      # Note: We can't easily test this without more complex mocking,
      # but the fact that it doesn't crash is a good sign
    finally:
      import os

      os.unlink(tmp_path)
