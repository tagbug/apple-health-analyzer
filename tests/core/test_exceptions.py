"""Tests for custom exception classes."""

import pytest

from src.core.exceptions import (
  AnalysisError,
  ConfigurationError,
  DataProcessingError,
  DataValidationError,
  FileOperationError,
  HealthAnalyzerError,
  VisualizationError,
  XMLParseError,
)


def test_exception_inheritance():
  """Ensure custom exceptions inherit from HealthAnalyzerError."""
  exceptions = [
    ConfigurationError,
    XMLParseError,
    DataValidationError,
    FileOperationError,
    DataProcessingError,
    AnalysisError,
    VisualizationError,
  ]

  for exc in exceptions:
    instance = exc("test")
    assert isinstance(instance, HealthAnalyzerError)
    assert isinstance(instance, Exception)


def test_exception_raising():
  """Ensure exceptions raise with the provided message."""
  with pytest.raises(ConfigurationError, match="invalid config"):
    raise ConfigurationError("invalid config")
