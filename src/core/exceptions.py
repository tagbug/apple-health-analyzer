"""Custom exceptions for Apple Health Analyzer.

Provides specific exception types for different error conditions.
"""


class HealthAnalyzerError(Exception):
  """Base exception for all health analyzer errors."""

  pass


class ConfigurationError(HealthAnalyzerError):
  """Raised when there are configuration-related errors."""

  pass


class XMLParseError(HealthAnalyzerError):
  """Raised when XML parsing fails."""

  pass


class DataValidationError(HealthAnalyzerError):
  """Raised when data validation fails."""

  pass


class FileOperationError(HealthAnalyzerError):
  """Raised when file operations fail."""

  pass


class DataProcessingError(HealthAnalyzerError):
  """Raised when data processing operations fail."""

  pass


class AnalysisError(HealthAnalyzerError):
  """Raised when analysis operations fail."""

  pass


class VisualizationError(HealthAnalyzerError):
  """Raised when visualization operations fail."""

  pass
