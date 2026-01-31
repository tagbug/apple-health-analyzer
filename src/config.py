"""Configuration management for Apple Health Analyzer.

Supports environment-based configuration with validation and type safety.
"""

import os
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class Environment(Enum):
  """Application environment enumeration."""

  DEV = "dev"
  PROD = "prod"


class Config(BaseModel):
  """Main configuration class for the application."""

  # Environment settings
  environment: Environment = Field(default=Environment.DEV)
  debug: bool = Field(default=True)

  # Path configurations
  export_xml_path: Path = Field(...)
  output_dir: Path = Field(default=Path("./output"))

  # Data source priorities
  apple_watch_priority: int = Field(default=3, ge=1, le=10)
  xiaomi_health_priority: int = Field(default=2, ge=1, le=10)
  iphone_priority: int = Field(default=1, ge=1, le=10)

  # Logging configuration
  log_level: str = Field(default="INFO")
  log_file: Path | None = Field(default=None)

  # Localization
  locale: str = Field(default="en")

  # Performance settings
  batch_size: int = Field(default=1000, gt=0)
  memory_limit_mb: int = Field(default=500, gt=0)

  class Config:
    """Pydantic configuration."""

    validate_assignment = True
    frozen = False  # Allow updates for testing

  @field_validator("export_xml_path")
  @classmethod
  def validate_export_xml_path(cls, v: Path) -> Path:
    """Validate that export XML path exists and is readable."""
    if not v.exists():
      raise ValueError(f"Export XML file does not exist: {v}")
    if not v.is_file():
      raise ValueError(f"Export XML path is not a file: {v}")
    if not os.access(v, os.R_OK):
      raise ValueError(f"Export XML file is not readable: {v}")
    return v

  @field_validator("output_dir")
  @classmethod
  def validate_output_dir(cls, v: Path) -> Path:
    """Ensure output directory exists and is writable."""
    try:
      v.mkdir(parents=True, exist_ok=True)
      # Test write access
      test_file = v / ".write_test"
      test_file.write_text("test")
      test_file.unlink()
    except Exception as e:
      raise ValueError(f"Output directory is not writable: {v} ({e})") from e
    return v

  @field_validator("log_level")
  @classmethod
  def validate_log_level(cls, v: str) -> str:
    """Validate log level is a valid logging level."""
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if v.upper() not in valid_levels:
      raise ValueError(f"Invalid log level '{v}'. Must be one of: {valid_levels}")
    return v.upper()

  @field_validator("locale")
  @classmethod
  def validate_locale(cls, v: str) -> str:
    valid_locales = {"en", "zh"}
    if v not in valid_locales:
      raise ValueError(f"Invalid locale '{v}'. Must be one of: {sorted(valid_locales)}")
    return v

  @property
  def source_priority_map(self) -> dict[str, int]:
    """Get source name to priority mapping."""
    return {
      "Watch": self.apple_watch_priority,
      "Xiaomi Home": self.xiaomi_health_priority,
      "Phone": self.iphone_priority,
    }

  @property
  def is_development(self) -> bool:
    """Check if running in development mode."""
    return self.environment == Environment.DEV

  @property
  def is_production(self) -> bool:
    """Check if running in production mode."""
    return self.environment == Environment.PROD


def load_config() -> Config:
  """Load configuration from environment variables and .env file.

  Priority order:
  1. Environment variables
  2. .env file
  3. Default values
  """
  from dotenv import load_dotenv

  # Load .env file if it exists
  env_path = Path(".env")
  if env_path.exists():
    load_dotenv(env_path)

  # Build configuration from environment
  config_data = {}

  # Environment
  env_value = os.getenv("ENVIRONMENT", "dev").lower()
  config_data["environment"] = Environment(env_value)

  # Debug mode
  debug_value = os.getenv("DEBUG", "true").lower()
  config_data["debug"] = debug_value in ("true", "1", "yes", "on")

  # Paths
  export_xml = os.getenv("EXPORT_XML_PATH", "./export_data/export.xml")
  config_data["export_xml_path"] = Path(export_xml)

  output_dir = os.getenv("OUTPUT_DIR", "./output")
  config_data["output_dir"] = Path(output_dir)

  # Data source priorities
  config_data["apple_watch_priority"] = int(os.getenv("APPLE_WATCH_PRIORITY", "3"))
  config_data["xiaomi_health_priority"] = int(os.getenv("XIAOMI_HEALTH_PRIORITY", "2"))
  config_data["iphone_priority"] = int(os.getenv("IPHONE_PRIORITY", "1"))

  # Logging
  config_data["log_level"] = os.getenv("LOG_LEVEL", "INFO")
  log_file = os.getenv("LOG_FILE")
  if log_file:
    config_data["log_file"] = Path(log_file)

  # Localization
  config_data["locale"] = os.getenv("LOCALE", "en")

  # Performance
  config_data["batch_size"] = int(os.getenv("BATCH_SIZE", "1000"))
  config_data["memory_limit_mb"] = int(os.getenv("MEMORY_LIMIT_MB", "500"))

  return Config(**config_data)


# Global configuration instance
_config: Config | None = None


def get_config() -> Config:
  """Get the global configuration instance."""
  global _config
  if _config is None:
    _config = load_config()
  return _config


def reload_config() -> Config:
  """Reload configuration from environment."""
  global _config
  _config = load_config()
  return _config
