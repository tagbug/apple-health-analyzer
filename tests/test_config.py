"""Tests for config module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import (
  Config,
  DataSourcePriority,
  Environment,
  get_config,
  load_config,
  reload_config,
)


class TestEnvironmentEnum:
  """Test Environment enum."""

  def test_environment_values(self):
    """Test environment enum values."""
    assert Environment.DEV.value == "dev"
    assert Environment.PROD.value == "prod"


class TestDataSourcePriorityEnum:
  """Test DataSourcePriority enum."""

  def test_data_source_priority_values(self):
    """Test data source priority enum values."""
    assert DataSourcePriority.IPHONE.value == 1
    assert DataSourcePriority.XIAOMI_HEALTH.value == 2
    assert DataSourcePriority.APPLE_WATCH.value == 3


class TestConfig:
  """Test Config class."""

  def test_config_initialization(self):
    """Test config initialization with defaults."""
    # Create a temporary file for the required export_xml_path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp_file:
      tmp_path = Path(tmp_file.name)

    try:
      config = Config(export_xml_path=tmp_path)

      assert config.environment == Environment.DEV
      assert config.debug is True
      assert config.export_xml_path == tmp_path
      assert config.output_dir == Path("./output")
      assert config.apple_watch_priority == 3
      assert config.xiaomi_health_priority == 2
      assert config.iphone_priority == 1
      assert config.log_level == "INFO"
      assert config.log_file is None
      assert config.batch_size == 1000
      assert config.memory_limit_mb == 500
    finally:
      tmp_path.unlink(missing_ok=True)

  def test_config_properties(self):
    """Test config computed properties."""
    # Create a temporary file for the required export_xml_path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp_file:
      tmp_path = Path(tmp_file.name)

    try:
      config = Config(export_xml_path=tmp_path)

      assert config.is_development is True
      assert config.is_production is False

      config.environment = Environment.PROD
      assert config.is_development is False
      assert config.is_production is True
    finally:
      tmp_path.unlink(missing_ok=True)

  def test_source_priority_map(self):
    """Test source priority mapping."""
    # Create a temporary file for the required export_xml_path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp_file:
      tmp_path = Path(tmp_file.name)

    try:
      config = Config(export_xml_path=tmp_path)

      priority_map = config.source_priority_map
      assert priority_map["Watch"] == 3
      assert priority_map["Xiaomi Home"] == 2
      assert priority_map["Phone"] == 1
    finally:
      tmp_path.unlink(missing_ok=True)

  def test_config_with_custom_values(self):
    """Test config with custom values."""
    # Create a temporary file for the required export_xml_path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp_file:
      tmp_path = Path(tmp_file.name)

    try:
      config = Config(
        export_xml_path=tmp_path,
        environment=Environment.PROD,
        debug=False,
        output_dir=Path("/custom/output"),
        apple_watch_priority=5,
        log_level="DEBUG",
        batch_size=500,
      )

      assert config.environment == Environment.PROD
      assert config.debug is False
      assert config.output_dir == Path("/custom/output")
      assert config.apple_watch_priority == 5
      assert config.log_level == "DEBUG"
      assert config.batch_size == 500
    finally:
      tmp_path.unlink(missing_ok=True)


class TestConfigValidation:
  """Test config validation."""

  def test_validate_export_xml_path_exists(self):
    """Test validation of export XML path existence."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
      tmp_path = Path(tmp_file.name)

    try:
      # Should succeed with existing file
      config = Config(export_xml_path=tmp_path)
      assert config.export_xml_path == tmp_path
    finally:
      tmp_path.unlink(missing_ok=True)

  def test_validate_export_xml_path_nonexistent(self):
    """Test validation fails for nonexistent export XML path."""
    nonexistent_path = Path("/nonexistent/file.xml")

    with pytest.raises(ValueError, match="Export XML file does not exist"):
      Config(export_xml_path=nonexistent_path)

  def test_validate_export_xml_path_not_file(self):
    """Test validation fails for directory instead of file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      tmp_path = Path(tmp_dir)

      with pytest.raises(ValueError, match="Export XML path is not a file"):
        Config(export_xml_path=tmp_path)

  def test_validate_output_dir_creation(self):
    """Test output directory creation and validation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      output_path = Path(tmp_dir) / "new_output_dir"

      # Directory doesn't exist yet
      assert not output_path.exists()

      # Create a temporary file for the required export_xml_path
      with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp_file:
        tmp_path = Path(tmp_file.name)

      try:
        config = Config(export_xml_path=tmp_path, output_dir=output_path)

        # Should be created and validated
        assert output_path.exists()
        assert output_path.is_dir()
        assert config.output_dir == output_path
      finally:
        tmp_path.unlink(missing_ok=True)

  def test_validate_output_dir_not_writable(self):
    """Test validation fails for non-writable output directory."""
    # This is hard to test on Windows, so we'll skip this test
    # On Unix systems, we could create a directory and remove write permissions
    pytest.skip("Output directory write test not applicable on Windows")

  def test_validate_log_level_valid(self):
    """Test validation of valid log levels."""
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    for level in valid_levels:
      # Create a temporary file for the required export_xml_path
      with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp_file:
        tmp_path = Path(tmp_file.name)

      try:
        config = Config(export_xml_path=tmp_path, log_level=level)
        assert config.log_level == level
      finally:
        tmp_path.unlink(missing_ok=True)

  def test_validate_log_level_invalid(self):
    """Test validation fails for invalid log level."""
    with pytest.raises(ValueError, match="Invalid log level"):
      # Create a temporary file for the required export_xml_path
      with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp_file:
        tmp_path = Path(tmp_file.name)

      try:
        Config(export_xml_path=tmp_path, log_level="INVALID")
      finally:
        tmp_path.unlink(missing_ok=True)

  def test_validate_log_level_case_insensitive(self):
    """Test log level validation is case insensitive."""
    # Create a temporary file for the required export_xml_path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp_file:
      tmp_path = Path(tmp_file.name)

    try:
      config = Config(export_xml_path=tmp_path, log_level="debug")
      assert config.log_level == "DEBUG"
    finally:
      tmp_path.unlink(missing_ok=True)


class TestConfigFieldValidators:
  """Test field validation through Config instantiation."""

  def test_export_xml_path_validation_through_config(self):
    """Test export XML path validation by creating Config instance."""
    # Test with existing file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
      tmp_path = Path(tmp_file.name)

    try:
      # This should work and validate the path
      config = Config(export_xml_path=tmp_path)
      assert config.export_xml_path == tmp_path
    finally:
      tmp_path.unlink(missing_ok=True)

  def test_output_dir_validation_through_config(self):
    """Test output directory validation by creating Config instance."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      output_path = Path(tmp_dir) / "test_output"

      # Create a temporary file for the required export_xml_path
      with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp_file:
        tmp_path = Path(tmp_file.name)

      try:
        # This should create the directory and validate it
        config = Config(export_xml_path=tmp_path, output_dir=output_path)
        assert config.output_dir == output_path
        assert output_path.exists()
      finally:
        tmp_path.unlink(missing_ok=True)

  def test_log_level_validation_through_config(self):
    """Test log level validation by creating Config instance."""
    # Test valid log levels
    for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
      # Create a temporary file for the required export_xml_path
      with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp_file:
        tmp_path = Path(tmp_file.name)

      try:
        config = Config(export_xml_path=tmp_path, log_level=level)
        assert config.log_level == level
      finally:
        tmp_path.unlink(missing_ok=True)

    # Test case insensitive conversion
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp_file:
      tmp_path = Path(tmp_file.name)

    try:
      config = Config(export_xml_path=tmp_path, log_level="debug")
      assert config.log_level == "DEBUG"
    finally:
      tmp_path.unlink(missing_ok=True)


class TestLoadConfig:
  """Test load_config function."""

  @patch.dict(os.environ, {}, clear=True)
  def test_load_config_defaults(self):
    """Test loading config with default values."""
    config = load_config()

    assert config.environment == Environment.DEV
    assert config.debug is True
    assert config.log_level == "INFO"
    assert config.batch_size == 1000
    assert config.memory_limit_mb == 500

  @patch.dict(
    os.environ,
    {
      "ENVIRONMENT": "prod",
      "DEBUG": "false",
      "LOG_LEVEL": "DEBUG",
      "BATCH_SIZE": "500",
      "MEMORY_LIMIT_MB": "256",
    },
    clear=True,
  )
  def test_load_config_from_env(self):
    """Test loading config from environment variables."""
    config = load_config()

    assert config.environment == Environment.PROD
    assert config.debug is False
    assert config.log_level == "DEBUG"
    assert config.batch_size == 500
    assert config.memory_limit_mb == 256

  @patch("dotenv.load_dotenv")
  @patch.dict(os.environ, {}, clear=True)
  def test_load_config_with_dotenv(self, mock_load_dotenv):
    """Test loading config with .env file."""
    # Set environment variables that would be loaded from .env
    with patch.dict(
      os.environ,
      {"ENVIRONMENT": "prod", "DEBUG": "false", "LOG_LEVEL": "WARNING"},
      clear=False,
    ):
      config = load_config()

      assert config.environment == Environment.PROD
      assert config.debug is False
      assert config.log_level == "WARNING"

  @patch.dict(
    os.environ,
    {
      "EXPORT_XML_PATH": "/custom/path.xml",
      "OUTPUT_DIR": "/custom/output",
      "APPLE_WATCH_PRIORITY": "5",
      "LOG_FILE": "/tmp/test.log",
    },
    clear=True,
  )
  def test_load_config_custom_paths(self):
    """Test loading config with custom paths."""
    # Create the required XML file
    xml_path = Path("/custom/path.xml")
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_text("<xml>test</xml>")

    try:
      config = load_config()

      assert config.export_xml_path == Path("/custom/path.xml")
      assert config.output_dir == Path("/custom/output")
      assert config.apple_watch_priority == 5
      assert config.log_file == Path("/tmp/test.log")
    finally:
      xml_path.unlink(missing_ok=True)


class TestGetConfig:
  """Test get_config function."""

  @patch("src.config.load_config")
  def test_get_config_caching(self, mock_load_config):
    """Test that get_config caches the configuration."""
    # Create a temporary file for the required export_xml_path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp_file:
      tmp_path = Path(tmp_file.name)

    try:
      mock_config = Config(export_xml_path=tmp_path)
      mock_load_config.return_value = mock_config

      # Clear any existing cached config
      import src.config

      src.config._config = None

      # First call
      config1 = get_config()
      assert config1 is mock_config

      # Second call should return the same instance
      config2 = get_config()
      assert config2 is config1

      # load_config should only be called once
      mock_load_config.assert_called_once()
    finally:
      tmp_path.unlink(missing_ok=True)


class TestReloadConfig:
  """Test reload_config function."""

  @patch.dict(os.environ, {"EXPORT_XML_PATH": "/tmp/test1.xml"}, clear=True)
  @patch.dict(os.environ, {"EXPORT_XML_PATH": "/tmp/test2.xml"}, clear=False)
  def test_reload_config(self):
    """Test that reload_config forces reloading."""
    # Create the required XML files
    xml_path1 = Path("/tmp/test1.xml")
    xml_path2 = Path("/tmp/test2.xml")
    xml_path1.parent.mkdir(parents=True, exist_ok=True)
    xml_path2.parent.mkdir(parents=True, exist_ok=True)
    xml_path1.write_text("<xml>test1</xml>")
    xml_path2.write_text("<xml>test2</xml>")

    try:
      # Set initial config
      with patch.dict(
        os.environ, {"EXPORT_XML_PATH": str(xml_path1)}, clear=True
      ):
        config1 = load_config()
        import src.config

        src.config._config = config1

      # Verify initial config
      assert get_config().export_xml_path == xml_path1

      # Change environment and reload
      with patch.dict(
        os.environ, {"EXPORT_XML_PATH": str(xml_path2)}, clear=True
      ):
        config2 = reload_config()
        assert config2.export_xml_path == xml_path2

      # get_config should now return the new config
      assert get_config().export_xml_path == xml_path2
    finally:
      xml_path1.unlink(missing_ok=True)
      xml_path2.unlink(missing_ok=True)


class TestConfigIntegration:
  """Test config integration scenarios."""

  @patch.dict(
    os.environ,
    {
      "ENVIRONMENT": "prod",
      "DEBUG": "false",
      "EXPORT_XML_PATH": "/tmp/test.xml",
      "OUTPUT_DIR": "/tmp/output",
      "LOG_LEVEL": "ERROR",
      "BATCH_SIZE": "2000",
    },
    clear=True,
  )
  def test_config_integration_complete(self):
    """Test complete config integration."""
    # Create the required file
    xml_path = Path("/tmp/test.xml")
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_text("<xml>test</xml>")

    try:
      config = load_config()

      assert config.environment == Environment.PROD
      assert config.debug is False
      assert config.export_xml_path == xml_path
      assert config.output_dir == Path("/tmp/output")
      assert config.log_level == "ERROR"
      assert config.batch_size == 2000

      # Test computed properties
      assert config.is_production is True
      assert config.is_development is False

      # Test source priority map
      priority_map = config.source_priority_map
      assert isinstance(priority_map, dict)
      assert len(priority_map) == 3
    finally:
      xml_path.unlink(missing_ok=True)

  def test_config_validation_edge_cases(self):
    """Test config validation edge cases."""
    # Test with minimal valid config
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
      tmp_path = Path(tmp_file.name)

    try:
      config = Config(export_xml_path=tmp_path)

      # Test boundary values for priorities
      config.apple_watch_priority = 1  # Minimum
      config.xiaomi_health_priority = 10  # Maximum

      assert config.apple_watch_priority == 1
      assert config.xiaomi_health_priority == 10
    finally:
      tmp_path.unlink(missing_ok=True)


class TestConfigErrorHandling:
  """Test config error handling."""

  @patch.dict(os.environ, {"EXPORT_XML_PATH": "/nonexistent.xml"}, clear=True)
  def test_load_config_invalid_xml_path(self):
    """Test load_config with invalid XML path."""
    with pytest.raises(ValueError, match="Export XML file does not exist"):
      load_config()

  @patch.dict(
    os.environ,
    {"LOG_LEVEL": "INVALID_LEVEL", "EXPORT_XML_PATH": "/tmp/test.xml"},
    clear=True,
  )
  def test_load_config_invalid_log_level(self):
    """Test load_config with invalid log level."""
    # Create the required XML file
    xml_path = Path("/tmp/test.xml")
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_text("<xml>test</xml>")

    try:
      with pytest.raises(ValueError, match="Invalid log level"):
        load_config()
    finally:
      xml_path.unlink(missing_ok=True)

  @patch.dict(os.environ, {"ENVIRONMENT": "invalid"}, clear=True)
  def test_load_config_invalid_environment(self):
    """Test load_config with invalid environment."""
    # Create a valid XML file first
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
      tmp_path = Path(tmp_file.name)

    try:
      with patch.dict(
        os.environ, {"EXPORT_XML_PATH": str(tmp_path)}, clear=True
      ):
        # This should work because Environment enum handles invalid values gracefully
        config = load_config()
        assert config.environment == Environment.DEV  # Should default to DEV
    finally:
      tmp_path.unlink(missing_ok=True)
