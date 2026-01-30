"""Tests for CLI interface."""

from click.testing import CliRunner

from src.cli import cli


class TestCLI:
  """Test CLI interface functionality."""

  def setup_method(self):
    """Set up test fixtures."""
    self.runner = CliRunner()

  def test_cli_main_command(self):
    """Test main CLI command shows help."""
    result = self.runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Apple Health Data Analyzer" in result.output

  def test_cli_verbose_flag(self):
    """Test verbose flag with help command."""
    result = self.runner.invoke(cli, ["--verbose", "--help"])
    assert result.exit_code == 0
