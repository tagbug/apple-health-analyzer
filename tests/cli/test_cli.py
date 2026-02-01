"""Tests for CLI interface."""

from click.testing import CliRunner

from src.cli import cli
from src.i18n import Translator, resolve_locale


class TestCLI:
  """Test CLI interface functionality."""

  def setup_method(self):
    """Set up test fixtures."""
    self.runner = CliRunner()

  def test_cli_main_command(self):
    """Test main CLI command shows help."""
    result = self.runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    translator = Translator(resolve_locale())
    expected = translator.t("cli.help.root")
    assert expected.split("\n\n", maxsplit=1)[0] in result.output
    expected_body = expected.split("\n\n", maxsplit=1)[1]
    assert " ".join(expected_body.split()) in " ".join(result.output.split())

  def test_cli_verbose_flag(self):
    """Test verbose flag with help command."""
    result = self.runner.invoke(cli, ["--verbose", "--help"])
    assert result.exit_code == 0
