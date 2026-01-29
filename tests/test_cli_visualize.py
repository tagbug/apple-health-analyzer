"""Tests for cli_visualize module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from src.cli_visualize import report, visualize


class TestReportCommand:
  """Test report command functionality."""

  @patch("src.cli_visualize.StreamingXMLParser")
  @patch("src.cli_visualize.HeartRateAnalyzer")
  @patch("src.cli_visualize.SleepAnalyzer")
  @patch("src.cli_visualize.HighlightsGenerator")
  @patch("src.cli_visualize.ReportGenerator")
  @patch("src.cli_visualize.get_config")
  def test_report_command_success(
    self,
    mock_get_config,
    mock_report_generator,
    mock_highlights_generator,
    mock_sleep_analyzer,
    mock_hr_analyzer,
    mock_xml_parser,
  ):
    """Test successful report generation."""
    # Setup mocks
    mock_config = MagicMock()
    mock_config.output_dir = Path("/tmp/output")
    mock_get_config.return_value = mock_config

    # Mock XML parser
    mock_parser_instance = MagicMock()
    mock_hr_record = MagicMock()
    mock_hr_record.type = "HKQuantityTypeIdentifierHeartRate"
    mock_parser_instance.parse_records.return_value = [mock_hr_record]
    mock_xml_parser.return_value = mock_parser_instance

    # Mock analyzers
    mock_hr_analyzer_instance = MagicMock()
    mock_hr_report = MagicMock()
    mock_hr_analyzer_instance.analyze_comprehensive.return_value = (
      mock_hr_report
    )
    mock_hr_analyzer.return_value = mock_hr_analyzer_instance

    mock_sleep_analyzer_instance = MagicMock()
    mock_sleep_report = MagicMock()
    mock_sleep_analyzer_instance.analyze_comprehensive.return_value = (
      mock_sleep_report
    )
    mock_sleep_analyzer.return_value = mock_sleep_analyzer_instance

    # Mock highlights generator
    mock_highlights_instance = MagicMock()
    mock_highlights = MagicMock()
    mock_highlights_instance.generate_comprehensive_highlights.return_value = (
      mock_highlights
    )
    mock_highlights_generator.return_value = mock_highlights_instance

    # Mock report generator
    mock_report_gen_instance = MagicMock()
    mock_report_gen_instance.generate_html_report.return_value = Path(
      "/tmp/test.html"
    )
    mock_report_generator.return_value = mock_report_gen_instance

    runner = CliRunner()
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp_file:
      tmp_path = Path(tmp_file.name)

      try:
        result = runner.invoke(
          report,
          [
            str(tmp_path),
            "--age",
            "30",
            "--gender",
            "male",
            "--format",
            "html",
          ],
        )

        assert result.exit_code == 0
        assert "Report generation successful" in result.output
      finally:
        tmp_path.unlink(missing_ok=True)

  @patch("src.cli_visualize.get_config")
  def test_report_command_missing_file(self, mock_get_config):
    """Test report command with missing XML file."""
    mock_config = MagicMock()
    mock_config.output_dir = Path("/tmp/output")
    mock_get_config.return_value = mock_config

    runner = CliRunner()
    result = runner.invoke(report, ["/nonexistent/file.xml"])

    assert result.exit_code == 1
    assert "Error" in result.output

  @patch("src.cli_visualize.StreamingXMLParser")
  @patch("src.cli_visualize.get_config")
  def test_report_command_parsing_error(self, mock_get_config, mock_xml_parser):
    """Test report command with XML parsing error."""
    mock_config = MagicMock()
    mock_config.output_dir = Path("/tmp/output")
    mock_get_config.return_value = mock_config

    # Mock parser to raise exception
    mock_parser_instance = MagicMock()
    mock_parser_instance.parse_records.side_effect = Exception("Parsing failed")
    mock_xml_parser.return_value = mock_parser_instance

    runner = CliRunner()
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp_file:
      tmp_path = Path(tmp_file.name)

      try:
        result = runner.invoke(report, [str(tmp_path)])

        assert result.exit_code == 1
        assert "Error" in result.output
      finally:
        tmp_path.unlink(missing_ok=True)


class TestVisualizeCommand:
  """Test visualize command functionality."""

  @patch("src.cli_visualize.StreamingXMLParser")
  @patch("src.cli_visualize.ChartGenerator")
  @patch("src.cli_visualize.SleepAnalyzer")
  @patch("src.cli_visualize.DataConverter")
  @patch("src.cli_visualize.get_config")
  def test_visualize_command_success(
    self,
    mock_get_config,
    mock_data_converter,
    mock_sleep_analyzer,
    mock_chart_generator,
    mock_xml_parser,
  ):
    """Test successful chart generation."""
    # Setup mocks
    mock_config = MagicMock()
    mock_config.output_dir = Path("/tmp/output")
    mock_get_config.return_value = mock_config

    # Mock XML parser
    mock_parser_instance = MagicMock()
    mock_hr_record = MagicMock()
    mock_hr_record.type = "HKQuantityTypeIdentifierHeartRate"
    mock_parser_instance.parse_records.return_value = [mock_hr_record]
    mock_xml_parser.return_value = mock_parser_instance

    # Mock data converter
    mock_data_converter.heart_rate_to_df.return_value = MagicMock()
    mock_data_converter.heart_rate_to_df.return_value.empty = False

    # Mock chart generator
    mock_chart_instance = MagicMock()
    mock_chart_instance.plot_heart_rate_timeseries.return_value = MagicMock()
    mock_chart_generator.return_value = mock_chart_instance

    runner = CliRunner()
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp_file:
      tmp_path = Path(tmp_file.name)

      try:
        result = runner.invoke(
          visualize,
          [str(tmp_path), "--charts", "heart_rate_timeseries", "--static"],
        )

        assert result.exit_code == 0
        assert "Chart generation completed" in result.output
      finally:
        tmp_path.unlink(missing_ok=True)

  @patch("src.cli_visualize.get_config")
  def test_visualize_command_missing_file(self, mock_get_config):
    """Test visualize command with missing XML file."""
    mock_config = MagicMock()
    mock_config.output_dir = Path("/tmp/output")
    mock_get_config.return_value = mock_config

    runner = CliRunner()
    result = runner.invoke(visualize, ["/nonexistent/file.xml"])

    assert result.exit_code == 1
    assert "Error" in result.output

  @patch("src.cli_visualize.StreamingXMLParser")
  @patch("src.cli_visualize.get_config")
  def test_visualize_command_no_data(self, mock_get_config, mock_xml_parser):
    """Test visualize command with no data."""
    mock_config = MagicMock()
    mock_config.output_dir = Path("/tmp/output")
    mock_get_config.return_value = mock_config

    # Mock parser with no records
    mock_parser_instance = MagicMock()
    mock_parser_instance.parse_records.return_value = []
    mock_xml_parser.return_value = mock_parser_instance

    runner = CliRunner()
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp_file:
      tmp_path = Path(tmp_file.name)

      try:
        result = runner.invoke(visualize, [str(tmp_path), "--charts", "all"])

        assert result.exit_code == 0
        assert "No chart files were generated" in result.output
      finally:
        tmp_path.unlink(missing_ok=True)

  @patch("src.cli_visualize.StreamingXMLParser")
  @patch("src.cli_visualize.ChartGenerator")
  @patch("src.cli_visualize.DataConverter")
  @patch("src.cli_visualize.get_config")
  def test_visualize_command_chart_generation_error(
    self,
    mock_get_config,
    mock_data_converter,
    mock_chart_generator,
    mock_xml_parser,
  ):
    """Test visualize command with chart generation error."""
    # Setup mocks
    mock_config = MagicMock()
    mock_config.output_dir = Path("/tmp/output")
    mock_get_config.return_value = mock_config

    # Mock XML parser
    mock_parser_instance = MagicMock()
    mock_hr_record = MagicMock()
    mock_hr_record.type = "HKQuantityTypeIdentifierHeartRate"
    mock_parser_instance.parse_records.return_value = [mock_hr_record]
    mock_xml_parser.return_value = mock_parser_instance

    # Mock data converter
    mock_data_converter.heart_rate_to_df.return_value = MagicMock()
    mock_data_converter.heart_rate_to_df.return_value.empty = False

    # Mock chart generator to raise exception
    mock_chart_instance = MagicMock()
    mock_chart_instance.plot_heart_rate_timeseries.side_effect = Exception(
      "Chart generation failed"
    )
    mock_chart_generator.return_value = mock_chart_instance

    runner = CliRunner()
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp_file:
      tmp_path = Path(tmp_file.name)

      try:
        result = runner.invoke(
          visualize,
          [str(tmp_path), "--charts", "heart_rate_timeseries", "--static"],
        )

        assert result.exit_code == 1
        assert "Error" in result.output
      finally:
        tmp_path.unlink(missing_ok=True)


class TestCommandOptions:
  """Test command line options."""

  @patch("src.cli_visualize.StreamingXMLParser")
  @patch("src.cli_visualize.HeartRateAnalyzer")
  @patch("src.cli_visualize.SleepAnalyzer")
  @patch("src.cli_visualize.HighlightsGenerator")
  @patch("src.cli_visualize.ReportGenerator")
  @patch("src.cli_visualize.get_config")
  def test_report_format_options(
    self,
    mock_get_config,
    mock_report_generator,
    mock_highlights_generator,
    mock_sleep_analyzer,
    mock_hr_analyzer,
    mock_xml_parser,
  ):
    """Test different report format options."""
    # Setup mocks similar to success test
    mock_config = MagicMock()
    mock_config.output_dir = Path("/tmp/output")
    mock_get_config.return_value = mock_config

    mock_parser_instance = MagicMock()
    mock_hr_record = MagicMock()
    mock_hr_record.type = "HKQuantityTypeIdentifierHeartRate"
    mock_parser_instance.parse_records.return_value = [mock_hr_record]
    mock_xml_parser.return_value = mock_parser_instance

    mock_hr_analyzer_instance = MagicMock()
    mock_hr_report = MagicMock()
    mock_hr_analyzer_instance.analyze_comprehensive.return_value = (
      mock_hr_report
    )
    mock_hr_analyzer.return_value = mock_hr_analyzer_instance

    mock_sleep_analyzer_instance = MagicMock()
    mock_sleep_report = MagicMock()
    mock_sleep_analyzer_instance.analyze_comprehensive.return_value = (
      mock_sleep_report
    )
    mock_sleep_analyzer.return_value = mock_sleep_analyzer_instance

    mock_highlights_instance = MagicMock()
    mock_highlights = MagicMock()
    mock_highlights_instance.generate_comprehensive_highlights.return_value = (
      mock_highlights
    )
    mock_highlights_generator.return_value = mock_highlights_instance

    mock_report_gen_instance = MagicMock()
    mock_report_gen_instance.generate_html_report.return_value = Path(
      "/tmp/test.html"
    )
    mock_report_gen_instance.generate_markdown_report.return_value = Path(
      "/tmp/test.md"
    )
    mock_report_generator.return_value = mock_report_gen_instance

    runner = CliRunner()
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp_file:
      tmp_path = Path(tmp_file.name)

      try:
        # Test both format
        result = runner.invoke(report, [str(tmp_path), "--format", "both"])

        assert result.exit_code == 0
        assert "HTML report" in result.output
        assert "Markdown report" in result.output
      finally:
        tmp_path.unlink(missing_ok=True)

  @patch("src.cli_visualize.StreamingXMLParser")
  @patch("src.cli_visualize.ChartGenerator")
  @patch("src.cli_visualize.DataConverter")
  @patch("src.cli_visualize.get_config")
  def test_visualize_chart_selection(
    self,
    mock_get_config,
    mock_data_converter,
    mock_chart_generator,
    mock_xml_parser,
  ):
    """Test chart selection options."""
    # Setup mocks
    mock_config = MagicMock()
    mock_config.output_dir = Path("/tmp/output")
    mock_get_config.return_value = mock_config

    mock_parser_instance = MagicMock()
    mock_hr_record = MagicMock()
    mock_hr_record.type = "HKQuantityTypeIdentifierHeartRate"
    mock_parser_instance.parse_records.return_value = [mock_hr_record]
    mock_xml_parser.return_value = mock_parser_instance

    mock_data_converter.heart_rate_to_df.return_value = MagicMock()
    mock_data_converter.heart_rate_to_df.return_value.empty = False

    mock_chart_instance = MagicMock()
    mock_chart_instance.plot_heart_rate_timeseries.return_value = MagicMock()
    mock_chart_generator.return_value = mock_chart_instance

    runner = CliRunner()
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp_file:
      tmp_path = Path(tmp_file.name)

      try:
        # Test specific chart selection
        result = runner.invoke(
          visualize,
          [str(tmp_path), "--charts", "heart_rate_timeseries", "--static"],
        )

        assert result.exit_code == 0
        assert "Charts to generate: 1 charts" in result.output
      finally:
        tmp_path.unlink(missing_ok=True)


class TestOutputDirectoryHandling:
  """Test output directory handling."""

  @patch("src.cli_visualize.StreamingXMLParser")
  @patch("src.cli_visualize.HeartRateAnalyzer")
  @patch("src.cli_visualize.SleepAnalyzer")
  @patch("src.cli_visualize.HighlightsGenerator")
  @patch("src.cli_visualize.ReportGenerator")
  @patch("src.cli_visualize.get_config")
  def test_custom_output_directory(
    self,
    mock_get_config,
    mock_report_generator,
    mock_highlights_generator,
    mock_sleep_analyzer,
    mock_hr_analyzer,
    mock_xml_parser,
  ):
    """Test custom output directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
      custom_output = Path(temp_dir) / "custom_reports"

      # Setup mocks
      mock_config = MagicMock()
      mock_config.output_dir = Path("/tmp/default")
      mock_get_config.return_value = mock_config

      mock_parser_instance = MagicMock()
      mock_hr_record = MagicMock()
      mock_hr_record.type = "HKQuantityTypeIdentifierHeartRate"
      mock_parser_instance.parse_records.return_value = [mock_hr_record]
      mock_xml_parser.return_value = mock_parser_instance

      mock_hr_analyzer_instance = MagicMock()
      mock_hr_report = MagicMock()
      mock_hr_analyzer_instance.analyze_comprehensive.return_value = (
        mock_hr_report
      )
      mock_hr_analyzer.return_value = mock_hr_analyzer_instance

      mock_sleep_analyzer_instance = MagicMock()
      mock_sleep_report = MagicMock()
      mock_sleep_analyzer_instance.analyze_comprehensive.return_value = (
        mock_sleep_report
      )
      mock_sleep_analyzer.return_value = mock_sleep_analyzer_instance

      mock_highlights_instance = MagicMock()
      mock_highlights = MagicMock()
      mock_highlights_instance.generate_comprehensive_highlights.return_value = mock_highlights
      mock_highlights_generator.return_value = mock_highlights_instance

      mock_report_gen_instance = MagicMock()
      mock_report_gen_instance.generate_html_report.return_value = (
        custom_output / "test.html"
      )
      mock_report_generator.return_value = mock_report_gen_instance

      runner = CliRunner()
      with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

        try:
          result = runner.invoke(
            report, [str(tmp_path), "--output", str(custom_output)]
          )

          assert result.exit_code == 0
          assert str(custom_output) in result.output
        finally:
          tmp_path.unlink(missing_ok=True)
