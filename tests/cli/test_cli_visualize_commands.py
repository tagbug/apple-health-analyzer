"""Tests for visualize/report command flows."""

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from src.cli_visualize import report, visualize


def test_report_missing_file():
  """Report should exit with code 2 on missing file."""
  runner = CliRunner()
  result = runner.invoke(report, ["/nonexistent/file.xml"])

  assert result.exit_code == 2
  assert "XML file not found" in result.output


@patch("src.cli_visualize.ReportGenerator")
@patch("src.cli_visualize.HighlightsGenerator")
@patch("src.cli_visualize.SleepAnalyzer")
@patch("src.cli_visualize.HeartRateAnalyzer")
@patch("src.cli_visualize.StreamingXMLParser")
@patch("src.cli_visualize.get_config")
def test_report_success_markdown(
  mock_get_config,
  mock_parser,
  mock_hr_analyzer,
  mock_sleep_analyzer,
  mock_highlights,
  mock_report_generator,
  tmp_path,
):
  """Report should generate markdown output."""
  runner = CliRunner()
  xml_path = tmp_path / "data.xml"
  xml_path.write_text("<HealthData></HealthData>")

  mock_get_config.return_value = SimpleNamespace(output_dir=tmp_path)

  parser_instance = MagicMock()
  parser_instance.parse_records.return_value = []
  mock_parser.return_value = parser_instance

  mock_hr_analyzer.return_value.analyze_comprehensive.return_value = MagicMock()
  mock_sleep_analyzer.return_value.analyze_comprehensive.return_value = MagicMock()
  mock_highlights.return_value.generate_comprehensive_highlights.return_value = (
    MagicMock(insights=[], recommendations=[])
  )

  report_gen_instance = MagicMock()
  report_gen_instance.generate_markdown_report.return_value = tmp_path / "out.md"
  mock_report_generator.return_value = report_gen_instance

  result = runner.invoke(report, [str(xml_path), "--format", "markdown", "--no-charts"])

  assert result.exit_code == 0
  assert "Markdown report" in result.output


@patch("src.cli_visualize.StreamingXMLParser")
@patch("src.cli_visualize.get_config")
def test_visualize_no_data(mock_get_config, mock_parser, tmp_path):
  """Visualize should warn when no data is available."""
  runner = CliRunner()
  xml_path = tmp_path / "data.xml"
  xml_path.write_text("<HealthData></HealthData>")

  mock_get_config.return_value = SimpleNamespace(output_dir=tmp_path)
  parser_instance = MagicMock()
  parser_instance.parse_records.return_value = []
  mock_parser.return_value = parser_instance

  result = runner.invoke(visualize, [str(xml_path), "--charts", "all"])

  assert result.exit_code == 0
  assert "No chart files were generated" in result.output


@patch("src.visualization.data_converter.DataConverter")
@patch("src.visualization.charts.ChartGenerator")
@patch("src.cli_visualize.StreamingXMLParser")
@patch("src.cli_visualize.get_config")
def test_visualize_generates_index(
  mock_get_config,
  mock_parser,
  mock_chart_generator,
  mock_converter,
  tmp_path,
):
  """Visualize should create index file when charts are generated."""
  runner = CliRunner()
  xml_path = tmp_path / "data.xml"
  xml_path.write_text("<HealthData></HealthData>")

  output_dir = tmp_path / "charts"
  mock_get_config.return_value = SimpleNamespace(output_dir=tmp_path)

  record = MagicMock()
  record.type = "HKQuantityTypeIdentifierHeartRate"
  parser_instance = MagicMock()
  parser_instance.parse_records.return_value = [record]
  mock_parser.return_value = parser_instance

  mock_converter.heart_rate_to_df.return_value = __import__("pandas").DataFrame(
    {"timestamp": [datetime(2024, 1, 1)], "value": [70]}
  )
  mock_converter.sample_data_for_performance.return_value = (
    mock_converter.heart_rate_to_df.return_value
  )

  fig = MagicMock()
  chart_instance = MagicMock()
  chart_instance.plot_heart_rate_timeseries.return_value = fig
  mock_chart_generator.return_value = chart_instance

  result = runner.invoke(
    visualize,
    [
      str(xml_path),
      "--charts",
      "heart_rate_timeseries",
      "--output",
      str(output_dir),
    ],
  )

  assert result.exit_code == 0
  assert (output_dir / "index.md").exists()
