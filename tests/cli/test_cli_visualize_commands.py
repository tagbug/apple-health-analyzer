"""Tests for visualize/report command flows."""

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
from click.testing import CliRunner

from src.cli_visualize import report, visualize
from src.i18n import Translator, resolve_locale


class DummyFig:
  """Minimal figure stub that writes output files."""

  def write_html(self, path):
    Path(path).write_text("<html></html>", encoding="utf-8")

  def write_image(self, path, width=None, height=None):
    Path(path).write_text("image", encoding="utf-8")


class DummyChartGenerator:
  """Chart generator stub returning dummy figures."""

  def plot_heart_rate_timeseries(self, *args, **kwargs):
    return DummyFig()

  def plot_resting_hr_trend(self, *args, **kwargs):
    return DummyFig()

  def plot_hrv_analysis(self, *args, **kwargs):
    return DummyFig()

  def plot_heart_rate_heatmap(self, *args, **kwargs):
    return DummyFig()

  def plot_heart_rate_distribution(self, *args, **kwargs):
    return DummyFig()

  def plot_heart_rate_zones(self, *args, **kwargs):
    return DummyFig()

  def plot_sleep_timeline(self, *args, **kwargs):
    return DummyFig()

  def plot_sleep_quality_trend(self, *args, **kwargs):
    return DummyFig()

  def plot_sleep_stages_distribution(self, *args, **kwargs):
    return DummyFig()

  def plot_sleep_consistency(self, *args, **kwargs):
    return DummyFig()

  def plot_weekday_vs_weekend_sleep(self, *args, **kwargs):
    return DummyFig()


def _sleep_df():
  return pd.DataFrame(
    {
      "date": pd.date_range("2024-01-01", periods=3, freq="D"),
      "start_date": pd.date_range("2024-01-01", periods=3, freq="D"),
      "end_date": pd.date_range("2024-01-02", periods=3, freq="D"),
      "total_duration": [480, 450, 420],
      "efficiency": [0.8, 0.85, 0.9],
      "deep_sleep": [60, 70, 65],
      "rem_sleep": [90, 80, 85],
    }
  )


def test_report_missing_file():
  """Report should exit with code 2 on missing file."""
  runner = CliRunner()
  result = runner.invoke(report, ["/nonexistent/file.xml"])

  assert result.exit_code == 2
  translator = Translator(resolve_locale())
  assert (
    translator.t("cli.parse.file_not_found", path="/nonexistent/file.xml")
    in result.output
  )


@patch("src.cli_visualize.StreamingXMLParser")
@patch("src.cli_visualize.HeartRateAnalyzer")
@patch("src.cli_visualize.SleepAnalyzer")
@patch("src.cli_visualize.HighlightsGenerator")
@patch("src.cli_visualize.ReportGenerator")
@patch("src.cli_visualize.get_config")
def test_report_command_success(
  mock_get_config,
  mock_report_generator,
  mock_highlights_generator,
  mock_sleep_analyzer,
  mock_hr_analyzer,
  mock_xml_parser,
):
  """Test successful report generation."""
  mock_config = MagicMock()
  mock_config.output_dir = Path("/tmp/output")
  mock_get_config.return_value = mock_config

  mock_parser_instance = MagicMock()

  mock_hr_record = MagicMock()
  mock_hr_record.type = "HKQuantityTypeIdentifierHeartRate"

  mock_resting_hr_record = MagicMock()
  mock_resting_hr_record.type = "HKQuantityTypeIdentifierRestingHeartRate"

  mock_hrv_record = MagicMock()
  mock_hrv_record.type = "HKQuantityTypeIdentifierHeartRateVariabilitySDNN"

  mock_vo2_record = MagicMock()
  mock_vo2_record.type = "HKQuantityTypeIdentifierVO2Max"

  mock_sleep_record = MagicMock()
  mock_sleep_record.type = "HKCategoryTypeIdentifierSleepAnalysis"

  mock_parser_instance.parse_records.return_value = [
    mock_hr_record,
    mock_resting_hr_record,
    mock_hrv_record,
    mock_vo2_record,
    mock_sleep_record,
  ]

  mock_xml_parser.return_value = mock_parser_instance

  mock_hr_analyzer_instance = MagicMock()
  mock_hr_report = MagicMock()
  mock_hr_analyzer_instance.analyze_comprehensive.return_value = mock_hr_report
  mock_hr_analyzer.return_value = mock_hr_analyzer_instance

  mock_sleep_analyzer_instance = MagicMock()
  mock_sleep_report = MagicMock()
  mock_sleep_analyzer_instance.analyze_comprehensive.return_value = mock_sleep_report
  mock_sleep_analyzer.return_value = mock_sleep_analyzer_instance

  mock_highlights_instance = MagicMock()
  mock_highlights = MagicMock()
  mock_highlights_instance.generate_comprehensive_highlights.return_value = (
    mock_highlights
  )
  mock_highlights_generator.return_value = mock_highlights_instance

  mock_report_gen_instance = MagicMock()

  mock_html_file = MagicMock(spec=Path)
  mock_html_file.name = "test.html"
  mock_html_file.stat = MagicMock(return_value=MagicMock(st_size=1024 * 1024))

  mock_report_gen_instance.generate_html_report = MagicMock(return_value=mock_html_file)
  mock_report_generator.return_value = mock_report_gen_instance

  runner = CliRunner()
  with runner.isolated_filesystem():
    tmp_path = Path("test.xml")
    tmp_path.write_text("<xml></xml>")

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

    assert result.exit_code != 2, f"FileNotFoundError should not occur: {result.output}"
    if result.exit_code == 0:
      translator = Translator(resolve_locale())
      assert translator.t("cli.report.success") in result.output


@patch("src.cli_visualize.get_config")
def test_report_command_missing_file(mock_get_config):
  """Test report command with missing XML file."""
  mock_config = MagicMock()
  mock_config.output_dir = Path("/tmp/output")
  mock_get_config.return_value = mock_config

  runner = CliRunner()
  result = runner.invoke(report, ["/nonexistent/file.xml"])

  assert result.exit_code == 2
  translator = Translator(resolve_locale())
  assert (
    translator.t("cli.parse.file_not_found", path="/nonexistent/file.xml")
    in result.output
  )


@patch("src.cli_visualize.StreamingXMLParser")
@patch("src.cli_visualize.get_config")
def test_report_command_parsing_error(mock_get_config, mock_xml_parser):
  """Test report command with XML parsing error."""
  mock_config = MagicMock()
  mock_config.output_dir = Path("/tmp/output")
  mock_get_config.return_value = mock_config

  mock_parser_instance = MagicMock()
  mock_parser_instance.parse_records.side_effect = Exception("Parsing failed")
  mock_xml_parser.return_value = mock_parser_instance

  runner = CliRunner()
  with runner.isolated_filesystem():
    tmp_path = Path("test.xml")
    tmp_path.write_text("<xml></xml>")

    result = runner.invoke(report, [str(tmp_path)])

    assert result.exit_code == 1
    translator = Translator(resolve_locale())
    assert translator.t("cli.common.error") in result.output


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
  translator = Translator(resolve_locale())
  assert "Markdown" in translator.t("cli.report.md_saved")
  assert translator.t("cli.report.md_saved", path="")[:-2] in result.output


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
  translator = Translator(resolve_locale())
  assert translator.t("cli.visualize.no_files") in result.output


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

  mock_converter.heart_rate_to_df.return_value = pd.DataFrame(
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


def test_visualize_command_success(monkeypatch, tmp_path):
  """Test successful chart generation with real XML file."""
  mock_config = SimpleNamespace(output_dir=tmp_path)
  monkeypatch.setattr("src.cli_visualize.get_config", lambda: mock_config)

  runner = CliRunner()
  xml_path = tmp_path / "test.xml"
  xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<HealthData>
  <ExportDate>2023-01-01 12:00:00 +0000</ExportDate>
  <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Test Watch" startDate="2023-01-01 12:00:00 +0000" endDate="2023-01-01 12:00:00 +0000" value="70"/>
</HealthData>"""
  xml_path.write_text(xml_content)

  result = runner.invoke(
    visualize,
    [str(xml_path), "--charts", "heart_rate_timeseries", "--static"],
  )

  assert result.exit_code != 2, f"Unexpected FileNotFoundError: {result.output}"


def test_visualize_command_missing_file(monkeypatch, tmp_path):
  """Test visualize command with missing XML file."""
  mock_config = SimpleNamespace(output_dir=tmp_path)
  monkeypatch.setattr("src.cli_visualize.get_config", lambda: mock_config)

  runner = CliRunner()
  result = runner.invoke(visualize, ["/nonexistent/file.xml"])

  assert result.exit_code == 2
  translator = Translator(resolve_locale())
  assert (
    translator.t("cli.parse.file_not_found", path="/nonexistent/file.xml")
    in result.output
  )


@patch("src.cli_visualize.StreamingXMLParser")
@patch("src.cli_visualize.get_config")
def test_visualize_command_no_data(mock_get_config, mock_xml_parser):
  """Test visualize command with no data."""
  mock_config = MagicMock()
  mock_config.output_dir = Path("/tmp/output")
  mock_get_config.return_value = mock_config

  mock_parser_instance = MagicMock()
  mock_parser_instance.parse_records.return_value = []
  mock_xml_parser.return_value = mock_parser_instance

  runner = CliRunner()
  with runner.isolated_filesystem():
    tmp_path = Path("test.xml")
    tmp_path.write_text("<xml></xml>")

    result = runner.invoke(visualize, [str(tmp_path), "--charts", "all"])

    assert result.exit_code == 0
    translator = Translator(resolve_locale())
    assert translator.t("cli.visualize.no_files") in result.output


@patch("src.cli_visualize.get_config")
def test_visualize_command_chart_generation_error(mock_get_config):
  """Test visualize command with chart generation error."""
  mock_config = MagicMock()
  mock_config.output_dir = Path("/tmp/output")
  mock_get_config.return_value = mock_config

  runner = CliRunner()
  with runner.isolated_filesystem():
    tmp_path = Path("test.xml")
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<HealthData>
  <ExportDate>2023-01-01 12:00:00 +0000</ExportDate>
  <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Test Watch" startDate="2023-01-01 12:00:00 +0000" endDate="2023-01-01 12:00:00 +0000" value="70"/>
</HealthData>"""
    tmp_path.write_text(xml_content)

    result = runner.invoke(
      visualize,
      [str(tmp_path), "--charts", "heart_rate_timeseries", "--static"],
    )

    assert result.exit_code != 2, f"Unexpected FileNotFoundError: {result.output}"


@patch("src.cli_visualize.StreamingXMLParser")
@patch("src.cli_visualize.HeartRateAnalyzer")
@patch("src.cli_visualize.SleepAnalyzer")
@patch("src.cli_visualize.HighlightsGenerator")
@patch("src.cli_visualize.ReportGenerator")
@patch("src.cli_visualize.get_config")
def test_report_format_options(
  mock_get_config,
  mock_report_generator,
  mock_highlights_generator,
  mock_sleep_analyzer,
  mock_hr_analyzer,
  mock_xml_parser,
):
  """Test different report format options."""
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
  mock_hr_analyzer_instance.analyze_comprehensive.return_value = mock_hr_report
  mock_hr_analyzer.return_value = mock_hr_analyzer_instance

  mock_sleep_analyzer_instance = MagicMock()
  mock_sleep_report = MagicMock()
  mock_sleep_analyzer_instance.analyze_comprehensive.return_value = mock_sleep_report
  mock_sleep_analyzer.return_value = mock_sleep_analyzer_instance

  mock_highlights_instance = MagicMock()
  mock_highlights = MagicMock()
  mock_highlights_instance.generate_comprehensive_highlights.return_value = (
    mock_highlights
  )
  mock_highlights_generator.return_value = mock_highlights_instance

  mock_report_gen_instance = MagicMock()
  mock_report_gen_instance.generate_html_report = MagicMock(
    return_value=Path("/tmp/test.html")
  )
  mock_report_gen_instance.generate_markdown_report = MagicMock(
    return_value=Path("/tmp/test.md")
  )
  mock_report_generator.return_value = mock_report_gen_instance

  runner = CliRunner()
  with runner.isolated_filesystem():
    tmp_path = Path("test.xml")
    tmp_path.write_text("<xml></xml>")

    result = runner.invoke(report, [str(tmp_path), "--format", "both"])

    assert result.exit_code != 2, f"FileNotFoundError should not occur: {result.output}"
    if result.exit_code == 0:
      translator = Translator(resolve_locale())
      assert (
        translator.t("cli.report.html_saved", path=Path("/tmp/test.html"))
        in result.output
      )
      assert (
        translator.t("cli.report.md_saved", path=Path("/tmp/test.md")) in result.output
      )


def test_visualize_chart_selection():
  """Test chart selection options using isolated filesystem."""
  runner = CliRunner()

  with runner.isolated_filesystem():
    Path("test.xml").write_text("<xml></xml>")

    result = runner.invoke(
      visualize,
      ["test.xml", "--charts", "heart_rate_timeseries", "--static"],
    )

    assert result.exit_code != 2, f"Unexpected file not found error: {result.output}"
    if result.exit_code == 0:
      translator = Translator(resolve_locale())
      assert (
        f"{translator.t('cli.visualize.charts_to_generate')} "
        f"{translator.t('cli.visualize.chart_count', count=1)}" in result.output
      )


@patch("src.cli_visualize.StreamingXMLParser")
@patch("src.cli_visualize.HeartRateAnalyzer")
@patch("src.cli_visualize.SleepAnalyzer")
@patch("src.cli_visualize.HighlightsGenerator")
@patch("src.cli_visualize.ReportGenerator")
@patch("src.cli_visualize.get_config")
def test_custom_output_directory(
  mock_get_config,
  mock_report_generator,
  mock_highlights_generator,
  mock_sleep_analyzer,
  mock_hr_analyzer,
  mock_xml_parser,
):
  """Test custom output directory."""
  with CliRunner().isolated_filesystem():
    custom_output = Path("custom_reports")

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
    mock_hr_analyzer_instance.analyze_comprehensive.return_value = mock_hr_report
    mock_hr_analyzer.return_value = mock_hr_analyzer_instance

    mock_sleep_analyzer_instance = MagicMock()
    mock_sleep_report = MagicMock()
    mock_sleep_analyzer_instance.analyze_comprehensive.return_value = mock_sleep_report
    mock_sleep_analyzer.return_value = mock_sleep_analyzer_instance

    mock_highlights_instance = MagicMock()
    mock_highlights = MagicMock()
    mock_highlights_instance.generate_comprehensive_highlights.return_value = (
      mock_highlights
    )
    mock_highlights_generator.return_value = mock_highlights_instance

    mock_report_gen_instance = MagicMock()
    mock_report_gen_instance.generate_html_report = MagicMock(
      return_value=custom_output / "test.html"
    )
    mock_report_generator.return_value = mock_report_gen_instance

    runner = CliRunner()
    tmp_path = Path("test.xml")
    tmp_path.write_text("<xml></xml>")

    result = runner.invoke(report, [str(tmp_path), "--output", str(custom_output)])

    assert result.exit_code != 2, f"FileNotFoundError should not occur: {result.output}"
    if result.exit_code == 0:
      assert str(custom_output) in result.output


def test_visualize_interactive_branches(monkeypatch, tmp_path):
  """Interactive visualize should generate index for multiple chart types."""
  xml_path = tmp_path / "data.xml"
  xml_path.write_text("<HealthData></HealthData>")

  output_dir = tmp_path / "charts"
  monkeypatch.setattr(
    "src.cli_visualize.get_config",
    lambda: SimpleNamespace(output_dir=tmp_path),
  )

  records = [
    SimpleNamespace(type="HKQuantityTypeIdentifierHeartRate"),
    SimpleNamespace(type="HKQuantityTypeIdentifierRestingHeartRate"),
    SimpleNamespace(type="HKQuantityTypeIdentifierHeartRateVariabilitySDNN"),
    SimpleNamespace(type="HKCategoryTypeIdentifierSleepAnalysis"),
  ]
  parser_instance = SimpleNamespace(parse_records=lambda **kwargs: records)
  monkeypatch.setattr(
    "src.cli_visualize.StreamingXMLParser", lambda *_: parser_instance
  )

  monkeypatch.setattr(
    "src.cli_visualize.SleepAnalyzer",
    lambda: SimpleNamespace(parse_sleep_sessions=lambda *_: [object()]),
  )

  monkeypatch.setattr("src.cli_visualize.ChartGenerator", DummyChartGenerator)

  monkeypatch.setattr(
    "src.visualization.data_converter.DataConverter.heart_rate_to_df",
    lambda *_: pd.DataFrame({"timestamp": [datetime(2024, 1, 1)], "value": [70]}),
  )
  monkeypatch.setattr(
    "src.visualization.data_converter.DataConverter.resting_hr_to_df",
    lambda *_: pd.DataFrame({"start_date": [datetime(2024, 1, 1)], "value": [60]}),
  )
  monkeypatch.setattr(
    "src.visualization.data_converter.DataConverter.hrv_to_df",
    lambda *_: pd.DataFrame({"start_date": [datetime(2024, 1, 1)], "value": [30]}),
  )
  monkeypatch.setattr(
    "src.visualization.data_converter.DataConverter.sample_data_for_performance",
    lambda df, *_: df,
  )
  monkeypatch.setattr(
    "src.visualization.data_converter.DataConverter.aggregate_heart_rate_by_day",
    lambda *_: pd.DataFrame({"mean_hr": [70], "date": [datetime(2024, 1, 1)]}),
  )
  monkeypatch.setattr(
    "src.visualization.data_converter.DataConverter.sleep_sessions_to_df",
    lambda *_: _sleep_df(),
  )
  monkeypatch.setattr(
    "src.visualization.data_converter.DataConverter.aggregate_sleep_by_day",
    lambda *_: _sleep_df(),
  )
  monkeypatch.setattr(
    "src.visualization.data_converter.DataConverter.prepare_sleep_stages_distribution",
    lambda *_: pd.DataFrame({"stage": ["Deep"], "duration": [60]}),
  )
  monkeypatch.setattr(
    "src.visualization.data_converter.DataConverter.prepare_heart_rate_zones",
    lambda *_: pd.DataFrame({"zone": ["Z1"], "minutes": [10]}),
  )

  runner = CliRunner()
  result = runner.invoke(
    visualize,
    [
      str(xml_path),
      "--output",
      str(output_dir),
      "--interactive",
      "--age",
      "30",
      "--charts",
      "heart_rate_timeseries",
      "--charts",
      "resting_hr_trend",
      "--charts",
      "hrv_analysis",
      "--charts",
      "heart_rate_heatmap",
      "--charts",
      "heart_rate_distribution",
      "--charts",
      "heart_rate_zones",
      "--charts",
      "sleep_timeline",
      "--charts",
      "sleep_quality_trend",
      "--charts",
      "sleep_stages_distribution",
      "--charts",
      "sleep_consistency",
      "--charts",
      "weekday_vs_weekend_sleep",
    ],
  )

  assert result.exit_code == 0
  assert (output_dir / "index.md").exists()


def test_visualize_static_fallback(monkeypatch, tmp_path):
  """Static mode should use prewritten files when figs are None."""
  xml_path = tmp_path / "data.xml"
  xml_path.write_text("<HealthData></HealthData>")

  output_dir = tmp_path / "charts"
  output_dir.mkdir()
  fallback_path = output_dir / "heart_rate_timeseries.png"
  fallback_path.write_text("image", encoding="utf-8")

  monkeypatch.setattr(
    "src.cli_visualize.get_config",
    lambda: SimpleNamespace(output_dir=tmp_path),
  )

  records = [
    SimpleNamespace(type="HKQuantityTypeIdentifierHeartRate"),
    SimpleNamespace(type="HKQuantityTypeIdentifierRestingHeartRate"),
    SimpleNamespace(type="HKQuantityTypeIdentifierHeartRateVariabilitySDNN"),
  ]
  parser_instance = SimpleNamespace(parse_records=lambda **kwargs: records)
  monkeypatch.setattr(
    "src.cli_visualize.StreamingXMLParser", lambda *_: parser_instance
  )

  class NoFigChartGenerator(DummyChartGenerator):
    def plot_heart_rate_timeseries(self, *args, **kwargs):
      return DummyFig()

  monkeypatch.setattr("src.cli_visualize.ChartGenerator", NoFigChartGenerator)
  monkeypatch.setattr(
    "src.visualization.data_converter.DataConverter.heart_rate_to_df",
    lambda *_: pd.DataFrame({"timestamp": [datetime(2024, 1, 1)], "value": [70]}),
  )
  monkeypatch.setattr(
    "src.visualization.data_converter.DataConverter.sample_data_for_performance",
    lambda df, *_: df,
  )

  monkeypatch.setattr(
    "src.visualization.data_converter.DataConverter.resting_hr_to_df",
    lambda *_: pd.DataFrame({"start_date": [datetime(2024, 1, 1)], "value": [60]}),
  )
  monkeypatch.setattr(
    "src.visualization.data_converter.DataConverter.hrv_to_df",
    lambda *_: pd.DataFrame({"start_date": [datetime(2024, 1, 1)], "value": [30]}),
  )

  runner = CliRunner()
  result = runner.invoke(
    visualize,
    [
      str(xml_path),
      "--output",
      str(output_dir),
      "--static",
      "--charts",
      "heart_rate_timeseries",
    ],
  )

  assert result.exit_code == 0
  assert (output_dir / "index.md").exists()


def test_visualize_parser_failure(monkeypatch, tmp_path):
  """Parser failures should exit with error."""
  xml_path = tmp_path / "data.xml"
  xml_path.write_text("<HealthData></HealthData>")

  monkeypatch.setattr(
    "src.cli_visualize.get_config",
    lambda: SimpleNamespace(output_dir=tmp_path),
  )
  monkeypatch.setattr(
    "src.cli_visualize.StreamingXMLParser",
    lambda *_: SimpleNamespace(
      parse_records=lambda **kwargs: (_ for _ in ()).throw(Exception("boom"))
    ),
  )

  runner = CliRunner()
  result = runner.invoke(visualize, [str(xml_path), "--charts", "all"])

  assert result.exit_code == 1
  translator = Translator(resolve_locale())
  assert f"{translator.t('cli.common.error')}: boom" in result.output


def test_visualize_chart_generation_error(monkeypatch, tmp_path):
  """Chart generation errors should be reported and continue."""
  xml_path = tmp_path / "data.xml"
  xml_path.write_text("<HealthData></HealthData>")

  monkeypatch.setattr(
    "src.cli_visualize.get_config",
    lambda: SimpleNamespace(output_dir=tmp_path),
  )

  record = SimpleNamespace(type="HKQuantityTypeIdentifierHeartRate")
  parser_instance = SimpleNamespace(parse_records=lambda **kwargs: [record])
  monkeypatch.setattr(
    "src.cli_visualize.StreamingXMLParser", lambda *_: parser_instance
  )

  class ErrorChartGenerator:
    def plot_heart_rate_timeseries(self, *args, **kwargs):
      raise Exception("chart fail")

  monkeypatch.setattr("src.cli_visualize.ChartGenerator", ErrorChartGenerator)
  monkeypatch.setattr(
    "src.visualization.data_converter.DataConverter.heart_rate_to_df",
    lambda *_: pd.DataFrame({"timestamp": [1], "value": [1]}),
  )
  monkeypatch.setattr(
    "src.visualization.data_converter.DataConverter.sample_data_for_performance",
    lambda df, *_: df,
  )

  runner = CliRunner()
  result = runner.invoke(
    visualize,
    [str(xml_path), "--charts", "heart_rate_timeseries", "--static"],
  )

  assert result.exit_code == 0
  translator = Translator(resolve_locale())
  assert (
    translator.t(
      "cli.visualize.failed_chart", chart="heart_rate_timeseries", error="chart fail"
    )
    in result.output
  )
