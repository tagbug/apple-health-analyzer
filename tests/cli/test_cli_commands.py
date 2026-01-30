"""Tests for CLI command flows."""

"""End-to-end CLI command tests."""

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from src.cli import cli


def _sample_stats():
  return {
    "total_records": 2,
    "processed_records": 2,
    "skipped_records": 0,
    "invalid_records": 0,
    "success_rate": 1.0,
    "date_range": {"start": "2024-01-01", "end": "2024-01-02"},
    "record_types": {"HKQuantityTypeIdentifierHeartRate": 2},
    "sources": {"Watch": 2},
  }


def test_parse_empty_file(tmp_path, monkeypatch):
  """Empty files should exit with error."""
  runner = CliRunner()
  xml_path = tmp_path / "empty.xml"
  xml_path.write_text("")

  monkeypatch.setattr("src.cli.UnifiedProgress", MagicMock())

  result = runner.invoke(cli, ["parse", str(xml_path)])

  assert result.exit_code == 1
  assert "File is empty" in result.output


@patch("src.cli._save_parsed_data")
@patch("src.cli.get_export_file_info")
@patch("src.cli.StreamingXMLParser")
def test_parse_success_with_preview(
  mock_parser,
  mock_file_info,
  mock_save,
  tmp_path,
  monkeypatch,
):
  """Parse should preview and save when output is provided."""
  runner = CliRunner()
  xml_path = tmp_path / "data.xml"
  xml_path.write_text("<HealthData></HealthData>")

  mock_file_info.return_value = {
    "file_path": xml_path,
    "file_size_mb": 0.1,
    "estimated_record_count": 2,
    "last_modified": datetime(2024, 1, 1),
  }

  record = SimpleNamespace(
    type="HKQuantityTypeIdentifierHeartRate",
    source_name="Watch",
    start_date=datetime(2024, 1, 1, 8, 0, 0),
    value=70.0,
  )

  parser_instance = MagicMock()
  parser_instance.parse_records.return_value = [record]
  parser_instance.get_statistics.return_value = _sample_stats()
  mock_parser.return_value = parser_instance

  output_dir = tmp_path / "out"

  progress = MagicMock(
    __enter__=lambda s: s, __exit__=lambda *a: None, update=lambda *a, **k: None
  )
  monkeypatch.setattr("src.cli.UnifiedProgress", MagicMock(return_value=progress))

  result = runner.invoke(
    cli, ["parse", str(xml_path), "--preview", "--output", str(output_dir)]
  )

  assert result.exit_code == 0
  mock_save.assert_called_once()


@patch("src.cli.StreamingXMLParser")
def test_parse_no_records(mock_parser, tmp_path, monkeypatch):
  """Empty parsing results should exit with warning."""
  runner = CliRunner()
  xml_path = tmp_path / "data.xml"
  xml_path.write_text("<HealthData></HealthData>")

  parser_instance = MagicMock()
  parser_instance.parse_records.return_value = []
  parser_instance.get_statistics.return_value = _sample_stats()
  mock_parser.return_value = parser_instance

  progress = MagicMock(
    __enter__=lambda s: s, __exit__=lambda *a: None, update=lambda *a, **k: None
  )
  monkeypatch.setattr("src.cli.UnifiedProgress", MagicMock(return_value=progress))

  result = runner.invoke(cli, ["parse", str(xml_path)])

  assert result.exit_code == 1
  assert "No records were parsed" in result.output


@patch("src.cli.StreamingXMLParser", side_effect=Exception("boom"))
def test_parse_parser_init_error(mock_parser, tmp_path, monkeypatch):
  """Parser initialization errors should exit with code 1."""
  runner = CliRunner()
  xml_path = tmp_path / "data.xml"
  xml_path.write_text("<HealthData></HealthData>")

  monkeypatch.setattr("src.cli.UnifiedProgress", MagicMock())

  result = runner.invoke(cli, ["parse", str(xml_path)])

  assert result.exit_code == 1
  assert "Failed to initialize parser" in result.output


@patch("src.cli.get_export_file_info")
def test_info_success(mock_file_info, tmp_path):
  """Info should display file details and sample types."""
  runner = CliRunner()
  xml_path = tmp_path / "data.xml"
  xml_path.write_text(
    """<?xml version="1.0" encoding="UTF-8"?>
<HealthData>
  <ExportDate>2024-01-01 12:00:00 +0000</ExportDate>
  <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Watch" startDate="2024-01-01 12:00:00 +0000" endDate="2024-01-01 12:00:00 +0000" value="70"/>
</HealthData>"""
  )

  mock_file_info.return_value = {
    "file_path": xml_path,
    "file_size_mb": 0.1,
    "estimated_record_count": 1,
    "last_modified": datetime(2024, 1, 1),
  }

  result = runner.invoke(cli, ["info", str(xml_path)])

  assert result.exit_code == 0
  assert "File Information" in result.output


@patch("src.cli.get_export_file_info", return_value=None)
def test_info_missing_file_info(mock_file_info, tmp_path):
  """Info should return when file info cannot be read."""
  runner = CliRunner()
  xml_path = tmp_path / "data.xml"
  xml_path.write_text("<HealthData></HealthData>")

  result = runner.invoke(cli, ["info", str(xml_path)])

  assert result.exit_code == 0
  assert "Failed to analyze file" in result.output


@patch("src.cli.ET.iterparse", side_effect=Exception("parse error"))
@patch("src.cli.get_export_file_info")
def test_info_parse_warning(mock_file_info, mock_iterparse, tmp_path):
  """Info should warn when sample parsing fails."""
  runner = CliRunner()
  xml_path = tmp_path / "data.xml"
  xml_path.write_text("<HealthData></HealthData>")

  mock_file_info.return_value = {
    "file_path": xml_path,
    "file_size_mb": 0.1,
    "estimated_record_count": 1,
    "last_modified": datetime(2024, 1, 1),
  }

  result = runner.invoke(cli, ["info", str(xml_path)])

  assert result.exit_code == 0
  assert "Warning: Could not parse all records" in result.output


def test_export_success(tmp_path, monkeypatch):
  """Export should render summary for non-empty results."""
  runner = CliRunner()
  xml_path = tmp_path / "data.xml"
  xml_path.write_text("<HealthData></HealthData>")

  class DummyExporter:
    def __init__(self, output_dir):
      self.output_dir = Path(output_dir)

    def export_by_category(self, *args, **kwargs):
      return {"HeartRate": {"csv": 1, "json": 1}}

  monkeypatch.setattr(
    "src.processors.exporter.DataExporter", DummyExporter, raising=False
  )

  progress = MagicMock(
    __enter__=lambda s: s,
    __exit__=lambda *a: None,
    update=lambda *a, **k: None,
  )
  monkeypatch.setattr("src.cli.UnifiedProgress", MagicMock(return_value=progress))

  output_dir = tmp_path / "out"
  output_dir.mkdir()
  (output_dir / "dummy.txt").write_text("ok")

  result = runner.invoke(cli, ["export", str(xml_path), "--output", str(output_dir)])

  assert result.exit_code == 0
  assert "Export completed successfully" in result.output


def test_export_no_records(tmp_path, monkeypatch):
  """Export should exit when no records are exported."""
  runner = CliRunner()
  xml_path = tmp_path / "data.xml"
  xml_path.write_text("<HealthData></HealthData>")

  class DummyExporter:
    def __init__(self, output_dir):
      self.output_dir = Path(output_dir)

    def export_by_category(self, *args, **kwargs):
      return {"HeartRate": {"csv": 0, "json": 0}}

  monkeypatch.setattr(
    "src.processors.exporter.DataExporter", DummyExporter, raising=False
  )

  progress = MagicMock(
    __enter__=lambda s: s,
    __exit__=lambda *a: None,
    update=lambda *a, **k: None,
  )
  monkeypatch.setattr("src.cli.UnifiedProgress", MagicMock(return_value=progress))

  result = runner.invoke(cli, ["export", str(xml_path)])

  assert result.exit_code == 1
  assert "No records were exported" in result.output


def test_export_error(tmp_path, monkeypatch):
  """Export should exit on exporter errors."""
  runner = CliRunner()
  xml_path = tmp_path / "data.xml"
  xml_path.write_text("<HealthData></HealthData>")

  class DummyExporter:
    def __init__(self, output_dir):
      self.output_dir = Path(output_dir)

    def export_by_category(self, *args, **kwargs):
      raise Exception("boom")

  monkeypatch.setattr(
    "src.processors.exporter.DataExporter", DummyExporter, raising=False
  )

  progress = MagicMock(
    __enter__=lambda s: s,
    __exit__=lambda *a: None,
    update=lambda *a, **k: None,
  )
  monkeypatch.setattr("src.cli.UnifiedProgress", MagicMock(return_value=progress))

  result = runner.invoke(cli, ["export", str(xml_path)])

  assert result.exit_code == 1
  assert "Error: boom" in result.output


def test_analyze_requires_age_gender(tmp_path):
  """Cardio fitness analysis should require age and gender."""
  runner = CliRunner()
  xml_path = tmp_path / "data.xml"
  xml_path.write_text("<HealthData></HealthData>")

  result = runner.invoke(cli, ["analyze", str(xml_path), "--types", "cardio_fitness"])

  assert result.exit_code == 1
  assert "Age and gender are required" in result.output


def test_analyze_invalid_date_range(tmp_path):
  """Invalid date ranges should exit with error."""
  runner = CliRunner()
  xml_path = tmp_path / "data.xml"
  xml_path.write_text("<HealthData></HealthData>")

  result = runner.invoke(
    cli, ["analyze", str(xml_path), "--types", "heart_rate", "--date-range", "bad"]
  )

  assert result.exit_code == 1
  assert "Invalid date range format" in result.output


@patch("src.cli._display_highlights")
@patch("src.cli._display_sleep_results")
@patch("src.cli._display_heart_rate_results")
@patch("src.cli._save_analysis_results_text")
@patch("src.cli._save_analysis_results_json")
@patch("src.cli.StreamingXMLParser")
def test_analyze_success_json(
  mock_parser,
  mock_save_json,
  mock_save_text,
  _display_hr,
  _display_sleep,
  _display_highlights,
  tmp_path,
  monkeypatch,
):
  """Analyze should run and save JSON results."""
  runner = CliRunner()
  xml_path = tmp_path / "data.xml"
  xml_path.write_text("<HealthData></HealthData>")

  record = SimpleNamespace(
    type="HKQuantityTypeIdentifierHeartRate",
    start_date=datetime(2024, 1, 1, 8, 0, 0),
  )
  parser_instance = MagicMock()
  parser_instance.parse_records.return_value = [record]
  mock_parser.return_value = parser_instance

  class DummyHeartRateAnalyzer:
    def __init__(self, *args, **kwargs):
      pass

    def analyze_comprehensive(self, **kwargs):
      return MagicMock()

  class DummySleepAnalyzer:
    def analyze_comprehensive(self, *args, **kwargs):
      return MagicMock()

  class DummyHighlightsGenerator:
    def generate_comprehensive_highlights(self, **kwargs):
      return SimpleNamespace(insights=[], recommendations=[])

  monkeypatch.setattr(
    "src.processors.heart_rate.HeartRateAnalyzer", DummyHeartRateAnalyzer
  )
  monkeypatch.setattr("src.processors.sleep.SleepAnalyzer", DummySleepAnalyzer)
  monkeypatch.setattr(
    "src.analyzers.highlights.HighlightsGenerator", DummyHighlightsGenerator
  )

  progress = MagicMock(
    __enter__=lambda s: s, __exit__=lambda *a: None, update=lambda *a, **k: None
  )
  monkeypatch.setattr("src.cli.UnifiedProgress", MagicMock(return_value=progress))

  output_dir = tmp_path / "out"

  result = runner.invoke(
    cli,
    [
      "analyze",
      str(xml_path),
      "--types",
      "heart_rate",
      "--types",
      "sleep",
      "--format",
      "json",
      "--date-range",
      "2024-01-01:2024-01-02",
      "--output",
      str(output_dir),
    ],
  )

  assert result.exit_code == 0
  mock_save_json.assert_called_once()
  mock_save_text.assert_not_called()


@patch("src.processors.benchmark.run_benchmark")
def test_benchmark_success(mock_benchmark, tmp_path):
  """Benchmark should call runner and exit 0."""
  runner = CliRunner()
  xml_path = tmp_path / "data.xml"
  xml_path.write_text("<HealthData></HealthData>")

  result = runner.invoke(cli, ["benchmark", str(xml_path), "--output", str(tmp_path)])

  assert result.exit_code == 0
  mock_benchmark.assert_called_once()


@patch("src.processors.benchmark.run_benchmark", side_effect=Exception("boom"))
def test_benchmark_error(mock_benchmark, tmp_path):
  """Benchmark should exit when runner fails."""
  runner = CliRunner()
  xml_path = tmp_path / "data.xml"
  xml_path.write_text("<HealthData></HealthData>")

  result = runner.invoke(cli, ["benchmark", str(xml_path), "--output", str(tmp_path)])

  assert result.exit_code == 1
  assert "Error: boom" in result.output
