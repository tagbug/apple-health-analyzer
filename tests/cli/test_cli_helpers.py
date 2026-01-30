"""Tests for CLI helper utilities."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from src import cli as cli_module


@dataclass
class DummyReport:
  """Simple report with minimal fields for serialization."""

  analysis_date: datetime
  data_range: tuple[datetime, datetime]
  record_count: int
  data_quality_score: float
  resting_hr_analysis: object | None = None
  hrv_analysis: object | None = None
  cardio_fitness: object | None = None
  quality_metrics: object | None = None


def test_display_parsing_results_outputs(monkeypatch):
  """Ensure parsing results render without errors."""
  console = MagicMock()
  console.print = MagicMock()
  monkeypatch.setattr(cli_module, "console", console)

  stats = {
    "total_records": 10,
    "processed_records": 9,
    "skipped_records": 1,
    "invalid_records": 0,
    "success_rate": 0.9,
    "date_range": {"start": "2024-01-01", "end": "2024-01-02"},
    "record_types": {"TypeA": 5, "TypeB": 4},
    "sources": {"Watch": 6, "Phone": 3},
  }

  cli_module._display_parsing_results(stats)

  assert console.print.call_count > 0


def test_display_records_preview_handles_values(monkeypatch):
  """Ensure record preview handles value and sleep stage fields."""
  console = MagicMock()
  console.print = MagicMock()
  monkeypatch.setattr(cli_module, "console", console)

  record_with_value = SimpleNamespace(
    type="HKQuantityTypeIdentifierHeartRate",
    source_name="Watch",
    start_date=datetime(2024, 1, 1, 8, 0, 0),
    value=72.5,
  )
  record_with_stage = SimpleNamespace(
    type="HKCategoryTypeIdentifierSleepAnalysis",
    source_name="Watch",
    start_date=datetime(2024, 1, 1, 23, 0, 0),
    sleep_stage=SimpleNamespace(value="Asleep"),
  )

  cli_module._display_records_preview([record_with_value, record_with_stage])

  assert console.print.call_count > 0


def test_report_to_dict_serializes_fields():
  """Ensure report serialization returns expected keys."""
  report = DummyReport(
    analysis_date=datetime(2024, 1, 1, 12, 0, 0),
    data_range=(
      datetime(2024, 1, 1, 0, 0, 0),
      datetime(2024, 1, 2, 0, 0, 0),
    ),
    record_count=5,
    data_quality_score=0.9,
  )

  payload = cli_module._report_to_dict(report)

  assert payload["analysis_date"] is not None
  assert payload["data_range"][0].startswith("2024-01-01")
  assert payload["record_count"] == 5
  assert payload["data_quality_score"] == 0.9


def test_save_analysis_results_json_and_text(tmp_path, monkeypatch):
  """Ensure analysis results are written to disk."""
  console = MagicMock()
  console.print = MagicMock()
  monkeypatch.setattr(cli_module, "console", console)

  report = DummyReport(
    analysis_date=datetime(2024, 1, 1, 12, 0, 0),
    data_range=(
      datetime(2024, 1, 1, 0, 0, 0),
      datetime(2024, 1, 2, 0, 0, 0),
    ),
    record_count=3,
    data_quality_score=0.8,
    resting_hr_analysis=SimpleNamespace(
      current_value=65.0, trend_direction="stable", health_rating="good"
    ),
    hrv_analysis=SimpleNamespace(
      current_sdnn=40.0, stress_level="low", recovery_status="good"
    ),
    cardio_fitness=SimpleNamespace(current_vo2_max=40.0, age_adjusted_rating="good"),
    quality_metrics=SimpleNamespace(
      average_duration=7.5, average_efficiency=0.9, consistency_score=0.8
    ),
  )
  highlights = SimpleNamespace(
    insights=[
      SimpleNamespace(
        category="heart_rate",
        priority="high",
        title="Insight",
        message="Message",
        confidence=0.9,
      )
    ],
    recommendations=["Rec1"],
    summary={"total_insights": 1},
  )

  cli_module._save_analysis_results_json(tmp_path, report, report, highlights)
  cli_module._save_analysis_results_text(tmp_path, report, report, highlights)

  assert (tmp_path / "analysis_results.json").exists()
  assert (tmp_path / "analysis_results.txt").exists()


def test_display_results_helpers(monkeypatch):
  """Ensure display helpers render for summary objects."""
  console = MagicMock()
  console.print = MagicMock()
  monkeypatch.setattr(cli_module, "console", console)

  heart_rate_report = SimpleNamespace(
    resting_hr_analysis=SimpleNamespace(
      current_value=65.0, trend_direction="decreasing", health_rating="excellent"
    ),
    hrv_analysis=SimpleNamespace(
      current_sdnn=40.0, stress_level="low", recovery_status="good"
    ),
    cardio_fitness=SimpleNamespace(current_vo2_max=40.0, age_adjusted_rating="good"),
    data_quality_score=0.9,
    record_count=100,
  )
  sleep_report = SimpleNamespace(
    quality_metrics=SimpleNamespace(
      average_duration=7.5, average_efficiency=0.9, consistency_score=0.8
    ),
    data_quality_score=0.85,
    record_count=50,
  )
  highlights = SimpleNamespace(
    insights=[SimpleNamespace(priority="high", title="Insight", message="Message")],
    recommendations=["Rec1"],
  )

  cli_module._display_heart_rate_results(heart_rate_report)
  cli_module._display_sleep_results(sleep_report)
  cli_module._display_highlights(highlights)

  assert console.print.call_count > 0


def test_save_parsed_data_with_validation(tmp_path, monkeypatch):
  """Ensure parsed data persistence runs with mocked exporters."""

  class DummyValidationResult:
    def __init__(self):
      self.errors = []
      self.warnings = []
      self.outliers_detected = []
      self.issues_by_type = {}
      self.consistency_checks = {}

    def get_summary(self):
      return {"quality_score": 0.9, "total_warnings": 0, "outliers_count": 0}

  class DummyExporter:
    def __init__(self, output_dir):
      self.output_dir = Path(output_dir)

    def export_to_csv(self, records, path):
      Path(path).write_text("col\nvalue\n", encoding="utf-8")
      return len(records)

    def export_to_json(self, records, path):
      Path(path).write_text("[]", encoding="utf-8")
      return len(records)

  console = MagicMock()
  console.print = MagicMock()
  monkeypatch.setattr(cli_module, "console", console)
  monkeypatch.setattr(
    "src.processors.exporter.DataExporter", DummyExporter, raising=False
  )
  monkeypatch.setattr(
    "src.processors.validator.validate_health_data",
    lambda records: DummyValidationResult(),
    raising=False,
  )

  record = SimpleNamespace(
    type="HKQuantityTypeIdentifierHeartRate",
    source_name="Watch",
    start_date=datetime(2024, 1, 1, 8, 0, 0),
    end_date=datetime(2024, 1, 1, 8, 1, 0),
    value=70.0,
  )

  output_dir = tmp_path / "out"
  stats = {
    "total_records": 1,
    "processed_records": 1,
    "skipped_records": 0,
    "invalid_records": 0,
    "success_rate": 1.0,
    "date_range": {"start": "2024-01-01", "end": "2024-01-01"},
    "record_types": {"HKQuantityTypeIdentifierHeartRate": 1},
    "sources": {"Watch": 1},
  }

  cli_module._save_parsed_data([record], stats, output_dir)

  assert (output_dir / "HeartRate.csv").exists()
  assert (output_dir / "HeartRate.json").exists()
  assert (output_dir / "parsing_stats.json").exists()
  assert (output_dir / "data_manifest.txt").exists()
