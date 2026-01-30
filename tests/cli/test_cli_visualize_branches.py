"""Branch coverage for visualize command flows."""

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from click.testing import CliRunner

from src.cli_visualize import visualize


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
    lambda: SimpleNamespace(_parse_sleep_sessions=lambda *_: [object()]),
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
      return None

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
