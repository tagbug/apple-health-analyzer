"""Coverage for report chart generation helpers."""

from datetime import datetime
from types import SimpleNamespace

import pandas as pd

from src.visualization.charts import ChartGenerator


def _daily_stats_df():
  return pd.DataFrame(
    {
      "interval_start": pd.date_range("2024-01-01", periods=3, freq="D"),
      "mean_value": [70, 72, 68],
      "value": [70, 72, 68],
      "start_date": pd.date_range("2024-01-01", periods=3, freq="D"),
    }
  )


def test_generate_heart_rate_report_charts(tmp_path):
  """Ensure heart rate report charts are generated when data is present."""
  report = SimpleNamespace(
    resting_hr_analysis=object(),
    hrv_analysis=object(),
    daily_stats=_daily_stats_df(),
  )

  generator = ChartGenerator(width=600, height=400)
  charts = generator.generate_heart_rate_report_charts(report, tmp_path)

  assert "resting_hr_trend" in charts
  assert "hrv_analysis" in charts
  assert "heart_rate_heatmap" in charts
  assert "heart_rate_distribution" in charts
  assert "heart_rate_zones" in charts

  for path in charts.values():
    assert path.exists()


def test_generate_sleep_report_charts(tmp_path):
  """Ensure sleep report charts are generated when data is present."""
  report = SimpleNamespace(
    sleep_sessions=[
      SimpleNamespace(
        start_date=datetime(2024, 1, 1, 22, 0, 0),
        end_date=datetime(2024, 1, 2, 6, 0, 0),
      )
    ],
    daily_summary=pd.DataFrame(
      {
        "date": pd.date_range("2024-01-01", periods=2, freq="D"),
        "total_duration": [480, 450],
        "efficiency": [0.8, 0.85],
        "deep_sleep": [60, 70],
        "rem_sleep": [90, 80],
      }
    ),
  )

  generator = ChartGenerator(width=600, height=400)
  charts = generator.generate_sleep_report_charts(report, tmp_path)

  assert "sleep_timeline" in charts
  assert "sleep_quality_trend" in charts
  assert "sleep_stages_distribution" in charts
  assert "sleep_consistency" in charts
  assert "weekday_vs_weekend_sleep" in charts

  for path in charts.values():
    assert path.exists()
