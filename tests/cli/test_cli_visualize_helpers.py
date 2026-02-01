"""Chart helper tests for visualize CLI."""

from datetime import datetime

import pandas as pd

from src.visualization.charts import ChartGenerator


def test_plot_sleep_quality_trend_with_efficiency():
  """Ensure sleep quality trend renders with efficiency series."""
  data = pd.DataFrame(
    {
      "date": pd.date_range("2024-01-01", periods=3, freq="D"),
      "total_duration": [420, 450, 480],
      "efficiency": [0.8, 0.85, 0.9],
    }
  )

  generator = ChartGenerator(width=600, height=400)
  fig = generator.plot_sleep_quality_trend(data)

  assert fig is not None
  data_series = fig.to_dict().get("data", [])
  assert len(data_series) == 2


def test_plot_sleep_quality_trend_without_efficiency():
  """Ensure sleep quality trend handles missing efficiency column."""
  data = pd.DataFrame(
    {
      "date": pd.date_range("2024-01-01", periods=2, freq="D"),
      "total_duration": [420, 460],
    }
  )

  generator = ChartGenerator(width=600, height=400)
  fig = generator.plot_sleep_quality_trend(data)

  assert fig is not None
  data_series = fig.to_dict().get("data", [])
  assert len(data_series) == 1


def test_plot_sleep_stages_distribution_with_data():
  """Ensure sleep stages distribution renders when durations exist."""
  data = pd.DataFrame(
    {
      "date": pd.date_range("2024-01-01", periods=4, freq="D"),
      "stage": ["Deep", "REM", "Core", "Deep"],
      "duration": [60, 90, 120, 30],
    }
  )

  generator = ChartGenerator(width=600, height=400)
  fig = generator.plot_sleep_stages_distribution(data)

  assert fig is not None
  data_series = fig.to_dict().get("data", [])
  assert len(data_series) == 3


def test_plot_sleep_consistency_with_times():
  """Ensure sleep consistency handles time conversion."""
  data = pd.DataFrame(
    {
      "date": pd.date_range("2024-01-01", periods=2, freq="D"),
      "bedtime": [
        datetime(2024, 1, 1, 23, 0, 0),
        datetime(2024, 1, 2, 23, 30, 0),
      ],
      "wake_time": [
        datetime(2024, 1, 2, 7, 0, 0),
        datetime(2024, 1, 3, 7, 15, 0),
      ],
    }
  )

  generator = ChartGenerator(width=600, height=400)
  fig = generator.plot_sleep_consistency(data)

  assert fig is not None
  data_series = fig.to_dict().get("data", [])
  assert len(data_series) == 2


def test_plot_weekday_vs_weekend_sleep():
  """Ensure weekday vs weekend sleep chart renders."""
  data = pd.DataFrame(
    {
      "date": pd.date_range("2024-01-01", periods=4, freq="D"),
      "duration": [7.5, 7.0, 8.2, 8.0],
      "is_weekend": [False, False, True, True],
    }
  )

  generator = ChartGenerator(width=600, height=400)
  fig = generator.plot_weekday_vs_weekend_sleep(data)

  assert fig is not None
  data_series = fig.to_dict().get("data", [])
  assert len(data_series) == 2


def test_plot_sleep_timeline_renders():
  """Ensure sleep timeline renders for basic session data."""
  data = pd.DataFrame(
    {
      "start_date": [datetime(2024, 1, 1, 23, 0, 0)],
      "end_date": [datetime(2024, 1, 2, 7, 0, 0)],
      "value": ["Asleep"],
    }
  )

  generator = ChartGenerator(width=600, height=400)
  fig = generator.plot_sleep_timeline(data)

  assert fig is not None
  data_series = fig.to_dict().get("data", [])
  assert len(data_series) == 1
