"""Unit tests for chart generation functionality."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.visualization.charts import ChartGenerator


class TestChartGenerator:
  """ChartGenerator tests."""

  @pytest.fixture
  def chart_generator(self):
    """Create ChartGenerator fixture."""
    return ChartGenerator(width=800, height=600)

  @pytest.fixture
  def sample_heart_rate_data(self):
    """Create sample heart rate data."""
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    data = pd.DataFrame(
      {
        "timestamp": dates,
        "value": [70 + (i % 24) * 2 + (i % 10) for i in range(100)],
      }
    )
    return data

  @pytest.fixture
  def sample_resting_hr_data(self):
    """Create sample resting heart rate data."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    data = pd.DataFrame(
      {"start_date": dates, "value": [72 - i * 0.1 for i in range(30)]}
    )
    return data

  @pytest.fixture
  def sample_hrv_data(self):
    """Create sample HRV data."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    data = pd.DataFrame(
      {"start_date": dates, "value": [35 + i * 0.2 for i in range(30)]}
    )
    return data

  @pytest.fixture
  def sample_heatmap_data(self):
    """Create sample heatmap data."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    data = []
    for date in dates:
      for hour in range(24):
        data.append(
          {
            "date": date,
            "hour": hour,
            "activity_level": (hour % 12) * 10 + (date.day % 5),
          }
        )
    return pd.DataFrame(data)

  @pytest.fixture
  def sample_timeline_data(self):
    """Create sample timeline data."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    data = []
    for date in dates:
      data.extend(
        [
          {
            "date": date,
            "metric": "Heart Rate",
            "value": 70 + (date.day % 10),
            "category": "Cardio",
          },
          {
            "date": date,
            "metric": "Sleep",
            "value": 8 - (date.day % 3),
            "category": "Sleep",
          },
          {
            "date": date,
            "metric": "Steps",
            "value": 8000 + (date.day % 2000),
            "category": "Activity",
          },
        ]
      )
    return pd.DataFrame(data)

  def test_initialization(self, chart_generator):
    """Test initialization."""
    assert isinstance(chart_generator, ChartGenerator)
    assert chart_generator.width == 800
    assert chart_generator.height == 600
    assert chart_generator.theme == "health"

  def test_plot_health_dashboard(self, chart_generator):
    """Test health dashboard plotting."""
    wellness_score = 0.85
    metrics = {"Sleep Quality": 0.8, "Activity Level": 0.7, "Stress": 0.6}

    fig = chart_generator.plot_health_dashboard(wellness_score, metrics)

    assert fig is not None
    assert len(fig.data) == 4  # Gauge, radar, bar, pie.

  def test_plot_health_dashboard_empty_metrics(self, chart_generator):
    """Test health dashboard plotting with empty metrics."""
    wellness_score = 0.75
    metrics = {}

    fig = chart_generator.plot_health_dashboard(wellness_score, metrics)

    assert fig is not None
    assert len(fig.data) == 2  # Gauge + pie (radar/bar skipped).

  def test_plot_correlation_heatmap(self, chart_generator):
    """Test correlation heatmap plotting."""
    correlation_data = {
      "heart_rate_sleep": {"correlation": -0.3},
      "activity_stress": {"correlation": -0.5},
      "sleep_activity": {"correlation": 0.2},
    }

    fig = chart_generator.plot_correlation_heatmap(correlation_data)

    assert fig is not None
    assert len(fig.data) == 1
    assert fig.data[0].type == "heatmap"

  def test_plot_correlation_heatmap_empty(self, chart_generator):
    """Test correlation heatmap plotting with empty data."""
    correlation_data = {}

    fig = chart_generator.plot_correlation_heatmap(correlation_data)

    assert fig is None

  def test_plot_trend_analysis(self, chart_generator):
    """Test trend analysis plotting."""
    trend_data = {
      "Heart Rate": [70, 72, 68, 75, 71],
      "Sleep Duration": [8.0, 7.5, 8.5, 7.8, 8.2],
      "Activity Level": [0.7, 0.8, 0.6, 0.9, 0.75],
    }
    dates = [
      "2024-01-01",
      "2024-01-02",
      "2024-01-03",
      "2024-01-04",
      "2024-01-05",
    ]

    fig = chart_generator.plot_trend_analysis(trend_data, dates)

    assert fig is not None
    assert len(fig.data) == 3  # Three trend lines.

  def test_plot_activity_heatmap(self, chart_generator, sample_heatmap_data):
    """Test activity heatmap plotting."""
    fig = chart_generator.plot_activity_heatmap(sample_heatmap_data)

    assert fig is not None
    assert len(fig.data) == 1
    assert fig.data[0].type == "heatmap"

  def test_plot_activity_heatmap_empty(self, chart_generator):
    """Test activity heatmap plotting with empty data."""
    empty_data = pd.DataFrame()

    fig = chart_generator.plot_activity_heatmap(empty_data)

    assert fig is None

  def test_plot_circular_health_metrics(self, chart_generator):
    """Test circular health metrics plotting."""
    metrics = {"Sleep": 0.8, "Exercise": 0.7, "Nutrition": 0.6, "Stress": 0.5}

    fig = chart_generator.plot_circular_health_metrics(metrics)

    assert fig is not None
    assert len(fig.data) == 2  # Background + data rings.

  def test_plot_circular_health_metrics_empty(self, chart_generator):
    """Test circular health metrics plotting with empty metrics."""
    metrics = {}

    fig = chart_generator.plot_circular_health_metrics(metrics)

    assert fig is None

  def test_plot_health_timeline(self, chart_generator, sample_timeline_data):
    """Test health timeline plotting."""
    fig = chart_generator.plot_health_timeline(sample_timeline_data)

    assert fig is not None
    assert len(fig.data) == 3  # Three categories.

  def test_plot_health_timeline_empty(self, chart_generator):
    """Test health timeline plotting with empty data."""
    empty_data = pd.DataFrame()

    fig = chart_generator.plot_health_timeline(empty_data)

    assert fig is None

  def test_plot_risk_assessment(self, chart_generator):
    """Test risk assessment plotting."""
    risk_factors = {
      "Stress": 0.8,
      "Sleep Debt": 0.6,
      "Low Activity": 0.4,
      "Heart Rate Anomalies": 0.2,
    }

    fig = chart_generator.plot_risk_assessment(risk_factors)

    assert fig is not None
    assert len(fig.data) == 1  # Bar chart.
    assert len(fig.layout.shapes) == 2  # Two reference lines.

  def test_plot_risk_assessment_empty(self, chart_generator):
    """Test risk assessment plotting with empty data."""
    risk_factors = {}

    fig = chart_generator.plot_risk_assessment(risk_factors)

    assert fig is None

  def test_generate_comprehensive_report_charts(self, chart_generator):
    """Test comprehensive report chart generation."""
    # Create mock report object.
    mock_report = Mock()
    mock_report.overall_wellness_score = 0.85

    # Sleep quality.
    mock_sleep = Mock()
    mock_sleep.average_duration_hours = 7.5
    mock_report.sleep_quality = mock_sleep

    # Activity patterns.
    mock_activity = Mock()
    mock_activity.daily_step_average = 8500
    mock_report.activity_patterns = mock_activity

    # Metabolic health.
    mock_metabolic = Mock()
    mock_metabolic.metabolic_health_score = 0.75
    mock_report.metabolic_health = mock_metabolic

    # Stress resilience.
    mock_stress = Mock()
    mock_stress.stress_accumulation_score = 0.3
    mock_stress.recovery_capacity_score = 0.8
    mock_report.stress_resilience = mock_stress

    # Correlation data.
    mock_report.health_correlations = {
      "sleep_activity": {"correlation": 0.4},
      "stress_heart_rate": {"correlation": 0.6},
    }

    with tempfile.TemporaryDirectory() as temp_dir:
      output_dir = Path(temp_dir) / "charts"
      charts = chart_generator.generate_comprehensive_report_charts(
        mock_report, output_dir
      )

      assert isinstance(charts, dict)
      assert "dashboard" in charts
      assert "correlation" in charts
      assert "risk_assessment" in charts

      # Verify chart files exist.
      for chart_path in charts.values():
        assert chart_path.exists()
        assert chart_path.suffix == ".html"

  @patch("src.visualization.charts.logger")
  def test_plot_correlation_heatmap_error_handling(self, mock_logger, chart_generator):
    """Test correlation heatmap error handling."""
    correlation_data = {"test": {"correlation": 0.5}}

    with patch("plotly.graph_objects.Figure") as mock_fig:
      mock_fig.side_effect = Exception("Plotly error")

      fig = chart_generator.plot_correlation_heatmap(correlation_data)

      assert fig is None
      mock_logger.error.assert_called_once()

  def test_save_plotly_figure_html(self, chart_generator):
    """Test saving Plotly chart to HTML."""
    import plotly.graph_objects as go

    fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))

    with tempfile.TemporaryDirectory() as temp_dir:
      output_path = Path(temp_dir) / "test_chart.html"

      chart_generator._save_plotly_figure(fig, output_path)

      assert output_path.exists()
      assert output_path.suffix == ".html"

      # Check file content (UTF-8).
      content = output_path.read_text(encoding="utf-8")
      assert "<html>" in content
      assert "plotly" in content.lower()

  def test_downsample_data(self, chart_generator, sample_heart_rate_data):
    """Test data downsampling."""
    large_data = pd.concat([sample_heart_rate_data] * 20)  # Large dataset.

    downsampled = chart_generator._downsample_data(large_data, 1000)

    assert len(downsampled) <= 1000
    assert len(downsampled) > 0

  def test_downsample_data_small_dataset(self, chart_generator, sample_heart_rate_data):
    """Test data downsampling with small dataset."""
    small_data = sample_heart_rate_data.head(50)

    downsampled = chart_generator._downsample_data(small_data, 1000)

    assert len(downsampled) == len(small_data)  # No downsampling.

  # Tests for inherited methods.
  def test_plot_heart_rate_timeseries(self, chart_generator, sample_heart_rate_data):
    """Test heart rate timeseries plotting."""
    fig = chart_generator.plot_heart_rate_timeseries(sample_heart_rate_data)

    assert fig is not None
    assert len(fig.data) >= 1

  def test_plot_resting_hr_trend(self, chart_generator, sample_resting_hr_data):
    """Test resting HR trend plotting."""
    fig = chart_generator.plot_resting_hr_trend(sample_resting_hr_data)

    assert fig is not None
    assert len(fig.data) >= 1

  def test_plot_hrv_analysis(self, chart_generator, sample_hrv_data):
    """Test HRV analysis plotting."""
    fig = chart_generator.plot_hrv_analysis(sample_hrv_data)

    assert fig is not None
    assert len(fig.data) >= 1

  def test_plot_heart_rate_heatmap(self, chart_generator):
    """Test heart rate heatmap plotting."""
    # Create heatmap data.
    dates = pd.date_range("2024-01-01", periods=14, freq="D")
    data = pd.DataFrame({"date": dates, "avg_hr": [70 + i for i in range(14)]})

    fig = chart_generator.plot_heart_rate_heatmap(data)

    assert fig is not None
    assert len(fig.data) == 1
    assert fig.data[0].type == "heatmap"

  def test_plot_heart_rate_distribution(self, chart_generator):
    """Test heart rate distribution plotting."""
    data = pd.DataFrame({"value": [65, 70, 75, 80, 85, 90] * 10})

    fig = chart_generator.plot_heart_rate_distribution(data)

    assert fig is not None
    assert len(fig.data) >= 2  # Histogram + normal curve.

  def test_plot_heart_rate_zones(self, chart_generator):
    """Test heart rate zones plotting."""
    data = pd.DataFrame({"value": [60, 70, 80, 90, 100, 110, 120] * 5})

    fig = chart_generator.plot_heart_rate_zones(data)

    assert fig is not None
    assert len(fig.data) == 1
    assert fig.data[0].type == "pie"
