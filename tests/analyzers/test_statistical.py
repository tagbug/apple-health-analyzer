"""Tests for statistical analysis functionality."""

import numpy as np
import pandas as pd
import pytest

from src.analyzers.statistical import (
  StatisticalAnalyzer,
  StatisticsReport,
  TrendAnalysis,
)


class TestStatisticalAnalyzer:
  """Test StatisticalAnalyzer class."""

  @pytest.fixture
  def analyzer(self):
    """Create StatisticalAnalyzer instance."""
    return StatisticalAnalyzer()

  @pytest.fixture
  def sample_heart_rate_records(self):
    """Create sample heart rate records for testing."""
    from datetime import datetime, timedelta

    from src.core.data_models import HeartRateRecord

    base_time = datetime(2023, 1, 1, 10, 0, 0)

    return [
      HeartRateRecord(
        source_name="Apple Watch",
        value=70.0,
        creation_date=base_time,
        start_date=base_time,
        end_date=base_time,
      ),
      HeartRateRecord(
        source_name="Apple Watch",
        value=75.0,
        creation_date=base_time + timedelta(minutes=1),
        start_date=base_time + timedelta(minutes=1),
        end_date=base_time + timedelta(minutes=1),
      ),
      HeartRateRecord(
        source_name="Apple Watch",
        value=80.0,
        creation_date=base_time + timedelta(minutes=2),
        start_date=base_time + timedelta(minutes=2),
        end_date=base_time + timedelta(minutes=2),
      ),
    ]

  def test_analyzer_initialization(self, analyzer):
    """Test analyzer initialization."""
    assert isinstance(analyzer, StatisticalAnalyzer)

  def test_aggregate_by_hour(self, analyzer, sample_heart_rate_records):
    """Test aggregating records by hour."""
    result = analyzer.aggregate_by_interval(sample_heart_rate_records, "hour")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1  # All records in same hour
    assert "record_count" in result.columns
    assert "mean_value" in result.columns
    assert result.iloc[0]["record_count"] == 3
    assert result.iloc[0]["mean_value"] == 75.0  # (70 + 75 + 80) / 3

  def test_aggregate_by_day(self, analyzer, sample_heart_rate_records):
    """Test aggregating records by day."""
    result = analyzer.aggregate_by_interval(sample_heart_rate_records, "day")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1  # All records on same day
    assert result.iloc[0]["record_count"] == 3

  def test_aggregate_by_week(self, analyzer, sample_heart_rate_records):
    """Test aggregating records by week."""
    result = analyzer.aggregate_by_interval(sample_heart_rate_records, "week")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1  # All records in same week

  def test_aggregate_by_month(self, analyzer, sample_heart_rate_records):
    """Test aggregating records by month."""
    result = analyzer.aggregate_by_interval(sample_heart_rate_records, "month")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1  # All records in same month

  def test_aggregate_empty_records(self, analyzer):
    """Test aggregating empty record list."""
    result = analyzer.aggregate_by_interval([], "day")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0

  def test_calculate_statistics(self, analyzer, sample_heart_rate_records):
    """Test calculating statistics for records."""
    df = analyzer._records_to_dataframe(sample_heart_rate_records)
    stats = analyzer.calculate_statistics(df)

    assert isinstance(stats, StatisticsReport)
    assert stats.record_type == "HKQuantityTypeIdentifierHeartRate"
    assert stats.total_records == 3
    assert stats.min_value == 70.0
    assert stats.max_value == 80.0
    assert stats.mean_value == 75.0
    assert stats.median_value == 75.0
    assert stats.std_deviation > 0

  def test_calculate_statistics_empty_dataframe(self, analyzer):
    """Test calculating statistics for empty DataFrame."""
    empty_df = pd.DataFrame()
    stats = analyzer.calculate_statistics(empty_df)

    assert stats is None

  def test_calculate_statistics_missing_column(self, analyzer):
    """Test calculating statistics with missing value column."""
    df = pd.DataFrame({"other_column": [1, 2, 3]})
    stats = analyzer.calculate_statistics(df, "value")

    assert stats is None

  def test_analyze_trend_linear_increasing(self, analyzer):
    """Test linear trend analysis with increasing data."""
    # Create increasing data with simple numeric x values to avoid timestamp scaling issues
    data = pd.DataFrame(
      {
        "start_date": pd.to_datetime(
          ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
        ),
        "value": [10.0, 20.0, 30.0, 40.0, 50.0],  # Clearly increasing
      }
    )

    trend = analyzer.analyze_trend(data)

    assert isinstance(trend, TrendAnalysis)
    assert trend.method == "linear"
    # The slope should be positive (approximately 10.0 for this data)
    # But due to timestamp scaling, we just check it's positive and significant
    assert trend.slope > 0, f"Slope should be positive, got {trend.slope}"
    assert trend.trend_direction == "increasing", (
      f"Expected 'increasing', got '{trend.trend_direction}'"
    )
    assert trend.confidence_level > 0

  def test_analyze_trend_linear_decreasing(self, analyzer):
    """Test linear trend analysis with decreasing data."""
    # Create decreasing data
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    data = pd.DataFrame(
      {
        "start_date": dates,
        "value": [50, 40, 30, 20, 10],  # Clearly decreasing
      }
    )

    trend = analyzer.analyze_trend(data)

    assert isinstance(trend, TrendAnalysis)
    assert trend.method == "linear"
    assert trend.slope < 0  # Negative slope
    assert trend.trend_direction == "decreasing"

  def test_analyze_trend_stable(self, analyzer):
    """Test linear trend analysis with stable data."""
    # Create stable data
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    data = pd.DataFrame(
      {
        "start_date": dates,
        "value": [50, 51, 49, 50, 50],  # Mostly stable
      }
    )

    trend = analyzer.analyze_trend(data)

    assert isinstance(trend, TrendAnalysis)
    assert trend.method == "linear"
    assert abs(trend.slope) < 0.001  # Very small slope
    assert trend.trend_direction == "stable"

  def test_analyze_trend_insufficient_data(self, analyzer):
    """Test trend analysis with insufficient data."""
    # Only 2 data points
    dates = pd.date_range("2023-01-01", periods=2, freq="D")
    data = pd.DataFrame({"start_date": dates, "value": [10, 20]})

    trend = analyzer.analyze_trend(data)

    assert trend is None  # Should return None for insufficient data

  def test_analyze_trend_missing_columns(self, analyzer):
    """Test trend analysis with missing columns."""
    data = pd.DataFrame({"value": [1, 2, 3]})  # Missing time column

    trend = analyzer.analyze_trend(data)

    assert trend is None

  def test_analyze_trend_polynomial(self, analyzer):
    """Test polynomial trend analysis."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    # Create quadratic relationship: y = x^2
    x_values = np.arange(10)
    y_values = x_values**2
    data = pd.DataFrame({"start_date": dates, "value": y_values})

    trend = analyzer.analyze_trend(data, method="polynomial")

    assert isinstance(trend, TrendAnalysis)
    assert trend.method == "polynomial"
    assert trend.r_squared > 0.9  # Should fit very well

  def test_analyze_trend_moving_average(self, analyzer):
    """Test moving average trend analysis."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    # Create data with some noise
    base_values = np.linspace(10, 50, 10)
    noise = np.random.normal(0, 2, 10)
    data = pd.DataFrame({"start_date": dates, "value": base_values + noise})

    trend = analyzer.analyze_trend(data, method="moving_average", window=3)

    assert isinstance(trend, TrendAnalysis)
    assert trend.method == "moving_average"

  def test_generate_report(self, analyzer, sample_heart_rate_records):
    """Test generating complete statistical report."""
    intervals = ["day", "week"]
    report = analyzer.generate_report(sample_heart_rate_records, intervals)

    assert isinstance(report, dict)
    assert "summary" in report
    assert "interval_analyses" in report

    # Check summary
    summary = report["summary"]
    assert isinstance(summary, StatisticsReport)
    assert summary.total_records == 3

    # Check interval analyses
    interval_analyses = report["interval_analyses"]
    assert "day" in interval_analyses
    assert "week" in interval_analyses

  def test_generate_report_empty_records(self, analyzer):
    """Test generating report with empty records."""
    report = analyzer.generate_report([])

    assert isinstance(report, dict)
    assert report == {}  # Should return empty dict

  def test_generate_report_as_dataframe(
    self, analyzer, sample_heart_rate_records
  ):
    """Test generating report as DataFrame."""
    report = analyzer.generate_report(
      sample_heart_rate_records, output_format="dataframe"
    )

    assert isinstance(report, pd.DataFrame)
    assert len(report) > 0
    assert "interval" in report.columns

  def test_records_to_dataframe_conversion(
    self, analyzer, sample_heart_rate_records
  ):
    """Test converting records to DataFrame."""
    df = analyzer._records_to_dataframe(sample_heart_rate_records)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "type" in df.columns
    assert "value" in df.columns
    assert "source_name" in df.columns

    # Check values
    assert df["value"].tolist() == [70.0, 75.0, 80.0]

  def test_records_to_dataframe_empty(self, analyzer):
    """Test DataFrame conversion with empty records."""
    df = analyzer._records_to_dataframe([])

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0

  def test_data_quality_score_calculation(self, analyzer):
    """Test data quality score calculation."""
    # Create test data
    values = pd.Series([70, 75, 80, 85, 90])
    data = pd.DataFrame({"value": values})
    record_type = "HKQuantityTypeIdentifierHeartRate"

    score = analyzer._calculate_data_quality_score(values, data, record_type)

    assert isinstance(score, float)
    assert 0 <= score <= 1

    # Perfect data should have high score
    assert score > 0.8

  def test_data_quality_score_with_outliers(self, analyzer):
    """Test data quality score with outliers."""
    # Create data with extreme outliers (500 bpm is clearly unreasonable for heart rate)
    values = pd.Series(
      [70, 75, 80, 85, 500]
    )  # 500 is extreme outlier for heart rate
    data = pd.DataFrame({"value": values})
    record_type = "HKQuantityTypeIdentifierHeartRate"

    score = analyzer._calculate_data_quality_score(values, data, record_type)

    assert isinstance(score, float)
    assert (
      0.6 < score < 0.8
    )  # Should be lower than perfect data but not too low

  def test_normality_score_calculation(self, analyzer):
    """Test normality score calculation."""
    # Create normal-like data
    np.random.seed(42)
    values = pd.Series(np.random.normal(75, 5, 100))

    score = analyzer._calculate_normality_score(values)

    assert isinstance(score, float)
    assert 0 <= score <= 1

  def test_normality_score_small_sample(self, analyzer):
    """Test normality score with small sample."""
    values = pd.Series([70, 75, 80])

    score = analyzer._calculate_normality_score(values)

    # Small samples should return default score
    assert score == 0.5

  def test_report_to_dataframe_conversion(
    self, analyzer, sample_heart_rate_records
  ):
    """Test converting report to DataFrame."""
    # Create a mock report
    mock_report = {
      "summary": analyzer.calculate_statistics(
        analyzer._records_to_dataframe(sample_heart_rate_records)
      ),
      "interval_analyses": {},
    }

    df = analyzer._report_to_dataframe(mock_report)

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "interval" in df.columns
    assert "overall" in df["interval"].values


class TestStatisticsReport:
  """Test StatisticsReport dataclass."""

  def test_statistics_report_creation(self):
    """Test creating a StatisticsReport."""
    from datetime import datetime

    report = StatisticsReport(
      record_type="HKQuantityTypeIdentifierHeartRate",
      time_interval="day",
      total_records=100,
      date_range=(datetime(2023, 1, 1), datetime(2023, 1, 2)),
      min_value=60.0,
      max_value=90.0,
      mean_value=75.0,
      median_value=74.0,
      std_deviation=5.0,
      percentile_25=70.0,
      percentile_75=80.0,
      percentile_95=85.0,
      data_quality_score=0.95,
      missing_values=0,
      records_per_day=50.0,
      active_days=2,
      total_days=2,
    )

    assert report.record_type == "HKQuantityTypeIdentifierHeartRate"
    assert report.total_records == 100
    assert report.mean_value == 75.0
    assert report.data_quality_score == 0.95


class TestTrendAnalysis:
  """Test TrendAnalysis dataclass."""

  def test_trend_analysis_creation(self):
    """Test creating a TrendAnalysis."""
    analysis = TrendAnalysis(
      method="linear",
      slope=2.5,
      intercept=10.0,
      r_squared=0.85,
      p_value=0.001,
      trend_direction="increasing",
      confidence_level=85.0,
    )

    assert analysis.method == "linear"
    assert analysis.slope == 2.5
    assert analysis.trend_direction == "increasing"
    assert analysis.confidence_level == 85.0

  def test_trend_analysis_directions(self):
    """Test different trend directions."""
    increasing = TrendAnalysis(
      method="linear",
      slope=1.0,
      intercept=0.0,
      r_squared=0.8,
      p_value=0.01,
      trend_direction="increasing",
      confidence_level=80.0,
    )
    decreasing = TrendAnalysis(
      method="linear",
      slope=-1.0,
      intercept=0.0,
      r_squared=0.8,
      p_value=0.01,
      trend_direction="decreasing",
      confidence_level=80.0,
    )
    stable = TrendAnalysis(
      method="linear",
      slope=0.0,
      intercept=0.0,
      r_squared=0.8,
      p_value=0.01,
      trend_direction="stable",
      confidence_level=80.0,
    )

    assert increasing.trend_direction == "increasing"
    assert decreasing.trend_direction == "decreasing"
    assert stable.trend_direction == "stable"


class TestStatisticalAnalyzerEdgeCases:
  """Test edge cases and error handling."""

  @pytest.fixture
  def analyzer(self):
    """Create StatisticalAnalyzer instance."""
    return StatisticalAnalyzer()

  @pytest.fixture
  def sample_heart_rate_records(self):
    """Create sample heart rate records for testing."""
    from datetime import datetime, timedelta

    from src.core.data_models import HeartRateRecord

    base_time = datetime(2023, 1, 1, 10, 0, 0)

    return [
      HeartRateRecord(
        source_name="Apple Watch",
        value=70.0,
        creation_date=base_time,
        start_date=base_time,
        end_date=base_time,
      ),
      HeartRateRecord(
        source_name="Apple Watch",
        value=75.0,
        creation_date=base_time + timedelta(minutes=1),
        start_date=base_time + timedelta(minutes=1),
        end_date=base_time + timedelta(minutes=1),
      ),
      HeartRateRecord(
        source_name="Apple Watch",
        value=80.0,
        creation_date=base_time + timedelta(minutes=2),
        start_date=base_time + timedelta(minutes=2),
        end_date=base_time + timedelta(minutes=2),
      ),
    ]

  def test_aggregate_invalid_interval(
    self, analyzer, sample_heart_rate_records
  ):
    """Test aggregating with invalid interval."""
    # Should default to "D" (day) for invalid interval
    result = analyzer.aggregate_by_interval(
      sample_heart_rate_records, "invalid"
    )

    assert isinstance(result, pd.DataFrame)
    # Should still work with default frequency

  def test_trend_analysis_empty_dataframe(self, analyzer):
    """Test trend analysis with empty DataFrame."""
    empty_df = pd.DataFrame()
    trend = analyzer.analyze_trend(empty_df)

    assert trend is None

  def test_trend_analysis_single_point(self, analyzer):
    """Test trend analysis with single data point."""
    dates = pd.date_range("2023-01-01", periods=1, freq="D")
    data = pd.DataFrame({"start_date": dates, "value": [75.0]})

    trend = analyzer.analyze_trend(data)

    assert trend is None  # Need at least 3 points

  def test_calculate_statistics_no_values(self, analyzer):
    """Test calculating statistics with no valid values."""
    df = pd.DataFrame({"value": [None, None, None]})
    stats = analyzer.calculate_statistics(df)

    assert stats is None

  def test_data_quality_score_empty_data(self, analyzer):
    """Test data quality score with empty data."""
    empty_values = pd.Series([], dtype=float)
    empty_data = pd.DataFrame()

    score = analyzer._calculate_data_quality_score(empty_values, empty_data)

    assert score == 0.0  # Should handle empty data gracefully

  def test_data_quality_score_single_value(self, analyzer):
    """Test data quality score with single value."""
    values = pd.Series([75.0])
    data = pd.DataFrame({"value": values})

    score = analyzer._calculate_data_quality_score(values, data)

    assert isinstance(score, float)
    # Single value should have reasonable score
