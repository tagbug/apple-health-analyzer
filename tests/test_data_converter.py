"""Tests for data_converter module."""

from datetime import datetime

import pandas as pd

from src.core.data_models import QuantityRecord
from src.visualization.data_converter import DataConverter


class TestDataConverter:
  """Test DataConverter class."""

  def test_heart_rate_to_df_empty(self):
    """Test converting empty heart rate records to DataFrame."""
    result = DataConverter.heart_rate_to_df([])

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["timestamp", "value", "source"]
    assert len(result) == 0

  def test_heart_rate_to_df_with_data(self):
    """Test converting heart rate records to DataFrame."""
    records = [
      QuantityRecord(
        type="HKQuantityTypeIdentifierHeartRate",
        source_name="TestWatch",
        source_version="1.0",
        device="TestDevice",
        unit="count/min",
        creation_date=datetime(2024, 1, 1, 10, 0),
        start_date=datetime(2024, 1, 1, 10, 0),
        end_date=datetime(2024, 1, 1, 10, 1),
        value=70,
      ),
      QuantityRecord(
        type="HKQuantityTypeIdentifierHeartRate",
        source_name="TestWatch",
        source_version="1.0",
        device="TestDevice",
        unit="count/min",
        creation_date=datetime(2024, 1, 1, 10, 5),
        start_date=datetime(2024, 1, 1, 10, 5),
        end_date=datetime(2024, 1, 1, 10, 6),
        value=75,
      ),
    ]

    result = DataConverter.heart_rate_to_df(records)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert list(result.columns) == ["timestamp", "value", "source", "unit"]
    assert result.iloc[0]["value"] == 70.0
    assert result.iloc[1]["value"] == 75.0
    assert result.iloc[0]["source"] == "TestWatch"
    assert result.iloc[0]["unit"] == "count/min"

  def test_resting_hr_to_df_empty(self):
    """Test converting empty resting HR records to DataFrame."""
    result = DataConverter.resting_hr_to_df([])

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["date", "value", "source"]
    assert len(result) == 0

  def test_resting_hr_to_df_with_data(self):
    """Test converting resting HR records to DataFrame."""
    records = [
      QuantityRecord(
        type="HKQuantityTypeIdentifierRestingHeartRate",
        source_name="TestWatch",
        source_version="1.0",
        device="TestDevice",
        unit="count/min",
        creation_date=datetime(2024, 1, 1, 8, 0),
        start_date=datetime(2024, 1, 1, 8, 0),
        end_date=datetime(2024, 1, 1, 8, 1),
        value=65,
      ),
      QuantityRecord(
        type="HKQuantityTypeIdentifierRestingHeartRate",
        source_name="TestWatch",
        source_version="1.0",
        device="TestDevice",
        unit="count/min",
        creation_date=datetime(2024, 1, 2, 8, 0),
        start_date=datetime(2024, 1, 2, 8, 0),
        end_date=datetime(2024, 1, 2, 8, 1),
        value=68,
      ),
    ]

    result = DataConverter.resting_hr_to_df(records)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert list(result.columns) == [
      "date",
      "timestamp",
      "value",
      "source",
      "unit",
    ]
    assert result.iloc[0]["value"] == 65.0
    assert result.iloc[1]["value"] == 68.0

  def test_hrv_to_df_empty(self):
    """Test converting empty HRV records to DataFrame."""
    result = DataConverter.hrv_to_df([])

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["date", "value", "source"]
    assert len(result) == 0

  def test_hrv_to_df_with_data(self):
    """Test converting HRV records to DataFrame."""
    records = [
      QuantityRecord(
        type="HKQuantityTypeIdentifierHeartRateVariabilitySDNN",
        source_name="TestWatch",
        source_version="1.0",
        device="TestDevice",
        unit="ms",
        creation_date=datetime(2024, 1, 1, 8, 0),
        start_date=datetime(2024, 1, 1, 8, 0),
        end_date=datetime(2024, 1, 1, 8, 1),
        value=45.2,
      ),
    ]

    result = DataConverter.hrv_to_df(records)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert list(result.columns) == [
      "date",
      "timestamp",
      "value",
      "source",
      "unit",
    ]
    assert result.iloc[0]["value"] == 45.2
    assert result.iloc[0]["unit"] == "ms"

  def test_sleep_sessions_to_df_empty(self):
    """Test converting empty sleep sessions to DataFrame."""
    result = DataConverter.sleep_sessions_to_df([])

    assert isinstance(result, pd.DataFrame)
    expected_columns = [
      "date",
      "start_time",
      "end_time",
      "total_duration",
      "sleep_duration",
      "efficiency",
      "deep_sleep",
      "rem_sleep",
      "awakenings",
    ]
    assert list(result.columns) == expected_columns
    assert len(result) == 0

  def test_aggregate_heart_rate_by_hour_empty(self):
    """Test aggregating empty heart rate data by hour."""
    df = pd.DataFrame()
    result = DataConverter.aggregate_heart_rate_by_hour(df)

    assert isinstance(result, pd.DataFrame)
    expected_columns = ["hour", "mean_hr", "min_hr", "max_hr", "count"]
    assert list(result.columns) == expected_columns
    assert len(result) == 0

  def test_aggregate_heart_rate_by_hour_with_data(self):
    """Test aggregating heart rate data by hour."""
    # Create test data
    timestamps = [
      datetime(2024, 1, 1, 10, 0),
      datetime(2024, 1, 1, 10, 15),
      datetime(2024, 1, 1, 10, 30),
      datetime(2024, 1, 1, 11, 0),
    ]
    values = [70, 75, 72, 80]

    df = pd.DataFrame({"timestamp": timestamps, "value": values})

    result = DataConverter.aggregate_heart_rate_by_hour(df)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2  # Two different hours
    assert list(result.columns) == [
      "hour",
      "mean_hr",
      "min_hr",
      "max_hr",
      "count",
    ]

    # Check first hour (10:00)
    hour_10 = result[result["hour"] == datetime(2024, 1, 1, 10, 0)]
    assert len(hour_10) == 1
    assert hour_10.iloc[0]["mean_hr"] == 72.3  # (70+75+72)/3
    assert hour_10.iloc[0]["min_hr"] == 70.0
    assert hour_10.iloc[0]["max_hr"] == 75.0
    assert hour_10.iloc[0]["count"] == 3

  def test_aggregate_heart_rate_by_day_empty(self):
    """Test aggregating empty heart rate data by day."""
    df = pd.DataFrame()
    result = DataConverter.aggregate_heart_rate_by_day(df)

    assert isinstance(result, pd.DataFrame)
    expected_columns = ["date", "mean_hr", "min_hr", "max_hr", "count"]
    assert list(result.columns) == expected_columns
    assert len(result) == 0

  def test_aggregate_heart_rate_by_day_with_data(self):
    """Test aggregating heart rate data by day."""
    # Create test data spanning two days
    timestamps = [
      datetime(2024, 1, 1, 10, 0),
      datetime(2024, 1, 1, 15, 0),
      datetime(2024, 1, 2, 10, 0),
    ]
    values = [70, 75, 80]

    df = pd.DataFrame({"timestamp": timestamps, "value": values})

    result = DataConverter.aggregate_heart_rate_by_day(df)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2  # Two different days
    assert list(result.columns) == [
      "date",
      "mean_hr",
      "min_hr",
      "max_hr",
      "count",
    ]

  def test_aggregate_sleep_by_day_empty(self):
    """Test aggregating empty sleep data by day."""
    df = pd.DataFrame()
    result = DataConverter.aggregate_sleep_by_day(df)

    assert isinstance(result, pd.DataFrame)
    expected_columns = [
      "date",
      "total_duration",
      "sleep_duration",
      "efficiency",
      "deep_sleep",
      "rem_sleep",
      "awakenings",
    ]
    assert list(result.columns) == expected_columns
    assert len(result) == 0

  def test_prepare_heart_rate_zones_empty(self):
    """Test preparing heart rate zones for empty data."""
    df = pd.DataFrame()
    result = DataConverter.prepare_heart_rate_zones(df)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0

  def test_prepare_heart_rate_zones_with_data(self):
    """Test preparing heart rate zones with data."""
    # Create test data with various heart rates
    values = [60, 80, 120, 150, 180, 200]  # Different zones
    df = pd.DataFrame({"value": values})

    result = DataConverter.prepare_heart_rate_zones(df, age=30)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 5  # 5 zones
    assert list(result.columns) == [
      "zone",
      "count",
      "percentage",
      "min_hr",
      "max_hr",
    ]

    # Check that zones are properly defined
    zones = result["zone"].tolist()
    assert "zone1" in zones
    assert "zone5" in zones

    # Check that percentages are reasonable (each zone has some percentage)
    assert all(result["percentage"] >= 0)
    assert all(result["percentage"] <= 100)
    # Check specific zone distributions for our test data
    # zone1 should have 2 values (33.33%), others should have 1 or 0
    assert abs(result["percentage"].iloc[0] - 33.33) < 1.0  # zone1
    assert result["count"].sum() == 5  # 200 is outside all zones

  def test_prepare_sleep_stages_distribution_empty(self):
    """Test preparing sleep stages distribution for empty data."""
    df = pd.DataFrame()
    result = DataConverter.prepare_sleep_stages_distribution(df)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["stage", "duration", "percentage"]
    assert len(result) == 0

  def test_prepare_sleep_stages_distribution_with_data(self):
    """Test preparing sleep stages distribution with data."""
    # Create test sleep data
    data = {
      "deep_sleep": [60, 70],  # 130 minutes total
      "rem_sleep": [80, 90],  # 170 minutes total
      "light_sleep": [100, 110],  # 210 minutes total
    }
    df = pd.DataFrame(data)

    result = DataConverter.prepare_sleep_stages_distribution(df)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3  # 3 stages
    assert list(result.columns) == ["stage", "duration", "percentage", "color"]

    # Check stages
    stages = result["stage"].tolist()
    assert "Deep Sleep" in stages
    assert "REM Sleep" in stages
    assert "Light Sleep" in stages

    # Check total duration
    total_duration = result["duration"].sum()
    assert total_duration == 510  # 130 + 170 + 210

    # Check percentages sum to 100
    total_percentage = result["percentage"].sum()
    assert abs(total_percentage - 100.0) < 1.0

  def test_sample_data_for_performance_no_sampling(self):
    """Test sampling when data is already small."""
    df = pd.DataFrame({"value": list(range(100))})
    result = DataConverter.sample_data_for_performance(df, max_points=1000)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 100  # No sampling needed
    assert result.equals(df)

  def test_sample_data_for_performance_with_sampling(self):
    """Test sampling when data is large."""
    df = pd.DataFrame({"value": list(range(20000))})
    result = DataConverter.sample_data_for_performance(df, max_points=100)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 100  # Sampled down
    assert len(result) < len(df)

  def test_aggregate_heart_rate_by_hour_missing_timestamp(self):
    """Test aggregating heart rate data with missing timestamp column."""
    df = pd.DataFrame({"value": [70, 75, 80]})
    result = DataConverter.aggregate_heart_rate_by_hour(df)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0  # Should return empty DataFrame

  def test_aggregate_heart_rate_by_day_missing_timestamp(self):
    """Test aggregating heart rate data by day with missing timestamp column."""
    df = pd.DataFrame({"value": [70, 75, 80]})
    result = DataConverter.aggregate_heart_rate_by_day(df)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0  # Should return empty DataFrame
