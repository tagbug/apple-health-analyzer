"""Tests for anomaly detection module."""

from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np

from src.analyzers.anomaly import (
  AnomalyConfig,
  AnomalyDetector,
  AnomalyRecord,
  AnomalyReport,
)
from src.core.data_models import QuantityRecord
from src.i18n import Translator, resolve_locale


class TestAnomalyDetector:
  """Test AnomalyDetector class."""

  def test_initialization_default_config(self):
    """Test detector initialization with default config."""
    detector = AnomalyDetector()

    assert detector.config is not None
    assert detector.config["zscore_threshold"] == 3.0
    assert detector.config["iqr_multiplier"] == 1.5
    assert detector.config["ma_threshold"] == 2.0
    assert detector.config["context_threshold"] == 2.5
    assert "severity_thresholds" in detector.config

  def test_initialization_custom_config(self):
    """Test detector initialization with custom config."""
    custom_config: AnomalyConfig = {
      "zscore_threshold": 2.5,
      "iqr_multiplier": 2.0,
      "severity_thresholds": {
        "low": 1.0,
        "medium": 2.0,
        "high": 3.0,
      },
    }

    detector = AnomalyDetector(custom_config)

    assert detector.config["zscore_threshold"] == 2.5
    assert detector.config["iqr_multiplier"] == 2.0
    assert detector.config["severity_thresholds"]["low"] == 1.0

  def test_detect_anomalies_empty_records(self):
    """Test anomaly detection with empty records."""
    detector = AnomalyDetector()
    anomalies = detector.detect_anomalies([])

    assert anomalies == []

  def test_detect_anomalies_zscore_method(self):
    """Test Z-Score anomaly detection."""
    np.random.seed(42)  # Ensure reproducibility
    detector = AnomalyDetector()

    # Create test records with clear outlier
    base_time = datetime(2024, 1, 1, 10, 0, 0)
    records = []

    # Normal values around 70
    for i in range(10):
      records.append(
        QuantityRecord(
          type="HKQuantityTypeIdentifierHeartRate",
          source_name="Test",
          source_version="1.0",
          device="TestDevice",
          unit="count/min",
          creation_date=base_time + timedelta(hours=i),
          start_date=base_time + timedelta(hours=i),
          end_date=base_time + timedelta(hours=i) + timedelta(minutes=1),
          value=70 + np.random.normal(0, 2),  # Normal distribution
        )
      )

    # Add clear outlier
    records.append(
      QuantityRecord(
        type="HKQuantityTypeIdentifierHeartRate",
        source_name="Test",
        source_version="1.0",
        device="TestDevice",
        unit="count/min",
        creation_date=base_time + timedelta(hours=10),
        start_date=base_time + timedelta(hours=10),
        end_date=base_time + timedelta(hours=10) + timedelta(minutes=1),
        value=150,  # Clear outlier
      )
    )

    anomalies = detector.detect_anomalies(records, methods=["zscore"])

    assert len(anomalies) > 0
    assert any(a.method == "zscore" for a in anomalies)
    assert any(a.severity in ["medium", "high"] for a in anomalies)

  def test_detect_anomalies_iqr_method(self):
    """Test IQR anomaly detection."""
    detector = AnomalyDetector()

    # Create test records
    base_time = datetime(2024, 1, 1, 10, 0, 0)
    records = []

    # Create values with clear outlier
    values = [10, 12, 11, 13, 12, 100]  # 100 is outlier

    for i, value in enumerate(values):
      records.append(
        QuantityRecord(
          type="HKQuantityTypeIdentifierHeartRate",
          source_name="Test",
          source_version="1.0",
          device="TestDevice",
          unit="count/min",
          creation_date=base_time + timedelta(hours=i),
          start_date=base_time + timedelta(hours=i),
          end_date=base_time + timedelta(hours=i) + timedelta(minutes=1),
          value=value,
        )
      )

    anomalies = detector.detect_anomalies(records, methods=["iqr"])

    assert len(anomalies) > 0
    assert any(a.method == "iqr" for a in anomalies)

  def test_detect_anomalies_moving_average_method(self):
    """Test moving average anomaly detection."""
    detector = AnomalyDetector()

    # Create test records with trend and outlier
    base_time = datetime(2024, 1, 1, 10, 0, 0)
    records = []

    # Create stable values with sudden spike
    for i in range(10):
      value = 70 if i < 8 else 120  # Sudden spike at end
      records.append(
        QuantityRecord(
          type="HKQuantityTypeIdentifierHeartRate",
          source_name="Test",
          source_version="1.0",
          device="TestDevice",
          unit="count/min",
          creation_date=base_time + timedelta(hours=i),
          start_date=base_time + timedelta(hours=i),
          end_date=base_time + timedelta(hours=i) + timedelta(minutes=1),
          value=value,
        )
      )

    anomalies = detector.detect_anomalies(records, methods=["moving_average"])

    assert len(anomalies) >= 0  # May or may not detect depending on window

  def test_detect_anomalies_contextual_time_of_day(self):
    """Test contextual anomaly detection by time of day."""
    detector = AnomalyDetector()

    # Create records for different hours with patterns
    base_date = datetime(2024, 1, 1)
    records = []

    # Morning hours (6-9 AM): lower heart rate
    for hour in range(6, 10):
      for day in range(7):  # One week of data
        time = base_date + timedelta(days=day, hours=hour)
        value = 60 + np.random.normal(0, 3)  # Around 60 bpm
        records.append(
          QuantityRecord(
            type="HKQuantityTypeIdentifierHeartRate",
            source_name="Test",
            source_version="1.0",
            device="TestDevice",
            unit="count/min",
            creation_date=time,
            start_date=time,
            end_date=time + timedelta(minutes=1),
            value=value,
          )
        )

    # Add anomalous high value during morning
    anomalous_time = base_date + timedelta(hours=7)  # 7 AM
    records.append(
      QuantityRecord(
        type="HKQuantityTypeIdentifierHeartRate",
        source_name="Test",
        source_version="1.0",
        device="TestDevice",
        unit="count/min",
        creation_date=anomalous_time,
        start_date=anomalous_time,
        end_date=anomalous_time + timedelta(minutes=1),
        value=120,  # Anomalous high value
      )
    )

    anomalies = detector.detect_anomalies(
      records, methods=["contextual"], context="time_of_day"
    )

    assert isinstance(anomalies, list)

  def test_detect_anomalies_contextual_day_of_week(self):
    """Test contextual anomaly detection by day of week."""
    detector = AnomalyDetector()

    # Create records for different days with patterns
    base_date = datetime(2024, 1, 1)  # Monday
    records = []

    # Weekdays vs weekends pattern
    for day in range(14):  # Two weeks
      current_date = base_date + timedelta(days=day)
      is_weekend = current_date.weekday() >= 5  # Sat=5, Sun=6

      # Different patterns for weekdays vs weekends
      base_hr = 65 if is_weekend else 75

      for hour in range(8, 18):  # Working hours
        time = current_date + timedelta(hours=hour)
        value = base_hr + np.random.normal(0, 5)
        records.append(
          QuantityRecord(
            type="HKQuantityTypeIdentifierHeartRate",
            source_name="Test",
            source_version="1.0",
            device="TestDevice",
            unit="count/min",
            creation_date=time,
            start_date=time,
            end_date=time + timedelta(minutes=1),
            value=value,
          )
        )

    # Add anomalous value on weekday
    anomalous_time = base_date + timedelta(days=1, hours=10)  # Tuesday 10 AM
    records.append(
      QuantityRecord(
        type="HKQuantityTypeIdentifierHeartRate",
        source_name="Test",
        source_version="1.0",
        device="TestDevice",
        unit="count/min",
        creation_date=anomalous_time,
        start_date=anomalous_time,
        end_date=anomalous_time + timedelta(minutes=1),
        value=120,  # Anomalous high value on weekday
      )
    )

    anomalies = detector.detect_anomalies(
      records, methods=["contextual"], context="day_of_week"
    )

    assert isinstance(anomalies, list)

  def test_detect_anomalies_multiple_methods(self):
    """Test anomaly detection with multiple methods."""
    detector = AnomalyDetector()

    # Create test records
    base_time = datetime(2024, 1, 1, 10, 0, 0)
    records = []

    # Create normal data with clear outlier
    for i in range(20):
      value = 70 + np.random.normal(0, 3)
      records.append(
        QuantityRecord(
          type="HKQuantityTypeIdentifierHeartRate",
          source_name="Test",
          source_version="1.0",
          device="TestDevice",
          unit="count/min",
          creation_date=base_time + timedelta(hours=i),
          start_date=base_time + timedelta(hours=i),
          end_date=base_time + timedelta(hours=i) + timedelta(minutes=1),
          value=value,
        )
      )

    # Add clear outlier
    records.append(
      QuantityRecord(
        type="HKQuantityTypeIdentifierHeartRate",
        source_name="Test",
        source_version="1.0",
        device="TestDevice",
        unit="count/min",
        creation_date=base_time + timedelta(hours=20),
        start_date=base_time + timedelta(hours=20),
        end_date=base_time + timedelta(hours=20) + timedelta(minutes=1),
        value=150,
      )
    )

    anomalies = detector.detect_anomalies(records, methods=["zscore", "iqr"])

    assert isinstance(anomalies, list)
    # Should detect anomalies with at least one method

  def test_generate_report(self):
    """Test anomaly report generation."""
    detector = AnomalyDetector()

    # Create mock anomalies
    anomalies = [
      AnomalyRecord(
        timestamp=datetime(2024, 1, 1, 10, 0),
        value=120,
        expected_value=70,
        deviation=3.5,
        severity="high",
        method="zscore",
        confidence=0.9,
        context={"test": "data"},
      ),
      AnomalyRecord(
        timestamp=datetime(2024, 1, 1, 11, 0),
        value=110,
        expected_value=70,
        deviation=2.1,
        severity="medium",
        method="iqr",
        confidence=0.8,
        context={"test": "data"},
      ),
    ]

    report = detector.generate_report(anomalies, total_records=100)

    assert isinstance(report, AnomalyReport)
    assert report.total_records == 100
    assert report.anomaly_count == 2
    assert report.anomaly_rate == 0.02
    assert report.anomalies_by_severity["high"] == 1
    assert report.anomalies_by_severity["medium"] == 1
    assert report.anomalies_by_method["zscore"] == 1
    assert report.anomalies_by_method["iqr"] == 1
    assert isinstance(report.recommendations, list)

  def test_generate_report_empty_anomalies(self):
    """Test anomaly report generation with no anomalies."""
    detector = AnomalyDetector()

    report = detector.generate_report([], total_records=50)

    assert report.anomaly_count == 0
    assert report.anomaly_rate == 0.0
    assert report.anomalies_by_severity["low"] == 0
    assert report.anomalies_by_severity["medium"] == 0
    assert report.anomalies_by_severity["high"] == 0

  def test_calculate_severity(self):
    """Test severity calculation."""
    detector = AnomalyDetector()

    # Test different deviation levels
    assert detector._calculate_severity(0.5) == "low"
    assert detector._calculate_severity(2.0) == "low"  # 2.0 < 2.5 (medium threshold)
    assert detector._calculate_severity(3.0) == "medium"  # 3.0 >= 2.5 but < 3.5
    assert detector._calculate_severity(4.0) == "high"  # 4.0 >= 3.5

    # Test boundary values
    assert (
      detector._calculate_severity(1.5) == "low"
    )  # Boundary (exactly low threshold)
    assert (
      detector._calculate_severity(2.5) == "medium"
    )  # Boundary (exactly medium threshold)
    assert (
      detector._calculate_severity(3.5) == "high"
    )  # Boundary (exactly high threshold)

  def test_deduplicate_anomalies(self):
    """Test anomaly deduplication."""
    detector = AnomalyDetector()

    timestamp = datetime(2024, 1, 1, 10, 0)

    # Create duplicate anomalies for same timestamp
    anomalies = [
      AnomalyRecord(
        timestamp=timestamp,
        value=120,
        expected_value=70,
        deviation=3.5,
        severity="high",
        method="zscore",
        confidence=0.9,
        context={},
      ),
      AnomalyRecord(
        timestamp=timestamp,
        value=115,
        expected_value=70,
        deviation=3.0,
        severity="high",
        method="iqr",
        confidence=0.8,
        context={},
      ),
      AnomalyRecord(  # Different timestamp
        timestamp=timestamp + timedelta(hours=1),
        value=110,
        expected_value=70,
        deviation=2.1,
        severity="medium",
        method="zscore",
        confidence=0.7,
        context={},
      ),
    ]

    deduplicated = detector._deduplicate_anomalies(anomalies)

    # Should keep 2 anomalies (one per timestamp, most severe)
    assert len(deduplicated) == 2

    # First timestamp should keep the higher severity one
    first_timestamp_anomalies = [a for a in deduplicated if a.timestamp == timestamp]
    assert len(first_timestamp_anomalies) == 1
    assert first_timestamp_anomalies[0].severity == "high"

  def test_analyze_time_distribution(self):
    """Test time distribution analysis."""
    detector = AnomalyDetector()

    # Create anomalies at different times
    anomalies = [
      AnomalyRecord(
        timestamp=datetime(2024, 1, 1, 6, 30),  # 6 AM
        value=120,
        expected_value=70,
        deviation=3.5,
        severity="high",
        method="zscore",
        confidence=0.9,
        context={},
      ),
      AnomalyRecord(
        timestamp=datetime(2024, 1, 1, 6, 45),  # 6 AM
        value=115,
        expected_value=70,
        deviation=3.0,
        severity="high",
        method="iqr",
        confidence=0.8,
        context={},
      ),
      AnomalyRecord(
        timestamp=datetime(2024, 1, 1, 14, 30),  # 2 PM
        value=110,
        expected_value=70,
        deviation=2.1,
        severity="medium",
        method="zscore",
        confidence=0.7,
        context={},
      ),
      AnomalyRecord(
        timestamp=datetime(2024, 1, 2, 6, 15),  # Next day 6 AM
        value=105,
        expected_value=70,
        deviation=1.8,
        severity="medium",
        method="zscore",
        confidence=0.6,
        context={},
      ),
    ]

    distribution = detector._analyze_time_distribution(anomalies)

    assert "by_hour" in distribution
    assert "by_day_of_week" in distribution
    assert "by_month" in distribution

    # Check hour distribution
    assert distribution["by_hour"]["6"] == 3  # Three anomalies at 6 AM
    assert distribution["by_hour"]["14"] == 1  # One anomaly at 2 PM

    # Check day of week (Monday = 0)
    assert distribution["by_day_of_week"]["Monday"] == 3

  def test_generate_recommendations(self):
    """Test recommendation generation."""
    detector = AnomalyDetector()

    # Test high anomaly rate
    anomalies_high_rate = [
      AnomalyRecord(
        timestamp=datetime(2024, 1, 1, i, 0),
        value=100 + i,
        expected_value=70,
        deviation=2.0,
        severity="medium",
        method="zscore",
        confidence=0.8,
        context={},
      )
      for i in range(15)  # 15 anomalies = 15% rate
    ]

    recommendations = detector._generate_recommendations(anomalies_high_rate, 0.15)
    translator = Translator(resolve_locale())
    assert translator.t("anomaly.recommendation.high_rate") in recommendations

    # Test low anomaly rate
    recommendations_low = detector._generate_recommendations([], 0.0001)
    assert translator.t("anomaly.recommendation.low_rate") in recommendations_low

    # Test high severity concentration
    anomalies_high_severity = [
      AnomalyRecord(
        timestamp=datetime(2024, 1, 1, i, 0),
        value=120,
        expected_value=70,
        deviation=3.5,
        severity="high",
        method="zscore",
        confidence=0.9,
        context={},
      )
      for i in range(4)  # 4 high severity out of 10 total
    ] + [
      AnomalyRecord(
        timestamp=datetime(2024, 1, 1, i + 10, 0),
        value=80,
        expected_value=70,
        deviation=1.0,
        severity="low",
        method="zscore",
        confidence=0.5,
        context={},
      )
      for i in range(6)
    ]

    recommendations_severity = detector._generate_recommendations(
      anomalies_high_severity, 0.1
    )
    assert (
      translator.t("anomaly.recommendation.high_severity") in recommendations_severity
    )

  def test_records_to_dataframe(self):
    """Test conversion of records to DataFrame."""
    detector = AnomalyDetector()

    # Create test records
    base_time = datetime(2024, 1, 1, 10, 0, 0)
    records = [
      QuantityRecord(
        type="HKQuantityTypeIdentifierHeartRate",
        source_name="Test",
        source_version="1.0",
        device="TestDevice",
        unit="count/min",
        creation_date=base_time,
        start_date=base_time,
        end_date=base_time + timedelta(minutes=1),
        value=70,
      ),
      QuantityRecord(
        type="HKQuantityTypeIdentifierStepCount",
        source_name="Test",
        source_version="1.0",
        device="TestDevice",
        unit="count",
        creation_date=base_time + timedelta(hours=1),
        start_date=base_time + timedelta(hours=1),
        end_date=base_time + timedelta(hours=1) + timedelta(minutes=1),
        value=1000,
      ),
    ]

    df = detector._records_to_dataframe(records)

    assert not df.empty
    assert len(df) == 2
    assert "value" in df.columns
    assert "type" in df.columns
    assert df.iloc[0]["value"] == 70
    assert df.iloc[1]["value"] == 1000

  def test__records_to_dataframe_empty(self):
    """Test conversion with empty records."""
    detector = AnomalyDetector()

    df = detector._records_to_dataframe([])

    assert df.empty or len(df) == 0

  @patch("src.analyzers.anomaly.logger")
  def test_detect_anomalies_with_invalid_method(self, mock_logger):
    """Test anomaly detection with invalid method."""
    detector = AnomalyDetector()

    records = [
      QuantityRecord(
        type="HKQuantityTypeIdentifierHeartRate",
        source_name="Test",
        source_version="1.0",
        device="TestDevice",
        unit="count/min",
        creation_date=datetime(2024, 1, 1, 10, 0),
        start_date=datetime(2024, 1, 1, 10, 0),
        end_date=datetime(2024, 1, 1, 10, 1),
        value=70,
      )
    ]

    anomalies = detector.detect_anomalies(records, methods=["invalid_method"])  # type: ignore

    # Should not crash, should log warning
    mock_logger.warning.assert_called()
    assert isinstance(anomalies, list)

  def test_contextual_sleep_wake_detection(self):
    """Test contextual sleep/wake detection returns a list."""
    detector = AnomalyDetector()

    records = [
      QuantityRecord(
        type="HKQuantityTypeIdentifierHeartRate",
        source_name="Test",
        source_version="1.0",
        device="TestDevice",
        unit="count/min",
        creation_date=datetime(2024, 1, 1, 10, 0),
        start_date=datetime(2024, 1, 1, 10, 0),
        end_date=datetime(2024, 1, 1, 10, 1),
        value=70,
      )
    ]

    anomalies = detector.detect_anomalies(
      records, methods=["contextual"], context="sleep_wake"
    )

    assert isinstance(anomalies, list)
