"""Unit tests for heart rate processor."""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from src.analyzers.anomaly import AnomalyDetector
from src.analyzers.statistical import StatisticalAnalyzer
from src.core.data_models import QuantityRecord
from src.processors.heart_rate import (
  CardioFitnessAnalysis,
  HeartRateAnalysisReport,
  HeartRateAnalyzer,
  HRVAnalysis,
  RestingHRAnalysis,
)


class TestHeartRateAnalyzer:
  """HeartRateAnalyzer tests."""

  @pytest.fixture
  def analyzer(self):
    """Create HeartRateAnalyzer fixture."""
    return HeartRateAnalyzer()

  @pytest.fixture
  def sample_hr_records(self):
    """Create sample heart rate records."""
    base_time = datetime(2024, 1, 1, 8, 0, 0)
    records = []

    # Generate 30 days of mock data.
    for day in range(30):
      for hour in range(24):
        # Simulate daily variation.
        base_hr = 70
        if 6 <= hour <= 8:  # Morning exercise.
          hr_variation = 20
        elif 22 <= hour <= 24:  # Night rest.
          hr_variation = -10
        else:
          hr_variation = 0

        hr_value = base_hr + hr_variation + (day % 5)  # Add slight variance.

        record_time = base_time + timedelta(days=day, hours=hour)
        records.append(
          QuantityRecord(
            type="HKQuantityTypeIdentifierHeartRate",
            source_name="Apple Watch",
            start_date=record_time,
            end_date=record_time + timedelta(minutes=1),
            creation_date=record_time,
            value=float(hr_value),
            unit="count/min",
            source_version="1.0",
            device="Apple Watch Series 8",
          )
        )

    return records

  @pytest.fixture
  def sample_resting_hr_records(self):
    """Create sample resting heart rate records."""
    base_time = datetime(2024, 1, 1, 8, 0, 0)
    records = []

    # Generate 30 days of resting HR data with decline.
    for day in range(30):
      resting_hr = 72 - (day * 0.1)  # Gradual decline.
      record_time = base_time + timedelta(days=day)

      records.append(
        QuantityRecord(
          type="HKQuantityTypeIdentifierRestingHeartRate",
          source_name="Apple Watch",
          start_date=record_time,
          end_date=record_time + timedelta(days=1),
          creation_date=record_time,
          value=float(resting_hr),
          unit="count/min",
          source_version="1.0",
          device="Apple Watch Series 8",
        )
      )

    return records

  @pytest.fixture
  def sample_hrv_records(self):
    """Create sample HRV records."""
    base_time = datetime(2024, 1, 1, 8, 0, 0)
    records = []

    # Generate 30 days of improving HRV data.
    for day in range(30):
      hrv_value = 35 + (day * 0.2)  # Gradual improvement.
      record_time = base_time + timedelta(days=day)

      records.append(
        QuantityRecord(
          type="HKQuantityTypeIdentifierHeartRateVariabilitySDNN",
          source_name="Apple Watch",
          start_date=record_time,
          end_date=record_time + timedelta(days=1),
          creation_date=record_time,
          value=float(hrv_value),
          unit="ms",
          source_version="1.0",
          device="Apple Watch Series 8",
        )
      )

    return records

  def test_initialization(self, analyzer):
    """Test initialization."""
    assert isinstance(analyzer, HeartRateAnalyzer)
    assert isinstance(analyzer.stat_analyzer, StatisticalAnalyzer)
    assert isinstance(analyzer.anomaly_detector, AnomalyDetector)

  def test_analyze_resting_heart_rate(self, analyzer, sample_resting_hr_records):
    """Test resting heart rate analysis."""
    analysis = analyzer.analyze_resting_heart_rate(sample_resting_hr_records)

    assert isinstance(analysis, RestingHRAnalysis)
    assert analysis.current_value < 72.0  # Should decline.
    assert (
      analysis.baseline_value >= analysis.current_value
    )  # Baseline should be >= current (declining trend).
    assert analysis.change_from_baseline <= 0  # Negative or zero change.
    assert analysis.trend_direction in ["increasing", "decreasing", "stable"]
    assert analysis.health_rating in ["excellent", "good", "fair", "poor"]

  def test_analyze_hrv(self, analyzer, sample_hrv_records):
    """Test HRV analysis."""
    analysis = analyzer.analyze_hrv(sample_hrv_records)

    assert isinstance(analysis, HRVAnalysis)
    assert analysis.current_sdnn > 35.0  # Should improve.
    assert (
      analysis.baseline_sdnn <= analysis.current_sdnn
    )  # Baseline should be <= current.
    assert analysis.change_from_baseline >= 0  # Positive or zero change.
    assert analysis.trend_direction in ["improving", "declining", "stable"]
    assert analysis.stress_level in ["low", "moderate", "high", "very_high"]
    assert analysis.recovery_status in ["poor", "fair", "good", "excellent"]

  def test_analyze_cardio_fitness_no_vo2_data(self, analyzer):
    """Test cardio fitness analysis with no VO2 data."""
    analysis = analyzer.analyze_cardio_fitness([])

    assert analysis is None

  def test_analyze_cardio_fitness_with_vo2_data(self, analyzer):
    """Test cardio fitness analysis with VO2 data."""
    # Create analyzer with age/gender.
    analyzer_with_age = HeartRateAnalyzer(age=30, gender="male")

    vo2_records = [
      QuantityRecord(
        type="HKQuantityTypeIdentifierVO2Max",
        source_name="Apple Watch",
        start_date=datetime(2024, 1, 15),
        end_date=datetime(2024, 1, 16),
        creation_date=datetime(2024, 1, 15),
        value=42.0,
        unit="mL/min·kg",
        source_version="1.0",
        device="Apple Watch Series 8",
      )
    ]

    analysis = analyzer_with_age.analyze_cardio_fitness(vo2_records)

    assert isinstance(analysis, CardioFitnessAnalysis)
    assert analysis.current_vo2_max == 42.0
    assert analysis.age_adjusted_rating in [
      "superior",
      "excellent",
      "good",
      "fair",
      "poor",
    ]
    assert isinstance(analysis.fitness_percentile, (int, float))
    assert isinstance(analysis.improvement_potential, (int, float))
    assert isinstance(analysis.training_recommendations, list)

  @patch.object(StatisticalAnalyzer, "analyze_trend")
  def test_analyze_comprehensive(self, mock_trend, analyzer, sample_hr_records):
    """Test comprehensive analysis."""
    # Mock trend analysis result.
    mock_trend.return_value = Mock(
      slope=-0.1, r_squared=0.8, trend_direction="decreasing"
    )

    report = analyzer.analyze_comprehensive(sample_hr_records)

    assert isinstance(report, HeartRateAnalysisReport)
    assert report.record_count == len(sample_hr_records)
    assert isinstance(report.data_quality_score, float)
    assert 0 <= report.data_quality_score <= 1

  def test_analyze_comprehensive_empty_records(self, analyzer):
    """Test comprehensive analysis with empty records."""
    report = analyzer.analyze_comprehensive([])

    assert isinstance(report, HeartRateAnalysisReport)
    assert report.record_count == 0

  def test_data_quality_assessment(self, analyzer, sample_hr_records):
    """Test data quality assessment."""
    quality = analyzer._assess_data_quality(sample_hr_records)

    assert isinstance(quality, float)
    assert 0 <= quality <= 1

  def test_data_quality_assessment_empty(self, analyzer):
    """Test data quality assessment with empty data."""
    quality = analyzer._assess_data_quality([])

    assert quality == 0.0
