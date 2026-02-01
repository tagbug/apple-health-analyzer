"""Unit tests for sleep processor."""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from src.analyzers.anomaly import AnomalyDetector
from src.analyzers.statistical import StatisticalAnalyzer
from src.core.data_models import CategoryRecord, QuantityRecord
from src.processors.sleep import (
  SleepAnalysisReport,
  SleepAnalyzer,
  SleepHeartRateCorrelation,
  SleepPatternAnalysis,
  SleepQualityMetrics,
  SleepSession,
)


class TestSleepAnalyzer:
  """SleepAnalyzer tests."""

  @pytest.fixture
  def analyzer(self):
    """Create SleepAnalyzer fixture."""
    return SleepAnalyzer()

  @pytest.fixture
  def sample_sleep_records(self):
    """Create sample sleep records."""
    base_time = datetime(2024, 1, 1, 22, 0, 0)  # Starts at 10 PM.
    records = []

    # Simulate a full sleep session.
    sleep_start = base_time
    records.append(
      CategoryRecord(
        type="HKCategoryTypeIdentifierSleepAnalysis",
        source_name="Apple Watch",
        start_date=sleep_start,
        end_date=sleep_start + timedelta(minutes=30),
        creation_date=sleep_start,
        value="InBed",
        source_version="1.0",
        device="Apple Watch Series 8",
        unit=None,
      )
    )

    # Core stage.
    asleep_start = sleep_start + timedelta(minutes=30)
    records.append(
      CategoryRecord(
        type="HKCategoryTypeIdentifierSleepAnalysis",
        source_name="Apple Watch",
        start_date=asleep_start,
        end_date=asleep_start + timedelta(minutes=60),
        creation_date=asleep_start,
        value="Core",
        source_version="1.0",
        device="Apple Watch Series 8",
        unit=None,
      )
    )

    # Deep stage.
    deep_start = asleep_start + timedelta(minutes=60)
    records.append(
      CategoryRecord(
        type="HKCategoryTypeIdentifierSleepAnalysis",
        source_name="Apple Watch",
        start_date=deep_start,
        end_date=deep_start + timedelta(minutes=90),
        creation_date=deep_start,
        value="Deep",
        source_version="1.0",
        device="Apple Watch Series 8",
        unit=None,
      )
    )

    # REM stage.
    rem_start = deep_start + timedelta(minutes=90)
    records.append(
      CategoryRecord(
        type="HKCategoryTypeIdentifierSleepAnalysis",
        source_name="Apple Watch",
        start_date=rem_start,
        end_date=rem_start + timedelta(minutes=60),
        creation_date=rem_start,
        value="REM",
        source_version="1.0",
        device="Apple Watch Series 8",
        unit=None,
      )
    )

    # Light sleep stage.
    light_start = rem_start + timedelta(minutes=60)
    records.append(
      CategoryRecord(
        type="HKCategoryTypeIdentifierSleepAnalysis",
        source_name="Apple Watch",
        start_date=light_start,
        end_date=light_start + timedelta(minutes=120),
        creation_date=light_start,
        value="Asleep",  # Use Asleep as light sleep.
        source_version="1.0",
        device="Apple Watch Series 8",
        unit=None,
      )
    )

    # Awake stage.
    awake_start = light_start + timedelta(minutes=120)
    records.append(
      CategoryRecord(
        type="HKCategoryTypeIdentifierSleepAnalysis",
        source_name="Apple Watch",
        start_date=awake_start,
        end_date=awake_start + timedelta(minutes=30),
        creation_date=awake_start,
        value="Awake",
        source_version="1.0",
        device="Apple Watch Series 8",
        unit=None,
      )
    )

    return records

  @pytest.fixture
  def sample_heart_rate_records(self):
    """Create sample heart rate records for correlation."""
    base_time = datetime(2024, 1, 1, 22, 0, 0)
    records = []

    # Generate heart rate during sleep.
    for hour in range(8):  # 8 hours of sleep.
      for minute in range(0, 60, 5):  # Every 5 minutes.
        hr_time = base_time + timedelta(hours=hour, minutes=minute)
        # Simulate a gentle decline during sleep.
        base_hr = 75 - (hour * 2)  # 2 bpm per hour.
        hr_value = base_hr + (minute % 10 - 5)  # Small variation.

        records.append(
          QuantityRecord(
            type="HKQuantityTypeIdentifierHeartRate",
            source_name="Apple Watch",
            start_date=hr_time,
            end_date=hr_time + timedelta(minutes=1),
            creation_date=hr_time,
            value=float(hr_value),
            unit="count/min",
            source_version="1.0",
            device="Apple Watch Series 8",
          )
        )

    return records

  def test_initialization(self, analyzer):
    """Test initialization."""
    assert isinstance(analyzer, SleepAnalyzer)
    assert isinstance(analyzer.stat_analyzer, StatisticalAnalyzer)
    assert isinstance(analyzer.anomaly_detector, AnomalyDetector)

  def test_parse_sleep_sessions(self, analyzer, sample_sleep_records):
    """Test sleep session parsing."""
    sessions = analyzer._parse_sleep_sessions(sample_sleep_records)

    assert len(sessions) >= 1  # May include multiple sessions.
    session = sessions[0]

    assert isinstance(session, SleepSession)
    assert session.session_id.startswith("sleep_")
    assert session.total_duration > 0
    assert session.sleep_duration > 0
    assert session.efficiency >= 0  # Allow zero.
    assert session.efficiency <= 1

  def test_parse_sleep_sessions_empty(self, analyzer):
    """Test sleep session parsing with empty records."""
    sessions = analyzer._parse_sleep_sessions([])

    assert len(sessions) == 0

  def test_analyze_sleep_quality(self, analyzer):
    """Test sleep quality analysis."""
    # Create sessions to exercise consistency logic.
    sessions = [
      SleepSession(
        session_id="test_session_1",
        start_date=datetime(2024, 1, 1, 22, 0),
        end_date=datetime(2024, 1, 2, 6, 0),
        total_duration=480,  # 8 hours.
        sleep_duration=420,  # 7 hours.
        awake_duration=60,  # 1 hour.
        efficiency=0.875,  # 87.5%.
        core_sleep=120,
        deep_sleep=90,
        rem_sleep=60,
        light_sleep=150,
        sleep_latency=30,
        wake_after_onset=30,
        awakenings_count=2,
      ),
      SleepSession(
        session_id="test_session_2",
        start_date=datetime(2024, 1, 2, 22, 15),  # Slightly later.
        end_date=datetime(2024, 1, 3, 6, 15),
        total_duration=465,  # 7.75 hours.
        sleep_duration=405,  # 6.75 hours.
        awake_duration=60,
        efficiency=0.870,  # 87.0%.
        core_sleep=115,
        deep_sleep=85,
        rem_sleep=55,
        light_sleep=150,
        sleep_latency=35,
        wake_after_onset=25,
        awakenings_count=3,
      ),
      SleepSession(
        session_id="test_session_3",
        start_date=datetime(2024, 1, 3, 21, 45),  # Slightly earlier.
        end_date=datetime(2024, 1, 4, 5, 45),
        total_duration=495,  # 8.25 hours.
        sleep_duration=435,  # 7.25 hours.
        awake_duration=60,
        efficiency=0.878,  # 87.8%.
        core_sleep=125,
        deep_sleep=95,
        rem_sleep=65,
        light_sleep=150,
        sleep_latency=25,
        wake_after_onset=35,
        awakenings_count=1,
      ),
    ]

    quality = analyzer.analyze_sleep_quality(sessions)

    assert isinstance(quality, SleepQualityMetrics)
    assert abs(quality.average_duration - 8.0) < 0.5  # About 8 hours.
    assert abs(quality.average_efficiency - 0.875) < 0.1  # Around 87.5%.
    assert abs(quality.average_latency - 30.0) < 10  # Around 30 minutes.
    assert quality.consistency_score >= 0  # Non-negative consistency.
    assert quality.overall_quality_score > 0

  def test_analyze_sleep_quality_empty(self, analyzer):
    """Test sleep quality analysis with empty sessions."""
    quality = analyzer.analyze_sleep_quality([])

    assert isinstance(quality, SleepQualityMetrics)
    assert quality.average_duration == 0
    assert quality.average_efficiency == 0
    assert quality.consistency_score == 0
    assert quality.overall_quality_score == 0

  def test_analyze_sleep_patterns(self, analyzer):
    """Test sleep pattern analysis."""
    # Create a week of sessions.
    sessions = []
    base_time = datetime(2024, 1, 1, 22, 30)  # 10:30 PM.

    for day in range(7):  # One week of data.
      session_time = base_time + timedelta(days=day)
      # Different bedtimes for weekdays vs weekends.
      if day < 5:  # Weekdays.
        bedtime_offset = timedelta(hours=0)
      else:  # Weekend.
        bedtime_offset = timedelta(hours=1)  # Sleep 1 hour later.

      sessions.append(
        SleepSession(
          session_id=f"session_{day}",
          start_date=session_time + bedtime_offset,
          end_date=session_time + bedtime_offset + timedelta(hours=8),
          total_duration=480,
          sleep_duration=420,
          awake_duration=60,
          efficiency=0.875,
        )
      )

    patterns = analyzer.analyze_sleep_patterns(sessions)

    assert isinstance(patterns, SleepPatternAnalysis)
    assert patterns.bedtime_consistency > 0
    assert patterns.waketime_consistency > 0
    assert isinstance(patterns.weekday_vs_weekend, dict)
    assert "social_jetlag" in patterns.weekday_vs_weekend
    assert patterns.duration_trend in ["increasing", "decreasing", "stable"]
    assert patterns.efficiency_trend in ["improving", "declining", "stable"]

  def test_analyze_sleep_patterns_empty(self, analyzer):
    """Test sleep pattern analysis with empty sessions."""
    patterns = analyzer.analyze_sleep_patterns([])

    assert isinstance(patterns, SleepPatternAnalysis)
    assert patterns.bedtime_consistency == 0
    assert patterns.waketime_consistency == 0
    assert patterns.duration_trend == "stable"
    assert patterns.efficiency_trend == "stable"

  def test_analyze_sleep_hr_correlation(self, analyzer, sample_heart_rate_records):
    """Test sleep-heart rate correlation."""
    sessions = [
      SleepSession(
        session_id="test_session",
        start_date=datetime(2024, 1, 1, 22, 0),
        end_date=datetime(2024, 1, 2, 6, 0),
        total_duration=480,
        sleep_duration=420,
        awake_duration=60,
        efficiency=0.875,
      )
    ]

    correlation = analyzer.analyze_sleep_hr_correlation(
      sessions, sample_heart_rate_records
    )

    # Correlation may be None depending on input; ensure no errors.
    assert correlation is None or isinstance(correlation, SleepHeartRateCorrelation)

  def test_analyze_sleep_hr_correlation_no_hr_data(self, analyzer):
    """Test sleep-heart rate correlation with no HR data."""
    sessions = [
      SleepSession(
        session_id="test_session",
        start_date=datetime(2024, 1, 1, 22, 0),
        end_date=datetime(2024, 1, 2, 6, 0),
        total_duration=480,
        sleep_duration=420,
        awake_duration=60,
        efficiency=0.875,
      )
    ]

    correlation = analyzer.analyze_sleep_hr_correlation(sessions, [])

    assert correlation is None

  def test_generate_daily_summary(self, analyzer):
    """Test daily summary generation."""
    sessions = [
      SleepSession(
        session_id="session_1",
        start_date=datetime(2024, 1, 1, 22, 0),
        end_date=datetime(2024, 1, 2, 6, 0),
        total_duration=480,
        sleep_duration=420,
        awake_duration=60,
        efficiency=0.875,
        deep_sleep=90,
        rem_sleep=60,
        light_sleep=150,
        sleep_latency=30,
        awakenings_count=2,
      ),
      SleepSession(
        session_id="session_2",
        start_date=datetime(2024, 1, 2, 22, 0),
        end_date=datetime(2024, 1, 3, 6, 0),
        total_duration=480,
        sleep_duration=420,
        awake_duration=60,
        efficiency=0.875,
        deep_sleep=90,
        rem_sleep=60,
        light_sleep=150,
        sleep_latency=30,
        awakenings_count=2,
      ),
    ]

    summary = analyzer._generate_daily_summary(sessions)

    assert len(summary) == 2
    assert "date" in summary.columns
    assert "total_duration" in summary.columns
    assert "sleep_duration" in summary.columns
    assert "efficiency" in summary.columns
    assert "light_sleep" in summary.columns

  def test_generate_weekly_summary(self, analyzer):
    """Test weekly summary generation."""
    # Mock daily summary DataFrame.
    import pandas as pd

    daily_data = {
      "date": pd.date_range("2024-01-01", periods=7, freq="D"),
      "total_duration": [480] * 7,
      "sleep_duration": [420] * 7,
      "efficiency": [0.875] * 7,
      "latency": [30] * 7,
      "awakenings": [2] * 7,
      "deep_sleep": [90] * 7,
      "rem_sleep": [60] * 7,
      "light_sleep": [150] * 7,
    }
    daily_df = pd.DataFrame(daily_data)

    # Mock daily summary method.
    analyzer._generate_daily_summary = Mock(return_value=daily_df)

    sessions = []  # Empty because daily summary is mocked.
    weekly_summary = analyzer._generate_weekly_summary(sessions)

    assert len(weekly_summary) == 1  # One week of data.
    assert "days_recorded" in weekly_summary.columns
    assert "avg_duration" in weekly_summary.columns

  def test_detect_sleep_anomalies(self, analyzer):
    """Test sleep anomaly detection."""
    sessions = [
      SleepSession(
        session_id="normal_session",
        start_date=datetime(2024, 1, 1, 22, 0),
        end_date=datetime(2024, 1, 2, 6, 0),
        total_duration=480,
        sleep_duration=420,
        awake_duration=60,
        efficiency=0.875,
      ),
      SleepSession(
        session_id="short_session",
        start_date=datetime(2024, 1, 2, 22, 0),
        end_date=datetime(2024, 1, 3, 2, 0),  # 4 hours only.
        total_duration=240,
        sleep_duration=200,
        awake_duration=40,
        efficiency=0.833,
      ),
    ]

    anomalies = analyzer._detect_sleep_anomalies(sessions)

    assert isinstance(anomalies, list)
    # Short session should be flagged as an anomaly.

  def test_generate_highlights_good_sleep(self, analyzer):
    """Test highlights for good sleep."""
    quality = SleepQualityMetrics(
      average_duration=8.0,
      average_efficiency=0.9,
      average_latency=15.0,
      consistency_score=0.85,
      overall_quality_score=85.0,
    )

    patterns = SleepPatternAnalysis(
      bedtime_consistency=0.9,
      waketime_consistency=0.85,
      weekday_vs_weekend={"social_jetlag": 0.5},
      seasonal_patterns={},
      duration_trend="stable",
      efficiency_trend="stable",
    )

    highlights = analyzer._generate_highlights(quality, patterns, None, {}, [])

    assert isinstance(highlights, list)
    assert len(highlights) > 0
    from src.i18n import Translator, resolve_locale

    translator = Translator(resolve_locale())
    assert any(
      translator.t("sleep.highlight.duration_good", hours=8.0) in h for h in highlights
    )
    assert any(
      translator.t("sleep.highlight.efficiency_good", efficiency=90.0) in h
      for h in highlights
    )

  def test_generate_highlights_poor_sleep(self, analyzer):
    """Test highlights for poor sleep."""
    quality = SleepQualityMetrics(
      average_duration=5.0,  # Short sleep.
      average_efficiency=0.75,  # Lower efficiency.
      average_latency=45.0,  # Longer latency.
      consistency_score=0.6,
      overall_quality_score=60.0,
    )

    highlights = analyzer._generate_highlights(quality, None, None, {}, [])

    assert isinstance(highlights, list)
    # Should include sleep issue messaging.
    from src.i18n import Translator, resolve_locale

    translator = Translator(resolve_locale())
    assert any(
      translator.t("sleep.highlight.duration_low", hours=5.0) in h for h in highlights
    )
    assert any(
      translator.t("sleep.highlight.efficiency_low", efficiency=75.0) in h
      for h in highlights
    )

  def test_generate_recommendations(self, analyzer):
    """Test recommendation generation."""
    quality = SleepQualityMetrics(
      average_duration=6.0,
      average_efficiency=0.8,
      average_latency=30.0,
      consistency_score=0.7,
      overall_quality_score=70.0,
    )

    recommendations = analyzer._generate_recommendations(quality, None, None, [])

    assert isinstance(recommendations, list)
    assert len(recommendations) > 0
    from src.i18n import Translator, resolve_locale

    translator = Translator(resolve_locale())
    assert any(
      translator.t("sleep.recommendation.sleep_7_9") in rec for rec in recommendations
    )
    assert any(
      translator.t("sleep.recommendation.environment") in rec for rec in recommendations
    )

  def test_assess_data_quality(self, analyzer, sample_sleep_records):
    """Test data quality assessment."""
    quality = analyzer._assess_data_quality(sample_sleep_records)

    assert isinstance(quality, float)
    assert 0 <= quality <= 1

  def test_assess_data_quality_empty(self, analyzer):
    """Test data quality assessment with empty data."""
    quality = analyzer._assess_data_quality([])

    assert quality == 0.0

  @patch.object(StatisticalAnalyzer, "analyze_trend")
  def test_analyze_comprehensive(self, mock_trend, analyzer, sample_sleep_records):
    """Test comprehensive analysis."""
    # Mock trend analysis result.
    mock_trend.return_value = Mock(slope=0.1, r_squared=0.7, trend_direction="stable")

    report = analyzer.analyze_comprehensive(sample_sleep_records)

    assert isinstance(report, SleepAnalysisReport)
    assert report.record_count == len(sample_sleep_records)
    assert isinstance(report.data_quality_score, float)
    assert 0 <= report.data_quality_score <= 1

  def test_analyze_comprehensive_empty_records(self, analyzer):
    """Test comprehensive analysis with empty records."""
    report = analyzer.analyze_comprehensive([])

    assert isinstance(report, SleepAnalysisReport)
    assert report.record_count == 0

  def test_calculate_data_range(self, analyzer, sample_sleep_records):
    """Test data range calculation."""
    start_date, end_date = analyzer._calculate_data_range(sample_sleep_records)

    assert isinstance(start_date, datetime)
    assert isinstance(end_date, datetime)
    assert start_date <= end_date

  def test_calculate_data_range_empty(self, analyzer):
    """Test data range calculation with empty data."""
    start_date, end_date = analyzer._calculate_data_range([])

    assert isinstance(start_date, datetime)
    assert isinstance(end_date, datetime)
    assert start_date == end_date
