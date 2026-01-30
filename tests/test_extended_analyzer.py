"""Tests for extended health analyzer module."""

from datetime import datetime, timedelta

from src.analyzers.extended_analyzer import (
  ActivityPatternAnalysis,
  ComprehensiveHealthReport,
  ExtendedHealthAnalyzer,
  MetabolicHealthAnalysis,
  SleepQualityAnalysis,
  StressResilienceAnalysis,
)
from src.core.data_models import QuantityRecord, SleepRecord


class TestExtendedHealthAnalyzer:
  """Test ExtendedHealthAnalyzer class."""

  def test_initialization(self):
    """Test analyzer initialization."""
    analyzer = ExtendedHealthAnalyzer()

    assert analyzer.stat_aggregator is not None
    assert analyzer.memory_optimizer is not None
    assert analyzer.performance_monitor is not None

  def test_analyze_comprehensive_health_empty_records(self):
    """Test comprehensive analysis with empty records."""
    analyzer = ExtendedHealthAnalyzer()
    report = analyzer.analyze_comprehensive_health([])

    assert isinstance(report, ComprehensiveHealthReport)
    assert report.sleep_quality is None
    assert report.activity_patterns is None
    assert report.metabolic_health is None
    assert report.stress_resilience is None
    assert report.overall_wellness_score == 0.0
    assert report.data_completeness_score == 0.0
    assert report.analysis_confidence == 0.0

  def test_analyze_comprehensive_health_with_data(self):
    """Test comprehensive analysis with health data."""
    analyzer = ExtendedHealthAnalyzer()

    # Create test records
    base_time = datetime(2024, 1, 1, 10, 0, 0)
    records = [
      # Sleep records
      SleepRecord(
        type="HKCategoryTypeIdentifierSleepAnalysis",
        source_name="Test",
        source_version="1.0",
        device="TestDevice",
        creation_date=base_time,
        start_date=base_time,
        end_date=base_time + timedelta(hours=8),
        value="HKCategoryValueSleepAnalysisAsleepCore",
        metadata={},
      ),
      # Activity records
      QuantityRecord(
        type="HKQuantityTypeIdentifierStepCount",
        source_name="Test",
        source_version="1.0",
        device="TestDevice",
        unit="count",
        creation_date=base_time,
        start_date=base_time,
        end_date=base_time + timedelta(minutes=1),
        value=8000,
      ),
      # Heart rate records
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
    ]

    report = analyzer.analyze_comprehensive_health(
      records, age=30, gender="male"
    )

    assert isinstance(report, ComprehensiveHealthReport)
    assert report.sleep_quality is not None
    assert report.activity_patterns is not None
    assert report.metabolic_health is None  # No body metrics provided
    assert report.stress_resilience is not None
    assert isinstance(report.overall_wellness_score, float)
    assert report.overall_wellness_score >= 0.0
    assert report.overall_wellness_score <= 1.0

  def test_categorize_records(self):
    """Test record categorization."""
    analyzer = ExtendedHealthAnalyzer()

    records = [
      SleepRecord(
        type="HKCategoryTypeIdentifierSleepAnalysis",
        source_name="Test",
        source_version="1.0",
        device="TestDevice",
        creation_date=datetime(2024, 1, 1, 22, 0),
        start_date=datetime(2024, 1, 1, 22, 0),
        end_date=datetime(2024, 1, 2, 6, 0),
        value="HKCategoryValueSleepAnalysisAsleepCore",
        metadata={},
      ),
      QuantityRecord(
        type="HKQuantityTypeIdentifierStepCount",
        source_name="Test",
        source_version="1.0",
        device="TestDevice",
        unit="count",
        creation_date=datetime(2024, 1, 1, 10, 0),
        start_date=datetime(2024, 1, 1, 10, 0),
        end_date=datetime(2024, 1, 1, 10, 1),
        value=5000,
      ),
      QuantityRecord(
        type="HKQuantityTypeIdentifierHeartRate",
        source_name="Test",
        source_version="1.0",
        device="TestDevice",
        unit="count/min",
        creation_date=datetime(2024, 1, 1, 10, 0),
        start_date=datetime(2024, 1, 1, 10, 0),
        end_date=datetime(2024, 1, 1, 10, 1),
        value=75,
      ),
      QuantityRecord(
        type="HKQuantityTypeIdentifierBodyMass",
        source_name="Test",
        source_version="1.0",
        device="TestDevice",
        unit="kg",
        creation_date=datetime(2024, 1, 1, 8, 0),
        start_date=datetime(2024, 1, 1, 8, 0),
        end_date=datetime(2024, 1, 1, 8, 1),
        value=70.0,
      ),
    ]

    categorized = analyzer._categorize_records(records)

    assert len(categorized["sleep"]) == 1
    assert len(categorized["activity"]) == 1
    assert len(categorized["heart_rate"]) == 1
    assert len(categorized["body_metrics"]) == 1
    assert len(categorized["nutrition"]) == 0
    assert len(categorized["stress"]) == 0

  def test_analyze_sleep_quality(self):
    """Test sleep quality analysis."""
    analyzer = ExtendedHealthAnalyzer()

    # Create sleep records
    base_time = datetime(2024, 1, 1, 22, 0, 0)
    sleep_records = [
      SleepRecord(
        type="HKCategoryTypeIdentifierSleepAnalysis",
        source_name="Test",
        source_version="1.0",
        device="TestDevice",
        creation_date=base_time,
        start_date=base_time,
        end_date=base_time + timedelta(hours=8),
        value="HKCategoryValueSleepAnalysisAsleepCore",
        metadata={},
      ),
      SleepRecord(
        type="HKCategoryTypeIdentifierSleepAnalysis",
        source_name="Test",
        source_version="1.0",
        device="TestDevice",
        creation_date=base_time + timedelta(days=1),
        start_date=base_time + timedelta(days=1),
        end_date=base_time + timedelta(days=1, hours=7.5),
        value="HKCategoryValueSleepAnalysisAsleepDeep",
        metadata={},
      ),
    ]

    sleep_analysis = analyzer._analyze_sleep_quality(sleep_records)

    assert isinstance(sleep_analysis, SleepQualityAnalysis)
    assert sleep_analysis.average_duration_hours > 0
    assert sleep_analysis.average_efficiency_percent > 0
    assert 0 <= sleep_analysis.consistency_score <= 1
    assert 0 <= sleep_analysis.deep_sleep_ratio <= 1
    assert 0 <= sleep_analysis.rem_sleep_ratio <= 1
    assert sleep_analysis.sleep_debt_hours >= 0
    assert 0 <= sleep_analysis.circadian_rhythm_score <= 1
    assert sleep_analysis.sleep_quality_trend in [
      "improving",
      "declining",
      "stable",
    ]

  def test_analyze_sleep_quality_empty(self):
    """Test sleep quality analysis with no records."""
    analyzer = ExtendedHealthAnalyzer()

    sleep_analysis = analyzer._analyze_sleep_quality([])

    assert sleep_analysis is None

  def test_analyze_activity_patterns(self):
    """Test activity pattern analysis."""
    analyzer = ExtendedHealthAnalyzer()

    # Create activity records
    base_time = datetime(2024, 1, 1, 8, 0, 0)
    activity_records = [
      QuantityRecord(
        type="HKQuantityTypeIdentifierStepCount",
        source_name="Test",
        source_version="1.0",
        device="TestDevice",
        unit="count",
        creation_date=base_time,
        start_date=base_time,
        end_date=base_time + timedelta(minutes=1),
        value=8000,
      ),
      QuantityRecord(
        type="HKQuantityTypeIdentifierDistanceWalkingRunning",
        source_name="Test",
        source_version="1.0",
        device="TestDevice",
        unit="km",
        creation_date=base_time,
        start_date=base_time,
        end_date=base_time + timedelta(minutes=1),
        value=6.0,
      ),
    ]

    activity_analysis = analyzer._analyze_activity_patterns(activity_records)

    assert isinstance(activity_analysis, ActivityPatternAnalysis)
    assert activity_analysis.daily_step_average >= 0
    assert activity_analysis.weekly_exercise_frequency >= 0
    assert activity_analysis.sedentary_hours_daily >= 0
    assert activity_analysis.active_hours_daily >= 0
    assert 0 <= activity_analysis.activity_consistency_score <= 1
    assert isinstance(activity_analysis.exercise_intensity_distribution, dict)

  def test_analyze_activity_patterns_empty(self):
    """Test activity pattern analysis with no records."""
    analyzer = ExtendedHealthAnalyzer()

    activity_analysis = analyzer._analyze_activity_patterns([])

    assert activity_analysis is None

  def test_analyze_metabolic_health(self):
    """Test metabolic health analysis."""
    analyzer = ExtendedHealthAnalyzer()

    categorized_records = {
      "body_metrics": [
        QuantityRecord(
          type="HKQuantityTypeIdentifierBodyFatPercentage",
          source_name="Test",
          source_version="1.0",
          device="TestDevice",
          unit="%",
          creation_date=datetime(2024, 1, 1, 8, 0),
          start_date=datetime(2024, 1, 1, 8, 0),
          end_date=datetime(2024, 1, 1, 8, 1),
          value=20.0,
        ),
        QuantityRecord(
          type="HKQuantityTypeIdentifierLeanBodyMass",
          source_name="Test",
          source_version="1.0",
          device="TestDevice",
          unit="kg",
          creation_date=datetime(2024, 1, 1, 8, 0),
          start_date=datetime(2024, 1, 1, 8, 0),
          end_date=datetime(2024, 1, 1, 8, 1),
          value=55.0,
        ),
      ]
    }

    metabolic_analysis = analyzer._analyze_metabolic_health(
      categorized_records,
      age=30,
      gender="male",
      weight_kg=75.0,
      height_cm=175.0,
    )

    assert isinstance(metabolic_analysis, MetabolicHealthAnalysis)
    assert metabolic_analysis.basal_metabolic_rate is not None
    assert metabolic_analysis.basal_metabolic_rate > 0
    assert metabolic_analysis.body_fat_percentage == 20.0
    assert (
      metabolic_analysis.muscle_mass_percentage is None
    )  # Lean body mass != muscle mass
    assert 0 <= metabolic_analysis.hydration_score <= 1
    assert metabolic_analysis.metabolic_age == 30
    assert 0 <= metabolic_analysis.metabolic_health_score <= 1

  def test_analyze_metabolic_health_minimal_data(self):
    """Test metabolic health analysis with minimal data."""
    analyzer = ExtendedHealthAnalyzer()

    categorized_records = {}

    metabolic_analysis = analyzer._analyze_metabolic_health(
      categorized_records,
      age=30,
      gender="female",
      weight_kg=65.0,
      height_cm=165.0,
    )

    assert isinstance(metabolic_analysis, MetabolicHealthAnalysis)
    assert metabolic_analysis.basal_metabolic_rate is not None
    assert metabolic_analysis.body_fat_percentage is None
    assert metabolic_analysis.muscle_mass_percentage is None

  def test_analyze_metabolic_health_no_data(self):
    """Test metabolic health analysis with no data."""
    analyzer = ExtendedHealthAnalyzer()

    categorized_records = {}

    metabolic_analysis = analyzer._analyze_metabolic_health(
      categorized_records, age=None, gender=None, weight_kg=None, height_cm=None
    )

    assert metabolic_analysis is None

  def test_analyze_stress_resilience(self):
    """Test stress resilience analysis."""
    analyzer = ExtendedHealthAnalyzer()

    categorized_records = {
      "heart_rate": [
        QuantityRecord(
          type="HKQuantityTypeIdentifierHeartRate",
          source_name="Test",
          source_version="1.0",
          device="TestDevice",
          unit="count/min",
          creation_date=datetime(2024, 1, 1, 10, 0),
          start_date=datetime(2024, 1, 1, 10, 0),
          end_date=datetime(2024, 1, 1, 10, 1),
          value=75,
        )
      ]
    }

    stress_analysis = analyzer._analyze_stress_resilience(categorized_records)

    assert isinstance(stress_analysis, StressResilienceAnalysis)
    assert 0 <= stress_analysis.stress_accumulation_score <= 1
    assert 0 <= stress_analysis.recovery_capacity_score <= 1
    assert stress_analysis.burnout_risk_level in [
      "low",
      "moderate",
      "high",
      "critical",
    ]
    assert stress_analysis.resilience_trend in [
      "improving",
      "declining",
      "stable",
    ]
    assert isinstance(stress_analysis.recommended_rest_periods, list)

  def test_analyze_stress_resilience_no_data(self):
    """Test stress resilience analysis with no data."""
    analyzer = ExtendedHealthAnalyzer()

    categorized_records = {}

    stress_analysis = analyzer._analyze_stress_resilience(categorized_records)

    assert stress_analysis is None

  def test_analyze_health_correlations(self):
    """Test health correlations analysis."""
    analyzer = ExtendedHealthAnalyzer()

    categorized_records = {
      "sleep": [
        SleepRecord(
          type="HKCategoryTypeIdentifierSleepAnalysis",
          source_name="Test",
          source_version="1.0",
          device="TestDevice",
          creation_date=datetime(2024, 1, 1, 22, 0),
          start_date=datetime(2024, 1, 1, 22, 0),
          end_date=datetime(2024, 1, 2, 6, 0),
          value="HKCategoryValueSleepAnalysisAsleepCore",
          metadata={},
        )
      ],
      "activity": [
        QuantityRecord(
          type="HKQuantityTypeIdentifierStepCount",
          source_name="Test",
          source_version="1.0",
          device="TestDevice",
          unit="count",
          creation_date=datetime(2024, 1, 1, 10, 0),
          start_date=datetime(2024, 1, 1, 10, 0),
          end_date=datetime(2024, 1, 1, 10, 1),
          value=8000,
        )
      ],
      "heart_rate": [
        QuantityRecord(
          type="HKQuantityTypeIdentifierHeartRate",
          source_name="Test",
          source_version="1.0",
          device="TestDevice",
          unit="count/min",
          creation_date=datetime(2024, 1, 1, 10, 0),
          start_date=datetime(2024, 1, 1, 10, 0),
          end_date=datetime(2024, 1, 1, 10, 1),
          value=75,
        )
      ],
      "stress": [
        QuantityRecord(
          type="HKQuantityTypeIdentifierStressLevel",
          source_name="Test",
          source_version="1.0",
          device="TestDevice",
          unit="score",
          creation_date=datetime(2024, 1, 1, 10, 0),
          start_date=datetime(2024, 1, 1, 10, 0),
          end_date=datetime(2024, 1, 1, 10, 1),
          value=3.0,
        )
      ],
    }

    correlations = analyzer._analyze_health_correlations(categorized_records)

    assert isinstance(correlations, dict)
    assert "sleep_activity" in correlations
    assert "hr_stress" in correlations

  def test_generate_predictive_insights(self):
    """Test predictive insights generation."""
    analyzer = ExtendedHealthAnalyzer()

    # Create mock analysis objects
    sleep_quality = SleepQualityAnalysis(
      average_duration_hours=6.0,
      average_efficiency_percent=80.0,
      consistency_score=0.7,
      deep_sleep_ratio=0.15,
      rem_sleep_ratio=0.20,
      sleep_debt_hours=2.0,
      circadian_rhythm_score=0.6,
      sleep_quality_trend="stable",
    )

    activity_patterns = ActivityPatternAnalysis(
      daily_step_average=4000,
      weekly_exercise_frequency=2.0,
      sedentary_hours_daily=18.0,
      active_hours_daily=1.5,
      peak_activity_hour=18,
      activity_consistency_score=0.5,
      exercise_intensity_distribution={
        "light": 0.6,
        "moderate": 0.3,
        "vigorous": 0.1,
      },
    )

    metabolic_health = MetabolicHealthAnalysis(
      basal_metabolic_rate=1800.0,
      body_fat_percentage=25.0,
      muscle_mass_percentage=30.0,
      hydration_score=0.7,
      metabolic_age=35,
      metabolic_health_score=0.5,
    )

    stress_resilience = StressResilienceAnalysis(
      stress_accumulation_score=0.8,
      recovery_capacity_score=0.4,
      burnout_risk_level="high",
      resilience_trend="declining",
      recommended_rest_periods=["建议每周安排1-2天完全休息日"],
    )

    insights = analyzer._generate_predictive_insights(
      sleep_quality, activity_patterns, metabolic_health, stress_resilience
    )

    assert isinstance(insights, list)
    assert len(insights) > 0
    # Should contain warnings for sleep debt, low activity, poor metabolic health, and high burnout risk

  def test_calculate_overall_wellness_score(self):
    """Test overall wellness score calculation."""
    analyzer = ExtendedHealthAnalyzer()

    # Create mock analysis objects
    sleep_quality = SleepQualityAnalysis(
      average_duration_hours=8.0,
      average_efficiency_percent=90.0,
      consistency_score=0.8,
      deep_sleep_ratio=0.20,
      rem_sleep_ratio=0.25,
      sleep_debt_hours=0.0,
      circadian_rhythm_score=0.8,
      sleep_quality_trend="stable",
    )

    activity_patterns = ActivityPatternAnalysis(
      daily_step_average=10000,
      weekly_exercise_frequency=4.0,
      sedentary_hours_daily=12.0,
      active_hours_daily=3.0,
      peak_activity_hour=18,
      activity_consistency_score=0.8,
      exercise_intensity_distribution={
        "light": 0.4,
        "moderate": 0.4,
        "vigorous": 0.2,
      },
    )

    metabolic_health = MetabolicHealthAnalysis(
      basal_metabolic_rate=1800.0,
      body_fat_percentage=20.0,
      muscle_mass_percentage=35.0,
      hydration_score=0.8,
      metabolic_age=30,
      metabolic_health_score=0.8,
    )

    stress_resilience = StressResilienceAnalysis(
      stress_accumulation_score=0.2,
      recovery_capacity_score=0.8,
      burnout_risk_level="low",
      resilience_trend="improving",
      recommended_rest_periods=[],
    )

    score = analyzer._calculate_overall_wellness_score(
      sleep_quality, activity_patterns, metabolic_health, stress_resilience
    )

    assert isinstance(score, float)
    assert 0 <= score <= 1

  def test_generate_priority_actions(self):
    """Test priority actions generation."""
    analyzer = ExtendedHealthAnalyzer()

    # Create mock analysis objects with issues
    sleep_quality = SleepQualityAnalysis(
      average_duration_hours=6.0,
      average_efficiency_percent=80.0,
      consistency_score=0.7,
      deep_sleep_ratio=0.15,
      rem_sleep_ratio=0.20,
      sleep_debt_hours=2.0,
      circadian_rhythm_score=0.6,
      sleep_quality_trend="stable",
    )

    activity_patterns = ActivityPatternAnalysis(
      daily_step_average=2000,
      weekly_exercise_frequency=1.0,
      sedentary_hours_daily=20.0,
      active_hours_daily=1.0,
      peak_activity_hour=18,
      activity_consistency_score=0.3,
      exercise_intensity_distribution={
        "light": 0.8,
        "moderate": 0.2,
        "vigorous": 0.0,
      },
    )

    metabolic_health = MetabolicHealthAnalysis(
      basal_metabolic_rate=1800.0,
      body_fat_percentage=30.0,
      muscle_mass_percentage=25.0,
      hydration_score=0.6,
      metabolic_age=40,
      metabolic_health_score=0.4,
    )

    stress_resilience = StressResilienceAnalysis(
      stress_accumulation_score=0.9,
      recovery_capacity_score=0.3,
      burnout_risk_level="critical",
      resilience_trend="declining",
      recommended_rest_periods=[],
    )

    actions = analyzer._generate_priority_actions(
      sleep_quality, activity_patterns, metabolic_health, stress_resilience
    )

    assert isinstance(actions, list)
    assert len(actions) > 0
    # Should contain actions for sleep debt, burnout risk, and low activity

  def test_assess_data_completeness(self):
    """Test data completeness assessment."""
    analyzer = ExtendedHealthAnalyzer()

    # Create mock records for each category
    base_time = datetime(2024, 1, 1, 10, 0, 0)

    categorized_records = {
      "sleep": [
        SleepRecord(
          type="HKCategoryTypeIdentifierSleepAnalysis",
          source_name="Test",
          source_version="1.0",
          device="TestDevice",
          creation_date=base_time + timedelta(hours=i),
          start_date=base_time + timedelta(hours=i),
          end_date=base_time + timedelta(hours=i, minutes=30),
          value="HKCategoryValueSleepAnalysisAsleepCore",
          metadata={},
        )
        for i in range(150)
      ],
      "activity": [
        QuantityRecord(
          type="HKQuantityTypeIdentifierStepCount",
          source_name="Test",
          source_version="1.0",
          device="TestDevice",
          unit="count",
          creation_date=base_time,
          start_date=base_time + timedelta(hours=i),
          end_date=base_time + timedelta(hours=i, minutes=1),
          value=1000,
        )
        for i in range(90)
      ],
      "heart_rate": [
        QuantityRecord(
          type="HKQuantityTypeIdentifierHeartRate",
          source_name="Test",
          source_version="1.0",
          device="TestDevice",
          unit="count/min",
          creation_date=base_time,
          start_date=base_time + timedelta(hours=i),
          end_date=base_time + timedelta(hours=i, minutes=1),
          value=70,
        )
        for i in range(240)
      ],
      "body_metrics": [
        QuantityRecord(
          type="HKQuantityTypeIdentifierBodyMass",
          source_name="Test",
          source_version="1.0",
          device="TestDevice",
          unit="kg",
          creation_date=base_time,
          start_date=base_time + timedelta(hours=i),
          end_date=base_time + timedelta(hours=i, minutes=1),
          value=70.0,
        )
        for i in range(15)
      ],
      "nutrition": [
        QuantityRecord(
          type="HKQuantityTypeIdentifierDietaryEnergyConsumed",
          source_name="Test",
          source_version="1.0",
          device="TestDevice",
          unit="kcal",
          creation_date=base_time,
          start_date=base_time + timedelta(hours=i),
          end_date=base_time + timedelta(hours=i, minutes=1),
          value=500,
        )
        for i in range(6)
      ],
      "stress": [
        QuantityRecord(
          type="HKQuantityTypeIdentifierStressLevel",
          source_name="Test",
          source_version="1.0",
          device="TestDevice",
          unit="score",
          creation_date=base_time,
          start_date=base_time + timedelta(hours=i),
          end_date=base_time + timedelta(hours=i, minutes=1),
          value=3.0,
        )
        for i in range(3)
      ],
    }

    completeness = analyzer._assess_data_completeness(categorized_records)

    assert isinstance(completeness, float)
    assert 0 <= completeness <= 1

  def test_calculate_analysis_confidence(self):
    """Test analysis confidence calculation."""
    analyzer = ExtendedHealthAnalyzer()

    confidence = analyzer._calculate_analysis_confidence(0.8)

    assert isinstance(confidence, float)
    assert 0.1 <= confidence <= 1.0  # Min confidence is 0.1

  def test_calculate_data_range(self):
    """Test data range calculation."""
    analyzer = ExtendedHealthAnalyzer()

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
      ),
      QuantityRecord(
        type="HKQuantityTypeIdentifierHeartRate",
        source_name="Test",
        source_version="1.0",
        device="TestDevice",
        unit="count/min",
        creation_date=datetime(2024, 1, 15, 10, 0),
        start_date=datetime(2024, 1, 15, 10, 0),
        end_date=datetime(2024, 1, 15, 10, 1),
        value=75,
      ),
    ]

    start_date, end_date = analyzer._calculate_data_range(records)

    assert start_date == datetime(2024, 1, 1, 10, 0)
    assert end_date == datetime(2024, 1, 15, 10, 0)

  def test_calculate_data_range_empty(self):
    """Test data range calculation with empty records."""
    analyzer = ExtendedHealthAnalyzer()

    start_date, end_date = analyzer._calculate_data_range([])

    assert start_date == end_date
    # Should be current datetime
