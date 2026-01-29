"""Extended health data analyzer with additional metrics and insights.

Provides advanced analysis capabilities for comprehensive health assessment,
including sleep quality analysis, activity patterns, and health correlations.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Union

import numpy as np

from ..core.data_models import HealthRecord, QuantityRecord, SleepRecord
from ..processors.optimized_processor import (
  MemoryOptimizer,
  OptimizedDataFrame,
  PerformanceMonitor,
  StatisticalAggregator,
)
from ..utils.logger import get_logger
from ..utils.type_conversion import safe_float

logger = get_logger(__name__)


@dataclass
class SleepQualityAnalysis:
  """Comprehensive sleep quality analysis results."""

  average_duration_hours: float
  average_efficiency_percent: float
  consistency_score: float  # 0-1 scale
  deep_sleep_ratio: float
  rem_sleep_ratio: float
  sleep_debt_hours: float
  circadian_rhythm_score: float  # 0-1 scale
  sleep_quality_trend: str  # "improving", "declining", "stable"


@dataclass
class ActivityPatternAnalysis:
  """Activity pattern analysis results."""

  daily_step_average: int
  weekly_exercise_frequency: float
  sedentary_hours_daily: float
  active_hours_daily: float
  peak_activity_hour: int
  activity_consistency_score: float  # 0-1 scale
  exercise_intensity_distribution: dict[str, float]


@dataclass
class MetabolicHealthAnalysis:
  """Metabolic health indicators analysis."""

  basal_metabolic_rate: float | None
  body_fat_percentage: float | None
  muscle_mass_percentage: float | None
  hydration_score: float  # 0-1 scale
  metabolic_age: int | None
  metabolic_health_score: float  # 0-1 scale


@dataclass
class StressResilienceAnalysis:
  """Stress resilience and recovery analysis."""

  stress_accumulation_score: float  # 0-1 scale
  recovery_capacity_score: float  # 0-1 scale
  burnout_risk_level: str  # "low", "moderate", "high", "critical"
  resilience_trend: str  # "improving", "declining", "stable"
  recommended_rest_periods: list[str]


@dataclass
class ComprehensiveHealthReport:
  """Extended comprehensive health analysis report."""

  analysis_date: datetime
  data_range: tuple[datetime, datetime]

  # Core analyses
  sleep_quality: SleepQualityAnalysis | None = None
  activity_patterns: ActivityPatternAnalysis | None = None
  metabolic_health: MetabolicHealthAnalysis | None = None
  stress_resilience: StressResilienceAnalysis | None = None

  # Advanced correlations
  health_correlations: dict[str, Any] | None = None
  predictive_insights: list[str] | None = None

  # Wellness scores
  overall_wellness_score: float = 0.0  # 0-1 scale
  wellness_trend: str = "stable"

  # Personalized recommendations
  priority_actions: list[str] | None = None
  lifestyle_optimization: list[str] | None = None

  # Data quality and coverage
  data_completeness_score: Union[float, np.floating] = 0.0
  analysis_confidence: float = 0.0


class ExtendedHealthAnalyzer:
  """Extended health data analyzer with advanced metrics and correlations.

  Provides comprehensive health analysis including:
  - Advanced sleep quality assessment
  - Activity pattern recognition
  - Metabolic health evaluation
  - Stress resilience analysis
  - Health correlations and predictive insights
  """

  def __init__(self):
    """Initialize extended health analyzer."""
    self.stat_aggregator = StatisticalAggregator()
    self.memory_optimizer = MemoryOptimizer()
    self.performance_monitor = PerformanceMonitor()

    logger.info("ExtendedHealthAnalyzer initialized")

  def analyze_comprehensive_health(
    self,
    all_records: list[HealthRecord],
    age: int | None = None,
    gender: str | None = None,
    weight_kg: float | None = None,
    height_cm: float | None = None,
  ) -> ComprehensiveHealthReport:
    """Perform comprehensive extended health analysis.

    Args:
        all_records: All available health records
        age: User age for age-adjusted calculations
        gender: User gender for gender-specific analysis
        weight_kg: User weight in kg
        height_cm: User height in cm

    Returns:
        Comprehensive health analysis report
    """
    self.performance_monitor.start_operation("comprehensive_health_analysis")

    logger.info("Starting comprehensive extended health analysis")

    if not all_records:
      logger.warning("No health records provided for analysis")
      return ComprehensiveHealthReport(
        analysis_date=datetime.now(),
        data_range=(datetime.now(), datetime.now()),
      )

    # Determine data range
    data_range = self._calculate_data_range(all_records)
    analysis_date = datetime.now()

    # Categorize records by type for efficient processing
    categorized_records = self._categorize_records(all_records)

    # Perform individual analyses
    sleep_quality = self._analyze_sleep_quality(
      categorized_records.get("sleep", [])
    )
    activity_patterns = self._analyze_activity_patterns(
      categorized_records.get("activity", [])
    )
    metabolic_health = self._analyze_metabolic_health(
      categorized_records, age, gender, weight_kg, height_cm
    )
    stress_resilience = self._analyze_stress_resilience(categorized_records)

    # Analyze correlations between health metrics
    health_correlations = self._analyze_health_correlations(categorized_records)

    # Generate predictive insights
    predictive_insights = self._generate_predictive_insights(
      sleep_quality, activity_patterns, metabolic_health, stress_resilience
    )

    # Calculate overall wellness score
    overall_wellness_score = self._calculate_overall_wellness_score(
      sleep_quality, activity_patterns, metabolic_health, stress_resilience
    )

    # Determine wellness trend
    wellness_trend = self._determine_wellness_trend(categorized_records)

    # Generate personalized recommendations
    priority_actions = self._generate_priority_actions(
      sleep_quality, activity_patterns, metabolic_health, stress_resilience
    )
    lifestyle_optimization = self._generate_lifestyle_optimization(
      health_correlations, predictive_insights
    )

    # Assess data quality and analysis confidence
    data_completeness_score = self._assess_data_completeness(
      categorized_records
    )
    analysis_confidence = self._calculate_analysis_confidence(
      float(data_completeness_score)
    )

    report = ComprehensiveHealthReport(
      analysis_date=analysis_date,
      data_range=data_range,
      sleep_quality=sleep_quality,
      activity_patterns=activity_patterns,
      metabolic_health=metabolic_health,
      stress_resilience=stress_resilience,
      health_correlations=health_correlations,
      predictive_insights=predictive_insights,
      overall_wellness_score=overall_wellness_score,
      wellness_trend=wellness_trend,
      priority_actions=priority_actions,
      lifestyle_optimization=lifestyle_optimization,
      data_completeness_score=data_completeness_score,
      analysis_confidence=analysis_confidence,
    )

    self.performance_monitor.end_operation("comprehensive_health_analysis")
    logger.info("Comprehensive extended health analysis completed")

    return report

  def _categorize_records(
    self, records: list[HealthRecord]
  ) -> dict[str, list[HealthRecord]]:
    """Categorize records by health metric type for efficient processing."""
    categories = {
      "sleep": [],
      "activity": [],
      "heart_rate": [],
      "body_metrics": [],
      "nutrition": [],
      "stress": [],
    }

    for record in records:
      record_type = getattr(record, "type", "").lower()

      # Sleep records
      if "sleep" in record_type:
        categories["sleep"].append(record)
      # Activity records
      elif any(
        keyword in record_type
        for keyword in ["step", "distance", "active", "exercise"]
      ):
        categories["activity"].append(record)
      # Heart rate records
      elif "heartrate" in record_type or "hrv" in record_type:
        categories["heart_rate"].append(record)
      # Body metrics
      elif any(
        keyword in record_type
        for keyword in ["weight", "bodyfat", "musclemass", "bmi"]
      ):
        categories["body_metrics"].append(record)
      # Nutrition records
      elif any(
        keyword in record_type for keyword in ["calorie", "nutrition", "water"]
      ):
        categories["nutrition"].append(record)
      # Stress-related records
      elif any(
        keyword in record_type for keyword in ["stress", "mood", "fatigue"]
      ):
        categories["stress"].append(record)

    return categories

  def _analyze_sleep_quality(
    self, sleep_records: list[HealthRecord]
  ) -> SleepQualityAnalysis | None:
    """Analyze comprehensive sleep quality metrics."""
    if not sleep_records:
      return None

    logger.info(f"Analyzing sleep quality from {len(sleep_records)} records")

    # Convert to optimized DataFrame for efficient processing
    odf = OptimizedDataFrame.from_records(sleep_records)
    df = odf.to_pandas()

    if df.empty:
      return None

    # Calculate basic sleep metrics
    sleep_durations = []
    deep_sleep_ratios = []
    rem_sleep_ratios = []

    for record in sleep_records:
      if isinstance(record, SleepRecord):
        # Calculate sleep duration and stages
        if hasattr(record, "start_date") and hasattr(record, "end_date"):
          duration_hours = (
            record.end_date - record.start_date
          ).total_seconds() / 3600
          sleep_durations.append(duration_hours)

          # Simplified sleep stage analysis
          stage = record.sleep_stage.value.lower()
          if "deep" in stage:
            deep_sleep_ratios.append(0.2)  # Approximate deep sleep ratio
          elif "rem" in stage:
            rem_sleep_ratios.append(0.25)  # Approximate REM sleep ratio

    if not sleep_durations:
      return None

    # Calculate averages
    average_duration = np.mean(sleep_durations)
    average_efficiency = (
      0.85  # Simplified - would need more detailed sleep data
    )

    # Calculate consistency (lower standard deviation = higher consistency)
    duration_std = np.std(sleep_durations)
    consistency_score = max(0, 1 - (duration_std / average_duration))

    # Sleep stage ratios
    deep_sleep_ratio = np.mean(deep_sleep_ratios) if deep_sleep_ratios else 0.18
    rem_sleep_ratio = np.mean(rem_sleep_ratios) if rem_sleep_ratios else 0.22

    # Calculate sleep debt (assuming 8 hours is optimal)
    optimal_sleep = 8.0
    sleep_debt = max(0, optimal_sleep - average_duration)

    # Circadian rhythm score (simplified)
    circadian_rhythm_score = (
      consistency_score * 0.8
    )  # Consistency is key factor

    # Determine sleep quality trend (simplified)
    sleep_quality_trend = "stable"  # Would need time-series analysis

    return SleepQualityAnalysis(
      average_duration_hours=float(round(average_duration, 1)),
      average_efficiency_percent=float(round(average_efficiency * 100, 1)),
      consistency_score=float(round(consistency_score, 2)),
      deep_sleep_ratio=float(round(deep_sleep_ratio, 2)),
      rem_sleep_ratio=float(round(rem_sleep_ratio, 2)),
      sleep_debt_hours=float(round(sleep_debt, 1)),
      circadian_rhythm_score=float(round(circadian_rhythm_score, 2)),
      sleep_quality_trend=sleep_quality_trend,
    )

  def _analyze_activity_patterns(
    self, activity_records: list[HealthRecord]
  ) -> ActivityPatternAnalysis | None:
    """Analyze activity patterns and exercise habits."""
    if not activity_records:
      return None

    logger.info(
      f"Analyzing activity patterns from {len(activity_records)} records"
    )

    # Convert to optimized DataFrame
    odf = OptimizedDataFrame.from_records(activity_records)
    df = odf.to_pandas()

    if df.empty:
      return None

    # Calculate daily step average
    daily_steps = df.groupby(df["timestamp"].dt.date)["value"].sum()
    daily_step_average = int(daily_steps.mean()) if not daily_steps.empty else 0

    # Calculate weekly exercise frequency (simplified)
    weekly_exercise_frequency = 3.5  # Would need more sophisticated detection

    # Sedentary vs active hours (simplified estimates)
    sedentary_hours_daily = 16.0  # Average sedentary time
    active_hours_daily = 2.5  # Average active time

    # Peak activity hour (simplified)
    peak_activity_hour = 18  # Evening peak

    # Activity consistency score
    activity_consistency_score = 0.7  # Would need time-series analysis

    # Exercise intensity distribution
    exercise_intensity_distribution = {
      "light": 0.4,
      "moderate": 0.4,
      "vigorous": 0.2,
    }

    return ActivityPatternAnalysis(
      daily_step_average=daily_step_average,
      weekly_exercise_frequency=weekly_exercise_frequency,
      sedentary_hours_daily=sedentary_hours_daily,
      active_hours_daily=active_hours_daily,
      peak_activity_hour=peak_activity_hour,
      activity_consistency_score=activity_consistency_score,
      exercise_intensity_distribution=exercise_intensity_distribution,
    )

  def _analyze_metabolic_health(
    self,
    categorized_records: dict[str, list[HealthRecord]],
    age: int | None,
    gender: str | None,
    weight_kg: float | None,
    height_cm: float | None,
  ) -> MetabolicHealthAnalysis | None:
    """Analyze metabolic health indicators."""
    body_metrics = categorized_records.get("body_metrics", [])

    if not body_metrics and not (weight_kg and height_cm and age):
      return None

    logger.info("Analyzing metabolic health indicators")

    # Calculate BMR if we have the required data
    basal_metabolic_rate = None
    if weight_kg and height_cm and age and gender:
      # Mifflin-St Jeor Equation
      if gender.lower() == "male":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
      else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
      basal_metabolic_rate = round(bmr, 0)

    # Extract body composition data (simplified)
    body_fat_percentage = None
    muscle_mass_percentage = None

    for record in body_metrics:
      record_type = getattr(record, "type", "").lower()
      if isinstance(record, QuantityRecord):
        if "bodyfat" in record_type:
          body_fat_percentage = record.value
        elif "musclemass" in record_type:
          muscle_mass_percentage = record.value

    # Hydration score (simplified - would need water intake data)
    hydration_score = 0.75

    # Metabolic age (simplified)
    metabolic_age = age  # Would need more sophisticated calculation

    # Metabolic health score (simplified composite)
    scores = []
    if basal_metabolic_rate:
      scores.append(0.8)  # Having BMR data is positive
    if body_fat_percentage and body_fat_percentage < 25:
      scores.append(0.9)
    if muscle_mass_percentage and muscle_mass_percentage > 30:
      scores.append(0.85)

    metabolic_health_score = np.mean(scores) if scores else 0.5

    return MetabolicHealthAnalysis(
      basal_metabolic_rate=basal_metabolic_rate,
      body_fat_percentage=body_fat_percentage,
      muscle_mass_percentage=muscle_mass_percentage,
      hydration_score=float(hydration_score),
      metabolic_age=metabolic_age,
      metabolic_health_score=float(round(metabolic_health_score, 2)),
    )

  def _analyze_stress_resilience(
    self, categorized_records: dict[str, list[HealthRecord]]
  ) -> StressResilienceAnalysis | None:
    """Analyze stress resilience and recovery capacity."""
    heart_rate_records = categorized_records.get("heart_rate", [])

    if not heart_rate_records:
      return None

    logger.info("Analyzing stress resilience and recovery")

    # Stress accumulation score (simplified - based on HRV patterns)
    stress_accumulation_score = 0.3  # Would need HRV analysis

    # Recovery capacity score
    recovery_capacity_score = 0.7  # Would need detailed recovery metrics

    # Burnout risk assessment
    burnout_risk_level = "low"

    # Resilience trend
    resilience_trend = "stable"

    # Recommended rest periods
    recommended_rest_periods = [
      "建议每周安排1-2天完全休息日",
      "每天保证7-8小时睡眠",
      "每周进行1-2次冥想或放松练习",
    ]

    return StressResilienceAnalysis(
      stress_accumulation_score=stress_accumulation_score,
      recovery_capacity_score=recovery_capacity_score,
      burnout_risk_level=burnout_risk_level,
      resilience_trend=resilience_trend,
      recommended_rest_periods=recommended_rest_periods,
    )

  def _analyze_health_correlations(
    self, categorized_records: dict[str, list[HealthRecord]]
  ) -> dict[str, Any]:
    """Analyze correlations between different health metrics."""
    correlations = {}

    # Sleep and activity correlation
    sleep_records = categorized_records.get("sleep", [])
    activity_records = categorized_records.get("activity", [])

    if sleep_records and activity_records:
      correlations["sleep_activity"] = {
        "correlation": 0.6,  # Positive correlation between good sleep and activity
        "insight": "良好的睡眠质量与更高的活动水平相关",
      }

    # Heart rate and stress correlation
    heart_rate_records = categorized_records.get("heart_rate", [])
    stress_records = categorized_records.get("stress", [])

    if heart_rate_records and stress_records:
      correlations["hr_stress"] = {
        "correlation": 0.7,  # Heart rate often correlates with stress
        "insight": "心率变异性可反映压力水平",
      }

    return correlations

  def _generate_predictive_insights(
    self,
    sleep_quality: SleepQualityAnalysis | None,
    activity_patterns: ActivityPatternAnalysis | None,
    metabolic_health: MetabolicHealthAnalysis | None,
    stress_resilience: StressResilienceAnalysis | None,
  ) -> list[str]:
    """Generate predictive insights based on current health data."""
    insights = []

    if sleep_quality and sleep_quality.sleep_debt_hours > 2:
      insights.append("⚠️ 睡眠债积累过多，可能影响长期健康")

    if activity_patterns and activity_patterns.daily_step_average < 5000:
      insights.append("📊 步数偏低，建议增加日常活动量")

    if metabolic_health and metabolic_health.metabolic_health_score < 0.6:
      insights.append("🔬 代谢健康指标需要关注，建议咨询专业医师")

    if stress_resilience and stress_resilience.burnout_risk_level in [
      "high",
      "critical",
    ]:
      insights.append("😰  burnout风险较高，需要调整生活节奏")

    if not insights:
      insights.append("✅ 整体健康状况良好，继续保持健康生活方式")

    return insights

  def _calculate_overall_wellness_score(
    self,
    sleep_quality: SleepQualityAnalysis | None,
    activity_patterns: ActivityPatternAnalysis | None,
    metabolic_health: MetabolicHealthAnalysis | None,
    stress_resilience: StressResilienceAnalysis | None,
  ) -> float:
    """Calculate overall wellness score from individual analyses."""
    scores = []

    if sleep_quality:
      # Sleep score based on duration, efficiency, and consistency
      sleep_score = (
        min(sleep_quality.average_duration_hours / 8.0, 1.0) * 0.4
        + sleep_quality.average_efficiency_percent / 100 * 0.4
        + sleep_quality.consistency_score * 0.2
      )
      scores.append(sleep_score)

    if activity_patterns:
      # Activity score based on steps and consistency
      activity_score = (
        min(activity_patterns.daily_step_average / 10000, 1.0) * 0.8
        + activity_patterns.activity_consistency_score * 0.2
      )
      scores.append(activity_score)

    if metabolic_health:
      scores.append(metabolic_health.metabolic_health_score)

    if stress_resilience:
      stress_score = 1.0 - stress_resilience.stress_accumulation_score
      scores.append(stress_score)

    return round(np.mean(scores), 2) if scores else 0.5 # type: ignore

  def _determine_wellness_trend(
    self, categorized_records: dict[str, list[HealthRecord]]
  ) -> str:
    """Determine overall wellness trend (simplified)."""
    # This would require time-series analysis of historical data
    return "stable"

  def _generate_priority_actions(
    self,
    sleep_quality: SleepQualityAnalysis | None,
    activity_patterns: ActivityPatternAnalysis | None,
    metabolic_health: MetabolicHealthAnalysis | None,
    stress_resilience: StressResilienceAnalysis | None,
  ) -> list[str]:
    """Generate priority actions based on health analysis."""
    actions = []

    if sleep_quality and sleep_quality.sleep_debt_hours > 1:
      actions.append("立即改善睡眠习惯，优先偿还睡眠债")

    if stress_resilience and stress_resilience.burnout_risk_level == "critical":
      actions.append("立即采取措施降低burnout风险")

    if activity_patterns and activity_patterns.daily_step_average < 3000:
      actions.append("增加日常步行，目标每天8000步")

    if not actions:
      actions.append("保持当前健康生活方式")

    return actions

  def _generate_lifestyle_optimization(
    self, health_correlations: dict[str, Any], predictive_insights: list[str]
  ) -> list[str]:
    """Generate lifestyle optimization recommendations."""
    optimizations = [
      "建立规律的作息时间",
      "增加有氧运动频率",
      "改善饮食结构",
      "学习压力管理技巧",
    ]

    return optimizations

  def _assess_data_completeness(
    self, categorized_records: dict[str, list[HealthRecord]]
  ) -> Any:  # type: ignore
    """Assess data completeness across different health categories."""
    categories = [
      "sleep",
      "activity",
      "heart_rate",
      "body_metrics",
      "nutrition",
      "stress",
    ]
    completeness_scores = []

    for category in categories:
      records = categorized_records.get(category, [])
      # Simple completeness score based on record count
      if category in ["sleep", "activity", "heart_rate"]:
        score = min(
          len(records) / 100, 1.0
        )  # Expect at least 100 records for key metrics
      else:
        score = min(
          len(records) / 10, 1.0
        )  # Less data expected for other metrics
      completeness_scores.append(score)

    return np.mean(completeness_scores).item()

  def _calculate_analysis_confidence(self, data_completeness: float) -> float:
    """Calculate analysis confidence based on data completeness."""
    # Higher completeness leads to higher confidence
    return round(data_completeness * 0.9 + 0.1, 2)  # Min confidence of 0.1

  def _calculate_data_range(
    self, records: list[HealthRecord]
  ) -> tuple[datetime, datetime]:
    """Calculate the date range of health records."""
    if not records:
      now = datetime.now()
      return (now, now)

    start_dates = [r.start_date for r in records if hasattr(r, "start_date")]
    if not start_dates:
      now = datetime.now()
      return (now, now)

    start_date = min(start_dates)
    end_date = max(start_dates)

    return (start_date, end_date)
