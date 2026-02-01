"""Extended health data analyzer with additional metrics and insights.

Provides advanced analysis capabilities for comprehensive health assessment,
including sleep quality analysis, activity patterns, and health correlations.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from ..core.data_models import HealthRecord, QuantityRecord, SleepRecord
from ..i18n import Translator, resolve_locale
from ..processors.optimized_processor import (
  MemoryOptimizer,
  OptimizedDataFrame,
  PerformanceMonitor,
  StatisticalAggregator,
)
from ..utils.logger import get_logger
from ..utils.type_conversion import numpy_to_python_scalar, safe_float

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
  data_completeness_score: float | np.floating = 0.0
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

  def __init__(self, locale: str | None = None):
    """Initialize extended health analyzer."""
    self.translator = Translator(resolve_locale(locale))
    self.stat_aggregator = StatisticalAggregator()
    self.memory_optimizer = MemoryOptimizer()
    self.performance_monitor = PerformanceMonitor()

    logger.info(self.translator.t("log.extended_analyzer.initialized"))

  def analyze_comprehensive_health(
    self,
    all_records: Sequence[HealthRecord],
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

    logger.info(self.translator.t("log.extended_analyzer.start_comprehensive"))

    if not all_records:
      logger.warning(self.translator.t("log.extended_analyzer.no_records"))
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
    sleep_quality = self._analyze_sleep_quality(categorized_records.get("sleep", []))
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
    data_completeness_score = self._assess_data_completeness(categorized_records)
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
    logger.info(self.translator.t("log.extended_analyzer.completed"))

    return report

  def _categorize_records(
    self, records: Sequence[HealthRecord]
  ) -> dict[str, list[HealthRecord]]:
    """Categorize records by health metric type for efficient processing."""
    categories = {
      "sleep": [],
      "activity": [],
      "heart_rate": [],
      "hrv": [],
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
        keyword in record_type for keyword in ["step", "distance", "active", "exercise"]
      ):
        categories["activity"].append(record)
      # Heart rate records
      elif "heartrate" in record_type:
        categories["heart_rate"].append(record)
      # HRV records
      elif "hrv" in record_type or "variability" in record_type:
        categories["hrv"].append(record)
      # Body metrics
      elif any(
        keyword in record_type
        for keyword in ["weight", "bodyfat", "musclemass", "bmi", "bodymass"]
      ):
        categories["body_metrics"].append(record)
      # Nutrition records
      elif any(keyword in record_type for keyword in ["calorie", "nutrition", "water"]):
        categories["nutrition"].append(record)
      # Stress-related records
      elif any(keyword in record_type for keyword in ["stress", "mood", "fatigue"]):
        categories["stress"].append(record)

    return categories

  def _analyze_sleep_quality(
    self, sleep_records: Sequence[HealthRecord]
  ) -> SleepQualityAnalysis | None:
    """Analyze comprehensive sleep quality metrics."""
    if not sleep_records:
      return None

    logger.info(
      self.translator.t(
        "log.extended_analyzer.sleep_quality",
        count=len(sleep_records),
      )
    )

    # Convert to optimized DataFrame for efficient processing
    odf = OptimizedDataFrame.from_records(sleep_records)
    if odf.record_count == 0:
      return None

    # Aggregate by day using actual stage durations
    daily_sleep_hours: dict[date, float] = {}
    daily_in_bed_hours: dict[date, float] = {}
    daily_stage_hours: dict[date, dict[str, float]] = {}
    bedtimes: list[float] = []

    for record in sleep_records:
      if not isinstance(record, SleepRecord):
        continue

      duration_hours = (record.end_date - record.start_date).total_seconds() / 3600
      if duration_hours <= 0:
        continue

      date_key = record.start_date.date()
      daily_stage_hours.setdefault(
        date_key,
        {
          "core": 0.0,
          "deep": 0.0,
          "rem": 0.0,
          "asleep": 0.0,
        },
      )

      if record.is_in_bed:
        daily_in_bed_hours[date_key] = (
          daily_in_bed_hours.get(date_key, 0.0) + duration_hours
        )
        bedtime_hour = record.start_date.hour + record.start_date.minute / 60
        bedtimes.append(bedtime_hour)
      elif record.is_awake:
        # Awake time does not count toward sleep
        pass
      else:
        daily_sleep_hours[date_key] = (
          daily_sleep_hours.get(date_key, 0.0) + duration_hours
        )
        stage_value = record.sleep_stage.value.lower()
        if "core" in stage_value:
          daily_stage_hours[date_key]["core"] += duration_hours
        elif "deep" in stage_value:
          daily_stage_hours[date_key]["deep"] += duration_hours
        elif "rem" in stage_value:
          daily_stage_hours[date_key]["rem"] += duration_hours
        else:
          daily_stage_hours[date_key]["asleep"] += duration_hours

    if not daily_sleep_hours:
      return None

    sleep_durations = list(daily_sleep_hours.values())
    average_duration = float(np.mean(sleep_durations))
    if daily_in_bed_hours:
      efficiencies = []
      for date_key, sleep_hours in daily_sleep_hours.items():
        in_bed = daily_in_bed_hours.get(date_key, 0.0)
        if in_bed > 0:
          efficiencies.append(min(sleep_hours / in_bed, 1.0))
      average_efficiency = float(np.mean(efficiencies)) if efficiencies else 0.85
    else:
      average_efficiency = 0.85

    # Calculate consistency (lower standard deviation = higher consistency)
    duration_std = np.std(sleep_durations)
    consistency_score = max(0, 1 - (duration_std / average_duration))

    # Sleep stage ratios from aggregated stage data
    total_core = sum(stage["core"] for stage in daily_stage_hours.values())
    total_deep = sum(stage["deep"] for stage in daily_stage_hours.values())
    total_rem = sum(stage["rem"] for stage in daily_stage_hours.values())
    total_asleep = sum(stage["asleep"] for stage in daily_stage_hours.values())
    total_stage_sleep = total_core + total_deep + total_rem + total_asleep
    if total_stage_sleep > 0:
      deep_sleep_ratio = total_deep / total_stage_sleep
      rem_sleep_ratio = total_rem / total_stage_sleep
    else:
      deep_sleep_ratio = 0.18
      rem_sleep_ratio = 0.22

    # Calculate sleep debt (assuming 8 hours is optimal)
    optimal_sleep = 8.0
    sleep_debt = max(0, optimal_sleep - average_duration)

    # Circadian rhythm score based on bedtime consistency
    if bedtimes:
      bedtime_std = float(np.std(bedtimes))
      circadian_rhythm_score = max(0.0, 1.0 - bedtime_std / 3.0)
    else:
      circadian_rhythm_score = consistency_score * 0.8

    # Determine sleep quality trend from daily sleep duration
    sleep_quality_trend = self._calculate_trend_label(daily_sleep_hours)

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
    self, activity_records: Sequence[HealthRecord]
  ) -> ActivityPatternAnalysis | None:
    """Analyze activity patterns and exercise habits."""
    if not activity_records:
      return None

    logger.info(
      self.translator.t(
        "log.extended_analyzer.activity_patterns",
        count=len(activity_records),
      )
    )

    # Convert to optimized DataFrame
    odf = OptimizedDataFrame.from_records(activity_records)
    if odf.record_count == 0:
      return None

    df = odf.to_pandas()

    # Focus on step count where possible
    step_df = df[df["type"].str.contains("step", case=False, na=False)].copy()
    if step_df.empty:
      step_df = df.copy()

    daily_steps = step_df.groupby(step_df["timestamp"].dt.date)["value"].sum()
    daily_step_average = int(daily_steps.mean()) if not daily_steps.empty else 0

    # Calculate weekly exercise frequency from activity data
    # Simplified: count days with activity above threshold
    if not df.empty:
      active_days = (
        df.groupby(df["timestamp"].dt.date)["value"].sum() > 1000
      )  # Arbitrary threshold
      weekly_exercise_frequency = active_days.sum() / max(
        1, len(active_days.unique()) / 7
      )
    else:
      weekly_exercise_frequency = 0.0

    # Estimate sedentary and active hours based on activity data
    # Simplified: assume 24 hours total, estimate based on activity levels
    total_hours = 24.0
    if not daily_steps.empty:
      # Rough estimate: higher steps = more active time
      activity_ratio = min(daily_steps.mean() / 10000, 1.0)
      active_hours_daily = activity_ratio * 8  # Max 8 hours active
      sedentary_hours_daily = total_hours - active_hours_daily - 8  # Subtract sleep
    else:
      sedentary_hours_daily = 16.0
      active_hours_daily = 0.0

    # Determine peak activity hour from hourly data
    if not df.empty:
      hourly_activity = df.groupby(df["timestamp"].dt.hour)["value"].sum()
      peak_activity_hour = (
        int(hourly_activity.idxmax()) if not hourly_activity.empty else 12
      )
    else:
      peak_activity_hour = 12  # Default midday

    # Calculate activity consistency score based on daily variation
    if not daily_steps.empty and len(daily_steps) > 1:
      activity_consistency_score = max(0, 1 - (daily_steps.std() / daily_steps.mean()))
    else:
      activity_consistency_score = 0.0

    # Exercise intensity distribution derived from daily step quantiles
    if len(daily_steps) >= 3:
      p33 = float(daily_steps.quantile(0.33))
      p66 = float(daily_steps.quantile(0.66))
      light_days = int((daily_steps <= p33).sum())
      moderate_days = int(((daily_steps > p33) & (daily_steps <= p66)).sum())
      vigorous_days = int((daily_steps > p66).sum())
      total_days = len(daily_steps)
      exercise_intensity_distribution = {
        "light": round(light_days / total_days, 2),
        "moderate": round(moderate_days / total_days, 2),
        "vigorous": round(vigorous_days / total_days, 2),
      }
    else:
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
    categorized_records: Mapping[str, Sequence[HealthRecord]],
    age: int | None,
    gender: str | None,
    weight_kg: float | None,
    height_cm: float | None,
  ) -> MetabolicHealthAnalysis | None:
    """Analyze metabolic health indicators."""
    body_metrics = categorized_records.get("body_metrics", [])

    if not body_metrics and not (weight_kg and height_cm and age):
      return None

    logger.info(self.translator.t("log.extended_analyzer.metabolic_health"))

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

    # Calculate hydration score based on available data or default
    # Would need water intake records for accurate calculation
    nutrition_records = categorized_records.get("nutrition", [])
    if nutrition_records:
      # Simple proxy: presence of nutrition data suggests hydration tracking
      hydration_score = 0.8
    else:
      hydration_score = 0.5  # Default when no data

    # Calculate metabolic age based on BMR and body composition
    if basal_metabolic_rate and age:
      # Simplified: metabolic age is estimated from BMR deviation from expected
      expected_bmr = (
        10 * (70 if gender and gender.lower() == "male" else 60)
        + 6.25 * 170
        - 5 * age
        + (5 if gender and gender.lower() == "male" else -161)
      )
      if expected_bmr > 0:
        ratio = basal_metabolic_rate / expected_bmr
        metabolic_age = age if abs(1 - ratio) < 0.1 else int(age * ratio)
      else:
        metabolic_age = age
    else:
      metabolic_age = age

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
    self, categorized_records: Mapping[str, Sequence[HealthRecord]]
  ) -> StressResilienceAnalysis | None:
    """Analyze stress resilience and recovery capacity."""
    hrv_records = categorized_records.get("hrv", [])
    heart_rate_records = categorized_records.get("heart_rate", [])

    if not heart_rate_records and not hrv_records:
      return None

    logger.info(self.translator.t("log.extended_analyzer.stress_resilience"))

    # Prefer HRV-based stress scoring when available
    stress_accumulation_score = 0.3
    if hrv_records:
      odf_hrv = OptimizedDataFrame.from_records(hrv_records)
      if odf_hrv.record_count > 0:
        hrv_mean = float(np.nanmean(odf_hrv.values))
        if np.isfinite(hrv_mean) and hrv_mean > 0:
          stress_accumulation_score = float(max(0.0, min(1.0, 1.0 - (hrv_mean / 100))))
    elif heart_rate_records:
      odf_hr = OptimizedDataFrame.from_records(heart_rate_records)
      if odf_hr.record_count > 0:
        hr_values = odf_hr.values
        hr_std = float(np.nanstd(hr_values))
        hr_mean = float(np.nanmean(hr_values))
        if hr_mean > 0 and np.isfinite(hr_std) and np.isfinite(hr_mean):
          stress_accumulation_score = min(hr_std / hr_mean, 1.0)

    # Recovery capacity score based on stress level
    recovery_capacity_score = 1.0 - stress_accumulation_score

    # Assess burnout risk based on stress accumulation
    if stress_accumulation_score > 0.8:
      burnout_risk_level = "critical"
    elif stress_accumulation_score > 0.6:
      burnout_risk_level = "high"
    elif stress_accumulation_score > 0.4:
      burnout_risk_level = "moderate"
    else:
      burnout_risk_level = "low"

    # Resilience trend (simplified)
    resilience_trend = "stable"  # Would need historical data

    # Recommended rest periods
    recommended_rest_periods = [
      self.translator.t("extended.recommendation.rest_days"),
      self.translator.t("extended.recommendation.sleep_7_8"),
      self.translator.t("extended.recommendation.meditation"),
    ]

    return StressResilienceAnalysis(
      stress_accumulation_score=stress_accumulation_score,
      recovery_capacity_score=recovery_capacity_score,
      burnout_risk_level=burnout_risk_level,
      resilience_trend=resilience_trend,
      recommended_rest_periods=recommended_rest_periods,
    )

  def _analyze_health_correlations(
    self, categorized_records: Mapping[str, Sequence[HealthRecord]]
  ) -> dict[str, Any]:
    """Analyze correlations between different health metrics."""
    correlations: dict[str, Any] = {}

    sleep_records = categorized_records.get("sleep", [])
    activity_records = categorized_records.get("activity", [])
    heart_rate_records = categorized_records.get("heart_rate", [])

    sleep_daily = self._daily_sleep_hours(sleep_records)
    activity_daily = self._daily_activity_values(activity_records)

    if sleep_daily and activity_daily:
      sleep_series = pd.Series(sleep_daily)
      activity_series = pd.Series(activity_daily)
      aligned = pd.concat([sleep_series, activity_series], axis=1).dropna()
      if len(aligned) >= 3:
        corr_val = safe_float(
          numpy_to_python_scalar(aligned.corr(numeric_only=True).iloc[0, 1])
        )
        correlations["sleep_activity"] = {
          "correlation": round(corr_val, 3),
          "sample_size": len(aligned),
          "confidence": self._correlation_confidence(len(aligned)),
          "insight": self.translator.t("extended.correlation.sleep_activity"),
        }

    if sleep_daily and heart_rate_records:
      hr_daily = self._daily_heart_rate_mean(heart_rate_records)
      if hr_daily:
        sleep_series = pd.Series(sleep_daily)
        hr_series = pd.Series(hr_daily)
        aligned = pd.concat([sleep_series, hr_series], axis=1).dropna()
        if len(aligned) >= 3:
          corr_val = safe_float(
            numpy_to_python_scalar(aligned.corr(numeric_only=True).iloc[0, 1])
          )
          correlations["sleep_hr"] = {
            "correlation": round(corr_val, 3),
            "sample_size": len(aligned),
            "confidence": self._correlation_confidence(len(aligned)),
            "insight": self.translator.t("extended.correlation.hr_stress"),
          }

    return correlations

  def _daily_sleep_hours(
    self, sleep_records: Sequence[HealthRecord]
  ) -> dict[str, float]:
    """Aggregate daily sleep hours from sleep records."""
    daily: dict[str, float] = {}
    for record in sleep_records:
      if isinstance(record, SleepRecord) and record.is_asleep:
        duration_hours = (record.end_date - record.start_date).total_seconds() / 3600
        if duration_hours > 0:
          key = record.start_date.date().isoformat()
          daily[key] = daily.get(key, 0.0) + duration_hours
    return daily

  def _daily_activity_values(
    self, activity_records: Sequence[HealthRecord]
  ) -> dict[str, float]:
    """Aggregate daily activity values from activity records."""
    daily: dict[str, float] = {}
    for record in activity_records:
      if isinstance(record, QuantityRecord):
        value = float(record.value)
        if value >= 0:
          key = record.start_date.date().isoformat()
          daily[key] = daily.get(key, 0.0) + value
    return daily

  def _daily_heart_rate_mean(
    self, heart_rate_records: Sequence[HealthRecord]
  ) -> dict[str, float]:
    """Aggregate daily mean heart rate."""
    daily_values: dict[str, list[float]] = {}
    for record in heart_rate_records:
      if isinstance(record, QuantityRecord):
        key = record.start_date.date().isoformat()
        daily_values.setdefault(key, []).append(float(record.value))
    return {
      key: float(np.mean(values)) for key, values in daily_values.items() if values
    }

  def _correlation_confidence(self, sample_size: int) -> float:
    """Estimate correlation confidence from sample size."""
    if sample_size <= 3:
      return 0.3
    if sample_size <= 7:
      return 0.5
    if sample_size <= 14:
      return 0.7
    return 0.85

  def _calculate_trend_label(self, daily_values: dict[date, float]) -> str:
    """Calculate trend label for daily values."""
    if len(daily_values) < 3:
      return "stable"
    df = pd.DataFrame(
      {
        "date": pd.to_datetime(list(daily_values.keys())),
        "value": list(daily_values.values()),
      }
    ).sort_values("date")
    df["index"] = np.arange(len(df))
    slope = np.polyfit(df["index"], df["value"], 1)[0]
    if abs(slope) < 0.02:
      return "stable"
    return "improving" if slope > 0 else "declining"

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
      insights.append(self.translator.t("extended.predictive.sleep_debt"))

    if activity_patterns and activity_patterns.daily_step_average < 5000:
      insights.append(self.translator.t("extended.predictive.low_steps"))

    if metabolic_health and metabolic_health.metabolic_health_score < 0.6:
      insights.append(self.translator.t("extended.predictive.metabolic_attention"))

    if stress_resilience and stress_resilience.burnout_risk_level in [
      "high",
      "critical",
    ]:
      insights.append(self.translator.t("extended.predictive.burnout_risk"))

    if not insights:
      insights.append(self.translator.t("extended.predictive.overall_good"))

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

    if not scores:
      return 0.5

    valid_scores = [score for score in scores if np.isfinite(score)]
    if not valid_scores:
      return 0.5

    return round(float(np.mean(valid_scores)), 2)

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
      actions.append(self.translator.t("extended.priority_actions.sleep_debt"))

    if stress_resilience and stress_resilience.burnout_risk_level == "critical":
      actions.append(self.translator.t("extended.priority_actions.reduce_burnout"))

    if activity_patterns and activity_patterns.daily_step_average < 3000:
      actions.append(self.translator.t("extended.priority_actions.increase_steps"))

    if not actions:
      actions.append(self.translator.t("extended.priority_actions.maintain"))

    return actions

  def _generate_lifestyle_optimization(
    self, health_correlations: dict[str, Any], predictive_insights: list[str]
  ) -> list[str]:
    """Generate lifestyle optimization recommendations."""
    optimizations = [
      self.translator.t("extended.optimization.routine"),
      self.translator.t("extended.optimization.aerobic"),
      self.translator.t("extended.optimization.diet"),
      self.translator.t("extended.optimization.stress_management"),
    ]

    return optimizations

  def _assess_data_completeness(
    self, categorized_records: Mapping[str, Sequence[HealthRecord]]
  ) -> Any:  # type: ignore
    """Assess data completeness across different health categories."""
    categories = [
      "sleep",
      "activity",
      "heart_rate",
      "hrv",
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
        score = min(len(records) / 10, 1.0)  # Less data expected for other metrics
      completeness_scores.append(score)

    return np.mean(completeness_scores).item()

  def _calculate_analysis_confidence(self, data_completeness: float) -> float:
    """Calculate analysis confidence based on data completeness."""
    # Higher completeness leads to higher confidence
    return round(data_completeness * 0.9 + 0.1, 2)  # Min confidence of 0.1

  def _calculate_data_range(
    self, records: Sequence[HealthRecord]
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
