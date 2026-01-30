"""Heart rate data specialized analysis module.

Provides in-depth analysis of heart rate related data, including resting heart rate, HRV, exercise heart rate, and cardio fitness.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Sequence

import pandas as pd

from ..analyzers.anomaly import AnomalyDetector, AnomalyReport
from ..analyzers.statistical import StatisticalAnalyzer
from ..core.data_models import HealthRecord, QuantityRecord
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HeartRateMetrics:
  """Basic heart rate metrics"""

  resting_hr: float | None = None
  hrv_sdnn: float | None = None
  walking_hr_avg: float | None = None
  hr_recovery_1min: float | None = None
  vo2_max: float | None = None
  timestamp: datetime | None = None


@dataclass
class RestingHRAnalysis:
  """Resting heart rate analysis results"""

  current_value: float
  baseline_value: float
  change_from_baseline: float
  trend_direction: Literal["increasing", "decreasing", "stable"]
  age_adjusted_percentile: float | None = None
  health_rating: Literal["excellent", "good", "fair", "poor"] = "good"


@dataclass
class HRVAnalysis:
  """Heart rate variability analysis results"""

  current_sdnn: float
  baseline_sdnn: float
  change_from_baseline: float
  stress_level: Literal["low", "moderate", "high", "very_high"]
  recovery_status: Literal["excellent", "good", "fair", "poor"]
  trend_direction: Literal["improving", "declining", "stable"]


@dataclass
class CardioFitnessAnalysis:
  """Cardio fitness analysis results"""

  current_vo2_max: float
  age_adjusted_rating: Literal["superior", "excellent", "good", "fair", "poor"]
  fitness_percentile: float
  improvement_potential: float
  training_recommendations: list[str]


@dataclass
class HeartRateAnalysisReport:
  """Comprehensive heart rate analysis report"""

  analysis_date: datetime
  data_range: tuple[datetime, datetime]

  # Core metrics
  resting_hr_analysis: RestingHRAnalysis | None = None
  hrv_analysis: HRVAnalysis | None = None
  cardio_fitness: CardioFitnessAnalysis | None = None

  # Statistical analysis
  daily_stats: pd.DataFrame | None = None
  weekly_stats: pd.DataFrame | None = None
  monthly_stats: pd.DataFrame | None = None

  # Anomaly detection
  anomalies: list[Any] | None = None
  anomaly_report: AnomalyReport | None = None

  # Trend analysis
  trends: dict[str, Any] | None = None

  # Highlights and recommendations
  highlights: list[str] | None = None
  recommendations: list[str] | None = None

  # Data quality
  data_quality_score: float = 0.0
  record_count: int = 0


class HeartRateAnalyzer:
  """Specialized heart rate data analyzer

  Provides in-depth analysis of heart rate related data, including:
  - Resting heart rate trend analysis
  - Heart rate variability (HRV) assessment
  - Exercise heart rate analysis
  - Cardio fitness rating
  - Anomaly detection and health insights
  """

  def __init__(
    self,
    age: int | None = None,
    gender: Literal["male", "female"] | None = None,
  ):
    """Initialize heart rate analyzer

    Args:
        age: Age (used for normal range calculations)
        gender: Gender (used for cardio fitness rating)
    """
    self.age = age
    self.gender = gender

    # Initialize analysis components.
    self.stat_analyzer = StatisticalAnalyzer()
    self.anomaly_detector = AnomalyDetector()

    logger.info(f"HeartRateAnalyzer initialized (age: {age}, gender: {gender})")

  def analyze_comprehensive(
    self,
    heart_rate_records: list[HealthRecord],
    resting_hr_records: list[HealthRecord] | None = None,
    hrv_records: list[HealthRecord] | None = None,
    walking_hr_records: list[HealthRecord] | None = None,
    vo2_max_records: list[HealthRecord] | None = None,
  ) -> HeartRateAnalysisReport:
    """Perform comprehensive heart rate data analysis

    Args:
        heart_rate_records: Basic heart rate records
        resting_hr_records: Resting heart rate records
        hrv_records: Heart rate variability records
        walking_hr_records: Walking heart rate records
        vo2_max_records: VO2Max records

    Returns:
        Comprehensive analysis report
    """
    logger.info("Starting comprehensive heart rate analysis")

    # Determine data time range.
    all_records = (
      heart_rate_records
      + (resting_hr_records or [])
      + (hrv_records or [])
      + (walking_hr_records or [])
      + (vo2_max_records or [])
    )

    if not all_records:
      logger.warning("No heart rate records provided for analysis")
      return HeartRateAnalysisReport(
        analysis_date=datetime.now(),
        data_range=(datetime.now(), datetime.now()),
      )

    data_range = self._calculate_data_range(all_records)
    analysis_date = datetime.now()

    # Analyze metrics.
    resting_hr_analysis = None
    if resting_hr_records:
      resting_hr_analysis = self.analyze_resting_heart_rate(resting_hr_records)

    hrv_analysis = None
    if hrv_records:
      hrv_analysis = self.analyze_hrv(hrv_records)

    cardio_fitness = None
    if vo2_max_records:
      # Filter QuantityRecord entries.
      quantity_records = [r for r in vo2_max_records if isinstance(r, QuantityRecord)]
      cardio_fitness = self.analyze_cardio_fitness(quantity_records)

    # Statistical analysis (based on base heart rate data).
    daily_stats = self.stat_analyzer.aggregate_by_interval(heart_rate_records, "day")
    weekly_stats = self.stat_analyzer.aggregate_by_interval(heart_rate_records, "week")
    monthly_stats = self.stat_analyzer.aggregate_by_interval(
      heart_rate_records, "month"
    )

    # Anomaly detection.
    anomalies = self.anomaly_detector.detect_anomalies(
      heart_rate_records, ["zscore", "iqr"]
    )
    anomaly_report = self.anomaly_detector.generate_report(
      anomalies, len(heart_rate_records)
    )

    # Trend analysis.
    trends = {}
    if not daily_stats.empty:
      hr_trend = self.stat_analyzer.analyze_trend(
        daily_stats, "interval_start", "mean_value"
      )
      if hr_trend:
        trends["heart_rate"] = hr_trend

    # Generate highlights and recommendations.
    highlights = self._generate_highlights(
      resting_hr_analysis, hrv_analysis, cardio_fitness, trends, anomalies
    )
    recommendations = self._generate_recommendations(
      resting_hr_analysis, hrv_analysis, cardio_fitness, anomalies
    )

    # Data quality assessment.
    data_quality = self._assess_data_quality(heart_rate_records)

    report = HeartRateAnalysisReport(
      analysis_date=analysis_date,
      data_range=data_range,
      resting_hr_analysis=resting_hr_analysis,
      hrv_analysis=hrv_analysis,
      cardio_fitness=cardio_fitness,
      daily_stats=daily_stats,
      weekly_stats=weekly_stats,
      monthly_stats=monthly_stats,
      anomalies=anomalies,
      anomaly_report=anomaly_report,
      trends=trends,
      highlights=highlights,
      recommendations=recommendations,
      data_quality_score=data_quality,
      record_count=len(heart_rate_records),
    )

    logger.info("Comprehensive heart rate analysis completed")
    return report

  def analyze_resting_heart_rate(
    self, records: list[HealthRecord]
  ) -> RestingHRAnalysis | None:
    """Analyze resting heart rate

    Args:
        records: Resting heart rate records

    Returns:
        Resting heart rate analysis results
    """
    if not records:
      return None

    logger.info(f"Analyzing resting heart rate from {len(records)} records")

    # Convert to DataFrame.
    df = self._records_to_dataframe(records)

    if df.empty or "value" not in df.columns:
      return None

    # Calculate current value (average of last 30 days).
    recent_data = df[df["start_date"] >= df["start_date"].max() - pd.Timedelta(days=30)]
    current_value = (
      recent_data["value"].mean() if not recent_data.empty else df["value"].mean()
    )

    # Calculate baseline value (average of earliest 30 days).
    baseline_data = df[
      df["start_date"] <= df["start_date"].min() + pd.Timedelta(days=30)
    ]
    baseline_value = (
      baseline_data["value"].mean() if not baseline_data.empty else df["value"].mean()
    )

    # Calculate change.
    change_from_baseline = current_value - baseline_value

    # Determine trend direction.
    if abs(change_from_baseline) < 1:  # Change below 1 bpm is stable.
      trend_direction = "stable"
    elif change_from_baseline < 0:
      trend_direction = "decreasing"  # Decrease is favorable.
    else:
      trend_direction = "increasing"  # Increase may need attention.

    # Age-adjusted percentile (when age is available).
    age_adjusted_percentile = None
    if self.age:
      age_adjusted_percentile = self._calculate_age_adjusted_percentile(
        current_value, self.age
      )

    # Health rating.
    health_rating = self._rate_resting_hr_health(current_value, self.age)

    return RestingHRAnalysis(
      current_value=round(float(current_value), 1),
      baseline_value=round(float(baseline_value), 1),
      change_from_baseline=round(float(change_from_baseline), 1),
      trend_direction=trend_direction,
      age_adjusted_percentile=age_adjusted_percentile,
      health_rating=health_rating,
    )

  def analyze_hrv(self, records: list[HealthRecord]) -> HRVAnalysis | None:
    """Analyze heart rate variability (HRV)

    Args:
        records: HRV records (SDNN values)

    Returns:
        HRV analysis results
    """
    if not records:
      return None

    logger.info(f"Analyzing HRV from {len(records)} records")

    # Convert to DataFrame.
    df = self._records_to_dataframe(records)

    if df.empty or "value" not in df.columns:
      return None

    # Calculate current value (average of last 30 days).
    recent_data = df[df["start_date"] >= df["start_date"].max() - pd.Timedelta(days=30)]
    current_sdnn = (
      recent_data["value"].mean() if not recent_data.empty else df["value"].mean()
    )

    # Calculate baseline value.
    baseline_data = df[
      df["start_date"] <= df["start_date"].min() + pd.Timedelta(days=30)
    ]
    baseline_sdnn = (
      baseline_data["value"].mean() if not baseline_data.empty else df["value"].mean()
    )

    # Calculate change.
    change_from_baseline = current_sdnn - baseline_sdnn

    # Assess stress level (based on SDNN).
    stress_level = self._assess_stress_level(current_sdnn)

    # Assess recovery status.
    recovery_status = self._assess_recovery_status(current_sdnn)

    # Determine trend direction.
    if abs(change_from_baseline) < 2:  # SDNN change below 2ms is stable.
      trend_direction = "stable"
    elif change_from_baseline > 0:
      trend_direction = "improving"  # HRV increase is favorable.
    else:
      trend_direction = "declining"  # HRV decline needs attention.

    return HRVAnalysis(
      current_sdnn=round(float(current_sdnn), 1),
      baseline_sdnn=round(float(baseline_sdnn), 1),
      change_from_baseline=round(float(change_from_baseline), 1),
      stress_level=stress_level,
      recovery_status=recovery_status,
      trend_direction=trend_direction,
    )

  def analyze_cardio_fitness(
    self, records: list[QuantityRecord]
  ) -> CardioFitnessAnalysis | None:
    """Analyze cardio fitness

    Args:
        records: VO2Max records

    Returns:
        Cardio fitness analysis results
    """
    if not records or not self.age or not self.gender:
      logger.warning("VO2Max analysis requires age and gender information")
      return None

    logger.info(f"Analyzing cardio fitness from {len(records)} VO2Max records")

    # Convert to DataFrame.
    df = self._records_to_dataframe(records)  # type: ignore

    if df.empty or "value" not in df.columns:
      return None

    # Get current VO2Max value (latest record).
    current_vo2_max = df["value"].iloc[-1]  # Assume records are time-sorted.

    # Age- and gender-adjusted rating.
    age_adjusted_rating = self._rate_vo2_max(current_vo2_max, self.age, self.gender)

    # Calculate percentile.
    fitness_percentile = self._calculate_vo2_max_percentile(
      current_vo2_max, self.age, self.gender
    )

    # Assess improvement potential.
    improvement_potential = self._calculate_improvement_potential(
      current_vo2_max, self.age, self.gender
    )

    # Generate training recommendations.
    training_recommendations = self._generate_training_recommendations(
      current_vo2_max, self.age, self.gender, age_adjusted_rating
    )

    return CardioFitnessAnalysis(
      current_vo2_max=round(float(current_vo2_max), 1),
      age_adjusted_rating=age_adjusted_rating,
      fitness_percentile=round(fitness_percentile, 1),
      improvement_potential=round(improvement_potential, 1),
      training_recommendations=training_recommendations,
    )

  def _calculate_data_range(
    self, records: list[HealthRecord]
  ) -> tuple[datetime, datetime]:
    """Compute the data time range."""
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

  def _records_to_dataframe(self, records: Sequence[HealthRecord]) -> pd.DataFrame:
    """Convert health records into a DataFrame."""
    data = []
    for record in records:
      # Read value.
      value = None
      if isinstance(record, (QuantityRecord)):
        value = record.value

      data.append(
        {
          "type": record.type,
          "source_name": record.source_name,
          "start_date": record.start_date,
          "end_date": record.end_date,
          "value": value,
          "unit": record.unit,
        }
      )

    return pd.DataFrame(data)

  def _calculate_age_adjusted_percentile(self, resting_hr: float, age: int) -> float:
    """Calculate age-adjusted resting heart rate percentile."""
    # Age-based normal resting HR range (simplified model).
    # A more precise percentile table should be used in production.
    if age < 30:
      normal_range = (50, 80)
    elif age < 50:
      normal_range = (55, 85)
    else:
      normal_range = (60, 90)

    if resting_hr <= normal_range[0]:
      return 25.0  # Lower percentile.
    elif resting_hr >= normal_range[1]:
      return 75.0  # Higher percentile.
    else:
      # Linear interpolation.
      return (
        25 + (resting_hr - normal_range[0]) / (normal_range[1] - normal_range[0]) * 50
      )

  def _rate_resting_hr_health(
    self, resting_hr: float, age: int | None
  ) -> Literal["excellent", "good", "fair", "poor"]:
    """Assess resting heart rate health level."""
    if age and age < 30:
      if resting_hr < 60:
        return "excellent"
      elif resting_hr < 70:
        return "good"
      elif resting_hr < 80:
        return "fair"
      else:
        return "poor"
    else:
      if resting_hr < 65:
        return "excellent"
      elif resting_hr < 75:
        return "good"
      elif resting_hr < 85:
        return "fair"
      else:
        return "poor"

  def _assess_stress_level(
    self, sdnn: float
  ) -> Literal["low", "moderate", "high", "very_high"]:
    """Assess stress level (based on SDNN)."""
    if sdnn >= 50:
      return "low"
    elif sdnn >= 30:
      return "moderate"
    elif sdnn >= 15:
      return "high"
    else:
      return "very_high"

  def _assess_recovery_status(
    self, sdnn: float
  ) -> Literal["excellent", "good", "fair", "poor"]:
    """Assess recovery status (based on SDNN)."""
    if sdnn >= 60:
      return "excellent"
    elif sdnn >= 40:
      return "good"
    elif sdnn >= 20:
      return "fair"
    else:
      return "poor"

  def _rate_vo2_max(
    self, vo2_max: float, age: int, gender: str
  ) -> Literal["superior", "excellent", "good", "fair", "poor"]:
    """Rate VO2Max level."""
    # Simplified VO2Max rating table (ml/kg/min).
    if gender == "male":
      if age < 30:
        thresholds = {"superior": 50, "excellent": 45, "good": 40, "fair": 35}
      elif age < 40:
        thresholds = {"superior": 48, "excellent": 43, "good": 38, "fair": 33}
      else:
        thresholds = {"superior": 45, "excellent": 40, "good": 35, "fair": 30}
    else:  # female
      if age < 30:
        thresholds = {"superior": 45, "excellent": 40, "good": 35, "fair": 30}
      elif age < 40:
        thresholds = {"superior": 42, "excellent": 37, "good": 32, "fair": 27}
      else:
        thresholds = {"superior": 40, "excellent": 35, "good": 30, "fair": 25}

    if vo2_max >= thresholds["superior"]:
      return "superior"
    elif vo2_max >= thresholds["excellent"]:
      return "excellent"
    elif vo2_max >= thresholds["good"]:
      return "good"
    elif vo2_max >= thresholds["fair"]:
      return "fair"
    else:
      return "poor"

  def _calculate_vo2_max_percentile(
    self, vo2_max: float, age: int, gender: str
  ) -> float:
    """Calculate VO2Max percentile (simplified)."""
    # Simplified percentile calculation.
    # A more accurate distribution should be used in production.
    rating = self._rate_vo2_max(vo2_max, age, gender)

    rating_to_percentile = {
      "superior": 90,
      "excellent": 75,
      "good": 50,
      "fair": 25,
      "poor": 10,
    }

    return rating_to_percentile[rating]

  def _calculate_improvement_potential(
    self, vo2_max: float, age: int, gender: str
  ) -> float:
    """Calculate improvement potential (0-100)."""
    current_rating = self._rate_vo2_max(vo2_max, age, gender)

    # Distance to the next rating level.
    rating_order = ["poor", "fair", "good", "excellent", "superior"]
    current_index = rating_order.index(current_rating)

    if current_index >= len(rating_order) - 1:
      return 0.0  # Already at the highest level.

    # Simplified improvement potential calculation.
    return (len(rating_order) - 1 - current_index) * 25

  def _generate_training_recommendations(
    self, vo2_max: float, age: int, gender: str, rating: str
  ) -> list[str]:
    """Generate training recommendations."""
    recommendations = []

    if rating in ["poor", "fair"]:
      recommendations.extend(
        [
          "建议每周进行3-4次有氧运动，每次30-45分钟",
          "结合力量训练，每周2-3次",
          "逐渐增加运动强度，避免过度疲劳",
        ]
      )
    elif rating == "good":
      recommendations.extend(
        [
          "保持当前训练强度，每周4-5次有氧运动",
          "尝试间歇训练来提升心肺适能",
          "定期监测VO2Max变化",
        ]
      )
    elif rating in ["excellent", "superior"]:
      recommendations.extend(
        [
          "维持高强度训练，考虑竞技运动",
          "关注恢复和营养补充",
          "可以尝试更高级的训练方法",
        ]
      )

    return recommendations

  def _generate_highlights(
    self,
    resting_hr: RestingHRAnalysis | None,
    hrv: HRVAnalysis | None,
    cardio: CardioFitnessAnalysis | None,
    trends: dict[str, Any],
    anomalies: list[Any],
  ) -> list[str]:
    """Generate highlights."""
    highlights = []

    # Resting heart rate highlights.
    if resting_hr:
      if resting_hr.trend_direction == "decreasing":
        highlights.append(
          f"🏆 静息心率下降{abs(resting_hr.change_from_baseline):.1f} bpm，健康状况改善"
        )
      elif resting_hr.trend_direction == "increasing":
        highlights.append(
          f"⚠️ 静息心率上升{resting_hr.change_from_baseline:.1f} bpm，建议关注"
        )

      if resting_hr.health_rating in ["excellent", "good"]:
        highlights.append(
          f"💚 静息心率{resting_hr.current_value:.0f} bpm，处于{resting_hr.health_rating}水平"
        )

    # HRV Highlights
    if hrv:
      if hrv.trend_direction == "improving":
        highlights.append(
          f"📈 HRV改善{abs(hrv.change_from_baseline):.1f} ms，恢复能力增强"
        )
      elif hrv.trend_direction == "declining":
        highlights.append(
          f"⚠️ HRV下降{abs(hrv.change_from_baseline):.1f} ms，建议管理压力"
        )

      if hrv.stress_level == "low":
        highlights.append("😌 压力水平较低，心率变异性良好")
      elif hrv.stress_level in ["high", "very_high"]:
        highlights.append("😰 检测到较高压力水平，建议放松")

    # Cardio fitness highlights.
    if cardio:
      rating_desc = {
        "superior": "卓越",
        "excellent": "优秀",
        "good": "良好",
        "fair": "一般",
        "poor": "需要改善",
      }
      highlights.append(
        f"🏃 心肺适能评级：{rating_desc[cardio.age_adjusted_rating]}（VO2Max: {cardio.current_vo2_max:.1f}）"
      )

    # Anomaly highlights.
    if anomalies:
      anomaly_count = len(anomalies)
      if anomaly_count > 0:
        highlights.append(f"🔍 检测到{anomaly_count}个心率异常事件，建议查看详细报告")

    return highlights

  def _generate_recommendations(
    self,
    resting_hr: RestingHRAnalysis | None,
    hrv: HRVAnalysis | None,
    cardio: CardioFitnessAnalysis | None,
    anomalies: list[Any],
  ) -> list[str]:
    """Generate recommendations."""
    recommendations = []

    # Recommendations based on resting heart rate.
    if resting_hr and resting_hr.health_rating == "poor":
      recommendations.append("建议增加有氧运动，如快走、跑步或骑行，每周至少150分钟")

    # Recommendations based on HRV.
    if hrv and hrv.stress_level in ["high", "very_high"]:
      recommendations.extend(
        [
          "建议进行压力管理，如冥想、深呼吸或适量运动",
          "保证充足睡眠，每晚7-9小时",
          "考虑咨询专业医师了解健康状况",
        ]
      )

    # Recommendations based on cardio fitness.
    if cardio and cardio.training_recommendations:
      recommendations.extend(cardio.training_recommendations[:2])  # Use first two.

    # Recommendations based on anomalies.
    if anomalies and len(anomalies) > 10:  # Many anomalies.
      recommendations.append("心率异常较多，建议咨询心脏科医师进行检查")

    # General recommendations.
    if not recommendations:
      recommendations.append("保持规律运动和健康生活方式")
      recommendations.append("定期监测心率指标，关注身体变化")

    return recommendations

  def _assess_data_quality(self, records: list[HealthRecord]) -> float:
    """Assess data quality."""
    if not records:
      return 0.0

    # Simplified quality assessment.
    # Can be extended with more advanced logic.
    df = self._records_to_dataframe(records)

    if df.empty:
      return 0.0

    # Check data completeness.
    completeness = df["value"].notna().mean()

    # Check value sanity (heart rate range).
    reasonable = ((df["value"] >= 40) & (df["value"] <= 200)).mean()

    # Combined score.
    quality_score = (completeness + reasonable) / 2

    return round(float(quality_score), 3)
