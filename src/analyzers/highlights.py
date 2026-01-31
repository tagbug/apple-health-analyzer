"""Health highlights generator module for key insights and recommendations."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from ..i18n import Translator, resolve_locale
from ..processors.heart_rate import HeartRateAnalysisReport
from ..processors.sleep import SleepAnalysisReport
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HealthInsight:
  """Health insight data model."""

  category: Literal["heart_rate", "sleep", "correlation", "general"]
  priority: Literal["high", "medium", "low"]
  title: str
  message: str
  details: dict[str, Any] | None = None
  confidence: float = 1.0  # Confidence (0-1)
  timestamp: datetime | None = None


@dataclass
class HealthHighlights:
  """Health highlights summary."""

  analysis_date: datetime
  insights: list[HealthInsight]
  summary: dict[str, Any]
  recommendations: list[str]


class HighlightsGenerator:
  """Health highlights generator.

  Extracts key insights from heart rate and sleep reports.
  """

  def __init__(self, locale: str | None = None):
    """Initialize the highlights generator."""
    self.translator = Translator(resolve_locale(locale))
    logger.info(self.translator.t("log.highlights_generator.initialized"))

  def generate_comprehensive_highlights(
    self,
    heart_rate_report: HeartRateAnalysisReport | None = None,
    sleep_report: SleepAnalysisReport | None = None,
    correlation_data: dict[str, Any] | None = None,
  ) -> HealthHighlights:
    """Generate comprehensive health highlights.

    Args:
        heart_rate_report: Heart rate analysis report.
        sleep_report: Sleep analysis report.
        correlation_data: Correlation insights.

    Returns:
        Health highlights summary.
    """
    logger.info(self.translator.t("log.highlights_generator.generating"))

    insights = []

    # Heart rate insights.
    if heart_rate_report:
      hr_insights = self._generate_heart_rate_insights(heart_rate_report)
      insights.extend(hr_insights)

    # Sleep insights.
    if sleep_report:
      sleep_insights = self._generate_sleep_insights(sleep_report)
      insights.extend(sleep_insights)

    # Correlation insights.
    if correlation_data:
      corr_insights = self._generate_correlation_insights(correlation_data)
      insights.extend(corr_insights)

    # Rank and filter insights.
    insights = self._rank_and_filter_insights(insights)

    # Generate summary.
    summary = self._generate_summary(insights, heart_rate_report, sleep_report)

    # Generate recommendations.
    recommendations = self._generate_recommendations(insights)

    highlights = HealthHighlights(
      analysis_date=datetime.now(),
      insights=insights,
      summary=summary,
      recommendations=recommendations,
    )

    logger.info(
      self.translator.t(
        "log.highlights_generator.generated",
        insights=len(insights),
        recommendations=len(recommendations),
      )
    )
    return highlights

  def _generate_heart_rate_insights(
    self, report: HeartRateAnalysisReport
  ) -> list[HealthInsight]:
    """Generate heart rate related insights."""
    insights = []

    # Resting heart rate insights.
    if report.resting_hr_analysis:
      resting_hr = report.resting_hr_analysis

      # Trend insights.
      if abs(resting_hr.change_from_baseline) > 2:
        if resting_hr.trend_direction == "decreasing":
          insights.append(
            HealthInsight(
              category="heart_rate",
              priority="high",
              title=self.translator.t(
                "highlights.heart_rate.resting_hr_improved.title"
              ),
              message=self.translator.t(
                "highlights.heart_rate.resting_hr_improved.message",
                change=abs(resting_hr.change_from_baseline),
              ),
              details={
                "current": resting_hr.current_value,
                "baseline": resting_hr.baseline_value,
                "change": resting_hr.change_from_baseline,
                "trend": resting_hr.trend_direction,
              },
              confidence=0.9,
            )
          )
        elif resting_hr.trend_direction == "increasing":
          insights.append(
            HealthInsight(
              category="heart_rate",
              priority="medium",
              title=self.translator.t(
                "highlights.heart_rate.resting_hr_increased.title"
              ),
              message=self.translator.t(
                "highlights.heart_rate.resting_hr_increased.message",
                change=resting_hr.change_from_baseline,
              ),
              details={
                "current": resting_hr.current_value,
                "baseline": resting_hr.baseline_value,
                "change": resting_hr.change_from_baseline,
                "trend": resting_hr.trend_direction,
              },
              confidence=0.8,
            )
          )

      # Health rating insights.
      if resting_hr.health_rating == "excellent":
        insights.append(
          HealthInsight(
            category="heart_rate",
            priority="low",
            title=self.translator.t("highlights.heart_rate.resting_hr_excellent.title"),
            message=self.translator.t(
              "highlights.heart_rate.resting_hr_excellent.message",
              value=resting_hr.current_value,
            ),
            details={
              "rating": resting_hr.health_rating,
              "value": resting_hr.current_value,
            },
            confidence=0.95,
          )
        )
      elif resting_hr.health_rating == "poor":
        insights.append(
          HealthInsight(
            category="heart_rate",
            priority="high",
            title=self.translator.t("highlights.heart_rate.resting_hr_poor.title"),
            message=self.translator.t(
              "highlights.heart_rate.resting_hr_poor.message",
              value=resting_hr.current_value,
            ),
            details={
              "rating": resting_hr.health_rating,
              "value": resting_hr.current_value,
              "issue": "resting_hr_poor",
            },
            confidence=0.9,
          )
        )

    # HRV insights.
    if report.hrv_analysis:
      hrv = report.hrv_analysis

      if hrv.trend_direction == "improving":
        insights.append(
          HealthInsight(
            category="heart_rate",
            priority="medium",
            title=self.translator.t("highlights.heart_rate.hrv_improving.title"),
            message=self.translator.t("highlights.heart_rate.hrv_improving.message"),
            details={
              "current_sdnn": hrv.current_sdnn,
              "change": hrv.change_from_baseline,
              "stress_level": hrv.stress_level,
              "recovery_status": hrv.recovery_status,
            },
            confidence=0.85,
          )
        )
      elif hrv.trend_direction == "declining":
        insights.append(
          HealthInsight(
            category="heart_rate",
            priority="high",
            title=self.translator.t("highlights.heart_rate.hrv_declining.title"),
            message=self.translator.t("highlights.heart_rate.hrv_declining.message"),
            details={
              "current_sdnn": hrv.current_sdnn,
              "change": hrv.change_from_baseline,
              "stress_level": hrv.stress_level,
              "recovery_status": hrv.recovery_status,
            },
            confidence=0.9,
          )
        )

      # Stress level insights.
      if hrv.stress_level in ["high", "very_high"]:
        insights.append(
          HealthInsight(
            category="heart_rate",
            priority="high",
            title=self.translator.t("highlights.heart_rate.stress_high.title"),
            message=self.translator.t("highlights.heart_rate.stress_high.message"),
            details={
              "stress_level": hrv.stress_level,
              "sdnn": hrv.current_sdnn,
              "issue": "stress_high",
            },
            confidence=0.8,
          )
        )

    # Cardio fitness insights.
    if report.cardio_fitness:
      cardio = report.cardio_fitness

      rating_descriptions = {
        "superior": self.translator.t("highlights.heart_rate.cardio_rating.superior"),
        "excellent": self.translator.t("highlights.heart_rate.cardio_rating.excellent"),
        "good": self.translator.t("highlights.heart_rate.cardio_rating.good"),
        "fair": self.translator.t("highlights.heart_rate.cardio_rating.fair"),
        "poor": self.translator.t("highlights.heart_rate.cardio_rating.poor"),
      }

      insights.append(
        HealthInsight(
          category="heart_rate",
          priority="medium",
          title=self.translator.t("highlights.heart_rate.cardio_rating.title"),
          message=self.translator.t(
            "highlights.heart_rate.cardio_rating.message",
            rating=rating_descriptions[cardio.age_adjusted_rating],
            vo2=cardio.current_vo2_max,
          ),
          details={
            "rating": cardio.age_adjusted_rating,
            "vo2_max": cardio.current_vo2_max,
            "percentile": cardio.fitness_percentile,
          },
          confidence=0.9,
        )
      )

    # Anomaly detection insights.
    if report.anomalies and len(report.anomalies) > 0:
      anomaly_count = len(report.anomalies)
      if anomaly_count > 10:
        insights.append(
          HealthInsight(
            category="heart_rate",
            priority="medium",
            title=self.translator.t("highlights.heart_rate.anomalies_many.title"),
            message=self.translator.t(
              "highlights.heart_rate.anomalies_many.message",
              count=anomaly_count,
            ),
            details={"anomaly_count": anomaly_count},
            confidence=0.8,
          )
        )

    return insights

  def _generate_sleep_insights(
    self, report: SleepAnalysisReport
  ) -> list[HealthInsight]:
    """Generate sleep-related insights."""
    insights = []

    # Sleep quality insights.
    if report.quality_metrics:
      quality = report.quality_metrics

      # Duration insights.
      if quality.average_duration < 7:
        insights.append(
          HealthInsight(
            category="sleep",
            priority="high",
            title=self.translator.t("highlights.sleep.duration_low.title"),
            message=self.translator.t(
              "highlights.sleep.duration_low.message",
              hours=quality.average_duration,
            ),
            details={
              "average_duration": quality.average_duration,
              "issue": "sleep_duration_low",
            },
            confidence=0.95,
          )
        )
      elif quality.average_duration >= 8:
        insights.append(
          HealthInsight(
            category="sleep",
            priority="low",
            title=self.translator.t("highlights.sleep.duration_good.title"),
            message=self.translator.t(
              "highlights.sleep.duration_good.message",
              hours=quality.average_duration,
            ),
            details={"average_duration": quality.average_duration},
            confidence=0.9,
          )
        )

      # Efficiency insights.
      if quality.average_efficiency < 0.85:
        insights.append(
          HealthInsight(
            category="sleep",
            priority="high",
            title=self.translator.t("highlights.sleep.efficiency_low.title"),
            message=self.translator.t(
              "highlights.sleep.efficiency_low.message",
              percent=quality.average_efficiency,
            ),
            details={
              "average_efficiency": quality.average_efficiency,
              "issue": "sleep_efficiency_low",
            },
            confidence=0.9,
          )
        )

      # Consistency insights.
      if quality.consistency_score < 0.7:
        insights.append(
          HealthInsight(
            category="sleep",
            priority="medium",
            title=self.translator.t("highlights.sleep.consistency_low.title"),
            message=self.translator.t("highlights.sleep.consistency_low.message"),
            details={
              "consistency_score": quality.consistency_score,
              "issue": "sleep_consistency_low",
            },
            confidence=0.85,
          )
        )

    # Sleep pattern insights.
    if report.pattern_analysis:
      patterns = report.pattern_analysis

      # Social jetlag insights.
      if (
        patterns.weekday_vs_weekend
        and patterns.weekday_vs_weekend.get("social_jetlag", 0) > 2
      ):
        social_jetlag = patterns.weekday_vs_weekend["social_jetlag"]
        insights.append(
          HealthInsight(
            category="sleep",
            priority="medium",
            title=self.translator.t("highlights.sleep.social_jetlag_high.title"),
            message=self.translator.t(
              "highlights.sleep.social_jetlag_high.message",
              hours=social_jetlag,
            ),
            details={"social_jetlag": social_jetlag},
            confidence=0.8,
          )
        )

    # Sleep-heart rate correlation insights.
    if report.hr_correlation:
      hr_corr = report.hr_correlation

      if hr_corr.recovery_quality < 70:
        insights.append(
          HealthInsight(
            category="correlation",
            priority="medium",
            title=self.translator.t("highlights.sleep.hr_recovery_low.title"),
            message=self.translator.t("highlights.sleep.hr_recovery_low.message"),
            details={"recovery_quality": hr_corr.recovery_quality},
            confidence=0.8,
          )
        )

    # Anomaly detection insights.
    if report.anomalies and len(report.anomalies) > 0:
      anomaly_count = len(report.anomalies)
      if anomaly_count > 5:
        insights.append(
          HealthInsight(
            category="sleep",
            priority="medium",
            title=self.translator.t("highlights.sleep.anomalies_many.title"),
            message=self.translator.t(
              "highlights.sleep.anomalies_many.message",
              count=anomaly_count,
            ),
            details={"anomaly_count": anomaly_count},
            confidence=0.8,
          )
        )

    return insights

  def _generate_correlation_insights(
    self, correlation_data: dict[str, Any]
  ) -> list[HealthInsight]:
    """Generate correlation insights."""
    insights = []

    if not correlation_data:
      return insights

    for correlation_key, data in correlation_data.items():
      correlation_value = data.get("correlation", 0.0)
      insight_text = data.get("insight", "")

      # Determine priority by correlation strength.
      abs_corr = abs(correlation_value)
      if abs_corr >= 0.7:
        priority = "high"
        confidence = 0.9
      elif abs_corr >= 0.4:
        priority = "medium"
        confidence = 0.8
      else:
        priority = "low"
        confidence = 0.6

      # Build title.
      if correlation_key == "sleep_activity":
        title = self.translator.t("highlights.correlation.sleep_activity.title")
      elif correlation_key == "hr_stress":
        title = self.translator.t("highlights.correlation.hr_stress.title")
      else:
        title = self.translator.t(
          "highlights.correlation.default.title",
          metric=correlation_key,
        )

      # Generate fallback insight text when missing.
      if not insight_text:
        direction = (
          self.translator.t("highlights.correlation.direction.positive")
          if correlation_value > 0
          else self.translator.t("highlights.correlation.direction.negative")
        )
        if abs_corr >= 0.7:
          strength = self.translator.t("highlights.correlation.strength.strong")
        elif abs_corr >= 0.4:
          strength = self.translator.t("highlights.correlation.strength.medium")
        else:
          strength = self.translator.t("highlights.correlation.strength.weak")
        insight_text = self.translator.t(
          "highlights.correlation.fallback",
          strength=strength,
          direction=direction,
          value=correlation_value,
        )

      insights.append(
        HealthInsight(
          category="correlation",
          priority=priority,
          title=title,
          message=insight_text,
          details={"correlation": correlation_value, "type": correlation_key},
          confidence=confidence,
        )
      )

    return insights

  def _rank_and_filter_insights(
    self, insights: list[HealthInsight]
  ) -> list[HealthInsight]:
    """Rank and filter insights."""
    if not insights:
      return []

    # Sort by priority and confidence.
    priority_order = {"high": 3, "medium": 2, "low": 1}

    def sort_key(insight: HealthInsight) -> tuple[int, float]:
      return (priority_order[insight.priority], insight.confidence)

    insights.sort(key=sort_key, reverse=True)

    # Limit count to avoid overload.
    max_insights = 10
    if len(insights) > max_insights:
      insights = insights[:max_insights]

    return insights

  def _generate_summary(
    self,
    insights: list[HealthInsight],
    heart_rate_report: HeartRateAnalysisReport | None,
    sleep_report: SleepAnalysisReport | None,
  ) -> dict[str, Any]:
    """Generate summary information."""
    summary: dict[str, Any] = {
      "total_insights": len(insights),
      "high_priority_count": sum(1 for i in insights if i.priority == "high"),
      "medium_priority_count": sum(1 for i in insights if i.priority == "medium"),
      "low_priority_count": sum(1 for i in insights if i.priority == "low"),
      "categories": {},
      "data_quality": {},
    }

    # Category summary.
    categories = summary["categories"]
    if isinstance(categories, dict):
      for insight in insights:
        categories[insight.category] = categories.get(insight.category, 0) + 1

    # Data quality summary.
    data_quality = summary["data_quality"]
    if isinstance(data_quality, dict):
      if heart_rate_report:
        data_quality["heart_rate_records"] = heart_rate_report.record_count
        data_quality["heart_rate_quality"] = heart_rate_report.data_quality_score

      if sleep_report:
        data_quality["sleep_records"] = sleep_report.record_count
        data_quality["sleep_quality"] = sleep_report.data_quality_score

    return summary

  def _generate_recommendations(self, insights: list[HealthInsight]) -> list[str]:
    """Generate recommendations based on insights."""
    recommendations = []

    # Identify key issues in insights.
    def has_issue(issue: str) -> bool:
      return any(
        insight.details and insight.details.get("issue") == issue
        for insight in insights
      )

    has_sleep_duration_issue = has_issue("sleep_duration_low")
    has_sleep_efficiency_issue = has_issue("sleep_efficiency_low")
    has_sleep_consistency_issue = has_issue("sleep_consistency_low")
    has_high_stress = has_issue("stress_high")
    has_poor_resting_hr = has_issue("resting_hr_poor")

    # Generate targeted recommendations.
    if has_sleep_duration_issue:
      recommendations.append(self.translator.t("highlights.recommendation.sleep_7_9"))
      recommendations.append(
        self.translator.t("highlights.recommendation.sleep_schedule_regular")
      )

    if has_sleep_efficiency_issue:
      recommendations.append(
        self.translator.t("highlights.recommendation.sleep_environment")
      )
      recommendations.append(
        self.translator.t("highlights.recommendation.sleep_avoid_screens_caffeine")
      )

    if has_sleep_consistency_issue:
      recommendations.append(
        self.translator.t("highlights.recommendation.sleep_fixed_times")
      )
      recommendations.append(
        self.translator.t("highlights.recommendation.sleep_relax_routine")
      )

    if has_high_stress:
      recommendations.append(
        self.translator.t("highlights.recommendation.stress_management")
      )
      recommendations.append(
        self.translator.t("highlights.recommendation.rest_recovery")
      )

    if has_poor_resting_hr:
      recommendations.append(self.translator.t("highlights.recommendation.aerobic_150"))
      recommendations.append(self.translator.t("highlights.recommendation.monitor_hr"))

    # Default recommendations.
    if not recommendations:
      recommendations.extend(
        [
          self.translator.t("highlights.recommendation.default_exercise"),
          self.translator.t("highlights.recommendation.default_checkup"),
          self.translator.t("highlights.recommendation.default_routine"),
        ]
      )

    return recommendations
