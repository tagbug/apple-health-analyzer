"""健康洞察生成模块 - 从分析结果中提取关键洞察和建议"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from ..processors.heart_rate import HeartRateAnalysisReport
from ..processors.sleep import SleepAnalysisReport
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HealthInsight:
  """健康洞察数据类"""

  category: Literal["heart_rate", "sleep", "correlation", "general"]
  priority: Literal["high", "medium", "low"]
  title: str
  message: str
  details: dict[str, Any] | None = None
  confidence: float = 1.0  # 置信度 (0-1)
  timestamp: datetime | None = None


@dataclass
class HealthHighlights:
  """健康洞察汇总"""

  analysis_date: datetime
  insights: list[HealthInsight]
  summary: dict[str, Any]
  recommendations: list[str]


class HighlightsGenerator:
  """健康洞察生成器

  从心率、睡眠等分析结果中提取关键洞察，生成可读性强的健康建议。
  """

  def __init__(self):
    """初始化洞察生成器"""
    logger.info("HighlightsGenerator initialized")

  def generate_comprehensive_highlights(
    self,
    heart_rate_report: HeartRateAnalysisReport | None = None,
    sleep_report: SleepAnalysisReport | None = None,
    correlation_data: dict[str, Any] | None = None,
  ) -> HealthHighlights:
    """生成综合健康洞察

    Args:
        heart_rate_report: 心率分析报告
        sleep_report: 睡眠分析报告
        correlation_data: 关联分析数据

    Returns:
        综合健康洞察
    """
    logger.info("Generating comprehensive health highlights")

    insights = []

    # 心率洞察
    if heart_rate_report:
      hr_insights = self._generate_heart_rate_insights(heart_rate_report)
      insights.extend(hr_insights)

    # 睡眠洞察
    if sleep_report:
      sleep_insights = self._generate_sleep_insights(sleep_report)
      insights.extend(sleep_insights)

    # 关联洞察
    if correlation_data:
      corr_insights = self._generate_correlation_insights(correlation_data)
      insights.extend(corr_insights)

    # 排序和过滤洞察
    insights = self._rank_and_filter_insights(insights)

    # 生成总结
    summary = self._generate_summary(insights, heart_rate_report, sleep_report)

    # 生成建议
    recommendations = self._generate_recommendations(insights)

    highlights = HealthHighlights(
      analysis_date=datetime.now(),
      insights=insights,
      summary=summary,
      recommendations=recommendations,
    )

    logger.info(
      f"Generated {len(insights)} insights and {len(recommendations)} recommendations"
    )
    return highlights

  def _generate_heart_rate_insights(
    self, report: HeartRateAnalysisReport
  ) -> list[HealthInsight]:
    """生成心率相关洞察"""
    insights = []

    # 静息心率洞察
    if report.resting_hr_analysis:
      resting_hr = report.resting_hr_analysis

      # 趋势洞察
      if abs(resting_hr.change_from_baseline) > 2:
        if resting_hr.trend_direction == "decreasing":
          insights.append(
            HealthInsight(
              category="heart_rate",
              priority="high",
              title="静息心率改善",
              message=f"静息心率下降{abs(resting_hr.change_from_baseline):.1f} bpm，表明心血管健康状况改善",
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
              title="静息心率上升",
              message=f"静息心率上升{resting_hr.change_from_baseline:.1f} bpm，建议关注压力管理和运动习惯",
              details={
                "current": resting_hr.current_value,
                "baseline": resting_hr.baseline_value,
                "change": resting_hr.change_from_baseline,
                "trend": resting_hr.trend_direction,
              },
              confidence=0.8,
            )
          )

      # 健康评级洞察
      if resting_hr.health_rating == "excellent":
        insights.append(
          HealthInsight(
            category="heart_rate",
            priority="low",
            title="心率健康优秀",
            message=f"静息心率为{resting_hr.current_value:.0f} bpm，处于优秀水平",
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
            title="心率健康需要关注",
            message=f"静息心率为{resting_hr.current_value:.0f} bpm，建议咨询医生并改善生活方式",
            details={
              "rating": resting_hr.health_rating,
              "value": resting_hr.current_value,
            },
            confidence=0.9,
          )
        )

    # HRV洞察
    if report.hrv_analysis:
      hrv = report.hrv_analysis

      if hrv.trend_direction == "improving":
        insights.append(
          HealthInsight(
            category="heart_rate",
            priority="medium",
            title="心率变异性改善",
            message="心率变异性改善，表明压力水平降低，恢复能力增强",
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
            title="心率变异性下降",
            message="心率变异性下降，可能表明压力过大或恢复不足",
            details={
              "current_sdnn": hrv.current_sdnn,
              "change": hrv.change_from_baseline,
              "stress_level": hrv.stress_level,
              "recovery_status": hrv.recovery_status,
            },
            confidence=0.9,
          )
        )

      # 压力水平洞察
      if hrv.stress_level in ["high", "very_high"]:
        insights.append(
          HealthInsight(
            category="heart_rate",
            priority="high",
            title="高压力水平",
            message="检测到较高压力水平，建议进行压力管理",
            details={
              "stress_level": hrv.stress_level,
              "sdnn": hrv.current_sdnn,
            },
            confidence=0.8,
          )
        )

    # 心肺适能洞察
    if report.cardio_fitness:
      cardio = report.cardio_fitness

      rating_descriptions = {
        "superior": "卓越",
        "excellent": "优秀",
        "good": "良好",
        "fair": "一般",
        "poor": "需要改善",
      }

      insights.append(
        HealthInsight(
          category="heart_rate",
          priority="medium",
          title="心肺适能评级",
          message=f"心肺适能评级：{rating_descriptions[cardio.age_adjusted_rating]}（VO2Max: {cardio.current_vo2_max:.1f}）",
          details={
            "rating": cardio.age_adjusted_rating,
            "vo2_max": cardio.current_vo2_max,
            "percentile": cardio.fitness_percentile,
          },
          confidence=0.9,
        )
      )

    # 异常检测洞察
    if report.anomalies and len(report.anomalies) > 0:
      anomaly_count = len(report.anomalies)
      if anomaly_count > 10:
        insights.append(
          HealthInsight(
            category="heart_rate",
            priority="medium",
            title="心率异常较多",
            message=f"检测到{anomaly_count}个心率异常事件，建议查看详细报告",
            details={"anomaly_count": anomaly_count},
            confidence=0.8,
          )
        )

    return insights

  def _generate_sleep_insights(
    self, report: SleepAnalysisReport
  ) -> list[HealthInsight]:
    """生成睡眠相关洞察"""
    insights = []

    # 睡眠质量洞察
    if report.quality_metrics:
      quality = report.quality_metrics

      # 时长洞察
      if quality.average_duration < 7:
        insights.append(
          HealthInsight(
            category="sleep",
            priority="high",
            title="睡眠时长不足",
            message=f"平均睡眠时长仅{quality.average_duration:.1f}小时，建议保证7-9小时睡眠",
            details={"average_duration": quality.average_duration},
            confidence=0.95,
          )
        )
      elif quality.average_duration >= 8:
        insights.append(
          HealthInsight(
            category="sleep",
            priority="low",
            title="睡眠时长充足",
            message=f"平均睡眠时长{quality.average_duration:.1f}小时，睡眠时间充足",
            details={"average_duration": quality.average_duration},
            confidence=0.9,
          )
        )

      # 效率洞察
      if quality.average_efficiency < 0.85:
        insights.append(
          HealthInsight(
            category="sleep",
            priority="high",
            title="睡眠效率低下",
            message=f"睡眠效率仅{quality.average_efficiency:.1%}，建议改善睡眠环境和习惯",
            details={"average_efficiency": quality.average_efficiency},
            confidence=0.9,
          )
        )

      # 规律性洞察
      if quality.consistency_score < 0.7:
        insights.append(
          HealthInsight(
            category="sleep",
            priority="medium",
            title="睡眠规律性差",
            message="睡眠时间不规律，建议保持固定的作息时间",
            details={"consistency_score": quality.consistency_score},
            confidence=0.85,
          )
        )

    # 睡眠模式洞察
    if report.pattern_analysis:
      patterns = report.pattern_analysis

      # 社会时差洞察
      if (
        patterns.weekday_vs_weekend
        and patterns.weekday_vs_weekend.get("social_jetlag", 0) > 2
      ):
        social_jetlag = patterns.weekday_vs_weekend["social_jetlag"]
        insights.append(
          HealthInsight(
            category="sleep",
            priority="medium",
            title="周末和工作日作息差异大",
            message=f"社会时差{social_jetlag:.1f}小时，建议减少周末和工作日的作息差异",
            details={"social_jetlag": social_jetlag},
            confidence=0.8,
          )
        )

    # 睡眠-心率关联洞察
    if report.hr_correlation:
      hr_corr = report.hr_correlation

      if hr_corr.recovery_quality < 70:
        insights.append(
          HealthInsight(
            category="correlation",
            priority="medium",
            title="睡眠期间心率恢复不佳",
            message="睡眠期间心率恢复质量较低，建议改善睡眠质量",
            details={"recovery_quality": hr_corr.recovery_quality},
            confidence=0.8,
          )
        )

    # 异常检测洞察
    if report.anomalies and len(report.anomalies) > 0:
      anomaly_count = len(report.anomalies)
      if anomaly_count > 5:
        insights.append(
          HealthInsight(
            category="sleep",
            priority="medium",
            title="睡眠异常较多",
            message=f"检测到{anomaly_count}个睡眠异常事件，建议关注睡眠质量",
            details={"anomaly_count": anomaly_count},
            confidence=0.8,
          )
        )

    return insights

  def _generate_correlation_insights(
    self, correlation_data: dict[str, Any]
  ) -> list[HealthInsight]:
    """生成关联分析洞察"""
    insights = []

    if not correlation_data:
      return insights

    for key, data in correlation_data.items():
      correlation_value = data.get("correlation", 0.0)
      insight_text = data.get("insight", "")
      
      # 根据相关性强度确定优先级
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
        
      # 生成标题
      if key == "sleep_activity":
        title = "睡眠与活动关联"
      elif key == "hr_stress":
        title = "心率与压力关联"
      else:
        title = f"健康指标关联 ({key})"
        
      # 如果没有预设的洞察文本，生成默认文本
      if not insight_text:
        direction = "正相关" if correlation_value > 0 else "负相关"
        strength = "强" if abs_corr >= 0.7 else "中等" if abs_corr >= 0.4 else "弱"
        insight_text = f"检测到{strength}{direction} (r={correlation_value:.2f})"

      insights.append(
        HealthInsight(
          category="correlation",
          priority=priority,
          title=title,
          message=insight_text,
          details={"correlation": correlation_value, "type": key},
          confidence=confidence,
        )
      )

    return insights

  def _rank_and_filter_insights(
    self, insights: list[HealthInsight]
  ) -> list[HealthInsight]:
    """对洞察进行排序和过滤"""
    if not insights:
      return []

    # 按优先级和置信度排序
    priority_order = {"high": 3, "medium": 2, "low": 1}

    def sort_key(insight: HealthInsight) -> tuple[int, float]:
      return (priority_order[insight.priority], insight.confidence)

    insights.sort(key=sort_key, reverse=True)

    # 限制数量，避免信息过载
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
    """生成总结信息"""
    summary: dict[str, Any] = {
      "total_insights": len(insights),
      "high_priority_count": sum(1 for i in insights if i.priority == "high"),
      "medium_priority_count": sum(
        1 for i in insights if i.priority == "medium"
      ),
      "low_priority_count": sum(1 for i in insights if i.priority == "low"),
      "categories": {},
      "data_quality": {},
    }

    # 分类统计
    categories = summary["categories"]
    if isinstance(categories, dict):
      for insight in insights:
        categories[insight.category] = categories.get(insight.category, 0) + 1

    # 数据质量信息
    data_quality = summary["data_quality"]
    if isinstance(data_quality, dict):
      if heart_rate_report:
        data_quality["heart_rate_records"] = heart_rate_report.record_count
        data_quality["heart_rate_quality"] = (
          heart_rate_report.data_quality_score
        )

      if sleep_report:
        data_quality["sleep_records"] = sleep_report.record_count
        data_quality["sleep_quality"] = sleep_report.data_quality_score

    return summary

  def _generate_recommendations(
    self, insights: list[HealthInsight]
  ) -> list[str]:
    """基于洞察生成建议"""
    recommendations = []

    # 分析洞察中的关键问题
    has_sleep_duration_issue = any("睡眠时长不足" in i.title for i in insights)
    has_sleep_efficiency_issue = any(
      "睡眠效率低下" in i.title for i in insights
    )
    has_sleep_consistency_issue = any(
      "睡眠规律性差" in i.title for i in insights
    )
    has_high_stress = any("高压力水平" in i.title for i in insights)
    has_poor_resting_hr = any("心率健康需要关注" in i.title for i in insights)

    # 生成针对性建议
    if has_sleep_duration_issue:
      recommendations.append("保证每晚7-9小时的睡眠时间，避免熬夜")
      recommendations.append("建立规律的作息时间表，包括周末")

    if has_sleep_efficiency_issue:
      recommendations.append("改善睡眠环境：保持卧室凉爽、黑暗和安静")
      recommendations.append("睡前2小时避免使用电子设备和摄入咖啡因")

    if has_sleep_consistency_issue:
      recommendations.append("保持固定的起床和睡觉时间，即使在周末")
      recommendations.append("建立睡前放松 routine，如阅读或冥想")

    if has_high_stress:
      recommendations.append("进行压力管理：尝试冥想、深呼吸或适量运动")
      recommendations.append("保证充足的休息和娱乐时间")

    if has_poor_resting_hr:
      recommendations.append(
        "增加有氧运动，如快走、跑步或骑行，每周至少150分钟"
      )
      recommendations.append("定期监测心率指标，如有异常及时咨询医生")

    # 通用建议
    if not recommendations:
      recommendations.extend(
        [
          "保持规律的运动和健康饮食习惯",
          "定期进行健康检查，关注身体变化",
          "保持良好的作息规律和压力管理",
        ]
      )

    return recommendations
