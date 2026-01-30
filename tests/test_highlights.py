"""Unit tests for HighlightsGenerator."""

from datetime import datetime, timedelta

import pytest

from src.analyzers.highlights import (
  HealthHighlights,
  HealthInsight,
  HighlightsGenerator,
)
from src.processors.heart_rate import (
  CardioFitnessAnalysis,
  HeartRateAnalysisReport,
  HRVAnalysis,
  RestingHRAnalysis,
)
from src.processors.sleep import (
  SleepAnalysisReport,
  SleepPatternAnalysis,
  SleepQualityMetrics,
)


class TestHighlightsGenerator:
  """HighlightsGenerator 测试类"""

  @pytest.fixture
  def generator(self):
    """创建测试用的HighlightsGenerator实例"""
    return HighlightsGenerator()

  @pytest.fixture
  def mock_heart_rate_report(self):
    """创建模拟的心率分析报告"""
    return HeartRateAnalysisReport(
      analysis_date=datetime.now(),
      data_range=(datetime.now() - timedelta(days=30), datetime.now()),
      resting_hr_analysis=RestingHRAnalysis(
        current_value=65.0,
        baseline_value=70.0,
        change_from_baseline=-5.0,
        trend_direction="decreasing",
        health_rating="excellent",
      ),
      hrv_analysis=HRVAnalysis(
        current_sdnn=45.0,
        baseline_sdnn=35.0,
        change_from_baseline=10.0,
        stress_level="low",
        recovery_status="excellent",
        trend_direction="improving",
      ),
      cardio_fitness=CardioFitnessAnalysis(
        current_vo2_max=42.0,
        age_adjusted_rating="excellent",
        fitness_percentile=75.0,
        improvement_potential=15.0,
        training_recommendations=["保持当前训练强度"],
      ),
      record_count=1000,
      data_quality_score=0.95,
    )

  @pytest.fixture
  def mock_sleep_report(self):
    """创建模拟的睡眠分析报告"""
    return SleepAnalysisReport(
      analysis_date=datetime.now(),
      data_range=(datetime.now() - timedelta(days=30), datetime.now()),
      quality_metrics=SleepQualityMetrics(
        average_duration=7.5,
        average_efficiency=0.88,
        average_latency=15.0,
        consistency_score=0.85,
        overall_quality_score=85.0,
      ),
      pattern_analysis=SleepPatternAnalysis(
        bedtime_consistency=0.8,
        waketime_consistency=0.75,
        weekday_vs_weekend={"social_jetlag": 1.5},
        seasonal_patterns={},
        duration_trend="stable",
        efficiency_trend="stable",
      ),
      record_count=30,
      data_quality_score=0.92,
    )

  def test_initialization(self, generator):
    """测试初始化"""
    assert isinstance(generator, HighlightsGenerator)

  def test_generate_heart_rate_insights_resting_hr_improvement(
    self, generator, mock_heart_rate_report
  ):
    """测试心率洞察生成 - 静息心率改善"""
    insights = generator._generate_heart_rate_insights(mock_heart_rate_report)

    # 应该生成静息心率改善洞察
    hr_improvement = next(
      (i for i in insights if "静息心率改善" in i.title), None
    )
    assert hr_improvement is not None
    assert hr_improvement.priority == "high"
    assert hr_improvement.category == "heart_rate"
    assert "下降5.0 bpm" in hr_improvement.message

  def test_generate_heart_rate_insights_hrv_improvement(
    self, generator, mock_heart_rate_report
  ):
    """测试心率洞察生成 - HRV改善"""
    insights = generator._generate_heart_rate_insights(mock_heart_rate_report)

    # 应该生成HRV改善洞察
    hrv_improvement = next(
      (i for i in insights if "心率变异性改善" in i.title), None
    )
    assert hrv_improvement is not None
    assert hrv_improvement.priority == "medium"
    assert "压力水平降低" in hrv_improvement.message

  def test_generate_heart_rate_insights_cardio_fitness(
    self, generator, mock_heart_rate_report
  ):
    """测试心率洞察生成 - 心肺适能"""
    insights = generator._generate_heart_rate_insights(mock_heart_rate_report)

    # 应该生成心肺适能洞察
    cardio = next((i for i in insights if "心肺适能评级" in i.title), None)
    assert cardio is not None
    assert cardio.priority == "medium"
    assert "优秀" in cardio.message

  def test_generate_sleep_insights_duration_insufficient(self, generator):
    """测试睡眠洞察生成 - 睡眠时长不足"""
    sleep_report = SleepAnalysisReport(
      analysis_date=datetime.now(),
      data_range=(datetime.now() - timedelta(days=30), datetime.now()),
      quality_metrics=SleepQualityMetrics(
        average_duration=6.0,  # 不足7小时
        average_efficiency=0.85,
        average_latency=20.0,
        consistency_score=0.8,
        overall_quality_score=70.0,
      ),
      record_count=30,
      data_quality_score=0.9,
    )

    insights = generator._generate_sleep_insights(sleep_report)

    # 应该生成睡眠时长不足洞察
    duration_issue = next(
      (i for i in insights if "睡眠时长不足" in i.title), None
    )
    assert duration_issue is not None
    assert duration_issue.priority == "high"
    assert "6.0小时" in duration_issue.message

  def test_generate_sleep_insights_efficiency_low(self, generator):
    """测试睡眠洞察生成 - 睡眠效率低下"""
    sleep_report = SleepAnalysisReport(
      analysis_date=datetime.now(),
      data_range=(datetime.now() - timedelta(days=30), datetime.now()),
      quality_metrics=SleepQualityMetrics(
        average_duration=8.0,
        average_efficiency=0.82,  # 低于85%
        average_latency=25.0,
        consistency_score=0.8,
        overall_quality_score=75.0,
      ),
      record_count=30,
      data_quality_score=0.9,
    )

    insights = generator._generate_sleep_insights(sleep_report)

    # 应该生成睡眠效率低下洞察
    efficiency_issue = next(
      (i for i in insights if "睡眠效率低下" in i.title), None
    )
    assert efficiency_issue is not None
    assert efficiency_issue.priority == "high"
    assert "82.0%" in efficiency_issue.message

  def test_generate_sleep_insights_social_jetlag(self, generator):
    """测试睡眠洞察生成 - 社会时差"""
    sleep_report = SleepAnalysisReport(
      analysis_date=datetime.now(),
      data_range=(datetime.now() - timedelta(days=30), datetime.now()),
      quality_metrics=SleepQualityMetrics(
        average_duration=8.0,
        average_efficiency=0.9,
        average_latency=15.0,
        consistency_score=0.8,
        overall_quality_score=85.0,
      ),
      pattern_analysis=SleepPatternAnalysis(
        bedtime_consistency=0.7,
        waketime_consistency=0.8,
        weekday_vs_weekend={"social_jetlag": 2.5},  # 超过2小时
        seasonal_patterns={},
        duration_trend="stable",
        efficiency_trend="stable",
      ),
      record_count=30,
      data_quality_score=0.9,
    )

    insights = generator._generate_sleep_insights(sleep_report)

    # 应该生成社会时差洞察
    jetlag_issue = next((i for i in insights if "作息差异大" in i.title), None)
    assert jetlag_issue is not None
    assert jetlag_issue.priority == "medium"
    assert "2.5小时" in jetlag_issue.message

  def test_generate_correlation_insights(self, generator):
    """测试关联洞察生成"""
    correlation_data = {
      "sleep_activity": {"correlation": 0.8, "insight": "强正相关"},
      "hr_stress": {"correlation": 0.5, "insight": "中等相关"},
      "other": {"correlation": -0.2},  # 无 insight 文本，低相关
    }

    insights = generator._generate_correlation_insights(correlation_data)

    assert len(insights) == 3

    # 检查高相关性
    high_corr = next((i for i in insights if i.title == "睡眠与活动关联"), None)
    assert high_corr is not None
    assert high_corr.priority == "high"
    assert high_corr.confidence == 0.9
    assert high_corr.message == "强正相关"

    # 检查中相关性
    med_corr = next((i for i in insights if i.title == "心率与压力关联"), None)
    assert med_corr is not None
    assert med_corr.priority == "medium"
    assert med_corr.confidence == 0.8

    # 检查低相关性及默认文本
    low_corr = next((i for i in insights if "other" in i.title), None)
    assert low_corr is not None
    assert low_corr.priority == "low"
    assert low_corr.confidence == 0.6
    assert "负相关" in low_corr.message

  def test_generate_correlation_insights_empty(self, generator):
    """测试关联洞察生成 - 空数据"""
    insights = generator._generate_correlation_insights({})
    assert len(insights) == 0

  def test_rank_and_filter_insights(self, generator):
    """测试洞察排序和过滤"""
    insights = [
      HealthInsight(
        category="heart_rate",
        priority="low",
        title="测试洞察1",
        message="消息1",
        confidence=0.5,
      ),
      HealthInsight(
        category="sleep",
        priority="high",
        title="测试洞察2",
        message="消息2",
        confidence=0.9,
      ),
      HealthInsight(
        category="general",
        priority="medium",
        title="测试洞察3",
        message="消息3",
        confidence=0.7,
      ),
    ]

    ranked = generator._rank_and_filter_insights(insights)

    # 应该按优先级和置信度排序
    assert len(ranked) == 3
    assert ranked[0].priority == "high"  # 高优先级排第一
    assert ranked[1].priority == "medium"  # 中优先级排第二
    assert ranked[2].priority == "low"  # 低优先级排第三

  def test_generate_summary(
    self, generator, mock_heart_rate_report, mock_sleep_report
  ):
    """测试总结生成"""
    summary = generator._generate_summary(
      [], mock_heart_rate_report, mock_sleep_report
    )

    assert isinstance(summary, dict)
    assert "total_insights" in summary
    assert "high_priority_count" in summary
    assert "categories" in summary
    assert "data_quality" in summary

    # 检查数据质量信息
    assert "heart_rate_records" in summary["data_quality"]
    assert "sleep_records" in summary["data_quality"]

  def test_generate_recommendations_sleep_issues(self, generator):
    """测试建议生成 - 睡眠问题"""
    insights = [
      HealthInsight(
        category="sleep",
        priority="high",
        title="睡眠时长不足",
        message="睡眠不足",
      ),
      HealthInsight(
        category="sleep",
        priority="high",
        title="睡眠效率低下",
        message="效率低下",
      ),
    ]

    recommendations = generator._generate_recommendations(insights)

    assert len(recommendations) > 0
    assert any("睡眠时间" in rec for rec in recommendations)
    assert any("电子设备" in rec for rec in recommendations)

  def test_generate_recommendations_stress_issues(self, generator):
    """测试建议生成 - 压力问题"""
    insights = [
      HealthInsight(
        category="heart_rate",
        priority="high",
        title="高压力水平",
        message="压力高",
      ),
    ]

    recommendations = generator._generate_recommendations(insights)

    assert len(recommendations) > 0
    assert any("压力管理" in rec for rec in recommendations)

  def test_generate_comprehensive_highlights(
    self, generator, mock_heart_rate_report, mock_sleep_report
  ):
    """测试综合洞察生成"""
    highlights = generator.generate_comprehensive_highlights(
      heart_rate_report=mock_heart_rate_report,
      sleep_report=mock_sleep_report,
    )

    assert isinstance(highlights, HealthHighlights)
    assert isinstance(highlights.insights, list)
    assert isinstance(highlights.summary, dict)
    assert isinstance(highlights.recommendations, list)

    # 应该有洞察生成
    assert len(highlights.insights) > 0
    assert len(highlights.recommendations) > 0

    # 检查洞察内容
    insight_titles = [i.title for i in highlights.insights]
    assert any("静息心率改善" in title for title in insight_titles)
    assert any("心率变异性改善" in title for title in insight_titles)
    assert any("心肺适能评级" in title for title in insight_titles)

  def test_generate_comprehensive_highlights_empty(self, generator):
    """测试综合洞察生成 - 空输入"""
    highlights = generator.generate_comprehensive_highlights()

    assert isinstance(highlights, HealthHighlights)
    assert len(highlights.insights) == 0
    assert len(highlights.recommendations) > 0  # 应该有通用建议

  def test_insight_data_types(self, generator, mock_heart_rate_report):
    """测试洞察数据类型"""
    insights = generator._generate_heart_rate_insights(mock_heart_rate_report)

    for insight in insights:
      assert isinstance(insight, HealthInsight)
      assert isinstance(insight.category, str)
      assert insight.priority in ["high", "medium", "low"]
      assert isinstance(insight.title, str)
      assert isinstance(insight.message, str)
      assert isinstance(insight.confidence, float)
      assert 0 <= insight.confidence <= 1
