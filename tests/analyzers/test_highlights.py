"""Unit tests for HighlightsGenerator."""

from datetime import datetime, timedelta

import pytest

from src.analyzers.highlights import (
  HealthHighlights,
  HealthInsight,
  HighlightsGenerator,
)
from src.i18n import Translator, resolve_locale
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
  """HighlightsGenerator tests."""

  @pytest.fixture
  def generator(self):
    """Create HighlightsGenerator fixture."""
    return HighlightsGenerator()

  @pytest.fixture
  def mock_heart_rate_report(self):
    """Create mock heart rate report."""
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
        training_recommendations=["Maintain current training intensity"],
      ),
      record_count=1000,
      data_quality_score=0.95,
    )

  @pytest.fixture
  def mock_sleep_report(self):
    """Create mock sleep report."""
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
    """Test initialization."""
    assert isinstance(generator, HighlightsGenerator)

  def test_generate_heart_rate_insights_resting_hr_improvement(
    self, generator, mock_heart_rate_report
  ):
    """Test heart rate insights for resting HR improvement."""
    insights = generator._generate_heart_rate_insights(mock_heart_rate_report)

    # Should include resting HR improvement insight.
    translator = Translator(resolve_locale())
    hr_improvement = next(
      (
        i
        for i in insights
        if translator.t("highlights.heart_rate.resting_hr_improved.title") in i.title
      ),
      None,
    )
    assert hr_improvement is not None
    assert hr_improvement.priority == "high"
    assert hr_improvement.category == "heart_rate"
    assert "5.0" in hr_improvement.message

  def test_generate_heart_rate_insights_hrv_improvement(
    self, generator, mock_heart_rate_report
  ):
    """Test heart rate insights for HRV improvement."""
    insights = generator._generate_heart_rate_insights(mock_heart_rate_report)

    # Should include HRV improvement insight.
    translator = Translator(resolve_locale())
    hrv_improvement = next(
      (
        i
        for i in insights
        if translator.t("highlights.heart_rate.hrv_improving.title") in i.title
      ),
      None,
    )
    assert hrv_improvement is not None
    assert hrv_improvement.priority == "medium"
    assert translator.t("highlights.heart_rate.hrv_improving.message") in (
      hrv_improvement.message
    )

  def test_generate_heart_rate_insights_cardio_fitness(
    self, generator, mock_heart_rate_report
  ):
    """Test heart rate insights for cardio fitness."""
    insights = generator._generate_heart_rate_insights(mock_heart_rate_report)

    # Should include cardio fitness insight.
    translator = Translator(resolve_locale())
    cardio = next(
      (
        i
        for i in insights
        if translator.t("highlights.heart_rate.cardio_rating.title") in i.title
      ),
      None,
    )
    assert cardio is not None
    assert cardio.priority == "medium"
    assert (
      translator.t(
        "highlights.heart_rate.cardio_rating.message",
        rating=translator.t("highlights.heart_rate.cardio_rating.excellent"),
        vo2=42.0,
      )
      in cardio.message
    )

  def test_generate_sleep_insights_duration_insufficient(self, generator):
    """Test sleep insights for insufficient duration."""
    sleep_report = SleepAnalysisReport(
      analysis_date=datetime.now(),
      data_range=(datetime.now() - timedelta(days=30), datetime.now()),
      quality_metrics=SleepQualityMetrics(
        average_duration=6.0,  # Less than 7 hours.
        average_efficiency=0.85,
        average_latency=20.0,
        consistency_score=0.8,
        overall_quality_score=70.0,
      ),
      record_count=30,
      data_quality_score=0.9,
    )

    insights = generator._generate_sleep_insights(sleep_report)

    # Should include insufficient duration insight.
    translator = Translator(resolve_locale())
    duration_issue = next(
      (
        i
        for i in insights
        if translator.t("highlights.sleep.duration_low.title") in i.title
      ),
      None,
    )
    assert duration_issue is not None
    assert duration_issue.priority == "high"
    assert "6.0" in duration_issue.message

  def test_generate_sleep_insights_efficiency_low(self, generator):
    """Test sleep insights for low efficiency."""
    sleep_report = SleepAnalysisReport(
      analysis_date=datetime.now(),
      data_range=(datetime.now() - timedelta(days=30), datetime.now()),
      quality_metrics=SleepQualityMetrics(
        average_duration=8.0,
        average_efficiency=0.82,  # Below 85%.
        average_latency=25.0,
        consistency_score=0.8,
        overall_quality_score=75.0,
      ),
      record_count=30,
      data_quality_score=0.9,
    )

    insights = generator._generate_sleep_insights(sleep_report)

    # Should include low efficiency insight.
    translator = Translator(resolve_locale())
    efficiency_issue = next(
      (
        i
        for i in insights
        if translator.t("highlights.sleep.efficiency_low.title") in i.title
      ),
      None,
    )
    assert efficiency_issue is not None
    assert efficiency_issue.priority == "high"
    assert "82.0%" in efficiency_issue.message

  def test_generate_sleep_insights_social_jetlag(self, generator):
    """Test sleep insights for social jetlag."""
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
        weekday_vs_weekend={"social_jetlag": 2.5},  # Over 2 hours.
        seasonal_patterns={},
        duration_trend="stable",
        efficiency_trend="stable",
      ),
      record_count=30,
      data_quality_score=0.9,
    )

    insights = generator._generate_sleep_insights(sleep_report)

    # Should include social jetlag insight.
    translator = Translator(resolve_locale())
    jetlag_issue = next(
      (
        i
        for i in insights
        if translator.t("highlights.sleep.social_jetlag_high.title") in i.title
      ),
      None,
    )
    assert jetlag_issue is not None
    assert jetlag_issue.priority == "medium"
    assert "2.5" in jetlag_issue.message

  def test_generate_correlation_insights(self, generator):
    """Test correlation insights."""
    correlation_data = {
      "sleep_activity": {"correlation": 0.8, "insight": "Strong positive"},
      "hr_stress": {"correlation": 0.5, "insight": "Moderate correlation"},
      "other": {"correlation": -0.2},  # No insight text, low correlation.
    }

    insights = generator._generate_correlation_insights(correlation_data)

    assert len(insights) == 3

    # High correlation.
    translator = Translator(resolve_locale())
    high_corr = next(
      (
        i
        for i in insights
        if i.title == translator.t("highlights.correlation.sleep_activity.title")
      ),
      None,
    )
    assert high_corr is not None
    assert high_corr.priority == "high"
    assert high_corr.confidence == 0.9
    assert high_corr.message == "Strong positive"

    # Medium correlation.
    med_corr = next(
      (
        i
        for i in insights
        if i.title == translator.t("highlights.correlation.hr_stress.title")
      ),
      None,
    )
    assert med_corr is not None
    assert med_corr.priority == "medium"
    assert med_corr.confidence == 0.8

    # Low correlation with default text.
    low_corr = next(
      (
        i
        for i in insights
        if translator.t("highlights.correlation.default.title", metric="other")
        in i.title
      ),
      None,
    )
    assert low_corr is not None
    assert low_corr.priority == "low"
    assert low_corr.confidence == 0.6
    assert translator.t("highlights.correlation.direction.negative") in low_corr.message

  def test_generate_correlation_insights_empty(self, generator):
    """Test correlation insights with empty data."""
    insights = generator._generate_correlation_insights({})
    assert len(insights) == 0

  def test_rank_and_filter_insights(self, generator):
    """Test insight ranking and filtering."""
    insights = [
      HealthInsight(
        category="heart_rate",
        priority="low",
        title="Insight 1",
        message="Message 1",
        confidence=0.5,
      ),
      HealthInsight(
        category="sleep",
        priority="high",
        title="Insight 2",
        message="Message 2",
        confidence=0.9,
      ),
      HealthInsight(
        category="general",
        priority="medium",
        title="Insight 3",
        message="Message 3",
        confidence=0.7,
      ),
    ]

    ranked = generator._rank_and_filter_insights(insights)

    # Should be ordered by priority and confidence.
    assert len(ranked) == 3
    assert ranked[0].priority == "high"  # Highest priority first.
    assert ranked[1].priority == "medium"  # Medium priority next.
    assert ranked[2].priority == "low"  # Lowest priority last.

  def test_generate_summary(self, generator, mock_heart_rate_report, mock_sleep_report):
    """Test summary generation."""
    summary = generator._generate_summary([], mock_heart_rate_report, mock_sleep_report)

    assert isinstance(summary, dict)
    assert "total_insights" in summary
    assert "high_priority_count" in summary
    assert "categories" in summary
    assert "data_quality" in summary

    # Check data quality info.
    assert "heart_rate_records" in summary["data_quality"]
    assert "sleep_records" in summary["data_quality"]

  def test_generate_recommendations_sleep_issues(self, generator):
    """Test recommendations for sleep issues."""
    insights = [
      HealthInsight(
        category="sleep",
        priority="high",
        title="睡眠时长不足",
        message="Sleep duration too short",
      ),
      HealthInsight(
        category="sleep",
        priority="high",
        title="睡眠效率低下",
        message="Low efficiency",
      ),
    ]

    recommendations = generator._generate_recommendations(insights)

    assert len(recommendations) > 0
    translator = Translator(resolve_locale())
    default_recommendations = {
      translator.t("highlights.recommendation.default_exercise"),
      translator.t("highlights.recommendation.default_checkup"),
      translator.t("highlights.recommendation.default_routine"),
    }
    assert any(rec in default_recommendations for rec in recommendations)

  def test_generate_recommendations_stress_issues(self, generator):
    """Test recommendations for stress issues."""
    insights = [
      HealthInsight(
        category="heart_rate",
        priority="high",
        title="高压力水平",
        message="High stress",
        details={"issue": "stress_high"},
      ),
    ]

    recommendations = generator._generate_recommendations(insights)

    assert len(recommendations) > 0
    translator = Translator(resolve_locale())
    assert any(
      translator.t("highlights.recommendation.stress_management") in rec
      for rec in recommendations
    )

  def test_generate_comprehensive_highlights(
    self, generator, mock_heart_rate_report, mock_sleep_report
  ):
    """Test comprehensive highlights generation."""
    highlights = generator.generate_comprehensive_highlights(
      heart_rate_report=mock_heart_rate_report,
      sleep_report=mock_sleep_report,
    )

    assert isinstance(highlights, HealthHighlights)
    assert isinstance(highlights.insights, list)
    assert isinstance(highlights.summary, dict)
    assert isinstance(highlights.recommendations, list)

    # Should include insights.
    assert len(highlights.insights) > 0
    assert len(highlights.recommendations) > 0

    # Check insight titles.
    insight_titles = [i.title for i in highlights.insights]
    translator = Translator(resolve_locale())
    assert any(
      translator.t("highlights.heart_rate.resting_hr_improved.title") in title
      for title in insight_titles
    )
    assert any(
      translator.t("highlights.heart_rate.hrv_improving.title") in title
      for title in insight_titles
    )
    assert any(
      translator.t("highlights.heart_rate.cardio_rating.title") in title
      for title in insight_titles
    )

  def test_generate_comprehensive_highlights_empty(self, generator):
    """Test comprehensive highlights with empty input."""
    highlights = generator.generate_comprehensive_highlights()

    assert isinstance(highlights, HealthHighlights)
    assert len(highlights.insights) == 0
    assert len(highlights.recommendations) > 0  # Should include defaults.

  def test_insight_data_types(self, generator, mock_heart_rate_report):
    """Test insight data types."""
    insights = generator._generate_heart_rate_insights(mock_heart_rate_report)

    for insight in insights:
      assert isinstance(insight, HealthInsight)
      assert isinstance(insight.category, str)
      assert insight.priority in ["high", "medium", "low"]
      assert isinstance(insight.title, str)
      assert isinstance(insight.message, str)
      assert isinstance(insight.confidence, float)
      assert 0 <= insight.confidence <= 1
