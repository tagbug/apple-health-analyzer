"""Unit tests for report generation functionality."""

import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.analyzers.highlights import HealthHighlights
from src.processors.heart_rate import HeartRateAnalysisReport
from src.processors.sleep import SleepAnalysisReport
from src.visualization.reports import ReportGenerator
from src.i18n import Translator, resolve_locale


class TestReportGenerator:
  """ReportGenerator tests."""

  @pytest.fixture
  def report_generator(self):
    """Create ReportGenerator fixture."""
    return ReportGenerator()

  @pytest.fixture
  def zh_translator(self):
    """Create Chinese translator fixture."""
    return Translator(resolve_locale("zh"))

  @pytest.fixture
  def sample_heart_rate_report(self):
    """Create sample heart rate report."""
    from datetime import datetime

    report = HeartRateAnalysisReport(
      analysis_date=datetime.now(),
      data_range=(datetime(2024, 1, 1), datetime(2024, 1, 31)),
      record_count=1000,
      data_quality_score=0.85,
    )

    # Add resting HR analysis.
    from src.processors.heart_rate import RestingHRAnalysis

    report.resting_hr_analysis = RestingHRAnalysis(
      current_value=68.5,
      baseline_value=72.0,
      change_from_baseline=-3.5,
      trend_direction="decreasing",
      health_rating="excellent",
    )

    return report

  @pytest.fixture
  def sample_sleep_report(self):
    """Create sample sleep report."""
    from datetime import datetime

    report = SleepAnalysisReport(
      analysis_date=datetime.now(),
      data_range=(datetime(2024, 1, 1), datetime(2024, 1, 30)),
      record_count=30,
      data_quality_score=0.9,
    )

    # Add sleep quality metrics.
    from src.processors.sleep import SleepQualityMetrics

    report.quality_metrics = SleepQualityMetrics(
      average_duration=7.5,
      average_efficiency=0.85,
      average_latency=15.0,
      consistency_score=0.75,
      overall_quality_score=80.0,
    )

    return report

  @pytest.fixture
  def sample_highlights(self):
    """Create sample health insights."""
    from datetime import datetime

    from src.analyzers.highlights import HealthInsight

    insights = [
      HealthInsight(
        category="heart_rate",
        priority="low",
        title="心率改善趋势",
        message="您的静息心率在过去一个月内稳步下降，这表明您的有氧健身水平正在提高。",
      ),
      HealthInsight(
        category="sleep",
        priority="medium",
        title="睡眠质量需要关注",
        message="您的睡眠效率低于推荐水平，建议改善睡眠环境和作息规律。",
      ),
    ]

    recommendations = [
      "保持规律的运动习惯",
      "改善睡眠环境，保持卧室凉爽黑暗",
      "定期监测血压和心率变化",
    ]

    return HealthHighlights(
      analysis_date=datetime.now(),
      insights=insights,
      summary={"total_insights": 2, "high_priority_count": 0},
      recommendations=recommendations,
    )

  def test_initialization(self, report_generator):
    """Test initialization."""
    assert isinstance(report_generator, ReportGenerator)
    assert report_generator.output_dir.exists()
    assert isinstance(report_generator.chart_generator, object)

  def test_generate_html_report(
    self,
    report_generator,
    sample_heart_rate_report,
    sample_sleep_report,
    sample_highlights,
  ):
    """Test HTML report generation."""
    with tempfile.TemporaryDirectory() as temp_dir:
      report_generator.output_dir = Path(temp_dir)

      report_path = report_generator.generate_html_report(
        title="测试健康报告",
        heart_rate_report=sample_heart_rate_report,
        sleep_report=sample_sleep_report,
        highlights=sample_highlights,
        include_charts=False,  # Skip charts to simplify test.
        locale="zh",
      )

      assert report_path.exists()
      assert report_path.suffix == ".html"

      # Verify report content.
      content = report_path.read_text(encoding="utf-8")
      assert "测试健康报告" in content
      assert "执行摘要" in content
      assert "心率分析" in content
      assert "睡眠分析" in content
      assert "关键发现" in content
      assert 'lang="zh' in content

  def test_generate_markdown_report(
    self,
    report_generator,
    sample_heart_rate_report,
    sample_sleep_report,
    sample_highlights,
  ):
    """Test Markdown report generation."""
    with tempfile.TemporaryDirectory() as temp_dir:
      report_generator.output_dir = Path(temp_dir)

      report_path = report_generator.generate_markdown_report(
        title="测试健康报告",
        heart_rate_report=sample_heart_rate_report,
        sleep_report=sample_sleep_report,
        highlights=sample_highlights,
        locale="zh",
      )

      assert report_path.exists()
      assert report_path.suffix == ".md"

      # Verify report content.
      content = report_path.read_text(encoding="utf-8")
      assert "# 测试健康报告" in content
      assert "## 执行摘要" in content
      assert "## 心率分析" in content
      assert "## 睡眠分析" in content
      assert "## 关键发现" in content

  def test_generate_comprehensive_report(self, report_generator):
    """Test comprehensive report generation."""
    # Create mock report.
    mock_report = Mock()
    mock_report.overall_wellness_score = 0.82
    from datetime import datetime

    mock_report.data_range = (datetime(2024, 1, 1), datetime(2024, 1, 31))
    mock_report.data_completeness_score = 0.88
    mock_report.analysis_confidence = 0.91

    # Sleep quality.
    mock_sleep = Mock()
    mock_sleep.average_duration_hours = 7.2
    mock_sleep.average_efficiency_percent = 85.0
    mock_sleep.sleep_debt_hours = 1.5
    mock_sleep.consistency_score = 0.8
    mock_report.sleep_quality = mock_sleep

    # Activity patterns.
    mock_activity = Mock()
    mock_activity.daily_step_average = 9200
    mock_activity.weekly_exercise_frequency = 4.5
    mock_activity.sedentary_hours_daily = 7.8
    mock_activity.activity_consistency_score = 0.85
    mock_report.activity_patterns = mock_activity

    # Stress resilience.
    mock_stress = Mock()
    mock_stress.stress_accumulation_score = 0.25
    mock_stress.recovery_capacity_score = 0.85
    mock_report.stress_resilience = mock_stress

    # Priority actions.
    mock_report.priority_actions = [
      "增加每日步行目标至10000步",
      "改善睡眠环境",
    ]

    # Lifestyle optimization.
    mock_report.lifestyle_optimization = [
      "保持规律作息时间",
      "增加有氧运动频率",
    ]

    # Predictive insights.
    mock_report.predictive_insights = [
      "📊 根据当前趋势，您的睡眠质量将在未来一个月内改善15%",
      "⚠️ 建议关注压力管理，当前压力累积水平较高",
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
      report_generator.output_dir = Path(temp_dir)

      report_path = report_generator.generate_comprehensive_report(
        report=mock_report,
        title="综合健康分析报告",
        include_charts=False,  # Skip charts to simplify test.
        locale="zh",
      )

      assert report_path.exists()
      assert report_path.suffix == ".html"

      # Verify report content.
      content = report_path.read_text(encoding="utf-8")
      assert "综合健康分析报告" in content
      assert "执行摘要" in content
      assert "😴 睡眠质量分析" in content
      assert "💡 个性化建议" in content

  def test_generate_comprehensive_report_minimal_data(self, report_generator):
    """Test comprehensive report with minimal data."""
    # Create mock report with minimal fields.
    mock_report = Mock()
    mock_report.overall_wellness_score = 0.75
    from datetime import datetime

    mock_report.data_range = (datetime(2024, 1, 1), datetime(2024, 1, 31))
    mock_report.data_completeness_score = 0.8
    mock_report.analysis_confidence = 0.85

    # Ensure optional attributes are absent or None.
    mock_report.sleep_quality = None
    mock_report.activity_patterns = None
    mock_report.priority_actions = None
    mock_report.lifestyle_optimization = None
    mock_report.predictive_insights = None

    with tempfile.TemporaryDirectory() as temp_dir:
      report_generator.output_dir = Path(temp_dir)

      report_path = report_generator.generate_comprehensive_report(
        report=mock_report,
        title="最小数据报告",
        include_charts=False,
        locale="zh",
      )

      assert report_path.exists()
      content = report_path.read_text(encoding="utf-8")
      assert "最小数据报告" in content
      assert "75.0%" in content  # Wellness score.

  def test_html_structure_creation(self, report_generator, zh_translator):
    """Test HTML structure creation."""
    title = "测试报告"

    html = report_generator._create_html_structure(title, zh_translator)

    assert "<!DOCTYPE html>" in html
    assert title in html
    assert "container" in html
    assert "header" in html

  def test_executive_summary_creation(
    self,
    report_generator,
    sample_heart_rate_report,
    sample_sleep_report,
    sample_highlights,
    zh_translator,
  ):
    """Test executive summary creation."""
    summary_html = report_generator._create_executive_summary(
      sample_heart_rate_report,
      sample_sleep_report,
      sample_highlights,
      zh_translator,
    )

    assert "执行摘要" in summary_html
    assert "metric-grid" in summary_html
    assert "1,000" in summary_html  # Heart rate records.
    assert "30" in summary_html  # Sleep records.

  def test_heart_rate_section_creation(
    self, report_generator, sample_heart_rate_report, zh_translator
  ):
    """Test heart rate section creation."""
    section_html = report_generator._create_heart_rate_section(
      sample_heart_rate_report,
      include_charts=False,
      translator=zh_translator,
    )

    assert zh_translator.t("report.section.heart_rate") in section_html
    assert zh_translator.t("report.section.data_overview") in section_html
    assert zh_translator.t("report.section.resting_hr") in section_html
    assert "68 bpm" in section_html  # Current value.
    assert "EXCELLENT" in section_html  # Rating.

  def test_sleep_section_creation(
    self, report_generator, sample_sleep_report, zh_translator
  ):
    """Test sleep section creation."""
    section_html = report_generator._create_sleep_section(
      sample_sleep_report, include_charts=False, translator=zh_translator
    )

    assert "睡眠分析" in section_html
    assert "数据概览" in section_html
    assert "睡眠质量指标" in section_html
    assert "7.5" in section_html  # Average duration.
    assert "85%" in section_html  # Average efficiency.

  def test_highlights_section_creation(
    self, report_generator, sample_highlights, zh_translator
  ):
    """Test highlights section creation."""
    section_html = report_generator._create_highlights_section(
      sample_highlights, zh_translator
    )

    assert zh_translator.t("report.section.key_findings") in section_html
    assert "insight-list" in section_html
    assert "心率改善趋势" in section_html
    assert "睡眠质量需要关注" in section_html
    assert "保持规律的运动习惯" in section_html

  def test_data_quality_section_creation(
    self,
    report_generator,
    sample_heart_rate_report,
    sample_sleep_report,
    zh_translator,
  ):
    """Test data quality section creation."""
    section_html = report_generator._create_data_quality_section(
      sample_heart_rate_report, sample_sleep_report, zh_translator
    )

    assert zh_translator.t("report.section.data_quality_info") in section_html
    assert zh_translator.t("report.section.heart_rate") in section_html
    assert zh_translator.t("report.section.sleep") in section_html

  def test_close_html_structure(self, report_generator, zh_translator):
    """Test HTML structure closing."""
    closing_html = report_generator._close_html_structure(zh_translator)

    assert "</body>" in closing_html
    assert "</html>" in closing_html
    assert "footer" in closing_html

  def test_comprehensive_summary_creation(self, report_generator, zh_translator):
    """Test comprehensive summary creation."""
    mock_report = Mock()
    mock_report.overall_wellness_score = 0.88
    from datetime import datetime

    mock_report.data_range = (datetime(2024, 1, 1), datetime(2024, 1, 31))
    mock_report.data_completeness_score = 0.92
    mock_report.analysis_confidence = 0.89

    summary_html = report_generator._create_comprehensive_summary(
      mock_report, zh_translator
    )

    assert "执行摘要" in summary_html
    assert "dashboard-grid" in summary_html
    assert "88.0%" in summary_html  # Wellness score.
    assert "92.0%" in summary_html  # Data completeness.
    assert "89.0%" in summary_html  # Confidence.

  def test_detailed_analysis_sections_creation(self, report_generator, zh_translator):
    """Test detailed analysis section creation."""
    mock_report = Mock()

    # Sleep quality.
    mock_sleep = Mock()
    mock_sleep.average_duration_hours = 7.8
    mock_sleep.average_efficiency_percent = 87.5
    mock_sleep.sleep_debt_hours = 2.1
    mock_sleep.consistency_score = 0.82
    mock_report.sleep_quality = mock_sleep

    # Activity patterns.
    mock_activity = Mock()
    mock_activity.daily_step_average = 9500
    mock_activity.weekly_exercise_frequency = 4.2
    mock_activity.sedentary_hours_daily = 8.5
    mock_activity.activity_consistency_score = 0.78
    mock_report.activity_patterns = mock_activity

    sections_html = report_generator._create_detailed_analysis_sections(
      mock_report, {}, zh_translator
    )

    assert "睡眠质量分析" in sections_html
    assert "活动模式分析" in sections_html
    assert "7.8" in sections_html  # Sleep duration.
    assert "9,500" in sections_html  # Steps.

  def test_recommendations_section_creation(self, report_generator, zh_translator):
    """Test recommendations section creation."""
    mock_report = Mock()

    # Priority actions.
    mock_report.priority_actions = [
      "增加有氧运动时间",
      "改善饮食习惯",
    ]

    # Lifestyle optimization.
    mock_report.lifestyle_optimization = [
      "保持规律作息",
      "增加蔬果摄入",
    ]

    # Predictive insights.
    mock_report.predictive_insights = [
      "📊 睡眠质量预计改善",
      "⚠️ 注意压力管理",
    ]

    recommendations_html = report_generator._create_recommendations_section(
      mock_report, zh_translator
    )

    assert "个性化建议" in recommendations_html
    assert "优先行动项目" in recommendations_html
    assert "生活方式优化建议" in recommendations_html
    assert "预测性洞察" in recommendations_html
    assert "增加有氧运动时间" in recommendations_html
    assert "睡眠质量预计改善" in recommendations_html

  @patch("src.visualization.reports.logger")
  def test_error_handling_in_comprehensive_report(self, mock_logger, report_generator):
    """Test comprehensive report error handling."""
    mock_report = Mock()
    mock_report.overall_wellness_score = 0.8
    from datetime import datetime

    mock_report.data_range = (datetime(2024, 1, 1), datetime(2024, 1, 31))
    mock_report.data_completeness_score = 0.85
    mock_report.analysis_confidence = 0.9

    # Explicitly set optional attributes to None.
    mock_report.configure_mock(
      **{
        "sleep_quality": None,
        "activity_patterns": None,
        "priority_actions": None,
        "lifestyle_optimization": None,
        "predictive_insights": None,
      }
    )

    # Simulate chart generator error.
    with patch.object(
      report_generator.chart_generator, "generate_comprehensive_report_charts"
    ) as mock_generate:
      mock_generate.side_effect = Exception("Chart generation error")

      with tempfile.TemporaryDirectory() as temp_dir:
        report_generator.output_dir = Path(temp_dir)

        # Report should still be generated without charts.
        report_path = report_generator.generate_comprehensive_report(
          report=mock_report,
          title="错误处理测试",
          include_charts=True,
          locale="zh",
        )

        assert report_path.exists()
        mock_logger.warning.assert_called_once()

  def test_report_file_naming(self, report_generator):
    """Test report file naming."""
    with tempfile.TemporaryDirectory() as temp_dir:
      report_generator.output_dir = Path(temp_dir)

      # Generate multiple reports and ensure unique names.
      paths = []
      for i in range(3):
        path = report_generator.generate_html_report(
          title=f"Report {i}",
          include_charts=False,
        )
        paths.append(path)
        # Delay 1ms.
        time.sleep(0.001)

      # Ensure filenames are unique.
      filenames = [p.name for p in paths]
      assert len(set(filenames)) == len(filenames)

      # Ensure filenames include timestamps.
      for filename in filenames:
        assert "health_report_" in filename
        assert filename.endswith(".html")

  def test_output_directory_creation(self, report_generator):
    """Test output directory creation."""
    with tempfile.TemporaryDirectory() as temp_base:
      custom_output_dir = Path(temp_base) / "custom_reports"

      # Directory should be created when missing.
      assert not custom_output_dir.exists()

      generator = ReportGenerator(output_dir=custom_output_dir)

      assert custom_output_dir.exists()
      assert generator.output_dir == custom_output_dir
