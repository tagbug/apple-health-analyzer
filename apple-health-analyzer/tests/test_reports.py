"""Unit tests for report generation functionality."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.analyzers.highlights import HealthHighlights
from src.processors.heart_rate import HeartRateAnalysisReport
from src.processors.sleep import SleepAnalysisReport
from src.visualization.reports import ReportGenerator


class TestReportGenerator:
  """ReportGenerator 测试类"""

  @pytest.fixture
  def report_generator(self):
    """创建测试用的ReportGenerator实例"""
    return ReportGenerator()

  @pytest.fixture
  def sample_heart_rate_report(self):
    """创建示例心率分析报告"""
    from datetime import datetime

    report = HeartRateAnalysisReport(
      analysis_date=datetime.now(),
      data_range=(datetime(2024, 1, 1), datetime(2024, 1, 31)),
      record_count=1000,
      data_quality_score=0.85,
    )

    # 添加静息心率分析
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
    """创建示例睡眠分析报告"""
    from datetime import datetime

    report = SleepAnalysisReport(
      analysis_date=datetime.now(),
      data_range=(datetime(2024, 1, 1), datetime(2024, 1, 30)),
      record_count=30,
      data_quality_score=0.9,
    )

    # 添加睡眠质量指标
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
    """创建示例健康洞察"""
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
    """测试初始化"""
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
    """测试HTML报告生成"""
    with tempfile.TemporaryDirectory() as temp_dir:
      report_generator.output_dir = Path(temp_dir)

      report_path = report_generator.generate_html_report(
        title="测试健康报告",
        heart_rate_report=sample_heart_rate_report,
        sleep_report=sample_sleep_report,
        highlights=sample_highlights,
        include_charts=False,  # 不包含图表以简化测试
      )

      assert report_path.exists()
      assert report_path.suffix == ".html"

      # 检查报告内容
      content = report_path.read_text(encoding="utf-8")
      assert "测试健康报告" in content
      assert "执行摘要" in content
      assert "心率分析" in content
      assert "睡眠分析" in content
      assert "关键发现与建议" in content

  def test_generate_markdown_report(
    self,
    report_generator,
    sample_heart_rate_report,
    sample_sleep_report,
    sample_highlights,
  ):
    """测试Markdown报告生成"""
    with tempfile.TemporaryDirectory() as temp_dir:
      report_generator.output_dir = Path(temp_dir)

      report_path = report_generator.generate_markdown_report(
        title="测试健康报告",
        heart_rate_report=sample_heart_rate_report,
        sleep_report=sample_sleep_report,
        highlights=sample_highlights,
      )

      assert report_path.exists()
      assert report_path.suffix == ".md"

      # 检查报告内容
      content = report_path.read_text(encoding="utf-8")
      assert "# 测试健康报告" in content
      assert "## 执行摘要" in content
      assert "## 心率分析" in content
      assert "## 睡眠分析" in content
      assert "## 关键发现" in content

  def test_generate_comprehensive_report(self, report_generator):
    """测试综合报告生成"""
    # 创建模拟综合报告
    mock_report = Mock()
    mock_report.overall_wellness_score = 0.82
    from datetime import datetime

    mock_report.data_range = (datetime(2024, 1, 1), datetime(2024, 1, 31))
    mock_report.data_completeness_score = 0.88
    mock_report.analysis_confidence = 0.91

    # 添加睡眠质量
    mock_sleep = Mock()
    mock_sleep.average_duration_hours = 7.2
    mock_sleep.average_efficiency_percent = 85.0
    mock_sleep.sleep_debt_hours = 1.5
    mock_sleep.consistency_score = 0.8
    mock_report.sleep_quality = mock_sleep

    # 添加活动模式
    mock_activity = Mock()
    mock_activity.daily_step_average = 9200
    mock_activity.weekly_exercise_frequency = 4.5
    mock_activity.sedentary_hours_daily = 7.8
    mock_activity.activity_consistency_score = 0.85
    mock_report.activity_patterns = mock_activity

    # 添加压力韧性
    mock_stress = Mock()
    mock_stress.stress_accumulation_score = 0.25
    mock_stress.recovery_capacity_score = 0.85
    mock_report.stress_resilience = mock_stress

    # 添加优先行动
    mock_report.priority_actions = [
      "增加每日步行目标至10000步",
      "改善睡眠环境",
    ]

    # 添加生活方式优化
    mock_report.lifestyle_optimization = [
      "保持规律作息时间",
      "增加有氧运动频率",
    ]

    # 添加预测洞察
    mock_report.predictive_insights = [
      "📊 根据当前趋势，您的睡眠质量将在未来一个月内改善15%",
      "⚠️ 建议关注压力管理，当前压力累积水平较高",
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
      report_generator.output_dir = Path(temp_dir)

      report_path = report_generator.generate_comprehensive_report(
        report=mock_report,
        title="综合健康分析报告",
        include_charts=False,  # 不包含图表以简化测试
      )

      assert report_path.exists()
      assert report_path.suffix == ".html"

      # 检查报告内容
      content = report_path.read_text(encoding="utf-8")
      assert "综合健康分析报告" in content
      assert "执行摘要" in content
      assert "😴 睡眠质量分析" in content
      assert "💡 个性化建议" in content

  def test_generate_comprehensive_report_minimal_data(self, report_generator):
    """测试综合报告生成 - 最小数据"""
    # 创建只有基本属性的模拟报告
    mock_report = Mock()
    mock_report.overall_wellness_score = 0.75
    from datetime import datetime
    mock_report.data_range = (datetime(2024, 1, 1), datetime(2024, 1, 31))
    mock_report.data_completeness_score = 0.8
    mock_report.analysis_confidence = 0.85

    # 确保没有可能导致问题的属性
    del mock_report.sleep_quality
    del mock_report.activity_patterns
    del mock_report.priority_actions
    del mock_report.lifestyle_optimization
    del mock_report.predictive_insights

    with tempfile.TemporaryDirectory() as temp_dir:
      report_generator.output_dir = Path(temp_dir)

      report_path = report_generator.generate_comprehensive_report(
        report=mock_report,
        title="最小数据报告",
        include_charts=False,
      )

      assert report_path.exists()
      content = report_path.read_text(encoding="utf-8")
      assert "最小数据报告" in content
      assert "75.0%" in content  # 健康评分

  def test_html_structure_creation(self, report_generator):
    """测试HTML结构创建"""
    title = "测试报告"

    html = report_generator._create_html_structure(title)

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
  ):
    """测试执行摘要创建"""
    summary_html = report_generator._create_executive_summary(
      sample_heart_rate_report, sample_sleep_report, sample_highlights
    )

    assert "执行摘要" in summary_html
    assert "metric-grid" in summary_html
    assert "1,000" in summary_html  # 心率记录数
    assert "30" in summary_html  # 睡眠记录数

  def test_heart_rate_section_creation(
    self, report_generator, sample_heart_rate_report
  ):
    """测试心率分析章节创建"""
    section_html = report_generator._create_heart_rate_section(
      sample_heart_rate_report, include_charts=False
    )

    assert "心率分析" in section_html
    assert "数据概览" in section_html
    assert "静息心率分析" in section_html
    assert "68 bpm" in section_html  # 当前值
    assert "EXCELLENT" in section_html  # 健康评级

  def test_sleep_section_creation(self, report_generator, sample_sleep_report):
    """测试睡眠分析章节创建"""
    section_html = report_generator._create_sleep_section(
      sample_sleep_report, include_charts=False
    )

    assert "睡眠分析" in section_html
    assert "数据概览" in section_html
    assert "睡眠质量指标" in section_html
    assert "7.5" in section_html  # 平均时长
    assert "85%" in section_html  # 平均效率

  def test_highlights_section_creation(
    self, report_generator, sample_highlights
  ):
    """测试Highlights章节创建"""
    section_html = report_generator._create_highlights_section(
      sample_highlights
    )

    assert "关键发现与建议" in section_html
    assert "insight-list" in section_html
    assert "心率改善趋势" in section_html
    assert "睡眠质量需要关注" in section_html
    assert "保持规律的运动习惯" in section_html

  def test_data_quality_section_creation(
    self, report_generator, sample_heart_rate_report, sample_sleep_report
  ):
    """测试数据质量章节创建"""
    section_html = report_generator._create_data_quality_section(
      sample_heart_rate_report, sample_sleep_report
    )

    assert "数据质量信息" in section_html
    assert "心率数据" in section_html
    assert "睡眠数据" in section_html

  def test_close_html_structure(self, report_generator):
    """测试HTML结构关闭"""
    closing_html = report_generator._close_html_structure()

    assert "</body>" in closing_html
    assert "</html>" in closing_html
    assert "footer" in closing_html

  def test_comprehensive_summary_creation(self, report_generator):
    """测试综合摘要创建"""
    mock_report = Mock()
    mock_report.overall_wellness_score = 0.88
    from datetime import datetime
    mock_report.data_range = (datetime(2024, 1, 1), datetime(2024, 1, 31))
    mock_report.data_completeness_score = 0.92
    mock_report.analysis_confidence = 0.89

    summary_html = report_generator._create_comprehensive_summary(mock_report)

    assert "执行摘要" in summary_html
    assert "dashboard-grid" in summary_html
    assert "88.0%" in summary_html  # 健康评分
    assert "92.0%" in summary_html  # 数据完整性
    assert "89.0%" in summary_html  # 分析置信度

  def test_detailed_analysis_sections_creation(self, report_generator):
    """测试详细分析章节创建"""
    mock_report = Mock()

    # 添加睡眠质量
    mock_sleep = Mock()
    mock_sleep.average_duration_hours = 7.8
    mock_sleep.average_efficiency_percent = 87.5
    mock_sleep.sleep_debt_hours = 2.1
    mock_sleep.consistency_score = 0.82
    mock_report.sleep_quality = mock_sleep

    # 添加活动模式
    mock_activity = Mock()
    mock_activity.daily_step_average = 9500
    mock_activity.weekly_exercise_frequency = 4.2
    mock_activity.sedentary_hours_daily = 8.5
    mock_activity.activity_consistency_score = 0.78
    mock_report.activity_patterns = mock_activity

    sections_html = report_generator._create_detailed_analysis_sections(
      mock_report, {}
    )

    assert "睡眠质量分析" in sections_html
    assert "活动模式分析" in sections_html
    assert "7.8" in sections_html  # 睡眠时长
    assert "9,500" in sections_html  # 步数

  def test_recommendations_section_creation(self, report_generator):
    """测试建议章节创建"""
    mock_report = Mock()

    # 添加优先行动
    mock_report.priority_actions = [
      "增加有氧运动时间",
      "改善饮食习惯",
    ]

    # 添加生活方式优化
    mock_report.lifestyle_optimization = [
      "保持规律作息",
      "增加蔬果摄入",
    ]

    # 添加预测洞察
    mock_report.predictive_insights = [
      "📊 睡眠质量预计改善",
      "⚠️ 注意压力管理",
    ]

    recommendations_html = report_generator._create_recommendations_section(
      mock_report
    )

    assert "个性化建议" in recommendations_html
    assert "优先行动项目" in recommendations_html
    assert "生活方式优化建议" in recommendations_html
    assert "预测性洞察" in recommendations_html
    assert "增加有氧运动时间" in recommendations_html
    assert "睡眠质量预计改善" in recommendations_html

  @patch("src.visualization.reports.logger")
  def test_error_handling_in_comprehensive_report(
    self, mock_logger, report_generator
  ):
    """测试综合报告错误处理"""
    mock_report = Mock()
    mock_report.overall_wellness_score = 0.8
    from datetime import datetime
    mock_report.data_range = (datetime(2024, 1, 1), datetime(2024, 1, 31))
    mock_report.data_completeness_score = 0.85
    mock_report.analysis_confidence = 0.9

    # 明确设置可能导致问题的属性为None
    mock_report.configure_mock(**{
      'sleep_quality': None,
      'activity_patterns': None,
      'priority_actions': None,
      'lifestyle_optimization': None,
      'predictive_insights': None,
    })

    # 模拟图表生成器错误
    with patch.object(
      report_generator.chart_generator, "generate_comprehensive_report_charts"
    ) as mock_generate:
      mock_generate.side_effect = Exception("Chart generation error")

      with tempfile.TemporaryDirectory() as temp_dir:
        report_generator.output_dir = Path(temp_dir)

        # 应该仍然生成报告，只是没有图表
        report_path = report_generator.generate_comprehensive_report(
          report=mock_report,
          title="错误处理测试",
          include_charts=True,
        )

        assert report_path.exists()
        mock_logger.warning.assert_called_once()

  def test_report_file_naming(self, report_generator):
    """测试报告文件命名"""
    with tempfile.TemporaryDirectory() as temp_dir:
      report_generator.output_dir = Path(temp_dir)

      # 生成多个报告，检查文件名是否唯一
      paths = []
      for i in range(3):
        path = report_generator.generate_html_report(
          title=f"报告{i}",
          include_charts=False,
        )
        paths.append(path)

      # 检查所有文件名都不同
      filenames = [p.name for p in paths]
      assert len(set(filenames)) == len(filenames)

      # 检查文件名包含时间戳
      for filename in filenames:
        assert "health_report_" in filename
        assert filename.endswith(".html")

  def test_output_directory_creation(self, report_generator):
    """测试输出目录创建"""
    with tempfile.TemporaryDirectory() as temp_base:
      custom_output_dir = Path(temp_base) / "custom_reports"

      # 目录不存在时应该自动创建
      assert not custom_output_dir.exists()

      generator = ReportGenerator(output_dir=custom_output_dir)

      assert custom_output_dir.exists()
      assert generator.output_dir == custom_output_dir
