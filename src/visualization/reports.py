"""Report generation module - generates comprehensive health analysis reports"""

from datetime import datetime
from pathlib import Path
from typing import Any

from ..analyzers.highlights import HealthHighlights
from ..processors.heart_rate import HeartRateAnalysisReport
from ..processors.sleep import SleepAnalysisReport
from ..utils.logger import get_logger
from .charts import ChartGenerator

logger = get_logger(__name__)


class ReportGenerator:
  """Health report generator

  Generates comprehensive reports including charts, statistical analysis, and health insights.
  Supports multiple formats including HTML and Markdown.
  """

  def __init__(self, output_dir: Path | None = None):
    """Initialize report generator

    Args:
        output_dir: Report output directory
    """
    self.output_dir = Path(output_dir) if output_dir else Path("./output/reports")
    self.output_dir.mkdir(parents=True, exist_ok=True)

    # Create chart generator.
    self.chart_generator = ChartGenerator()

    logger.info(f"ReportGenerator initialized: output_dir={self.output_dir}")

  def generate_html_report(
    self,
    title: str = "健康分析报告",
    heart_rate_report: HeartRateAnalysisReport | None = None,
    sleep_report: SleepAnalysisReport | None = None,
    highlights: HealthHighlights | None = None,
    include_charts: bool = True,
    heart_rate_data: list | None = None,
    sleep_data: list | None = None,
  ) -> Path:
    """Generate HTML format report

    Args:
        title: Report title
        heart_rate_report: Heart rate analysis report
        sleep_report: Sleep analysis report
        highlights: Health insights
        include_charts: Whether to include charts

    Returns:
        Report file path
    """
    logger.info("Generating HTML report")

    # Create report HTML content.
    html_content = self._create_html_structure(title)

    # Add executive summary.
    html_content += self._create_executive_summary(
      heart_rate_report, sleep_report, highlights
    )

    # Add heart rate analysis section.
    if heart_rate_report:
      html_content += self._create_heart_rate_section(
        heart_rate_report, include_charts, heart_rate_data
      )

    # Add sleep analysis section.
    if sleep_report:
      html_content += self._create_sleep_section(sleep_report, include_charts)

    # Add highlights section.
    if highlights:
      html_content += self._create_highlights_section(highlights)

    # Add data quality section.
    html_content += self._create_data_quality_section(heart_rate_report, sleep_report)

    # Close HTML.
    html_content += self._close_html_structure()

    # Save report.
    import time

    timestamp = (
      datetime.now().strftime("%Y%m%d_%H%M%S")
      + f"_{int(time.time() * 1000000) % 1000000:06d}"
    )
    report_path = self.output_dir / f"health_report_{timestamp}.html"
    report_path.write_text(html_content, encoding="utf-8")

    logger.info(f"HTML report saved to {report_path}")
    return report_path

  def generate_markdown_report(
    self,
    title: str = "健康分析报告",
    heart_rate_report: HeartRateAnalysisReport | None = None,
    sleep_report: SleepAnalysisReport | None = None,
    highlights: HealthHighlights | None = None,
  ) -> Path:
    """Generate a Markdown report.

    Args:
        title: Report title.
        heart_rate_report: Heart rate analysis report.
        sleep_report: Sleep analysis report.
        highlights: Health insights.

    Returns:
        Report file path.
    """
    logger.info("Generating Markdown report")

    md_content = f"# {title}\n\n"
    md_content += f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md_content += "---\n\n"

    # Executive summary.
    md_content += "## 执行摘要\n\n"
    if heart_rate_report:
      md_content += f"- **心率记录数**: {heart_rate_report.record_count}\n"
      md_content += f"- **数据质量**: {heart_rate_report.data_quality_score:.1%}\n"
    if sleep_report:
      md_content += f"- **睡眠记录数**: {sleep_report.record_count}\n"
      md_content += f"- **数据质量**: {sleep_report.data_quality_score:.1%}\n"
    md_content += "\n"

    # Highlights.
    if highlights:
      md_content += "## 关键发现\n\n"
      for i, insight in enumerate(highlights.insights[:5], 1):
        priority_emoji = {
          "high": "🔴",
          "medium": "🟡",
          "low": "🟢",
        }
        emoji = priority_emoji.get(insight.priority, "⚪")
        md_content += f"{i}. {emoji} **{insight.title}**\n"
        md_content += f"   - {insight.message}\n\n"

    # Heart rate analysis.
    if heart_rate_report:
      md_content += "## 心率分析\n\n"
      md_content += "### 数据概览\n\n"
      md_content += f"- 记录总数: {heart_rate_report.record_count}\n"
      md_content += f"- 时间范围: {heart_rate_report.data_range[0]} 至 {heart_rate_report.data_range[1]}\n"
      md_content += f"- 数据质量评分: {heart_rate_report.data_quality_score:.1%}\n\n"

      if heart_rate_report.resting_hr_analysis:
        rhr = heart_rate_report.resting_hr_analysis
        md_content += "### 静息心率\n\n"
        md_content += f"- 当前值: {rhr.current_value:.0f} bpm\n"
        md_content += f"- 基线值: {rhr.baseline_value:.0f} bpm\n"
        md_content += f"- 变化: {rhr.change_from_baseline:+.1f} bpm\n"
        md_content += f"- 趋势: {rhr.trend_direction}\n"
        md_content += f"- 健康评级: {rhr.health_rating}\n\n"

    # Sleep analysis.
    if sleep_report:
      md_content += "## 睡眠分析\n\n"
      md_content += "### 数据概览\n\n"
      md_content += f"- 记录总数: {sleep_report.record_count}\n"
      md_content += (
        f"- 时间范围: {sleep_report.data_range[0]} 至 {sleep_report.data_range[1]}\n"
      )
      md_content += f"- 数据质量评分: {sleep_report.data_quality_score:.1%}\n\n"

      if sleep_report.quality_metrics:
        quality = sleep_report.quality_metrics
        md_content += "### 睡眠质量指标\n\n"
        md_content += f"- 平均时长: {quality.average_duration:.1f} 小时\n"
        md_content += f"- 平均效率: {quality.average_efficiency:.1%}\n"
        md_content += f"- 规律性评分: {quality.consistency_score:.1%}\n\n"

    # Recommendations.
    if highlights and highlights.recommendations:
      md_content += "## 健康建议\n\n"
      for i, rec in enumerate(highlights.recommendations, 1):
        md_content += f"{i}. {rec}\n"
      md_content += "\n"

    # Save report.
    report_path = (
      self.output_dir / f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    report_path.write_text(md_content, encoding="utf-8")

    logger.info(f"Markdown report saved to {report_path}")
    return report_path

  def generate_comprehensive_report(
    self,
    report: Any,  # ComprehensiveHealthReport
    title: str = "综合健康分析报告",
    include_charts: bool = True,
  ) -> Path:
    """Generate comprehensive health analysis report.

    Args:
        report: Comprehensive health analysis report.
        title: Report title.
        include_charts: Whether to include charts.

    Returns:
        Report file path.
    """
    logger.info("Generating comprehensive health report")

    # Generate charts.
    charts = {}
    if include_charts:
      try:
        charts = self.chart_generator.generate_comprehensive_report_charts(
          report, self.output_dir / "charts"
        )
      except Exception as e:
        logger.warning(f"Chart generation failed; continuing with text report: {e}")
        charts = {}

    # Create HTML content.
    html_content = self._create_comprehensive_html_structure(title, report, charts)

    # Save report.
    report_path = (
      self.output_dir
      / f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    )
    report_path.write_text(html_content, encoding="utf-8")

    logger.info(f"Comprehensive report saved to {report_path}")
    return report_path

  def _create_comprehensive_html_structure(
    self,
    title: str,
    report: Any,
    charts: dict[str, Path],
  ) -> str:
    """Create comprehensive report HTML structure."""
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        :root {{
            --primary-color: #4CAF50;
            --secondary-color: #2196F3;
            --success-color: #8BC34A;
            --warning-color: #FF9800;
            --danger-color: #F44336;
            --info-color: #03A9F4;
            --light-bg: #f8f9fa;
            --card-shadow: 0 4px 6px rgba(0,0,0,0.1);
            --border-radius: 12px;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #2c3e50;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}

        .header {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 40px;
            text-align: center;
            border-radius: var(--border-radius);
            margin-bottom: 30px;
            box-shadow: var(--card-shadow);
        }}

        .header h1 {{
            font-size: 3em;
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}

        .header .subtitle {{
            color: #7f8c8d;
            font-size: 1.2em;
        }}

        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}

        .metric-card {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 25px;
            border-radius: var(--border-radius);
            box-shadow: var(--card-shadow);
            transition: transform 0.3s ease;
        }}

        .metric-card:hover {{
            transform: translateY(-5px);
        }}

        .metric-card.success {{ border-left: 4px solid var(--success-color); }}
        .metric-card.warning {{ border-left: 4px solid var(--warning-color); }}
        .metric-card.danger {{ border-left: 4px solid var(--danger-color); }}
        .metric-card.info {{ border-left: 4px solid var(--info-color); }}

        .metric-title {{
            font-size: 0.9em;
            color: #7f8c8d;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: var(--primary-color);
        }}

        .metric-unit {{
            font-size: 0.8em;
            color: #95a5a6;
        }}

        .section {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 40px;
            margin-bottom: 30px;
            border-radius: var(--border-radius);
            box-shadow: var(--card-shadow);
        }}

        .section h2 {{
            color: var(--primary-color);
            font-size: 2.2em;
            margin-bottom: 30px;
            padding-bottom: 15px;
            border-bottom: 3px solid var(--primary-color);
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .section h3 {{
            color: var(--secondary-color);
            font-size: 1.6em;
            margin: 30px 0 20px 0;
        }}

        .chart-container {{
            margin: 30px 0;
            padding: 20px;
            background: var(--light-bg);
            border-radius: var(--border-radius);
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
        }}

        .insights-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}

        .insight-card {{
            background: linear-gradient(135deg, #fff 0%, #f8f9fa 100%);
            padding: 25px;
            border-radius: var(--border-radius);
            box-shadow: var(--card-shadow);
            border-left: 4px solid var(--secondary-color);
        }}

        .insight-card.high {{ border-left-color: var(--danger-color); }}
        .insight-card.medium {{ border-left-color: var(--warning-color); }}
        .insight-card.low {{ border-left-color: var(--success-color); }}

        .insight-title {{
            font-weight: bold;
            font-size: 1.2em;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .insight-message {{
            color: #34495e;
            line-height: 1.6;
        }}

        .recommendations {{
            background: linear-gradient(135deg, #e8f5e9 0%, #d4edda 100%);
            padding: 30px;
            border-radius: var(--border-radius);
            border-left: 4px solid var(--success-color);
            margin: 30px 0;
        }}

        .recommendations ol {{
            margin-left: 20px;
        }}

        .recommendations li {{
            margin-bottom: 15px;
            padding: 10px;
            background: rgba(255, 255, 255, 0.7);
            border-radius: 8px;
        }}

        .footer {{
            text-align: center;
            padding: 30px;
            color: rgba(255, 255, 255, 0.8);
            font-size: 0.9em;
        }}

        @media (max-width: 768px) {{
            .dashboard-grid {{
                grid-template-columns: 1fr;
            }}

            .insights-grid {{
                grid-template-columns: 1fr;
            }}

            .header h1 {{
                font-size: 2em;
            }}

            .section {{
                padding: 20px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <p class="subtitle">生成时间: {datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}</p>
        </div>
"""

    # Executive summary.
    html_content += self._create_comprehensive_summary(report)

    # Health dashboard.
    if "dashboard" in charts:
      html_content += f"""
        <div class="section">
            <h2>📊 健康仪表盘</h2>
            <div class="chart-container">
                <iframe src="{charts["dashboard"].relative_to(self.output_dir)}"
                        width="100%" height="600" frameborder="0"></iframe>
            </div>
        </div>
      """

    # Detailed analysis sections.
    html_content += self._create_detailed_analysis_sections(report, charts)

    # Recommendations section.
    html_content += self._create_recommendations_section(report)

    # Footer.
    html_content += """
        <div class="footer">
            <p>本报告由 Apple Health Analyzer 自动生成 | 数据来源: Apple Health 导出数据</p>
        </div>
    </div>
</body>
</html>
"""

    return html_content

  def _create_comprehensive_summary(self, report: Any) -> str:
    """Create comprehensive summary."""
    content = '<div class="section">\n'
    content += "<h2>📈 执行摘要</h2>\n"
    content += '<div class="dashboard-grid">\n'

    # Overall wellness score.
    if hasattr(report, "overall_wellness_score"):
      score_class = (
        "success"
        if report.overall_wellness_score > 0.7
        else "warning"
        if report.overall_wellness_score > 0.4
        else "danger"
      )
      content += f'<div class="metric-card {score_class}">\n'
      content += '<div class="metric-title">整体健康评分</div>\n'
      content += (
        f'<div class="metric-value">{report.overall_wellness_score:.1%}</div>\n'
      )
      content += "</div>\n"

    # Data range.
    if hasattr(report, "data_range"):
      content += '<div class="metric-card info">\n'
      content += '<div class="metric-title">数据时间范围</div>\n'
      content += f'<div class="metric-value" style="font-size:1.2em">{report.data_range[0].strftime("%Y-%m-%d")}<br>至<br>{report.data_range[1].strftime("%Y-%m-%d")}</div>\n'
      content += "</div>\n"

    # Data completeness.
    if hasattr(report, "data_completeness_score"):
      completeness_class = (
        "success"
        if report.data_completeness_score > 0.8
        else "warning"
        if report.data_completeness_score > 0.5
        else "danger"
      )
      content += f'<div class="metric-card {completeness_class}">\n'
      content += '<div class="metric-title">数据完整性</div>\n'
      content += (
        f'<div class="metric-value">{report.data_completeness_score:.1%}</div>\n'
      )
      content += "</div>\n"

    # Analysis confidence.
    if hasattr(report, "analysis_confidence"):
      confidence_class = (
        "success"
        if report.analysis_confidence > 0.8
        else "warning"
        if report.analysis_confidence > 0.6
        else "danger"
      )
      content += f'<div class="metric-card {confidence_class}">\n'
      content += '<div class="metric-title">分析置信度</div>\n'
      content += f'<div class="metric-value">{report.analysis_confidence:.1%}</div>\n'
      content += "</div>\n"

    content += "</div>\n"
    content += "</div>\n"

    return content

  def _create_detailed_analysis_sections(
    self, report: Any, charts: dict[str, Path]
  ) -> str:
    """Create detailed analysis sections."""
    content = ""

    # Sleep analysis.
    if (
      hasattr(report, "sleep_quality")
      and getattr(report, "sleep_quality", None) is not None
    ):
      content += '<div class="section">\n'
      content += "<h2>😴 睡眠质量分析</h2>\n"
      content += '<div class="dashboard-grid">\n'

      sleep = report.sleep_quality
      content += '<div class="metric-card success">\n'
      content += '<div class="metric-title">平均睡眠时长</div>\n'
      content += f'<div class="metric-value">{sleep.average_duration_hours:.1f}<span class="metric-unit"> 小时</span></div>\n'
      content += "</div>\n"

      content += '<div class="metric-card info">\n'
      content += '<div class="metric-title">睡眠效率</div>\n'
      content += f'<div class="metric-value">{sleep.average_efficiency_percent:.1f}<span class="metric-unit">%</span></div>\n'
      content += "</div>\n"

      content += '<div class="metric-card warning">\n'
      content += '<div class="metric-title">睡眠债</div>\n'
      content += f'<div class="metric-value">{sleep.sleep_debt_hours:.1f}<span class="metric-unit"> 小时</span></div>\n'
      content += "</div>\n"

      content += '<div class="metric-card secondary">\n'
      content += '<div class="metric-title">规律性评分</div>\n'
      content += f'<div class="metric-value">{sleep.consistency_score:.1%}</div>\n'
      content += "</div>\n"

      content += "</div>\n"
      content += "</div>\n"

    # Activity pattern analysis.
    if getattr(report, "activity_patterns", None) is not None:
      content += '<div class="section">\n'
      content += "<h2>🏃 活动模式分析</h2>\n"
      content += '<div class="dashboard-grid">\n'

      activity = report.activity_patterns
      content += '<div class="metric-card success">\n'
      content += '<div class="metric-title">每日平均步数</div>\n'
      content += f'<div class="metric-value">{activity.daily_step_average:,}<span class="metric-unit"> 步</span></div>\n'
      content += "</div>\n"

      content += '<div class="metric-card info">\n'
      content += '<div class="metric-title">每周运动频率</div>\n'
      content += f'<div class="metric-value">{activity.weekly_exercise_frequency:.1f}<span class="metric-unit"> 次</span></div>\n'
      content += "</div>\n"

      content += '<div class="metric-card warning">\n'
      content += '<div class="metric-title">每日久坐时间</div>\n'
      content += f'<div class="metric-value">{activity.sedentary_hours_daily:.1f}<span class="metric-unit"> 小时</span></div>\n'
      content += "</div>\n"

      content += '<div class="metric-card secondary">\n'
      content += '<div class="metric-title">活动一致性</div>\n'
      content += (
        f'<div class="metric-value">{activity.activity_consistency_score:.1%}</div>\n'
      )
      content += "</div>\n"

      content += "</div>\n"
      content += "</div>\n"

    # Correlation analysis.
    if "correlation" in charts:
      content += f"""
        <div class="section">
            <h2>🔗 健康指标相关性</h2>
            <div class="chart-container">
                <iframe src="{charts["correlation"].relative_to(self.output_dir)}"
                        width="100%" height="600" frameborder="0"></iframe>
            </div>
        </div>
      """

    # Risk assessment.
    if "risk_assessment" in charts:
      content += f"""
        <div class="section">
            <h2>⚠️ 健康风险评估</h2>
            <div class="chart-container">
                <iframe src="{charts["risk_assessment"].relative_to(self.output_dir)}"
                        width="100%" height="500" frameborder="0"></iframe>
            </div>
        </div>
      """

    return content

  def _create_recommendations_section(self, report: Any) -> str:
    """Create recommendations section."""
    content = '<div class="section">\n'
    content += "<h2>💡 个性化建议</h2>\n"

    # Priority actions.
    if hasattr(report, "priority_actions") and report.priority_actions:
      content += "<h3>优先行动项目</h3>\n"
      content += '<div class="recommendations">\n'
      content += "<ol>\n"
      for action in report.priority_actions:
        content += f"<li>{action}</li>\n"
      content += "</ol>\n"
      content += "</div>\n"

    # Lifestyle optimization.
    if hasattr(report, "lifestyle_optimization") and report.lifestyle_optimization:
      content += "<h3>生活方式优化建议</h3>\n"
      content += '<div class="recommendations">\n'
      content += "<ol>\n"
      for optimization in report.lifestyle_optimization:
        content += f"<li>{optimization}</li>\n"
      content += "</ol>\n"
      content += "</div>\n"

    # Predictive insights.
    if hasattr(report, "predictive_insights") and report.predictive_insights:
      content += "<h3>预测性洞察</h3>\n"
      content += '<div class="insights-grid">\n'

      for insight in report.predictive_insights:
        priority_class = "low"  # Default to low priority.
        if "⚠️" in insight or "风险" in insight:
          priority_class = "high"
        elif "📊" in insight or "建议" in insight:
          priority_class = "medium"

        content += f'<div class="insight-card {priority_class}">\n'
        content += f'<div class="insight-message">{insight}</div>\n'
        content += "</div>\n"

      content += "</div>\n"

    content += "</div>\n"
    return content

  def _create_html_structure(self, title: str) -> str:
    """Create base HTML structure."""
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --primary-color: #4CAF50;
            --secondary-color: #2196F3;
            --warning-color: #FF9800;
            --danger-color: #F44336;
            --light-bg: #f5f5f5;
            --card-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: var(--light-bg);
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}

        header {{
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 40px 20px;
            text-align: center;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: var(--card-shadow);
        }}

        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        header .subtitle {{
            font-size: 1.1em;
            opacity: 0.9;
        }}

        .section {{
            background: white;
            padding: 30px;
            margin-bottom: 30px;
            border-radius: 8px;
            box-shadow: var(--card-shadow);
        }}

        .section h2 {{
            color: var(--primary-color);
            font-size: 1.8em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--primary-color);
        }}

        .section h3 {{
            color: var(--secondary-color);
            font-size: 1.4em;
            margin-top: 20px;
            margin-bottom: 15px;
        }}

        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}

        .metric-card {{
            background: var(--light-bg);
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid var(--primary-color);
        }}

        .metric-card.warning {{
            border-left-color: var(--warning-color);
        }}

        .metric-card.danger {{
            border-left-color: var(--danger-color);
        }}

        .metric-label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}

        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: var(--primary-color);
        }}

        .insight-list {{
            list-style: none;
        }}

        .insight-item {{
            background: var(--light-bg);
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 8px;
            border-left: 4px solid var(--secondary-color);
        }}

        .insight-item.high {{
            border-left-color: var(--danger-color);
        }}

        .insight-item.medium {{
            border-left-color: var(--warning-color);
        }}

        .insight-item.low {{
            border-left-color: var(--primary-color);
        }}

        .insight-title {{
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 8px;
        }}

        .insight-message {{
            color: #666;
        }}

        .recommendations {{
            background: #e8f5e9;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid var(--primary-color);
        }}

        .recommendations ol {{
            margin-left: 20px;
            margin-top: 10px;
        }}

        .recommendations li {{
            margin-bottom: 10px;
        }}

        footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }}

        .chart-container {{
            margin: 30px 0;
            text-align: center;
        }}

        @media (max-width: 768px) {{
            .metric-grid {{
                grid-template-columns: 1fr;
            }}

            header h1 {{
                font-size: 1.8em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <p class="subtitle">生成时间: {datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}</p>
        </header>
"""

  def _create_executive_summary(
    self,
    heart_rate_report: HeartRateAnalysisReport | None,
    sleep_report: SleepAnalysisReport | None,
    highlights: HealthHighlights | None,
  ) -> str:
    """Create executive summary section."""
    content = '<div class="section">\n'
    content += "<h2>📊 执行摘要</h2>\n"
    content += '<div class="metric-grid">\n'

    # Heart rate overview.
    if heart_rate_report:
      content += '<div class="metric-card">\n'
      content += '<div class="metric-label">心率记录数</div>\n'
      content += f'<div class="metric-value">{heart_rate_report.record_count:,}</div>\n'
      content += "</div>\n"

      content += '<div class="metric-card">\n'
      content += '<div class="metric-label">心率数据质量</div>\n'
      content += (
        f'<div class="metric-value">{heart_rate_report.data_quality_score:.0%}</div>\n'
      )
      content += "</div>\n"

    # Sleep data overview.
    if sleep_report:
      content += '<div class="metric-card">\n'
      content += '<div class="metric-label">睡眠记录数</div>\n'
      content += f'<div class="metric-value">{sleep_report.record_count}</div>\n'
      content += "</div>\n"

      content += '<div class="metric-card">\n'
      content += '<div class="metric-label">睡眠数据质量</div>\n'
      content += (
        f'<div class="metric-value">{sleep_report.data_quality_score:.0%}</div>\n'
      )
      content += "</div>\n"

    # Highlights summary.
    if highlights:
      high_count = sum(1 for i in highlights.insights if i.priority == "high")
      content += f'<div class="metric-card {"danger" if high_count > 0 else ""}">\n'
      content += '<div class="metric-label">高优先级洞察</div>\n'
      content += f'<div class="metric-value">{high_count}</div>\n'
      content += "</div>\n"

    content += "</div>\n"
    content += "</div>\n"

    return content

  def _create_heart_rate_section(
    self,
    report: HeartRateAnalysisReport,
    include_charts: bool,
    heart_rate_data: list | None = None,
  ) -> str:
    """Create heart rate analysis section."""
    content = '<div class="section">\n'
    content += "<h2>❤️ 心率分析</h2>\n"

    # Data range.
    content += "<h3>数据概览</h3>\n"
    content += f"<p>时间范围: {report.data_range[0]} 至 {report.data_range[1]}</p>\n"
    content += f"<p>记录总数: {report.record_count:,}</p>\n"
    content += f"<p>数据质量评分: {report.data_quality_score:.1%}</p>\n"

    # Resting heart rate.
    if report.resting_hr_analysis:
      rhr = report.resting_hr_analysis
      content += "<h3>静息心率分析</h3>\n"
      content += '<div class="metric-grid">\n'

      content += '<div class="metric-card">\n'
      content += '<div class="metric-label">当前值</div>\n'
      content += f'<div class="metric-value">{rhr.current_value:.0f} bpm</div>\n'
      content += "</div>\n"

      content += '<div class="metric-card">\n'
      content += '<div class="metric-label">基线值</div>\n'
      content += f'<div class="metric-value">{rhr.baseline_value:.0f} bpm</div>\n'
      content += "</div>\n"

      change_class = "danger" if rhr.change_from_baseline > 2 else ""
      content += f'<div class="metric-card {change_class}">\n'
      content += '<div class="metric-label">变化</div>\n'
      content += (
        f'<div class="metric-value">{rhr.change_from_baseline:+.1f} bpm</div>\n'
      )
      content += "</div>\n"

      content += '<div class="metric-card">\n'
      content += '<div class="metric-label">健康评级</div>\n'
      content += f'<div class="metric-value" style="font-size:1.5em">{rhr.health_rating.upper()}</div>\n'
      content += "</div>\n"

      content += "</div>\n"

    content += "</div>\n"
    return content

  def _create_sleep_section(
    self, report: SleepAnalysisReport, include_charts: bool
  ) -> str:
    """Create sleep analysis section."""
    content = '<div class="section">\n'
    content += "<h2>😴 睡眠分析</h2>\n"

    # Data range.
    content += "<h3>数据概览</h3>\n"
    content += f"<p>时间范围: {report.data_range[0]} 至 {report.data_range[1]}</p>\n"
    content += f"<p>记录总数: {report.record_count}</p>\n"
    content += f"<p>数据质量评分: {report.data_quality_score:.1%}</p>\n"

    # Sleep quality metrics.
    if report.quality_metrics:
      quality = report.quality_metrics
      content += "<h3>睡眠质量指标</h3>\n"
      content += '<div class="metric-grid">\n'

      duration_class = "danger" if quality.average_duration < 7 else ""
      content += f'<div class="metric-card {duration_class}">\n'
      content += '<div class="metric-label">平均睡眠时长</div>\n'
      content += (
        f'<div class="metric-value">{quality.average_duration:.1f} 小时</div>\n'
      )
      content += "</div>\n"

      efficiency_class = "warning" if quality.average_efficiency < 0.85 else ""
      content += f'<div class="metric-card {efficiency_class}">\n'
      content += '<div class="metric-label">平均睡眠效率</div>\n'
      content += f'<div class="metric-value">{quality.average_efficiency:.0%}</div>\n'
      content += "</div>\n"

      consistency_class = "warning" if quality.consistency_score < 0.7 else ""
      content += f'<div class="metric-card {consistency_class}">\n'
      content += '<div class="metric-label">规律性评分</div>\n'
      content += f'<div class="metric-value">{quality.consistency_score:.0%}</div>\n'
      content += "</div>\n"

      content += "</div>\n"

    content += "</div>\n"
    return content

  def _create_highlights_section(self, highlights: HealthHighlights) -> str:
    """Create highlights section."""
    content = '<div class="section">\n'
    content += "<h2>💡 关键发现与建议</h2>\n"

    # Insight list.
    if highlights.insights:
      content += "<h3>健康洞察</h3>\n"
      content += '<ul class="insight-list">\n'

      for insight in highlights.insights[:8]:  # Show the first 8 insights.
        content += f'<li class="insight-item {insight.priority}">\n'
        priority_emoji = {
          "high": "🔴",
          "medium": "🟡",
          "low": "🟢",
        }
        emoji = priority_emoji.get(insight.priority, "⚪")
        content += f'<div class="insight-title">{emoji} {insight.title}</div>\n'
        content += f'<div class="insight-message">{insight.message}</div>\n'
        content += "</li>\n"

      content += "</ul>\n"

    # Recommendations.
    if highlights.recommendations:
      content += "<h3>健康建议</h3>\n"
      content += '<div class="recommendations">\n'
      content += "<ol>\n"
      for rec in highlights.recommendations:
        content += f"<li>{rec}</li>\n"
      content += "</ol>\n"
      content += "</div>\n"

    content += "</div>\n"
    return content

  def _create_data_quality_section(
    self,
    heart_rate_report: HeartRateAnalysisReport | None,
    sleep_report: SleepAnalysisReport | None,
  ) -> str:
    """Create data quality section."""
    content = '<div class="section">\n'
    content += "<h2>📋 数据质量信息</h2>\n"

    if heart_rate_report:
      content += "<h3>心率数据</h3>\n"
      content += "<ul>\n"
      content += f"<li>记录总数: {heart_rate_report.record_count:,}</li>\n"
      content += f"<li>数据质量评分: {heart_rate_report.data_quality_score:.1%}</li>\n"
      content += f"<li>时间范围: {heart_rate_report.data_range[0]} 至 {heart_rate_report.data_range[1]}</li>\n"
      content += "</ul>\n"

    if sleep_report:
      content += "<h3>睡眠数据</h3>\n"
      content += "<ul>\n"
      content += f"<li>记录总数: {sleep_report.record_count}</li>\n"
      content += f"<li>数据质量评分: {sleep_report.data_quality_score:.1%}</li>\n"
      content += f"<li>时间范围: {sleep_report.data_range[0]} 至 {sleep_report.data_range[1]}</li>\n"
      content += "</ul>\n"

    content += "</div>\n"
    return content

  def _close_html_structure(self) -> str:
    """Close HTML structure."""
    return """
        <footer>
            <p>本报告由 Apple Health Analyzer 自动生成</p>
            <p>数据来源: Apple Health 导出数据</p>
        </footer>
    </div>
</body>
</html>
"""
