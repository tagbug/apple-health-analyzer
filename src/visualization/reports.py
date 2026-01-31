"""Report generation module - generates comprehensive health analysis reports"""

from datetime import datetime
from pathlib import Path
from typing import Any

from ..analyzers.highlights import HealthHighlights
from ..processors.heart_rate import HeartRateAnalysisReport
from ..processors.sleep import SleepAnalysisReport
from ..utils.logger import get_logger
from ..i18n import Translator, resolve_locale
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
    init_translator = Translator(resolve_locale())
    logger.info(
      init_translator.t(
        "log.report_generator_initialized",
        output_dir=str(self.output_dir),
      )
    )

  def generate_html_report(
    self,
    title: str | None = None,
    heart_rate_report: HeartRateAnalysisReport | None = None,
    sleep_report: SleepAnalysisReport | None = None,
    highlights: HealthHighlights | None = None,
    include_charts: bool = True,
    heart_rate_data: list | None = None,
    sleep_data: list | None = None,
    locale: str | None = None,
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
    translator = Translator(resolve_locale(locale))
    logger.info(translator.t("log.report_html_generating"))
    report_title = title or translator.t("report.title.health_analysis")

    # Create report HTML content.
    html_content = self._create_html_structure(report_title, translator)

    # Add executive summary.
    html_content += self._create_executive_summary(
      heart_rate_report, sleep_report, highlights, translator
    )

    # Add heart rate analysis section.
    if heart_rate_report:
      html_content += self._create_heart_rate_section(
        heart_rate_report, include_charts, heart_rate_data, translator
      )

    # Add sleep analysis section.
    if sleep_report:
      html_content += self._create_sleep_section(
        sleep_report, include_charts, translator
      )

    # Add highlights section.
    if highlights:
      html_content += self._create_highlights_section(highlights, translator)

    # Add data quality section.
    html_content += self._create_data_quality_section(
      heart_rate_report, sleep_report, translator
    )

    # Close HTML.
    html_content += self._close_html_structure(translator)

    # Save report.
    import time

    timestamp = (
      datetime.now().strftime("%Y%m%d_%H%M%S")
      + f"_{int(time.time() * 1000000) % 1000000:06d}"
    )
    report_path = self.output_dir / f"health_report_{timestamp}.html"
    report_path.write_text(html_content, encoding="utf-8")

    logger.info(translator.t("log.report_html_saved", path=report_path))
    return report_path

  def generate_markdown_report(
    self,
    title: str | None = None,
    heart_rate_report: HeartRateAnalysisReport | None = None,
    sleep_report: SleepAnalysisReport | None = None,
    highlights: HealthHighlights | None = None,
    locale: str | None = None,
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
    translator = Translator(resolve_locale(locale))
    logger.info(translator.t("log.report_md_generating"))
    content_timestamp = datetime.now()
    report_title = title or translator.t("report.title.health_analysis")

    md_content = self._markdown_header(report_title, content_timestamp, translator)
    md_content += self._markdown_executive_summary(
      heart_rate_report, sleep_report, translator
    )
    md_content += self._markdown_highlights(highlights, translator)
    md_content += self._markdown_heart_rate_section(heart_rate_report, translator)
    md_content += self._markdown_sleep_section(sleep_report, translator)
    md_content += self._markdown_recommendations(highlights, translator)

    # Save report.
    report_path = (
      self.output_dir / f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    report_path.write_text(md_content, encoding="utf-8")

    logger.info(translator.t("log.report_md_saved", path=report_path))
    return report_path

  def _markdown_header(
    self, title: str, timestamp: datetime, translator: Translator
  ) -> str:
    """Build the Markdown report header."""
    content = f"# {title}\n\n"
    content += (
      f"**{translator.t('report.generated_at')}**: "
      f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    content += "---\n\n"
    return content

  def _markdown_executive_summary(
    self,
    heart_rate_report: HeartRateAnalysisReport | None,
    sleep_report: SleepAnalysisReport | None,
    translator: Translator,
  ) -> str:
    """Build executive summary section."""
    content = f"## {translator.t('report.section.executive_summary')}\n\n"
    if heart_rate_report:
      content += (
        f"- **{translator.t('report.label.heart_rate_records')}**: "
        f"{heart_rate_report.record_count}\n"
      )
      content += (
        f"- **{translator.t('report.label.data_quality_score')}**: "
        f"{heart_rate_report.data_quality_score:.1%}\n"
      )
    if sleep_report:
      content += (
        f"- **{translator.t('report.label.sleep_records')}**: "
        f"{sleep_report.record_count}\n"
      )
      content += (
        f"- **{translator.t('report.label.data_quality_score')}**: "
        f"{sleep_report.data_quality_score:.1%}\n"
      )
    content += "\n"
    return content

  def _markdown_highlights(
    self, highlights: HealthHighlights | None, translator: Translator
  ) -> str:
    """Build highlights section."""
    if not highlights:
      return ""

    content = f"## {translator.t('report.section.key_findings')}\n\n"
    for i, insight in enumerate(highlights.insights[:5], 1):
      priority_emoji = {
        "high": "🔴",
        "medium": "🟡",
        "low": "🟢",
      }
      emoji = priority_emoji.get(insight.priority, "⚪")
      content += f"{i}. {emoji} **{insight.title}**\n"
      content += f"   - {insight.message}\n\n"
    return content

  def _markdown_heart_rate_section(
    self, heart_rate_report: HeartRateAnalysisReport | None, translator: Translator
  ) -> str:
    """Build heart rate section."""
    if not heart_rate_report:
      return ""

    content = f"## {translator.t('report.section.heart_rate')}\n\n"
    content += f"### {translator.t('report.section.data_overview')}\n\n"
    content += (
      f"- {translator.t('report.metric.record_count')}: "
      f"{heart_rate_report.record_count}\n"
    )
    content += (
      f"- {translator.t('report.metric.time_range')}: "
      f"{heart_rate_report.data_range[0]} "
      f"{translator.t('report.label.range_to')} "
      f"{heart_rate_report.data_range[1]}\n"
    )
    content += (
      f"- {translator.t('report.label.data_quality_score')}: "
      f"{heart_rate_report.data_quality_score:.1%}\n\n"
    )

    if heart_rate_report.resting_hr_analysis:
      rhr = heart_rate_report.resting_hr_analysis
      content += f"### {translator.t('report.section.resting_hr')}\n\n"
      content += (
        f"- {translator.t('report.metric.current_value')}: "
        f"{rhr.current_value:.0f} bpm\n"
      )
      content += (
        f"- {translator.t('report.metric.baseline_value')}: "
        f"{rhr.baseline_value:.0f} bpm\n"
      )
      content += (
        f"- {translator.t('report.metric.change')}: "
        f"{rhr.change_from_baseline:+.1f} bpm\n"
      )
      content += f"- {translator.t('report.metric.trend')}: {rhr.trend_direction}\n"
      content += (
        f"- {translator.t('report.metric.health_rating')}: {rhr.health_rating}\n\n"
      )

    return content

  def _markdown_sleep_section(
    self, sleep_report: SleepAnalysisReport | None, translator: Translator
  ) -> str:
    """Build sleep section."""
    if not sleep_report:
      return ""

    content = f"## {translator.t('report.section.sleep')}\n\n"
    content += f"### {translator.t('report.section.data_overview')}\n\n"
    content += (
      f"- {translator.t('report.metric.record_count')}: {sleep_report.record_count}\n"
    )
    content += (
      f"- {translator.t('report.metric.time_range')}: "
      f"{sleep_report.data_range[0]} "
      f"{translator.t('report.label.range_to')} "
      f"{sleep_report.data_range[1]}\n"
    )
    content += (
      f"- {translator.t('report.label.data_quality_score')}: "
      f"{sleep_report.data_quality_score:.1%}\n\n"
    )

    if sleep_report.quality_metrics:
      quality = sleep_report.quality_metrics
      content += f"### {translator.t('report.section.sleep_quality_metrics')}\n\n"
      content += (
        f"- {translator.t('report.metric.avg_duration')}: "
        f"{quality.average_duration:.1f} "
        f"{translator.t('report.metric.avg_sleep_duration_unit')}\n"
      )
      content += (
        f"- {translator.t('report.metric.avg_efficiency')}: "
        f"{quality.average_efficiency:.1%}\n"
      )
      content += (
        f"- {translator.t('report.metric.consistency_score')}: "
        f"{quality.consistency_score:.1%}\n\n"
      )

    return content

  def _markdown_recommendations(
    self, highlights: HealthHighlights | None, translator: Translator
  ) -> str:
    """Build recommendations section."""
    if not highlights or not highlights.recommendations:
      return ""

    content = f"## {translator.t('report.section.recommendations')}\n\n"
    for i, rec in enumerate(highlights.recommendations, 1):
      content += f"{i}. {rec}\n"
    content += "\n"
    return content

  def generate_comprehensive_report(
    self,
    report: Any,  # ComprehensiveHealthReport
    title: str | None = None,
    include_charts: bool = True,
    locale: str | None = None,
  ) -> Path:
    """Generate comprehensive health analysis report.

    Args:
        report: Comprehensive health analysis report.
        title: Report title.
        include_charts: Whether to include charts.

    Returns:
        Report file path.
    """
    translator = Translator(resolve_locale(locale))
    logger.info(translator.t("log.report_comprehensive_generating"))
    report_title = title or translator.t("report.title.comprehensive")

    # Generate charts.
    charts = {}
    if include_charts:
      try:
        charts = self.chart_generator.generate_comprehensive_report_charts(
          report, self.output_dir / "charts"
        )
      except Exception as e:
        logger.warning(translator.t("log.report_charts_failed", error=str(e)))
        charts = {}

    # Create HTML content.
    html_content = self._create_comprehensive_html_structure(
      report_title, report, charts, translator
    )

    # Save report.
    report_path = (
      self.output_dir
      / f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    )
    report_path.write_text(html_content, encoding="utf-8")

    logger.info(translator.t("log.report_comprehensive_saved", path=report_path))
    return report_path

  def _create_comprehensive_html_structure(
    self,
    title: str,
    report: Any,
    charts: dict[str, Path],
    translator: Translator,
  ) -> str:
    """Create comprehensive report HTML structure."""
    lang_code = "zh-CN" if translator.locale == "zh" else "en"
    html_content = f"""<!DOCTYPE html>
<html lang="{lang_code}">
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
            <p class="subtitle">{translator.t("report.generated_at")}: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
"""

    # Executive summary.
    html_content += self._create_comprehensive_summary(report, translator)

    # Health dashboard.
    if "dashboard" in charts:
      html_content += f"""
        <div class="section">
            <h2>📊 {translator.t("report.chart.dashboard")}</h2>
            <div class="chart-container">
                <iframe src="{charts["dashboard"].relative_to(self.output_dir)}"
                        width="100%" height="600" frameborder="0"></iframe>
            </div>
        </div>
      """

    # Detailed analysis sections.
    html_content += self._create_detailed_analysis_sections(report, charts, translator)

    # Recommendations section.
    html_content += self._create_recommendations_section(report, translator)

    # Footer.
    html_content += """
        <div class="footer">
            <p>{translator.t("report.footer.autogen_with_source")}</p>
        </div>
    </div>
</body>
</html>
"""

    return html_content

  def _create_comprehensive_summary(self, report: Any, translator: Translator) -> str:
    """Create comprehensive summary."""
    content = '<div class="section">\n'
    content += f"<h2>📈 {translator.t('report.section.executive_summary')}</h2>\n"
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
      content += (
        f'<div class="metric-title">'
        f"{translator.t('report.metric.overall_wellness')}</div>\n"
      )
      content += (
        f'<div class="metric-value">{report.overall_wellness_score:.1%}</div>\n'
      )
      content += "</div>\n"

    # Data range.
    if hasattr(report, "data_range"):
      content += '<div class="metric-card info">\n'
      content += (
        f'<div class="metric-title">{translator.t("report.metric.data_range")}</div>\n'
      )
      content += (
        f'<div class="metric-value" style="font-size:1.2em">'
        f"{report.data_range[0].strftime('%Y-%m-%d')}<br>"
        f"{translator.t('report.label.range_to')}<br>"
        f"{report.data_range[1].strftime('%Y-%m-%d')}</div>\n"
      )
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
      content += (
        f'<div class="metric-title">'
        f"{translator.t('report.metric.data_completeness')}</div>\n"
      )
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
      content += (
        f'<div class="metric-title">'
        f"{translator.t('report.metric.analysis_confidence')}</div>\n"
      )
      content += f'<div class="metric-value">{report.analysis_confidence:.1%}</div>\n'
      content += "</div>\n"

    content += "</div>\n"
    content += "</div>\n"

    return content

  def _create_detailed_analysis_sections(
    self, report: Any, charts: dict[str, Path], translator: Translator
  ) -> str:
    """Create detailed analysis sections."""
    content = ""

    # Sleep analysis.
    if (
      hasattr(report, "sleep_quality")
      and getattr(report, "sleep_quality", None) is not None
    ):
      content += '<div class="section">\n'
      content += f"<h2>😴 {translator.t('report.section.sleep_quality')}</h2>\n"
      content += '<div class="dashboard-grid">\n'

      sleep = report.sleep_quality
      content += '<div class="metric-card success">\n'
      content += (
        f'<div class="metric-title">'
        f"{translator.t('report.metric.avg_sleep_duration')}</div>\n"
      )
      content += (
        f'<div class="metric-value">{sleep.average_duration_hours:.1f}'
        f'<span class="metric-unit"> '
        f"{translator.t('report.metric.avg_sleep_duration_unit')}"
        f"</span></div>\n"
      )
      content += "</div>\n"

      content += '<div class="metric-card info">\n'
      content += (
        f'<div class="metric-title">'
        f"{translator.t('report.metric.sleep_efficiency')}</div>\n"
      )
      content += f'<div class="metric-value">{sleep.average_efficiency_percent:.1f}<span class="metric-unit">%</span></div>\n'
      content += "</div>\n"

      content += '<div class="metric-card warning">\n'
      content += (
        f'<div class="metric-title">{translator.t("report.metric.sleep_debt")}</div>\n'
      )
      content += (
        f'<div class="metric-value">{sleep.sleep_debt_hours:.1f}'
        f'<span class="metric-unit"> '
        f"{translator.t('report.metric.sleep_debt_unit')}"
        f"</span></div>\n"
      )
      content += "</div>\n"

      content += '<div class="metric-card secondary">\n'
      content += (
        f'<div class="metric-title">'
        f"{translator.t('report.metric.consistency_score')}</div>\n"
      )
      content += f'<div class="metric-value">{sleep.consistency_score:.1%}</div>\n'
      content += "</div>\n"

      content += "</div>\n"
      content += "</div>\n"

    # Activity pattern analysis.
    if getattr(report, "activity_patterns", None) is not None:
      content += '<div class="section">\n'
      content += f"<h2>🏃 {translator.t('report.section.activity_patterns')}</h2>\n"
      content += '<div class="dashboard-grid">\n'

      activity = report.activity_patterns
      content += '<div class="metric-card success">\n'
      content += (
        f'<div class="metric-title">'
        f"{translator.t('report.metric.daily_step_average')}</div>\n"
      )
      content += (
        f'<div class="metric-value">{activity.daily_step_average:,}'
        f'<span class="metric-unit"> '
        f"{translator.t('report.metric.daily_steps_unit')}"
        f"</span></div>\n"
      )
      content += "</div>\n"

      content += '<div class="metric-card info">\n'
      content += (
        f'<div class="metric-title">'
        f"{translator.t('report.metric.weekly_exercise_frequency')}</div>\n"
      )
      content += (
        f'<div class="metric-value">{activity.weekly_exercise_frequency:.1f}'
        f'<span class="metric-unit"> '
        f"{translator.t('report.metric.weekly_exercise_unit')}"
        f"</span></div>\n"
      )
      content += "</div>\n"

      content += '<div class="metric-card warning">\n'
      content += (
        f'<div class="metric-title">'
        f"{translator.t('report.metric.sedentary_hours_daily')}</div>\n"
      )
      content += (
        f'<div class="metric-value">{activity.sedentary_hours_daily:.1f}'
        f'<span class="metric-unit"> '
        f"{translator.t('report.metric.daily_sedentary_unit')}"
        f"</span></div>\n"
      )
      content += "</div>\n"

      content += '<div class="metric-card secondary">\n'
      content += (
        f'<div class="metric-title">'
        f"{translator.t('report.metric.activity_consistency_score')}</div>\n"
      )
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
            <h2>🔗 {translator.t("report.section.correlation")}</h2>
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
            <h2>⚠️ {translator.t("report.section.risk_assessment")}</h2>
            <div class="chart-container">
                <iframe src="{charts["risk_assessment"].relative_to(self.output_dir)}"
                        width="100%" height="500" frameborder="0"></iframe>
            </div>
        </div>
      """

    return content

  def _create_recommendations_section(self, report: Any, translator: Translator) -> str:
    """Create recommendations section."""
    content = '<div class="section">\n'
    content += (
      f"<h2>💡 {translator.t('report.section.personal_recommendations')}</h2>\n"
    )

    # Priority actions.
    if hasattr(report, "priority_actions") and report.priority_actions:
      content += f"<h3>{translator.t('report.section.priority_actions')}</h3>\n"
      content += '<div class="recommendations">\n'
      content += "<ol>\n"
      for action in report.priority_actions:
        content += f"<li>{action}</li>\n"
      content += "</ol>\n"
      content += "</div>\n"

    # Lifestyle optimization.
    if hasattr(report, "lifestyle_optimization") and report.lifestyle_optimization:
      content += f"<h3>{translator.t('report.section.lifestyle_optimization')}</h3>\n"
      content += '<div class="recommendations">\n'
      content += "<ol>\n"
      for optimization in report.lifestyle_optimization:
        content += f"<li>{optimization}</li>\n"
      content += "</ol>\n"
      content += "</div>\n"

    # Predictive insights.
    if hasattr(report, "predictive_insights") and report.predictive_insights:
      content += f"<h3>{translator.t('report.section.predictive_insights')}</h3>\n"
      content += '<div class="insights-grid">\n'

      risk_keyword = translator.t("report.keyword.risk").lower()
      recommendation_keyword = translator.t("report.keyword.recommendation").lower()
      insight_keyword = translator.t("report.keyword.insight").lower()

      for insight in report.predictive_insights:
        priority_class = "low"  # Default to low priority.
        insight_text = str(insight)
        insight_text_lower = insight_text.lower()
        if "⚠️" in insight_text or risk_keyword in insight_text_lower:
          priority_class = "high"
        elif "📊" in insight_text or recommendation_keyword in insight_text_lower:
          priority_class = "medium"
        elif insight_keyword in insight_text_lower:
          priority_class = "medium"

        content += f'<div class="insight-card {priority_class}">\n'
        content += f'<div class="insight-message">{insight_text}</div>\n'
        content += "</div>\n"

      content += "</div>\n"

    content += "</div>\n"
    return content

  def _create_html_structure(self, title: str, translator: Translator) -> str:
    """Create base HTML structure."""
    lang_code = "zh-CN" if translator.locale == "zh" else "en"
    return f"""<!DOCTYPE html>
<html lang="{lang_code}">
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
            <p class="subtitle">{translator.t("report.generated_at")}: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </header>
"""

  def _create_executive_summary(
    self,
    heart_rate_report: HeartRateAnalysisReport | None,
    sleep_report: SleepAnalysisReport | None,
    highlights: HealthHighlights | None,
    translator: Translator,
  ) -> str:
    """Create executive summary section."""
    content = '<div class="section">\n'
    content += f"<h2>📊 {translator.t('report.section.executive_summary')}</h2>\n"
    content += '<div class="metric-grid">\n'

    # Heart rate overview.
    if heart_rate_report:
      content += '<div class="metric-card">\n'
      content += (
        f'<div class="metric-label">'
        f"{translator.t('report.label.heart_rate_records')}</div>\n"
      )
      content += f'<div class="metric-value">{heart_rate_report.record_count:,}</div>\n'
      content += "</div>\n"

      content += '<div class="metric-card">\n'
      content += (
        f'<div class="metric-label">'
        f"{translator.t('report.label.data_quality_score')}</div>\n"
      )
      content += (
        f'<div class="metric-value">{heart_rate_report.data_quality_score:.0%}</div>\n'
      )
      content += "</div>\n"

    # Sleep data overview.
    if sleep_report:
      content += '<div class="metric-card">\n'
      content += (
        f'<div class="metric-label">'
        f"{translator.t('report.label.sleep_records')}</div>\n"
      )
      content += f'<div class="metric-value">{sleep_report.record_count}</div>\n'
      content += "</div>\n"

      content += '<div class="metric-card">\n'
      content += (
        f'<div class="metric-label">'
        f"{translator.t('report.label.data_quality_score')}</div>\n"
      )
      content += (
        f'<div class="metric-value">{sleep_report.data_quality_score:.0%}</div>\n'
      )
      content += "</div>\n"

    # Highlights summary.
    if highlights:
      high_count = sum(1 for i in highlights.insights if i.priority == "high")
      content += f'<div class="metric-card {"danger" if high_count > 0 else ""}">\n'
      content += (
        f'<div class="metric-label">'
        f"{translator.t('report.label.high_priority_insights')}</div>\n"
      )
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
    translator: Translator | None = None,
  ) -> str:
    """Create heart rate analysis section."""
    content = '<div class="section">\n'
    translator = translator or Translator()
    content += f"<h2>❤️ {translator.t('report.section.heart_rate')}</h2>\n"

    # Data range.
    content += f"<h3>{translator.t('report.section.data_overview')}</h3>\n"
    content += (
      f"<p>{translator.t('report.metric.time_range')}: "
      f"{report.data_range[0]} {translator.t('report.label.range_to')} "
      f"{report.data_range[1]}</p>\n"
    )
    content += (
      f"<p>{translator.t('report.metric.record_count')}: {report.record_count:,}</p>\n"
    )
    content += (
      f"<p>{translator.t('report.label.data_quality_score')}: "
      f"{report.data_quality_score:.1%}</p>\n"
    )

    # Resting heart rate.
    if report.resting_hr_analysis:
      rhr = report.resting_hr_analysis
      content += f"<h3>{translator.t('report.section.resting_hr')}</h3>\n"
      content += '<div class="metric-grid">\n'

      content += '<div class="metric-card">\n'
      content += (
        f'<div class="metric-label">'
        f"{translator.t('report.metric.current_value')}</div>\n"
      )
      content += f'<div class="metric-value">{rhr.current_value:.0f} bpm</div>\n'
      content += "</div>\n"

      content += '<div class="metric-card">\n'
      content += (
        f'<div class="metric-label">'
        f"{translator.t('report.metric.baseline_value')}</div>\n"
      )
      content += f'<div class="metric-value">{rhr.baseline_value:.0f} bpm</div>\n'
      content += "</div>\n"

      change_class = "danger" if rhr.change_from_baseline > 2 else ""
      content += f'<div class="metric-card {change_class}">\n'
      content += (
        f'<div class="metric-label">{translator.t("report.metric.change")}</div>\n'
      )
      content += (
        f'<div class="metric-value">{rhr.change_from_baseline:+.1f} bpm</div>\n'
      )
      content += "</div>\n"

      content += '<div class="metric-card">\n'
      content += (
        f'<div class="metric-label">'
        f"{translator.t('report.metric.health_rating')}</div>\n"
      )
      content += f'<div class="metric-value" style="font-size:1.5em">{rhr.health_rating.upper()}</div>\n'
      content += "</div>\n"

      content += "</div>\n"

    content += "</div>\n"
    return content

  def _create_sleep_section(
    self,
    report: SleepAnalysisReport,
    include_charts: bool,
    translator: Translator,
  ) -> str:
    """Create sleep analysis section."""
    content = '<div class="section">\n'
    content += f"<h2>😴 {translator.t('report.section.sleep')}</h2>\n"

    # Data range.
    content += f"<h3>{translator.t('report.section.data_overview')}</h3>\n"
    content += (
      f"<p>{translator.t('report.metric.time_range')}: "
      f"{report.data_range[0]} {translator.t('report.label.range_to')} "
      f"{report.data_range[1]}</p>\n"
    )
    content += (
      f"<p>{translator.t('report.metric.record_count')}: {report.record_count}</p>\n"
    )
    content += (
      f"<p>{translator.t('report.label.data_quality_score')}: "
      f"{report.data_quality_score:.1%}</p>\n"
    )

    # Sleep quality metrics.
    if report.quality_metrics:
      quality = report.quality_metrics
      content += f"<h3>{translator.t('report.section.sleep_quality_metrics')}</h3>\n"
      content += '<div class="metric-grid">\n'

      duration_class = "danger" if quality.average_duration < 7 else ""
      content += f'<div class="metric-card {duration_class}">\n'
      content += (
        f'<div class="metric-label">'
        f"{translator.t('report.metric.avg_sleep_duration')}</div>\n"
      )
      content += (
        f'<div class="metric-value">{quality.average_duration:.1f} '
        f"{translator.t('report.metric.avg_sleep_duration_unit')}</div>\n"
      )
      content += "</div>\n"

      efficiency_class = "warning" if quality.average_efficiency < 0.85 else ""
      content += f'<div class="metric-card {efficiency_class}">\n'
      content += (
        f'<div class="metric-label">'
        f"{translator.t('report.metric.sleep_efficiency')}</div>\n"
      )
      content += f'<div class="metric-value">{quality.average_efficiency:.0%}</div>\n'
      content += "</div>\n"

      consistency_class = "warning" if quality.consistency_score < 0.7 else ""
      content += f'<div class="metric-card {consistency_class}">\n'
      content += (
        f'<div class="metric-label">'
        f"{translator.t('report.metric.consistency_score')}</div>\n"
      )
      content += f'<div class="metric-value">{quality.consistency_score:.0%}</div>\n'
      content += "</div>\n"

      content += "</div>\n"

    content += "</div>\n"
    return content

  def _create_highlights_section(
    self, highlights: HealthHighlights, translator: Translator
  ) -> str:
    """Create highlights section."""
    content = '<div class="section">\n'
    content += f"<h2>💡 {translator.t('report.section.key_findings')}</h2>\n"

    # Insight list.
    if highlights.insights:
      content += f"<h3>{translator.t('report.section.insights')}</h3>\n"
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
      content += f"<h3>{translator.t('report.section.recommendations')}</h3>\n"
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
    translator: Translator,
  ) -> str:
    """Create data quality section."""
    content = '<div class="section">\n'
    content += f"<h2>📋 {translator.t('report.section.data_quality_info')}</h2>\n"

    if heart_rate_report:
      content += f"<h3>{translator.t('report.section.heart_rate')}</h3>\n"
      content += "<ul>\n"
      content += (
        f"<li>{translator.t('report.metric.record_count')}: "
        f"{heart_rate_report.record_count:,}</li>\n"
      )
      content += (
        f"<li>{translator.t('report.label.data_quality_score')}: "
        f"{heart_rate_report.data_quality_score:.1%}</li>\n"
      )
      content += (
        f"<li>{translator.t('report.metric.time_range')}: "
        f"{heart_rate_report.data_range[0]} "
        f"{translator.t('report.label.range_to')} "
        f"{heart_rate_report.data_range[1]}</li>\n"
      )
      content += "</ul>\n"

    if sleep_report:
      content += f"<h3>{translator.t('report.section.sleep')}</h3>\n"
      content += "<ul>\n"
      content += (
        f"<li>{translator.t('report.metric.record_count')}: "
        f"{sleep_report.record_count}</li>\n"
      )
      content += (
        f"<li>{translator.t('report.label.data_quality_score')}: "
        f"{sleep_report.data_quality_score:.1%}</li>\n"
      )
      content += (
        f"<li>{translator.t('report.metric.time_range')}: "
        f"{sleep_report.data_range[0]} "
        f"{translator.t('report.label.range_to')} "
        f"{sleep_report.data_range[1]}</li>\n"
      )
      content += "</ul>\n"

    content += "</div>\n"
    return content

  def _close_html_structure(self, translator: Translator) -> str:
    """Close HTML structure."""
    return """
        <footer>
            <p>{translator.t("report.footer.generated_by")}</p>
            <p>{translator.t("report.footer.data_source")}</p>
        </footer>
    </div>
</body>
</html>
"""
