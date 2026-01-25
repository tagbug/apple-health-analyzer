"""报告生成模块 - 生成完整的健康分析报告"""

from datetime import datetime
from pathlib import Path

from ..analyzers.highlights import HealthHighlights
from ..processors.heart_rate import HeartRateAnalysisReport
from ..processors.sleep import SleepAnalysisReport
from ..utils.logger import get_logger
from .charts import ChartGenerator

logger = get_logger(__name__)


class ReportGenerator:
  """健康报告生成器

  生成包含图表、统计分析和健康洞察的完整报告。
  支持HTML、Markdown等多种格式。
  """

  def __init__(self, output_dir: Path | None = None):
    """初始化报告生成器

    Args:
        output_dir: 报告输出目录
    """
    self.output_dir = Path(output_dir) if output_dir else Path("./reports")
    self.output_dir.mkdir(parents=True, exist_ok=True)

    # 创建图表生成器
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
    """生成HTML格式报告

    Args:
        title: 报告标题
        heart_rate_report: 心率分析报告
        sleep_report: 睡眠分析报告
        highlights: 健康洞察
        include_charts: 是否包含图表

    Returns:
        报告文件路径
    """
    logger.info("Generating HTML report")

    # 创建报告HTML内容
    html_content = self._create_html_structure(title)

    # 添加执行摘要
    html_content += self._create_executive_summary(
      heart_rate_report, sleep_report, highlights
    )

    # 添加心率分析章节
    if heart_rate_report:
      html_content += self._create_heart_rate_section(
        heart_rate_report, include_charts, heart_rate_data
      )

    # 添加睡眠分析章节
    if sleep_report:
      html_content += self._create_sleep_section(sleep_report, include_charts)

    # 添加Highlights章节
    if highlights:
      html_content += self._create_highlights_section(highlights)

    # 添加数据质量信息
    html_content += self._create_data_quality_section(
      heart_rate_report, sleep_report
    )

    # 关闭HTML
    html_content += self._close_html_structure()

    # 保存报告
    report_path = (
      self.output_dir
      / f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    )
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
    """生成Markdown格式报告

    Args:
        title: 报告标题
        heart_rate_report: 心率分析报告
        sleep_report: 睡眠分析报告
        highlights: 健康洞察

    Returns:
        报告文件路径
    """
    logger.info("Generating Markdown report")

    md_content = f"# {title}\n\n"
    md_content += (
      f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    md_content += "---\n\n"

    # 执行摘要
    md_content += "## 执行摘要\n\n"
    if heart_rate_report:
      md_content += f"- **心率记录数**: {heart_rate_report.record_count}\n"
      md_content += (
        f"- **数据质量**: {heart_rate_report.data_quality_score:.1%}\n"
      )
    if sleep_report:
      md_content += f"- **睡眠记录数**: {sleep_report.record_count}\n"
      md_content += f"- **数据质量**: {sleep_report.data_quality_score:.1%}\n"
    md_content += "\n"

    # Highlights
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

    # 心率分析
    if heart_rate_report:
      md_content += "## 心率分析\n\n"
      md_content += "### 数据概览\n\n"
      md_content += f"- 记录总数: {heart_rate_report.record_count}\n"
      md_content += f"- 时间范围: {heart_rate_report.data_range[0]} 至 {heart_rate_report.data_range[1]}\n"
      md_content += (
        f"- 数据质量评分: {heart_rate_report.data_quality_score:.1%}\n\n"
      )

      if heart_rate_report.resting_hr_analysis:
        rhr = heart_rate_report.resting_hr_analysis
        md_content += "### 静息心率\n\n"
        md_content += f"- 当前值: {rhr.current_value:.0f} bpm\n"
        md_content += f"- 基线值: {rhr.baseline_value:.0f} bpm\n"
        md_content += f"- 变化: {rhr.change_from_baseline:+.1f} bpm\n"
        md_content += f"- 趋势: {rhr.trend_direction}\n"
        md_content += f"- 健康评级: {rhr.health_rating}\n\n"

    # 睡眠分析
    if sleep_report:
      md_content += "## 睡眠分析\n\n"
      md_content += "### 数据概览\n\n"
      md_content += f"- 记录总数: {sleep_report.record_count}\n"
      md_content += f"- 时间范围: {sleep_report.data_range[0]} 至 {sleep_report.data_range[1]}\n"
      md_content += f"- 数据质量评分: {sleep_report.data_quality_score:.1%}\n\n"

      if sleep_report.quality_metrics:
        quality = sleep_report.quality_metrics
        md_content += "### 睡眠质量指标\n\n"
        md_content += f"- 平均时长: {quality.average_duration:.1f} 小时\n"
        md_content += f"- 平均效率: {quality.average_efficiency:.1%}\n"
        md_content += f"- 规律性评分: {quality.consistency_score:.1%}\n\n"

    # 建议
    if highlights and highlights.recommendations:
      md_content += "## 健康建议\n\n"
      for i, rec in enumerate(highlights.recommendations, 1):
        md_content += f"{i}. {rec}\n"
      md_content += "\n"

    # 保存报告
    report_path = (
      self.output_dir
      / f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    report_path.write_text(md_content, encoding="utf-8")

    logger.info(f"Markdown report saved to {report_path}")
    return report_path

  def _create_html_structure(self, title: str) -> str:
    """创建HTML基础结构"""
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
    """创建执行摘要章节"""
    content = '<div class="section">\n'
    content += "<h2>📊 执行摘要</h2>\n"
    content += '<div class="metric-grid">\n'

    # 心率数据概览
    if heart_rate_report:
      content += '<div class="metric-card">\n'
      content += '<div class="metric-label">心率记录数</div>\n'
      content += (
        f'<div class="metric-value">{heart_rate_report.record_count:,}</div>\n'
      )
      content += "</div>\n"

      content += '<div class="metric-card">\n'
      content += '<div class="metric-label">心率数据质量</div>\n'
      content += f'<div class="metric-value">{heart_rate_report.data_quality_score:.0%}</div>\n'
      content += "</div>\n"

    # 睡眠数据概览
    if sleep_report:
      content += '<div class="metric-card">\n'
      content += '<div class="metric-label">睡眠记录数</div>\n'
      content += (
        f'<div class="metric-value">{sleep_report.record_count}</div>\n'
      )
      content += "</div>\n"

      content += '<div class="metric-card">\n'
      content += '<div class="metric-label">睡眠数据质量</div>\n'
      content += f'<div class="metric-value">{sleep_report.data_quality_score:.0%}</div>\n'
      content += "</div>\n"

    # Highlights统计
    if highlights:
      high_count = sum(1 for i in highlights.insights if i.priority == "high")
      content += (
        f'<div class="metric-card {"danger" if high_count > 0 else ""}">\n'
      )
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
    """创建心率分析章节"""
    content = '<div class="section">\n'
    content += "<h2>❤️ 心率分析</h2>\n"

    # 数据范围
    content += "<h3>数据概览</h3>\n"
    content += (
      f"<p>时间范围: {report.data_range[0]} 至 {report.data_range[1]}</p>\n"
    )
    content += f"<p>记录总数: {report.record_count:,}</p>\n"
    content += f"<p>数据质量评分: {report.data_quality_score:.1%}</p>\n"

    # 静息心率
    if report.resting_hr_analysis:
      rhr = report.resting_hr_analysis
      content += "<h3>静息心率分析</h3>\n"
      content += '<div class="metric-grid">\n'

      content += '<div class="metric-card">\n'
      content += '<div class="metric-label">当前值</div>\n'
      content += (
        f'<div class="metric-value">{rhr.current_value:.0f} bpm</div>\n'
      )
      content += "</div>\n"

      content += '<div class="metric-card">\n'
      content += '<div class="metric-label">基线值</div>\n'
      content += (
        f'<div class="metric-value">{rhr.baseline_value:.0f} bpm</div>\n'
      )
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
    """创建睡眠分析章节"""
    content = '<div class="section">\n'
    content += "<h2>😴 睡眠分析</h2>\n"

    # 数据范围
    content += "<h3>数据概览</h3>\n"
    content += (
      f"<p>时间范围: {report.data_range[0]} 至 {report.data_range[1]}</p>\n"
    )
    content += f"<p>记录总数: {report.record_count}</p>\n"
    content += f"<p>数据质量评分: {report.data_quality_score:.1%}</p>\n"

    # 睡眠质量指标
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
      content += (
        f'<div class="metric-value">{quality.average_efficiency:.0%}</div>\n'
      )
      content += "</div>\n"

      consistency_class = "warning" if quality.consistency_score < 0.7 else ""
      content += f'<div class="metric-card {consistency_class}">\n'
      content += '<div class="metric-label">规律性评分</div>\n'
      content += (
        f'<div class="metric-value">{quality.consistency_score:.0%}</div>\n'
      )
      content += "</div>\n"

      content += "</div>\n"

    content += "</div>\n"
    return content

  def _create_highlights_section(self, highlights: HealthHighlights) -> str:
    """创建Highlights章节"""
    content = '<div class="section">\n'
    content += "<h2>💡 关键发现与建议</h2>\n"

    # 洞察列表
    if highlights.insights:
      content += "<h3>健康洞察</h3>\n"
      content += '<ul class="insight-list">\n'

      for insight in highlights.insights[:8]:  # 显示前8条
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

    # 建议
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
    """创建数据质量信息章节"""
    content = '<div class="section">\n'
    content += "<h2>📋 数据质量信息</h2>\n"

    if heart_rate_report:
      content += "<h3>心率数据</h3>\n"
      content += "<ul>\n"
      content += f"<li>记录总数: {heart_rate_report.record_count:,}</li>\n"
      content += (
        f"<li>数据质量评分: {heart_rate_report.data_quality_score:.1%}</li>\n"
      )
      content += f"<li>时间范围: {heart_rate_report.data_range[0]} 至 {heart_rate_report.data_range[1]}</li>\n"
      content += "</ul>\n"

    if sleep_report:
      content += "<h3>睡眠数据</h3>\n"
      content += "<ul>\n"
      content += f"<li>记录总数: {sleep_report.record_count}</li>\n"
      content += (
        f"<li>数据质量评分: {sleep_report.data_quality_score:.1%}</li>\n"
      )
      content += f"<li>时间范围: {sleep_report.data_range[0]} 至 {sleep_report.data_range[1]}</li>\n"
      content += "</ul>\n"

    content += "</div>\n"
    return content

  def _close_html_structure(self) -> str:
    """关闭HTML结构"""
    return """
        <footer>
            <p>本报告由 Apple Health Analyzer 自动生成</p>
            <p>数据来源: Apple Health 导出数据</p>
        </footer>
    </div>
</body>
</html>
"""
