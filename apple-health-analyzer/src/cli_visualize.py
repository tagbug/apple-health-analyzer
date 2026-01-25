"""可视化和报告生成CLI命令"""

import sys
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console

from src.analyzers.highlights import HighlightsGenerator
from src.config import get_config
from src.core.xml_parser import StreamingXMLParser
from src.processors.heart_rate import HeartRateAnalyzer
from src.processors.sleep import SleepAnalyzer
from src.utils.logger import get_logger
from src.visualization.charts import ChartGenerator
from src.visualization.reports import ReportGenerator

console = Console()
logger = get_logger(__name__)


@click.command()
@click.argument("xml_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="输出目录")
@click.option(
  "--format",
  "-f",
  type=click.Choice(["html", "markdown", "both"]),
  default="html",
  help="报告格式",
)
@click.option("--age", type=int, help="用户年龄（用于心肺适能分析）")
@click.option(
  "--gender",
  type=click.Choice(["male", "female"]),
  help="性别（用于心肺适能分析）",
)
@click.option("--no-charts", is_flag=True, help="不包含图表（仅生成文本报告）")
def report(
  xml_path: str,
  output: str | None,
  format: str,
  age: int | None,
  gender: str | None,
  no_charts: bool,
):
  """生成完整的健康分析报告（含图表和洞察）

  结合数据分析和可视化，生成包含完整分析结果、图表和健康洞察的报告。

  示例:
      # 生成HTML报告
      health-analyzer report export.xml --age 30 --gender male

      # 生成Markdown报告
      health-analyzer report export.xml --format markdown

      # 生成HTML和Markdown两种格式
      health-analyzer report export.xml --format both --age 30 --gender male
  """
  try:
    xml_file = Path(xml_path)
    output_dir = Path(output) if output else get_config().output_dir / "reports"

    console.print(f"[bold blue]生成健康报告:[/bold blue] {xml_file}")
    console.print(f"[bold blue]输出目录:[/bold blue] {output_dir}")
    console.print(f"[bold blue]报告格式:[/bold blue] {format}")

    # Step 1: 解析数据
    console.print("\n[bold]Step 1/4: 解析健康数据...[/bold]")
    with console.status("[bold green]解析记录..."):
      parser = StreamingXMLParser(xml_file)

      # 解析心率数据
      hr_types = [
        "HKQuantityTypeIdentifierHeartRate",
        "HKQuantityTypeIdentifierRestingHeartRate",
        "HKQuantityTypeIdentifierHeartRateVariabilitySDNN",
        "HKQuantityTypeIdentifierVO2Max",
      ]
      all_hr_records = list(parser.parse_records(record_types=hr_types))

      # 分类心率记录
      hr_records = []
      resting_hr_records = []
      hrv_records = []
      vo2_max_records = []

      for record in all_hr_records:
        record_type = getattr(record, "type", "")
        if record_type == "HKQuantityTypeIdentifierHeartRate":
          hr_records.append(record)
        elif record_type == "HKQuantityTypeIdentifierRestingHeartRate":
          resting_hr_records.append(record)
        elif record_type == "HKQuantityTypeIdentifierHeartRateVariabilitySDNN":
          hrv_records.append(record)
        elif record_type == "HKQuantityTypeIdentifierVO2Max":
          vo2_max_records.append(record)

      # 解析睡眠数据
      parser2 = StreamingXMLParser(xml_file)
      sleep_records = list(
        parser2.parse_records(
          record_types=["HKCategoryTypeIdentifierSleepAnalysis"]
        )
      )

    console.print(
      f"[green]✓ 解析完成: {len(hr_records)} 心率记录, {len(sleep_records)} 睡眠记录[/green]"
    )

    # Step 2: 分析数据
    console.print("\n[bold]Step 2/4: 分析健康数据...[/bold]")

    # 心率分析
    heart_rate_analyzer = HeartRateAnalyzer(age=age, gender=gender)  # type: ignore
    with console.status("[bold green]分析心率数据..."):
      heart_rate_report = heart_rate_analyzer.analyze_comprehensive(
        heart_rate_records=hr_records,
        resting_hr_records=resting_hr_records,
        hrv_records=hrv_records,
        vo2_max_records=vo2_max_records,
      )
    console.print("[green]✓ 心率分析完成[/green]")

    # 睡眠分析
    sleep_analyzer = SleepAnalyzer()
    sleep_report = None
    if sleep_records:
      with console.status("[bold green]分析睡眠数据..."):
        sleep_report = sleep_analyzer.analyze_comprehensive(sleep_records)  # type: ignore
      console.print("[green]✓ 睡眠分析完成[/green]")

    # Step 3: 生成Highlights
    console.print("\n[bold]Step 3/4: 生成健康洞察...[/bold]")
    highlights_generator = HighlightsGenerator()
    with console.status("[bold green]生成洞察和建议..."):
      highlights = highlights_generator.generate_comprehensive_highlights(
        heart_rate_report=heart_rate_report,
        sleep_report=sleep_report,
      )
    console.print(
      f"[green]✓ 生成 {len(highlights.insights)} 条洞察和 {len(highlights.recommendations)} 条建议[/green]"
    )

    # Step 4: 生成报告
    console.print("\n[bold]Step 4/4: 生成报告文件...[/bold]")
    report_generator = ReportGenerator(output_dir)

    report_files = []

    if format in ["html", "both"]:
      with console.status("[bold green]生成HTML报告..."):
        html_file = report_generator.generate_html_report(
          title="健康分析报告",
          heart_rate_report=heart_rate_report,
          sleep_report=sleep_report,
          highlights=highlights,
          include_charts=not no_charts,
        )
        report_files.append(html_file)
      console.print(f"[green]✓ HTML报告: {html_file}[/green]")

    if format in ["markdown", "both"]:
      with console.status("[bold green]生成Markdown报告..."):
        md_file = report_generator.generate_markdown_report(
          title="健康分析报告",
          heart_rate_report=heart_rate_report,
          sleep_report=sleep_report,
          highlights=highlights,
        )
        report_files.append(md_file)
      console.print(f"[green]✓ Markdown报告: {md_file}[/green]")

    # 总结
    console.print("\n[bold green]✅ 报告生成成功！[/bold green]")
    console.print("\n[bold]生成的文件:[/bold]")
    for file_path in report_files:
      size_mb = file_path.stat().st_size / (1024 * 1024)
      console.print(f"  • {file_path.name} ({size_mb:.2f} MB)")

  except Exception as e:
    logger.error(f"报告生成失败: {e}")
    console.print(f"[bold red]错误:[/bold red] {e}")
    sys.exit(1)


@click.command()
@click.argument("xml_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="输出目录")
@click.option(
  "--charts",
  "-c",
  multiple=True,
  type=click.Choice(
    [
      "heart_rate_timeseries",
      "resting_hr_trend",
      "hrv_analysis",
      "heart_rate_heatmap",
      "heart_rate_distribution",
      "heart_rate_zones",
      "sleep_timeline",
      "sleep_quality_trend",
      "sleep_stages_distribution",
      "sleep_consistency",
      "weekday_vs_weekend_sleep",
      "all",
    ]
  ),
  help="要生成的图表类型（可多选，默认all）",
)
@click.option(
  "--interactive/--static",
  default=True,
  help="交互式图表（HTML）或静态图表（PNG）",
)
@click.option("--age", type=int, help="用户年龄（用于心率区间计算）")
def visualize(
  xml_path: str,
  output: str | None,
  charts: tuple,
  interactive: bool,
  age: int | None,
):
  """生成健康数据可视化图表

  支持生成多种心率和睡眠相关的可视化图表。

  示例:
      # 生成所有图表
      health-analyzer visualize export.xml

      # 生成特定图表
      health-analyzer visualize export.xml -c heart_rate_timeseries -c sleep_quality_trend

      # 生成静态PNG图表
      health-analyzer visualize export.xml --static
  """
  try:
    xml_file = Path(xml_path)
    output_dir = Path(output) if output else get_config().output_dir / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold blue]生成可视化图表:[/bold blue] {xml_file}")
    console.print(f"[bold blue]输出目录:[/bold blue] {output_dir}")

    # 确定要生成的图表
    if not charts or "all" in charts:
      selected_charts = [
        "heart_rate_timeseries",
        "resting_hr_trend",
        "hrv_analysis",
        "heart_rate_distribution",
        "sleep_quality_trend",
        "sleep_stages_distribution",
      ]
    else:
      selected_charts = list(charts)

    console.print(
      f"[bold blue]要生成的图表:[/bold blue] {len(selected_charts)} 个"
    )

    # 解析数据
    console.print("\n[bold]解析数据...[/bold]")
    parser = StreamingXMLParser(xml_file)

    # 确定需要哪些数据类型
    need_hr = any(
      c.startswith("heart_rate")
      or c.startswith("resting")
      or c.startswith("hrv")
      for c in selected_charts
    )
    need_sleep = any(
      c.startswith("sleep") or c.startswith("weekday") for c in selected_charts
    )

    hr_data = {}
    sleep_data = {}

    if need_hr:
      with console.status("[bold green]解析心率数据..."):
        hr_types = [
          "HKQuantityTypeIdentifierHeartRate",
          "HKQuantityTypeIdentifierRestingHeartRate",
          "HKQuantityTypeIdentifierHeartRateVariabilitySDNN",
        ]
        records = list(parser.parse_records(record_types=hr_types))

        # 分类
        for record in records:
          record_type = getattr(record, "type", "")
          if record_type == "HKQuantityTypeIdentifierHeartRate":
            hr_data.setdefault("heart_rate", []).append(record)
          elif record_type == "HKQuantityTypeIdentifierRestingHeartRate":
            hr_data.setdefault("resting_hr", []).append(record)
          elif (
            record_type == "HKQuantityTypeIdentifierHeartRateVariabilitySDNN"
          ):
            hr_data.setdefault("hrv", []).append(record)

      console.print("[green]✓ 解析心率数据完成[/green]")

    if need_sleep:
      with console.status("[bold green]解析睡眠数据..."):
        parser2 = StreamingXMLParser(xml_file)
        sleep_records = list(
          parser2.parse_records(
            record_types=["HKCategoryTypeIdentifierSleepAnalysis"]
          )
        )

        # 解析睡眠会话
        sleep_analyzer = SleepAnalyzer()
        sleep_sessions = sleep_analyzer._parse_sleep_sessions(sleep_records)  # type: ignore

        sleep_data["sleep_records"] = sleep_records
        sleep_data["sleep_sessions"] = sleep_sessions
      console.print("[green]✓ 解析睡眠数据完成[/green]")

    # 生成图表
    console.print("\n[bold]生成图表...[/bold]")
    chart_generator = ChartGenerator()
    generated_files = []

    # 导入数据转换器
    from src.visualization.data_converter import DataConverter

    # 为每种图表类型生成文件
    for chart_type in selected_charts:
      try:
        console.print(f"[dim]生成 {chart_type}...[/dim]")

        if chart_type == "heart_rate_timeseries":
          if "heart_rate" in hr_data and hr_data["heart_rate"]:
            df = DataConverter.heart_rate_to_df(hr_data["heart_rate"])
            if not df.empty:
              df = DataConverter.sample_data_for_performance(
                df, 50000
              )  # 采样以提高性能
              fig = chart_generator.plot_heart_rate_timeseries(
                df,
                output_path=output_dir / f"{chart_type}.html"
                if interactive
                else output_dir / f"{chart_type}.png",
              )

              if fig:  # 如果返回了figure，说明是交互式模式且未保存
                if interactive:
                  file_path = output_dir / f"{chart_type}.html"
                  fig.write_html(file_path)
                  generated_files.append(file_path)
                else:
                  file_path = output_dir / f"{chart_type}.png"
                  fig.write_image(file_path, width=1200, height=600)
                  generated_files.append(file_path)
              else:  # 静态模式已保存
                file_path = output_dir / f"{chart_type}.png"
                if file_path.exists():
                  generated_files.append(file_path)
                else:
                  console.print(
                    f"[yellow]⚠ 静态图表文件未找到: {file_path}[/yellow]"
                  )

              console.print(f"[green]✓ 生成: {chart_type}[/green]")

        elif chart_type == "resting_hr_trend":
          if "resting_hr" in hr_data and hr_data["resting_hr"]:
            df = DataConverter.resting_hr_to_df(hr_data["resting_hr"])
            if not df.empty:
              fig = chart_generator.plot_resting_hr_trend(
                df,
                output_path=output_dir / f"{chart_type}.html"
                if interactive
                else output_dir / f"{chart_type}.png",
              )

              if fig:
                if interactive:
                  file_path = output_dir / f"{chart_type}.html"
                  fig.write_html(file_path)
                  generated_files.append(file_path)
                else:
                  file_path = output_dir / f"{chart_type}.png"
                  fig.write_image(file_path, width=1000, height=500)
                  generated_files.append(file_path)
              else:
                file_path = output_dir / f"{chart_type}.png"
                if file_path.exists():
                  generated_files.append(file_path)
                else:
                  console.print(
                    f"[yellow]⚠ 静态图表文件未找到: {file_path}[/yellow]"
                  )

              console.print(f"[green]✓ 生成: {chart_type}[/green]")

        elif chart_type == "hrv_analysis":
          if "hrv" in hr_data and hr_data["hrv"]:
            df = DataConverter.hrv_to_df(hr_data["hrv"])
            if not df.empty:
              fig = chart_generator.plot_hrv_analysis(
                df,
                output_path=output_dir / f"{chart_type}.html"
                if interactive
                else output_dir / f"{chart_type}.png",
              )

              if fig:
                if interactive:
                  file_path = output_dir / f"{chart_type}.html"
                  fig.write_html(file_path)
                  generated_files.append(file_path)
                else:
                  file_path = output_dir / f"{chart_type}.png"
                  fig.write_image(file_path, width=1000, height=500)
                  generated_files.append(file_path)
              else:
                file_path = output_dir / f"{chart_type}.png"
                if file_path.exists():
                  generated_files.append(file_path)
                else:
                  console.print(
                    f"[yellow]⚠ 静态图表文件未找到: {file_path}[/yellow]"
                  )

              console.print(f"[green]✓ 生成: {chart_type}[/green]")

        elif chart_type == "heart_rate_heatmap":
          if "heart_rate" in hr_data and hr_data["heart_rate"]:
            df = DataConverter.heart_rate_to_df(hr_data["heart_rate"])
            if not df.empty:
              df = DataConverter.sample_data_for_performance(df, 20000)
              # 准备日平均数据
              daily_df = DataConverter.aggregate_heart_rate_by_day(df)
              daily_df = daily_df.rename(columns={"mean_hr": "avg_hr"})
              fig = chart_generator.plot_heart_rate_heatmap(
                daily_df,
                output_path=output_dir / f"{chart_type}.html"
                if interactive
                else output_dir / f"{chart_type}.png",
              )

              if fig:
                if interactive:
                  file_path = output_dir / f"{chart_type}.html"
                  fig.write_html(file_path)
                  generated_files.append(file_path)
                else:
                  file_path = output_dir / f"{chart_type}.png"
                  fig.write_image(file_path, width=1200, height=800)
                  generated_files.append(file_path)
              else:
                file_path = output_dir / f"{chart_type}.png"
                if file_path.exists():
                  generated_files.append(file_path)
                else:
                  console.print(
                    f"[yellow]⚠ 静态图表文件未找到: {file_path}[/yellow]"
                  )

              console.print(f"[green]✓ 生成: {chart_type}[/green]")

        elif chart_type == "heart_rate_distribution":
          if "heart_rate" in hr_data and hr_data["heart_rate"]:
            df = DataConverter.heart_rate_to_df(hr_data["heart_rate"])
            if not df.empty:
              fig = chart_generator.plot_heart_rate_distribution(
                df,
                output_path=output_dir / f"{chart_type}.html"
                if interactive
                else output_dir / f"{chart_type}.png",
              )

              if fig:
                if interactive:
                  file_path = output_dir / f"{chart_type}.html"
                  fig.write_html(file_path)
                  generated_files.append(file_path)
                else:
                  file_path = output_dir / f"{chart_type}.png"
                  fig.write_image(file_path, width=1000, height=600)
                  generated_files.append(file_path)
              else:
                file_path = output_dir / f"{chart_type}.png"
                if file_path.exists():
                  generated_files.append(file_path)
                else:
                  console.print(
                    f"[yellow]⚠ 静态图表文件未找到: {file_path}[/yellow]"
                  )

              console.print(f"[green]✓ 生成: {chart_type}[/green]")

        elif chart_type == "heart_rate_zones":
          if "heart_rate" in hr_data and hr_data["heart_rate"]:
            df = DataConverter.heart_rate_to_df(hr_data["heart_rate"])
            if not df.empty:
              zones_df = DataConverter.prepare_heart_rate_zones(df, age)
              if not zones_df.empty:
                fig = chart_generator.plot_heart_rate_zones(
                  df,
                  age=age or 30,
                  output_path=output_dir / f"{chart_type}.html"
                  if interactive
                  else output_dir / f"{chart_type}.png",
                )

              if fig:
                if interactive:
                  file_path = output_dir / f"{chart_type}.html"
                  fig.write_html(file_path)
                  generated_files.append(file_path)
                else:
                  file_path = output_dir / f"{chart_type}.png"
                  fig.write_image(file_path, width=1000, height=600)
                  generated_files.append(file_path)
              else:
                file_path = output_dir / f"{chart_type}.png"
                if file_path.exists():
                  generated_files.append(file_path)
                else:
                  console.print(
                    f"[yellow]⚠ 静态图表文件未找到: {file_path}[/yellow]"
                  )

                console.print(f"[green]✓ 生成: {chart_type}[/green]")

        elif chart_type == "sleep_timeline":
          if "sleep_sessions" in sleep_data and sleep_data["sleep_sessions"]:
            df = DataConverter.sleep_sessions_to_df(
              sleep_data["sleep_sessions"]
            )
            if not df.empty:
              fig = chart_generator.plot_sleep_timeline(
                df,
                output_path=output_dir / f"{chart_type}.html"
                if interactive
                else output_dir / f"{chart_type}.png",
              )

              if fig:
                if interactive:
                  file_path = output_dir / f"{chart_type}.html"
                  fig.write_html(file_path)
                  generated_files.append(file_path)
                else:
                  file_path = output_dir / f"{chart_type}.png"
                  fig.write_image(file_path, width=1000, height=600)
                  generated_files.append(file_path)
              else:
                file_path = output_dir / f"{chart_type}.png"
                if file_path.exists():
                  generated_files.append(file_path)
                else:
                  console.print(
                    f"[yellow]⚠ 静态图表文件未找到: {file_path}[/yellow]"
                  )

              console.print(f"[green]✓ 生成: {chart_type}[/green]")

        elif chart_type == "sleep_quality_trend":
          if "sleep_sessions" in sleep_data and sleep_data["sleep_sessions"]:
            df = DataConverter.sleep_sessions_to_df(
              sleep_data["sleep_sessions"]
            )
            if not df.empty:
              daily_df = DataConverter.aggregate_sleep_by_day(df)
              if not daily_df.empty:
                fig = chart_generator.plot_sleep_quality_trend(
                  daily_df,
                  output_path=output_dir / f"{chart_type}.html"
                  if interactive
                  else output_dir / f"{chart_type}.png",
                )

              if fig:
                if interactive:
                  file_path = output_dir / f"{chart_type}.html"
                  fig.write_html(file_path)
                  generated_files.append(file_path)
                else:
                  file_path = output_dir / f"{chart_type}.png"
                  fig.write_image(file_path, width=1000, height=600)
                  generated_files.append(file_path)
              else:
                file_path = output_dir / f"{chart_type}.png"
                if file_path.exists():
                  generated_files.append(file_path)
                else:
                  console.print(
                    f"[yellow]⚠ 静态图表文件未找到: {file_path}[/yellow]"
                  )

                console.print(f"[green]✓ 生成: {chart_type}[/green]")

        elif chart_type == "sleep_stages_distribution":
          if "sleep_sessions" in sleep_data and sleep_data["sleep_sessions"]:
            df = DataConverter.sleep_sessions_to_df(
              sleep_data["sleep_sessions"]
            )
            if not df.empty:
              stages_df = DataConverter.prepare_sleep_stages_distribution(df)
              if not stages_df.empty:
                fig = chart_generator.plot_sleep_stages_distribution(
                  stages_df,
                  output_path=output_dir / f"{chart_type}.html"
                  if interactive
                  else output_dir / f"{chart_type}.png",
                )

              if fig:
                if interactive:
                  file_path = output_dir / f"{chart_type}.html"
                  fig.write_html(file_path)
                  generated_files.append(file_path)
                else:
                  file_path = output_dir / f"{chart_type}.png"
                  fig.write_image(file_path, width=1000, height=600)
                  generated_files.append(file_path)
              else:
                file_path = output_dir / f"{chart_type}.png"
                if file_path.exists():
                  generated_files.append(file_path)
                else:
                  console.print(
                    f"[yellow]⚠ 静态图表文件未找到: {file_path}[/yellow]"
                  )

                console.print(f"[green]✓ 生成: {chart_type}[/green]")

        elif chart_type == "sleep_consistency":
          if "sleep_sessions" in sleep_data and sleep_data["sleep_sessions"]:
            df = DataConverter.sleep_sessions_to_df(
              sleep_data["sleep_sessions"]
            )
            if not df.empty:
              fig = chart_generator.plot_sleep_consistency(
                df,
                output_path=output_dir / f"{chart_type}.html"
                if interactive
                else output_dir / f"{chart_type}.png",
              )

              if fig:
                if interactive:
                  file_path = output_dir / f"{chart_type}.html"
                  fig.write_html(file_path)
                  generated_files.append(file_path)
                else:
                  file_path = output_dir / f"{chart_type}.png"
                  fig.write_image(file_path, width=1000, height=600)
                  generated_files.append(file_path)
              else:
                file_path = output_dir / f"{chart_type}.png"
                if file_path.exists():
                  generated_files.append(file_path)
                else:
                  console.print(
                    f"[yellow]⚠ 静态图表文件未找到: {file_path}[/yellow]"
                  )

              console.print(f"[green]✓ 生成: {chart_type}[/green]")

        elif chart_type == "weekday_vs_weekend_sleep":
          if "sleep_sessions" in sleep_data and sleep_data["sleep_sessions"]:
            df = DataConverter.sleep_sessions_to_df(
              sleep_data["sleep_sessions"]
            )
            if not df.empty:
              # 添加周末标识
              df_copy = df.copy()
              df_copy["is_weekend"] = df_copy["date"].dt.dayofweek >= 5
              fig = chart_generator.plot_weekday_vs_weekend_sleep(
                df_copy,
                output_path=output_dir / f"{chart_type}.html"
                if interactive
                else output_dir / f"{chart_type}.png",
              )

              if fig:
                if interactive:
                  file_path = output_dir / f"{chart_type}.html"
                  fig.write_html(file_path)
                  generated_files.append(file_path)
                else:
                  file_path = output_dir / f"{chart_type}.png"
                  fig.write_image(file_path, width=1000, height=600)
                  generated_files.append(file_path)
              else:
                file_path = output_dir / f"{chart_type}.png"
                if file_path.exists():
                  generated_files.append(file_path)
                else:
                  console.print(
                    f"[yellow]⚠ 静态图表文件未找到: {file_path}[/yellow]"
                  )

              console.print(f"[green]✓ 生成: {chart_type}[/green]")

        else:
          console.print(f"[yellow]⚠ 暂不支持图表类型: {chart_type}[/yellow]")

      except Exception as e:
        console.print(f"[red]✗ 生成失败 {chart_type}: {e}[/red]")
        logger.error(f"Failed to generate chart {chart_type}: {e}")

    # 生成摘要信息
    if generated_files:
      # 创建图表索引文件
      index_content = f"""# 健康数据可视化图表

生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
数据来源: {xml_file.name}
图表数量: {len(generated_files)}
输出格式: {"交互式HTML" if interactive else "静态PNG"}

## 生成的图表

"""

      for i, file_path in enumerate(generated_files, 1):
        size_mb = file_path.stat().st_size / (1024 * 1024)
        index_content += f"{i}. **{file_path.stem}** - {size_mb:.2f} MB\n"

      index_file = output_dir / "index.md"
      index_file.write_text(index_content, encoding="utf-8")

      console.print("\n[bold green]✅ 图表生成完成！[/bold green]")
      console.print(f"[bold]生成文件数:[/bold] {len(generated_files)}")
      console.print(f"[bold]输出目录:[/bold] {output_dir}")
      console.print(f"[bold]图表索引:[/bold] {index_file}")

      # 显示文件列表
      console.print("\n[bold]生成的文件:[/bold]")
      for file_path in generated_files:
        size_mb = file_path.stat().st_size / (1024 * 1024)
        console.print(f"  • {file_path.name} ({size_mb:.2f} MB)")
    else:
      console.print("[yellow]⚠ 没有生成任何图表文件[/yellow]")
      console.print("[yellow]可能的原因：数据不足或图表类型不支持[/yellow]")

  except Exception as e:
    logger.error(f"图表生成失败: {e}")
    console.print(f"[bold red]错误:[/bold red] {e}")
    sys.exit(1)
