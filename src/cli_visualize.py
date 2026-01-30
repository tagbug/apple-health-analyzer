"""Visualization and report generation CLI commands."""

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
@click.argument("xml_path", type=click.Path())
@click.option("--output", "-o", type=click.Path(), help="Output directory")
@click.option(
  "--format",
  "-f",
  type=click.Choice(["html", "markdown", "both"]),
  default="html",
  help="Report format",
)
@click.option("--age", type=int, help="User age (for cardio fitness analysis)")
@click.option(
  "--gender",
  type=click.Choice(["male", "female"]),
  help="Gender (for cardio fitness analysis)",
)
@click.option("--no-charts", is_flag=True, help="Exclude charts (text report only)")
def report(
  xml_path: str,
  output: str | None,
  format: str,
  age: int | None,
  gender: str | None,
  no_charts: bool,
):
  """Generate comprehensive health analysis report (with charts and insights)

  Combines data analysis and visualization to generate reports with complete analysis results, charts, and health insights.

  Examples:\n
      # Generate HTML report\n
      health-analyzer report export.xml --age 30 --gender male

      # Generate Markdown report\n
      health-analyzer report export.xml --format markdown

      # Generate both HTML and Markdown formats\n
      health-analyzer report export.xml --format both --age 30 --gender male
  """
  # Validate input file exists
  xml_file = Path(xml_path)
  if not xml_file.exists():
    logger.error(f"XML file not found: {xml_file}")
    console.print(f"[bold red]Error:[/bold red] XML file not found: {xml_file}")
    sys.exit(2)

  try:
    output_dir = Path(output) if output else get_config().output_dir / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold blue]Generating health report:[/bold blue] {xml_file}")
    console.print(f"[bold blue]Output directory:[/bold blue] {output_dir}")
    console.print(f"[bold blue]Report format:[/bold blue] {format}")

    # Step 1: Parse data
    console.print("\n[bold]Step 1/4: Parsing health data...[/bold]")
    with console.status("[bold green]Parsing records..."):
      parser = StreamingXMLParser(xml_file)

      # Define all record types needed for report generation
      all_types = [
        "HKQuantityTypeIdentifierHeartRate",
        "HKQuantityTypeIdentifierRestingHeartRate",
        "HKQuantityTypeIdentifierHeartRateVariabilitySDNN",
        "HKQuantityTypeIdentifierVO2Max",
        "HKCategoryTypeIdentifierSleepAnalysis",
      ]

      # Parse all records in a single pass
      all_records = list(parser.parse_records(record_types=all_types))

      # Categorize records by type
      hr_records = []
      resting_hr_records = []
      hrv_records = []
      vo2_max_records = []
      sleep_records = []

      for record in all_records:
        record_type = getattr(record, "type", "")
        if record_type == "HKQuantityTypeIdentifierHeartRate":
          hr_records.append(record)
        elif record_type == "HKQuantityTypeIdentifierRestingHeartRate":
          resting_hr_records.append(record)
        elif record_type == "HKQuantityTypeIdentifierHeartRateVariabilitySDNN":
          hrv_records.append(record)
        elif record_type == "HKQuantityTypeIdentifierVO2Max":
          vo2_max_records.append(record)
        elif record_type == "HKCategoryTypeIdentifierSleepAnalysis":
          sleep_records.append(record)

    console.print(
      f"[green]✓ Parsing completed: {len(hr_records)} heart rate records, {len(sleep_records)} sleep records[/green]"
    )

    # Step 2: Analyze data
    console.print("\n[bold]Step 2/4: Analyzing health data...[/bold]")

    # Heart rate analysis
    heart_rate_analyzer = HeartRateAnalyzer(age=age, gender=gender)  # type: ignore
    with console.status("[bold green]Analyzing heart rate data..."):
      heart_rate_report = heart_rate_analyzer.analyze_comprehensive(
        heart_rate_records=hr_records,
        resting_hr_records=resting_hr_records,
        hrv_records=hrv_records,
        vo2_max_records=vo2_max_records,
      )
    console.print("[green]✓ Heart rate analysis completed[/green]")

    # Sleep analysis
    sleep_analyzer = SleepAnalyzer()
    sleep_report = None
    if sleep_records:
      with console.status("[bold green]Analyzing sleep data..."):
        sleep_report = sleep_analyzer.analyze_comprehensive(sleep_records)  # type: ignore
      console.print("[green]✓ Sleep analysis completed[/green]")

    # Step 3: Generate highlights
    console.print("\n[bold]Step 3/4: Generating health insights...[/bold]")
    highlights_generator = HighlightsGenerator()
    with console.status("[bold green]Generating insights and recommendations..."):
      highlights = highlights_generator.generate_comprehensive_highlights(
        heart_rate_report=heart_rate_report,
        sleep_report=sleep_report,
      )
    console.print(
      f"[green]✓ Generated {len(highlights.insights)} insights and {len(highlights.recommendations)} recommendations[/green]"
    )

    # Step 4: Generate reports
    console.print("\n[bold]Step 4/4: Generating report files...[/bold]")
    report_generator = ReportGenerator(output_dir)

    report_files = []

    if format in ["html", "both"]:
      with console.status("[bold green]Generating HTML report..."):
        html_file = report_generator.generate_html_report(
          title="Health Analysis Report",
          heart_rate_report=heart_rate_report,
          sleep_report=sleep_report,
          highlights=highlights,
          include_charts=not no_charts,
        )
        report_files.append(html_file)
      console.print(f"[green]✓ HTML report: {html_file}[/green]")

    if format in ["markdown", "both"]:
      with console.status("[bold green]Generating Markdown report..."):
        md_file = report_generator.generate_markdown_report(
          title="Health Analysis Report",
          heart_rate_report=heart_rate_report,
          sleep_report=sleep_report,
          highlights=highlights,
        )
        report_files.append(md_file)
      console.print(f"[green]✓ Markdown report: {md_file}[/green]")

    # Summary
    console.print("\n[bold green]✅ Report generation successful![/bold green]")
    console.print("\n[bold]Generated files:[/bold]")

    # Display file information with robust error handling
    for file_path in report_files:
      try:
        size_mb = file_path.stat().st_size / (1024 * 1024)
        console.print(f"  • {file_path.name} ({size_mb:.2f} MB)")
      except (FileNotFoundError, OSError) as e:
        # Report file might not exist or be inaccessible
        logger.warning(f"Could not get file size for {file_path}: {e}")
        console.print(f"  • {file_path.name} (size unknown)")

  except Exception as e:
    logger.error(f"Report generation failed: {e}")
    console.print(f"[bold red]Error:[/bold red] {e}")
    sys.exit(1)


@click.command()
@click.argument("xml_path", type=click.Path())
@click.option("--output", "-o", type=click.Path(), help="Output directory")
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
  help="Chart types to generate (multiple selection allowed, default: all)",
)
@click.option(
  "--interactive/--static",
  default=True,
  help="Interactive charts (HTML) or static charts (PNG)",
)
@click.option("--age", type=int, help="User age (for heart rate zone calculation)")
def visualize(
  xml_path: str,
  output: str | None,
  charts: tuple,
  interactive: bool,
  age: int | None,
):
  """Generate health data visualization charts

  Supports generating various heart rate and sleep-related visualization charts.

  Examples:\n
      # Generate all charts\n
      health-analyzer visualize export.xml

      # Generate specific charts\n
      health-analyzer visualize export.xml -c heart_rate_timeseries -c sleep_quality_trend

      # Generate static PNG charts\n
      health-analyzer visualize export.xml --static
  """
  # Validate input file exists
  xml_file = Path(xml_path)
  if not xml_file.exists():
    logger.error(f"XML file not found: {xml_file}")
    console.print(f"[bold red]Error:[/bold red] XML file not found: {xml_file}")
    sys.exit(2)

  try:
    output_dir = Path(output) if output else get_config().output_dir / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold blue]Generating visualization charts:[/bold blue] {xml_file}")
    console.print(f"[bold blue]Output directory:[/bold blue] {output_dir}")

    # Determine charts to generate
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
      f"[bold blue]Charts to generate:[/bold blue] {len(selected_charts)} charts"
    )

    # Parse data
    console.print("\n[bold]Parsing data...[/bold]")
    parser = StreamingXMLParser(xml_file)

    # Determine required data types
    need_hr = any(
      c.startswith("heart_rate") or c.startswith("resting") or c.startswith("hrv")
      for c in selected_charts
    )
    need_sleep = any(
      c.startswith("sleep") or c.startswith("weekday") for c in selected_charts
    )

    hr_data = {}
    sleep_data = {}

    # Define all record types needed for visualization
    all_types = []
    if need_hr:
      all_types.extend(
        [
          "HKQuantityTypeIdentifierHeartRate",
          "HKQuantityTypeIdentifierRestingHeartRate",
          "HKQuantityTypeIdentifierHeartRateVariabilitySDNN",
        ]
      )
    if need_sleep:
      all_types.append("HKCategoryTypeIdentifierSleepAnalysis")

    # Parse all records in a single pass if any data is needed
    if all_types:
      with console.status("[bold green]Parsing health data..."):
        all_records = list(parser.parse_records(record_types=all_types))

        # Categorize records by type
        for record in all_records:
          record_type = getattr(record, "type", "")
          if record_type == "HKQuantityTypeIdentifierHeartRate":
            hr_data.setdefault("heart_rate", []).append(record)
          elif record_type == "HKQuantityTypeIdentifierRestingHeartRate":
            hr_data.setdefault("resting_hr", []).append(record)
          elif record_type == "HKQuantityTypeIdentifierHeartRateVariabilitySDNN":
            hr_data.setdefault("hrv", []).append(record)
          elif record_type == "HKCategoryTypeIdentifierSleepAnalysis":
            sleep_data.setdefault("sleep_records", []).append(record)

      console.print("[green]✓ Data parsing completed[/green]")

    # Process sleep data if needed
    if need_sleep and "sleep_records" in sleep_data:
      with console.status("[bold green]Processing sleep sessions..."):
        # Parse sleep sessions
        sleep_analyzer = SleepAnalyzer()
        sleep_sessions = sleep_analyzer._parse_sleep_sessions(  # type: ignore
          sleep_data["sleep_records"]
        )

        sleep_data["sleep_sessions"] = sleep_sessions
      console.print("[green]✓ Sleep data processing completed[/green]")

    # Generate charts
    console.print("\n[bold]Generating charts...[/bold]")
    chart_generator = ChartGenerator()
    generated_files = []

    # Import data converter
    from src.visualization.data_converter import DataConverter

    # Generate files for each chart type
    # Algorithm: Iterate through chart types, dynamically dispatch to appropriate generation method
    for chart_type in selected_charts:
      try:
        console.print(f"[dim]Generating {chart_type}...[/dim]")

        if chart_type == "heart_rate_timeseries":
          if "heart_rate" in hr_data and hr_data["heart_rate"]:
            df = DataConverter.heart_rate_to_df(hr_data["heart_rate"])
            if not df.empty:
              df = DataConverter.sample_data_for_performance(
                df, 50000
              )  # Downsample to improve performance.
              fig = chart_generator.plot_heart_rate_timeseries(
                df,
                output_path=output_dir / f"{chart_type}.html"
                if interactive
                else output_dir / f"{chart_type}.png",
              )

              if fig:  # Interactive mode returns a figure without saving.
                if interactive:
                  file_path = output_dir / f"{chart_type}.html"
                  fig.write_html(file_path)
                  generated_files.append(file_path)
                else:
                  file_path = output_dir / f"{chart_type}.png"
                  fig.write_image(file_path, width=1200, height=600)
                  generated_files.append(file_path)
              else:  # Static mode already saved.
                file_path = output_dir / f"{chart_type}.png"
                if file_path.exists():
                  generated_files.append(file_path)
                else:
                  console.print(f"[yellow]⚠ 静态图表文件未找到: {file_path}[/yellow]")

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
                  console.print(f"[yellow]⚠ 静态图表文件未找到: {file_path}[/yellow]")

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
                  console.print(f"[yellow]⚠ 静态图表文件未找到: {file_path}[/yellow]")

              console.print(f"[green]✓ 生成: {chart_type}[/green]")

        elif chart_type == "heart_rate_heatmap":
          if "heart_rate" in hr_data and hr_data["heart_rate"]:
            df = DataConverter.heart_rate_to_df(hr_data["heart_rate"])
            if not df.empty:
              df = DataConverter.sample_data_for_performance(df, 20000)
              # Prepare daily average data.
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
                  console.print(f"[yellow]⚠ 静态图表文件未找到: {file_path}[/yellow]")

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
                  console.print(f"[yellow]⚠ 静态图表文件未找到: {file_path}[/yellow]")

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
                  console.print(f"[yellow]⚠ 静态图表文件未找到: {file_path}[/yellow]")

                console.print(f"[green]✓ 生成: {chart_type}[/green]")

        elif chart_type == "sleep_timeline":
          if "sleep_sessions" in sleep_data and sleep_data["sleep_sessions"]:
            df = DataConverter.sleep_sessions_to_df(sleep_data["sleep_sessions"])
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
                  console.print(f"[yellow]⚠ 静态图表文件未找到: {file_path}[/yellow]")

              console.print(f"[green]✓ 生成: {chart_type}[/green]")

        elif chart_type == "sleep_quality_trend":
          if "sleep_sessions" in sleep_data and sleep_data["sleep_sessions"]:
            df = DataConverter.sleep_sessions_to_df(sleep_data["sleep_sessions"])
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
                  console.print(f"[yellow]⚠ 静态图表文件未找到: {file_path}[/yellow]")

                console.print(f"[green]✓ 生成: {chart_type}[/green]")

        elif chart_type == "sleep_stages_distribution":
          if "sleep_sessions" in sleep_data and sleep_data["sleep_sessions"]:
            df = DataConverter.sleep_sessions_to_df(sleep_data["sleep_sessions"])
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
                  console.print(f"[yellow]⚠ 静态图表文件未找到: {file_path}[/yellow]")

                console.print(f"[green]✓ 生成: {chart_type}[/green]")

        elif chart_type == "sleep_consistency":
          if "sleep_sessions" in sleep_data and sleep_data["sleep_sessions"]:
            df = DataConverter.sleep_sessions_to_df(sleep_data["sleep_sessions"])
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
                  console.print(f"[yellow]⚠ 静态图表文件未找到: {file_path}[/yellow]")

              console.print(f"[green]✓ 生成: {chart_type}[/green]")

        elif chart_type == "weekday_vs_weekend_sleep":
          if "sleep_sessions" in sleep_data and sleep_data["sleep_sessions"]:
            df = DataConverter.sleep_sessions_to_df(sleep_data["sleep_sessions"])
            if not df.empty:
              # Add weekend indicator.
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
                  console.print(f"[yellow]⚠ 静态图表文件未找到: {file_path}[/yellow]")

              console.print(f"[green]✓ 生成: {chart_type}[/green]")

        else:
          console.print(f"[yellow]⚠ Chart type not supported: {chart_type}[/yellow]")

      except Exception as e:
        console.print(f"[red]✗ Failed to generate {chart_type}: {e}[/red]")
        logger.error(f"Failed to generate chart {chart_type}: {e}")

    # Generate summary information
    if generated_files:
      # Create chart index file
      index_content = f"""# Health Data Visualization Charts

Generation time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Data source: {xml_file.name}
Number of charts: {len(generated_files)}
Output format: {"Interactive HTML" if interactive else "Static PNG"}

## Generated Charts

"""

      for i, file_path in enumerate(generated_files, 1):
        size_mb = file_path.stat().st_size / (1024 * 1024)
        index_content += f"{i}. **{file_path.stem}** - {size_mb:.2f} MB\n"

      index_file = output_dir / "index.md"
      index_file.write_text(index_content, encoding="utf-8")

      console.print("\n[bold green]✅ Chart generation completed![/bold green]")
      console.print(f"[bold]Files generated:[/bold] {len(generated_files)}")
      console.print(f"[bold]Output directory:[/bold] {output_dir}")
      console.print(f"[bold]Chart index:[/bold] {index_file}")

      # Display file list
      console.print("\n[bold]Generated files:[/bold]")
      for file_path in generated_files:
        size_mb = file_path.stat().st_size / (1024 * 1024)
        console.print(f"  • {file_path.name} ({size_mb:.2f} MB)")
    else:
      console.print("[yellow]⚠ No chart files were generated[/yellow]")
      console.print(
        "[yellow]Possible reasons: insufficient data or unsupported chart types[/yellow]"
      )

  except FileNotFoundError:
    logger.error("XML file not found")
    console.print("[bold red]Error:[/bold red] XML file not found")
    sys.exit(2)
  except Exception as e:
    logger.error(f"Chart generation failed: {e}")
    console.print(f"[bold red]Error:[/bold red] {e}")
    sys.exit(1)
