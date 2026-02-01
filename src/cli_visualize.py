"""Visualization and report generation CLI commands."""

import sys
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import click
import pandas as pd
from rich.console import Console

from src.analyzers.highlights import HighlightsGenerator
from src.config import get_config
from src.core.xml_parser import StreamingXMLParser
from src.i18n import Translator, resolve_locale
from src.processors.heart_rate import HeartRateAnalyzer
from src.processors.sleep import SleepAnalyzer
from src.utils.logger import get_logger
from src.utils.record_categorizer import (
  HEART_RATE_TYPE,
  HRV_TYPE,
  RESTING_HR_TYPE,
  SLEEP_TYPE,
  categorize_chart_records,
  categorize_records,
)
from src.visualization.charts import ChartGenerator
from src.visualization.reports import ReportGenerator

console = Console()
logger = get_logger(__name__)


def _resolve_xml_path(xml_path: str | None) -> Path:
  if xml_path:
    return Path(xml_path)
  return get_config().export_xml_path


@click.command(
  help=Translator(resolve_locale()).t("cli.help.report_command"),
  short_help=Translator(resolve_locale()).t("cli.help.report_command_short"),
  add_help_option=False,
)
@click.argument("xml_path", type=click.Path(), required=False)
@click.option(
  "--output",
  "-o",
  type=click.Path(),
  help=Translator(resolve_locale()).t("cli.help.output_dir"),
)
@click.option(
  "--format",
  "-f",
  type=click.Choice(["html", "markdown", "both"]),
  default="html",
  help=Translator(resolve_locale()).t("cli.help.report_format"),
)
@click.option(
  "--age",
  type=int,
  help=Translator(resolve_locale()).t("cli.help.report_age"),
)
@click.option(
  "--gender",
  type=click.Choice(["male", "female"]),
  help=Translator(resolve_locale()).t("cli.help.report_gender"),
)
@click.option(
  "--no-charts",
  is_flag=True,
  help=Translator(resolve_locale()).t("cli.help.report_no_charts"),
)
@click.help_option(
  "--help",
  "-h",
  help=Translator(resolve_locale()).t("cli.help.help_option"),
)
def report(
  xml_path: str,
  output: str | None,
  format: str,
  age: int | None,
  gender: str | None,
  no_charts: bool,
):
  def _format_error(error: Exception | str) -> str:
    message = str(error)
    if not message:
      return message
    try:
      return translator.t(message)
    except Exception:
      return message

  # Validate input file exists
  xml_file = _resolve_xml_path(xml_path)
  if not xml_file.exists():
    translator = Translator(resolve_locale())
    logger.error(translator.t("log.cli.xml_not_found", path=xml_file))
    console.print(
      f"[bold red]{translator.t('cli.common.error')}:[/bold red] "
      f"{translator.t('cli.parse.file_not_found', path=xml_file)}"
    )
    sys.exit(2)

  try:
    translator = Translator(resolve_locale())
    output_dir = Path(output) if output else get_config().output_dir / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(
      f"[bold blue]{translator.t('cli.report.generating')}[/bold blue] {xml_file}"
    )
    console.print(
      f"[bold blue]{translator.t('cli.common.output_dir')}[/bold blue] {output_dir}"
    )
    console.print(
      f"[bold blue]{translator.t('cli.report.format')}[/bold blue] {format}"
    )

    # Step 1: Parse data
    console.print(f"\n[bold]{translator.t('cli.report.step_parse')}[/bold]")
    with console.status(f"[bold green]{translator.t('cli.report.status_parsing')}"):
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

      categorized = categorize_records(all_records)
      hr_records = categorized["heart_rate"]
      resting_hr_records = categorized["resting_hr"]
      hrv_records = categorized["hrv"]
      vo2_max_records = categorized["vo2_max"]
      sleep_records = categorized["sleep"]

    console.print(
      f"[green]✓ {translator.t('cli.report.parsing_completed', heart_rate=len(hr_records), sleep=len(sleep_records))}[/green]"
    )

    # Step 2: Analyze data
    console.print(f"\n[bold]{translator.t('cli.report.step_analyze')}[/bold]")

    # Heart rate analysis
    gender_value: Literal["male", "female"] | None
    if gender in ("male", "female"):
      gender_value = cast(Literal["male", "female"], gender)
    else:
      gender_value = None

    heart_rate_analyzer = HeartRateAnalyzer(age=age, gender=gender_value)
    with console.status(
      f"[bold green]{translator.t('cli.report.status_analyzing_hr')}"
    ):
      heart_rate_report = heart_rate_analyzer.analyze_comprehensive(
        heart_rate_records=hr_records,
        resting_hr_records=resting_hr_records,
        hrv_records=hrv_records,
        vo2_max_records=vo2_max_records,
      )
    console.print(f"[green]✓ {translator.t('cli.analyze.hr_completed')}[/green]")

    # Sleep analysis
    sleep_analyzer = SleepAnalyzer()
    sleep_report = None
    if sleep_records:
      with console.status(
        f"[bold green]{translator.t('cli.report.status_analyzing_sleep')}"
      ):
        sleep_report = sleep_analyzer.analyze_comprehensive(sleep_records)  # type: ignore
      console.print(f"[green]✓ {translator.t('cli.analyze.sleep_completed')}[/green]")

    # Step 3: Generate highlights
    console.print(f"\n[bold]{translator.t('cli.report.step_insights')}[/bold]")
    highlights_generator = HighlightsGenerator()
    with console.status(
      f"[bold green]{translator.t('cli.report.status_generating_insights')}"
    ):
      highlights = highlights_generator.generate_comprehensive_highlights(
        heart_rate_report=heart_rate_report,
        sleep_report=sleep_report,
      )
    console.print(
      f"[green]✓ {translator.t('cli.analyze.generated_insights', insights=len(highlights.insights), recommendations=len(highlights.recommendations))}[/green]"
    )

    # Step 4: Generate reports
    console.print(f"\n[bold]{translator.t('cli.report.step_generate')}[/bold]")
    report_generator = ReportGenerator(output_dir)
    locale = resolve_locale()

    report_files = []

    if format in ["html", "both"]:
      with console.status(f"[bold green]{translator.t('cli.report.status_html')}"):
        html_file = report_generator.generate_html_report(
          title=None,
          heart_rate_report=heart_rate_report,
          sleep_report=sleep_report,
          highlights=highlights,
          include_charts=not no_charts,
          locale=locale,
        )
        report_files.append(html_file)
      console.print(
        f"[green]✓ {translator.t('cli.report.html_saved', path=html_file)}[/green]"
      )

    if format in ["markdown", "both"]:
      with console.status(f"[bold green]{translator.t('cli.report.status_markdown')}"):
        md_file = report_generator.generate_markdown_report(
          title=None,
          heart_rate_report=heart_rate_report,
          sleep_report=sleep_report,
          highlights=highlights,
          locale=locale,
        )
        report_files.append(md_file)
      console.print(
        f"[green]✓ {translator.t('cli.report.md_saved', path=md_file)}[/green]"
      )

    # Summary
    console.print(f"\n[bold green]✅ {translator.t('cli.report.success')}[/bold green]")
    console.print(f"\n[bold]{translator.t('cli.export.generated_files')}[/bold]")

    # Display file information with robust error handling
    for file_path in report_files:
      try:
        size_mb = file_path.stat().st_size / (1024 * 1024)
        console.print(
          f"  • {file_path.name} "
          f"({translator.t('cli.common.file_size_mb', size=size_mb)})"
        )
      except (FileNotFoundError, OSError) as e:
        # Report file might not exist or be inaccessible
        logger.warning(
          translator.t(
            "log.cli.file_size_error",
            path=file_path,
            error=_format_error(e),
          )
        )
        console.print(
          f"  • {file_path.name} ({translator.t('cli.export.file_size_unknown')})"
        )

  except Exception as e:
    logger.error(translator.t("log.cli.report_failed", error=_format_error(e)))
    console.print(
      f"[bold red]{translator.t('cli.common.error')}:[/bold red] {_format_error(e)}"
    )
    sys.exit(1)


@click.command(
  help=Translator(resolve_locale()).t("cli.help.visualize_command"),
  short_help=Translator(resolve_locale()).t("cli.help.visualize_command_short"),
  add_help_option=False,
)
@click.argument("xml_path", type=click.Path(), required=False)
@click.option(
  "--output",
  "-o",
  type=click.Path(),
  help=Translator(resolve_locale()).t("cli.help.output_dir"),
)
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
  help=Translator(resolve_locale()).t("cli.help.visualize_charts"),
)
@click.option(
  "--interactive/--static",
  default=True,
  help=Translator(resolve_locale()).t("cli.help.visualize_interactive"),
)
@click.option(
  "--age",
  type=int,
  help=Translator(resolve_locale()).t("cli.help.visualize_age"),
)
@click.help_option(
  "--help",
  "-h",
  help=Translator(resolve_locale()).t("cli.help.help_option"),
)
def visualize(
  xml_path: str,
  output: str | None,
  charts: tuple,
  interactive: bool,
  age: int | None,
):
  def _format_error(error: Exception | str) -> str:
    message = str(error)
    if not message:
      return message
    try:
      return translator.t(message)
    except Exception:
      return message

  # Validate input file exists
  xml_file = _resolve_xml_path(xml_path)
  if not xml_file.exists():
    translator = Translator(resolve_locale())
    logger.error(translator.t("log.cli.xml_not_found", path=xml_file))
    console.print(
      f"[bold red]{translator.t('cli.common.error')}:[/bold red] "
      f"{translator.t('cli.parse.file_not_found', path=xml_file)}"
    )
    sys.exit(2)

  try:
    translator = Translator(resolve_locale())
    output_dir = Path(output) if output else get_config().output_dir / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(
      f"[bold blue]{translator.t('cli.visualize.generating')}[/bold blue] {xml_file}"
    )
    console.print(
      f"[bold blue]{translator.t('cli.common.output_dir')}[/bold blue] {output_dir}"
    )

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
      f"[bold blue]{translator.t('cli.visualize.charts_to_generate')}[/bold blue] "
      f"{translator.t('cli.visualize.chart_count', count=len(selected_charts))}"
    )

    # Parse data
    console.print(f"\n[bold]{translator.t('cli.visualize.parsing_data')}[/bold]")
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
      all_types.extend([HEART_RATE_TYPE, RESTING_HR_TYPE, HRV_TYPE])
    if need_sleep:
      all_types.append(SLEEP_TYPE)

    # Parse all records in a single pass if any data is needed
    if all_types:
      with console.status(
        f"[bold green]{translator.t('cli.visualize.status_parsing')}"
      ):
        all_records = list(parser.parse_records(record_types=all_types))

        categorized = categorize_chart_records(all_records)
        hr_data = {
          "heart_rate": categorized["heart_rate"],
          "resting_hr": categorized["resting_hr"],
          "hrv": categorized["hrv"],
        }
        sleep_data = {"sleep_records": categorized["sleep_records"]}

      console.print(
        f"[green]✓ {translator.t('cli.visualize.parsing_completed')}[/green]"
      )

    # Process sleep data if needed
    if need_sleep and "sleep_records" in sleep_data:
      with console.status(
        f"[bold green]{translator.t('cli.visualize.status_processing_sleep')}"
      ):
        # Parse sleep sessions
        sleep_analyzer = SleepAnalyzer()
        sleep_sessions = sleep_analyzer.parse_sleep_sessions(
          sleep_data["sleep_records"]
        )

        sleep_data["sleep_sessions"] = sleep_sessions
      console.print(
        f"[green]✓ {translator.t('cli.visualize.sleep_processing_completed')}[/green]"
      )

    # Generate charts
    console.print(f"\n[bold]{translator.t('cli.visualize.generating_charts')}[/bold]")
    chart_generator = ChartGenerator()
    generated_files = []

    # Import data converter
    from src.visualization.data_converter import DataConverter

    # Generate files for each chart type
    # Algorithm: Iterate through chart types, dynamically dispatch to appropriate generation method
    for chart_type in selected_charts:
      try:
        console.print(
          f"[dim]{translator.t('cli.visualize.generating_chart', chart=chart_type)}[/dim]"
        )

        if chart_type == "heart_rate_timeseries":
          if "heart_rate" in hr_data and hr_data["heart_rate"]:
            df = DataConverter.heart_rate_to_df(hr_data["heart_rate"])
            sleep_sessions = sleep_data.get("sleep_sessions", []) if sleep_data else []
            if not df.empty:
              df = DataConverter.sample_data_for_performance(
                df, 50000
              )  # Downsample to improve performance.
              fig = chart_generator.plot_heart_rate_timeseries(
                df,
                sleep_sessions=sleep_sessions,
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
                console.print(
                  f"[yellow]⚠ {translator.t('cli.visualize.static_not_found', path=file_path)}[/yellow]"
                )

            console.print(
              f"[green]✓ {translator.t('cli.visualize.generated', chart=chart_type)}[/green]"
            )

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
                    f"[yellow]⚠ {translator.t('cli.visualize.static_not_found', path=file_path)}[/yellow]"
                  )

              console.print(
                f"[green]✓ {translator.t('cli.visualize.generated', chart=chart_type)}[/green]"
              )

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
                    f"[yellow]⚠ {translator.t('cli.visualize.static_not_found', path=file_path)}[/yellow]"
                  )

              console.print(
                f"[green]✓ {translator.t('cli.visualize.generated', chart=chart_type)}[/green]"
              )

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
                  console.print(
                    f"[yellow]⚠ {translator.t('cli.visualize.static_not_found', path=file_path)}[/yellow]"
                  )

              console.print(
                f"[green]✓ {translator.t('cli.visualize.generated', chart=chart_type)}[/green]"
              )

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
                    f"[yellow]⚠ {translator.t('cli.visualize.static_not_found', path=file_path)}[/yellow]"
                  )

              console.print(
                f"[green]✓ {translator.t('cli.visualize.generated', chart=chart_type)}[/green]"
              )

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
                    f"[yellow]⚠ {translator.t('cli.visualize.static_not_found', path=file_path)}[/yellow]"
                  )

                console.print(
                  f"[green]✓ {translator.t('cli.visualize.generated', chart=chart_type)}[/green]"
                )

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
                  console.print(
                    f"[yellow]⚠ {translator.t('cli.visualize.static_not_found', path=file_path)}[/yellow]"
                  )

              console.print(
                f"[green]✓ {translator.t('cli.visualize.generated', chart=chart_type)}[/green]"
              )

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
                  console.print(
                    f"[yellow]⚠ {translator.t('cli.visualize.static_not_found', path=file_path)}[/yellow]"
                  )

                console.print(
                  f"[green]✓ {translator.t('cli.visualize.generated', chart=chart_type)}[/green]"
                )

        elif chart_type == "sleep_stages_distribution":
          if "sleep_sessions" in sleep_data and sleep_data["sleep_sessions"]:
            df = DataConverter.sleep_sessions_to_df(sleep_data["sleep_sessions"])
            if not df.empty:
              stages_data = []
              for _, row in df.iterrows():
                date_val = row.get("date")
                if pd.isna(date_val):
                  continue
                deep_val = float(row.get("deep_sleep", 0) or 0)
                rem_val = float(row.get("rem_sleep", 0) or 0)
                light_val = float(row.get("light_sleep", 0) or 0)
                total_sleep = float(row.get("sleep_duration", 0) or 0)
                if light_val <= 0:
                  light_val = max(0.0, total_sleep - deep_val - rem_val)
                row_added = False
                if deep_val > 0:
                  stages_data.append(
                    {"date": date_val, "stage": "Deep", "duration": deep_val / 60}
                  )
                  row_added = True
                if rem_val > 0:
                  stages_data.append(
                    {"date": date_val, "stage": "REM", "duration": rem_val / 60}
                  )
                  row_added = True
                if light_val > 0:
                  stages_data.append(
                    {"date": date_val, "stage": "Light", "duration": light_val / 60}
                  )
                  row_added = True
                if not row_added and total_sleep > 0:
                  stages_data.append(
                    {
                      "date": date_val,
                      "stage": "Light",
                      "duration": total_sleep / 60,
                    }
                  )

              if stages_data:
                stages_df = pd.DataFrame(stages_data)
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
                    f"[yellow]⚠ {translator.t('cli.visualize.static_not_found', path=file_path)}[/yellow]"
                  )

                console.print(
                  f"[green]✓ {translator.t('cli.visualize.generated', chart=chart_type)}[/green]"
                )

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
                  console.print(
                    f"[yellow]⚠ {translator.t('cli.visualize.static_not_found', path=file_path)}[/yellow]"
                  )

              console.print(
                f"[green]✓ {translator.t('cli.visualize.generated', chart=chart_type)}[/green]"
              )

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
                  console.print(
                    f"[yellow]⚠ {translator.t('cli.visualize.static_not_found', path=file_path)}[/yellow]"
                  )

              console.print(
                f"[green]✓ {translator.t('cli.visualize.generated', chart=chart_type)}[/green]"
              )

        else:
          console.print(
            f"[yellow]⚠ {translator.t('cli.visualize.unsupported_chart', chart=chart_type)}[/yellow]"
          )

      except Exception as e:
        console.print(
          f"[red]✗ {translator.t('cli.visualize.failed_chart', chart=chart_type, error=_format_error(e))}[/red]"
        )
        logger.error(
          translator.t(
            "log.cli.chart_failed",
            chart=chart_type,
            error=_format_error(e),
          )
        )

    # Generate summary information
    if generated_files:
      # Create chart index file
      index_content = f"""# {translator.t("cli.visualize.index_title")}

{translator.t("cli.visualize.index_generated_at")}: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{translator.t("cli.visualize.index_data_source")}: {xml_file.name}
{translator.t("cli.visualize.index_chart_count")}: {len(generated_files)}
{translator.t("cli.visualize.index_output_format")}: {translator.t("cli.visualize.index_format_html") if interactive else translator.t("cli.visualize.index_format_png")}

## {translator.t("cli.visualize.index_generated_charts")}

"""

      for i, file_path in enumerate(generated_files, 1):
        size_mb = file_path.stat().st_size / (1024 * 1024)
        index_content += (
          f"{i}. **{file_path.stem}** - "
          f"{translator.t('cli.common.file_size_mb', size=size_mb)}\n"
        )

      index_file = output_dir / "index.md"
      index_file.write_text(index_content, encoding="utf-8")

      console.print(
        f"\n[bold green]✅ {translator.t('cli.visualize.completed')}[/bold green]"
      )
      console.print(
        f"[bold]{translator.t('cli.visualize.files_generated')}[/bold] {len(generated_files)}"
      )
      console.print(
        f"[bold]{translator.t('cli.common.output_dir')}[/bold] {output_dir}"
      )
      console.print(
        f"[bold]{translator.t('cli.visualize.index_file')}[/bold] {index_file}"
      )

      # Display file list
      console.print(f"\n[bold]{translator.t('cli.export.generated_files')}[/bold]")
      for file_path in generated_files:
        size_mb = file_path.stat().st_size / (1024 * 1024)
        console.print(
          f"  • {file_path.name} "
          f"({translator.t('cli.common.file_size_mb', size=size_mb)})"
        )
    else:
      console.print(f"[yellow]⚠ {translator.t('cli.visualize.no_files')}[/yellow]")
      console.print(f"[yellow]{translator.t('cli.visualize.no_files_reason')}[/yellow]")

  except FileNotFoundError:
    logger.error(translator.t("log.cli.xml_not_found", path=xml_file))
    console.print(
      f"[bold red]{translator.t('cli.common.error')}:[/bold red] "
      f"{translator.t('cli.parse.file_not_found', path=xml_file)}"
    )
    sys.exit(2)
  except Exception as e:
    logger.error(translator.t("log.cli.visualize_failed", error=_format_error(e)))
    console.print(
      f"[bold red]{translator.t('cli.common.error')}:[/bold red] {_format_error(e)}"
    )
    sys.exit(1)
