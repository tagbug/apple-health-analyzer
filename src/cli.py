"""Command-line interface for Apple Health Analyzer.

Provides CLI commands for parsing, analyzing, and exporting health data.
"""

import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import click
from rich.console import Console
from rich.table import Table

from src.cli_visualize import report as report_command
from src.cli_visualize import visualize as visualize_command
from src.config import get_config, reload_config
from src.core.data_models import HealthRecord
from src.core.exceptions import HealthAnalyzerError
from src.core.xml_parser import StreamingXMLParser, get_export_file_info
from src.i18n import Translator, resolve_locale
from src.utils.logger import UnifiedProgress, get_logger
from src.utils.record_categorizer import (
  HEART_RATE_TYPE,
  HRV_TYPE,
  RESTING_HR_TYPE,
  SLEEP_TYPE,
  VO2_MAX_TYPE,
  categorize_records,
)

console = Console()
logger = get_logger(__name__)


def _t(locale: str | None = None) -> Translator:
  return Translator(resolve_locale(locale))


def _format_error(error: Exception | str, translator: Translator) -> str:
  message = str(error)
  if not message:
    return message
  try:
    return translator.t(message)
  except Exception:
    return message


def _resolve_xml_path(xml_path: str | None) -> Path:
  if xml_path:
    return Path(xml_path)
  return get_config().export_xml_path


@click.group(help=_t().t("cli.help.root"), add_help_option=False)
@click.option(
  "--config",
  "config_path",
  type=click.Path(exists=True),
  help=_t().t("cli.help.config_path"),
)
@click.option(
  "--verbose",
  "-v",
  is_flag=True,
  help=_t().t("cli.help.verbose"),
)
@click.option(
  "--locale",
  type=click.Choice(["en", "zh"]),
  help=_t().t("cli.help.locale"),
)
@click.version_option(help=_t().t("cli.help.version"))
@click.help_option("--help", "-h", help=_t().t("cli.help.help_option"))
def cli(config_path: str | None, verbose: bool, locale: str | None):
  if config_path:
    import os

    os.environ["CONFIG_FILE"] = config_path
    reload_config()

  if locale:
    import os

    os.environ["LOCALE"] = locale
    reload_config()

  if verbose:
    import logging

    logging.getLogger().setLevel(logging.DEBUG)


@cli.command(
  help=_t().t("cli.help.parse_command"),
  short_help=_t().t("cli.help.parse_command_short"),
  add_help_option=False,
)
@click.argument("xml_path", type=click.Path(exists=True), required=False)
@click.option(
  "--output",
  "-o",
  type=click.Path(),
  help=_t().t("cli.help.output_results"),
)
@click.option(
  "--types",
  "-t",
  multiple=True,
  help=_t().t("cli.help.parse_types"),
)
@click.option(
  "--preview",
  is_flag=True,
  help=_t().t("cli.help.preview"),
)
@click.help_option("--help", "-h", help=_t().t("cli.help.help_option"))
def parse(xml_path: str, output: str | None, types: list[str], preview: bool):
  translator = _t()
  try:
    xml_file = _resolve_xml_path(xml_path)

    # Validate input file
    _ensure_xml_file(xml_file)

    output_dir = Path(output) if output else get_config().output_dir

    console.print(
      f"[bold blue]{translator.t('cli.parse.parsing_export')}[/bold blue] {xml_file}"
    )
    console.print(
      f"[bold blue]{translator.t('cli.common.output_dir')}[/bold blue] {output_dir}"
    )

    _display_file_info(xml_file)

    # Initialize parser
    parser = _init_parser(xml_file)

    # Parse records with progress tracking
    record_types = list(types) if types else None
    records = []
    stats = {}

    try:
      records, stats = _parse_records_with_progress(parser, record_types)
    except Exception as e:
      _handle_parsing_exception(e)

    # Validate parsing results
    if not records:
      console.print(
        f"[yellow]{translator.t('cli.common.warning')}:[/yellow] "
        f"{translator.t('cli.parse.no_records')}"
      )
      if record_types:
        console.print(
          f"[yellow]{translator.t('cli.common.tip')}:[/yellow] "
          f"{translator.t('cli.parse.check_types', types=record_types)}"
        )
      else:
        console.print(
          f"[yellow]{translator.t('cli.common.tip')}:[/yellow] "
          f"{translator.t('cli.parse.verify_data')}"
        )
      sys.exit(1)

    # Display results
    _display_parsing_results(stats)

    if preview and records:
      console.print(f"\n[bold]{translator.t('cli.parse.data_preview')}:[/bold]")
      _display_records_preview(records[:10])

    # Save results if output specified
    if output:
      try:
        _save_parsed_data(records, stats, output_dir)
      except Exception as e:
        logger.error(
          translator.t("log.cli.save_failed", error=_format_error(e, translator))
        )
        console.print(
          f"[bold red]{translator.t('cli.common.error')}:[/bold red] "
          f"{translator.t('cli.parse.save_failed', error=_format_error(e, translator))}"
        )
        console.print(
          f"[yellow]{translator.t('cli.common.tip')}:[/yellow] "
          f"{translator.t('cli.parse.check_permissions', path=output_dir)}"
        )
        sys.exit(1)

    _print_parsing_success(stats)

  except KeyboardInterrupt:
    console.print(f"\n[yellow]{translator.t('cli.common.cancelled')}[/yellow]")
    sys.exit(1)
  except Exception as e:
    logger.error(
      translator.t("log.cli.parse_unexpected", error=_format_error(e, translator))
    )
    console.print(
      f"[bold red]{translator.t('cli.common.unexpected_error')}:[/bold red] "
      f"{_format_error(e, translator)}"
    )
    console.print(
      f"[yellow]{translator.t('cli.common.tip')}:[/yellow] "
      f"{translator.t('cli.common.report_issue')}"
    )
    sys.exit(1)


@cli.command(
  help=_t().t("cli.help.info_command"),
  short_help=_t().t("cli.help.info_command_short"),
  add_help_option=False,
)
@click.argument("xml_path", type=click.Path(exists=True), required=False)
@click.help_option("--help", "-h", help=_t().t("cli.help.help_option"))
def info(xml_path: str):
  translator = _t()
  try:
    xml_file = _resolve_xml_path(xml_path)

    console.print(
      f"[bold blue]{translator.t('cli.info.analyzing_file')}[/bold blue] {xml_file}"
    )

    # Get file information
    file_info = get_export_file_info(xml_file)

    if not file_info:
      console.print(f"[bold red]{translator.t('cli.info.failed_analysis')}[/bold red]")
      return

    # Display file information
    table = Table(title=translator.t("cli.info.file_info_title"))
    table.add_column(translator.t("cli.info.column.property"), style="cyan")
    table.add_column(translator.t("cli.info.column.value"), style="magenta")

    table.add_row(translator.t("cli.info.file_path"), str(file_info["file_path"]))
    table.add_row(
      translator.t("cli.info.file_size"),
      translator.t("cli.common.file_size_mb", size=file_info["file_size_mb"]),
    )
    table.add_row(
      translator.t("cli.info.estimated_records"),
      f"{file_info['estimated_record_count']:,}",
    )
    table.add_row(
      translator.t("cli.info.last_modified"),
      str(file_info["last_modified"]),
    )

    console.print(table)

    # Sample parsing for data range and type distribution
    # Algorithm: Stream parse first 100 records to estimate data characteristics
    sample_records = []
    try:
      context = ET.iterparse(xml_file, events=("start", "end"))
      context = iter(context)
      event, root = next(context)

      count = 0
      for event, elem in context:
        if event == "start" and elem.tag == "Record":
          try:
            from src.core.data_models import create_record_from_xml_element

            record, warnings = create_record_from_xml_element(elem)
            if record:
              sample_records.append(record)
              count += 1
              if count >= 100:  # Sample size limit for performance
                break
          except Exception:
            pass  # Skip invalid records in info mode

        if event == "end":
          elem.clear()  # Memory cleanup for large files

      root.clear()

    except Exception as parse_error:
      logger.error(translator.t("log.cli.parse_error", error=parse_error))
      import traceback

      logger.error(
        translator.t(
          "log.cli.traceback",
          error=traceback.format_exc(),
        )
      )
      console.print(
        f"[yellow]{translator.t('cli.common.warning')}: "
        f"{translator.t('cli.info.partial_parse', error=parse_error)}[/yellow]"
      )
      console.print(f"[yellow]{translator.t('cli.info.partial_results')}[/yellow]")

    if sample_records:
      dates = [r.start_date.date() for r in sample_records]
      min_date = min(dates)
      max_date = max(dates)

      console.print(
        f"\n[green]{translator.t('cli.info.sample_range')}:[/green] "
        f"{min_date} {translator.t('cli.info.range_to')} {max_date}"
      )

      # Show record type distribution in sample
      from collections import Counter

      type_counts = Counter(r.type for r in sample_records)

      console.print(f"\n[bold]{translator.t('cli.info.sample_record_types')}:[/bold]")
      for record_type, count in sorted(
        type_counts.items(), key=lambda x: x[1], reverse=True
      )[:10]:
        console.print(f"  {record_type}: {count}")
    else:
      console.print(f"[yellow]{translator.t('cli.info.no_records')}[/yellow]")

  except Exception as e:
    logger.error(
      translator.t("log.cli.info_failed", error=_format_error(e, translator))
    )
    console.print(
      f"[bold red]{translator.t('cli.common.error')}:[/bold red] "
      f"{_format_error(e, translator)}"
    )
    sys.exit(1)


@cli.command(
  help=_t().t("cli.help.export_command"),
  short_help=_t().t("cli.help.export_command_short"),
  add_help_option=False,
)
@click.argument("xml_path", type=click.Path(exists=True), required=False)
@click.option(
  "--output",
  "-o",
  type=click.Path(),
  help=_t().t("cli.help.output_dir"),
)
@click.option(
  "--format",
  "-f",
  type=click.Choice(["csv", "json", "both"]),
  default="both",
  help=_t().t("cli.help.export_format"),
)
@click.option(
  "--types",
  "-t",
  multiple=True,
  help=_t().t("cli.help.export_types"),
)
@click.option(
  "--clean/--no-clean",
  default=True,
  help=_t().t("cli.help.export_clean"),
)
@click.help_option("--help", "-h", help=_t().t("cli.help.help_option"))
def export(
  xml_path: str, output: str | None, format: str, types: list[str], clean: bool
):
  translator = _t()
  try:
    from src.processors.exporter import DataExporter

    xml_file = _resolve_xml_path(xml_path)
    output_dir = Path(output) if output else get_config().output_dir

    console.print(
      f"[bold blue]{translator.t('cli.export.exporting_from')}[/bold blue] {xml_file}"
    )
    console.print(
      f"[bold blue]{translator.t('cli.common.output_dir')}[/bold blue] {output_dir}"
    )
    console.print(
      f"[bold blue]{translator.t('cli.export.export_format')}[/bold blue] {format}"
    )

    if types:
      console.print(
        f"[bold blue]{translator.t('cli.export.record_types')}[/bold blue] "
        f"{', '.join(types)}"
      )

    # Determine export formats
    if format == "both":
      formats = ["csv", "json"]
    else:
      formats = [format]

    # Create exporter
    exporter = DataExporter(output_dir)

    # Perform export
    with UnifiedProgress(
      translator.t("cli.progress.exporting_data"), total=None
    ) as progress:  # Total unknown for export
      export_stats = exporter.export_by_category(
        xml_file,
        formats=formats,
        record_types=list(types) if types else None,
        enable_cleaning=clean,
      )
      progress.update(1, translator.t("cli.progress.export_completed"))

    # Validate export results
    total_exported = sum(
      sum(stats.get(fmt, 0) for fmt in ["csv", "json"])
      for stats in export_stats.values()
    )

    if total_exported == 0:
      console.print(f"[yellow]⚠ {translator.t('cli.export.no_records')}[/yellow]")
      console.print(f"\n{translator.t('cli.export.possible_reasons')}")
      if types:
        console.print(f"• {translator.t('cli.export.reason_types_missing')}")
        console.print(f"• {translator.t('cli.export.reason_use_full_types')}")
        console.print(f"• {translator.t('cli.export.reason_run_info')}")
      else:
        console.print(f"• {translator.t('cli.export.reason_invalid_data')}")
        console.print(f"• {translator.t('cli.export.reason_xml_corrupt')}")
      console.print(f"\n[yellow]{translator.t('cli.export.no_data_saved')}[/yellow]")
      sys.exit(1)

    # Display results
    console.print(f"\n[bold green]✓ {translator.t('cli.export.success')}[/bold green]")

    # Show export summary
    table = Table(title=translator.t("cli.export.summary_title"))
    table.add_column(translator.t("cli.export.column.record_type"), style="cyan")
    table.add_column(
      translator.t("cli.export.column.csv_records"), style="green", justify="right"
    )
    table.add_column(
      translator.t("cli.export.column.json_records"), style="blue", justify="right"
    )
    table.add_column(
      translator.t("cli.export.column.total"), style="magenta", justify="right"
    )

    total_csv = 0
    total_json = 0
    total_records = 0

    for record_type, stats in export_stats.items():
      csv_count = stats.get("csv", 0)
      json_count = stats.get("json", 0)
      type_total = csv_count + json_count

      table.add_row(
        record_type,
        f"{csv_count:,}" if csv_count > 0 else "-",
        f"{json_count:,}" if json_count > 0 else "-",
        f"{type_total:,}",
      )

      total_csv += csv_count
      total_json += json_count
      total_records += type_total

    # Add totals row
    table.add_row(
      f"[bold]{translator.t('cli.export.total')}[/bold]",
      f"[bold]{total_csv:,}[/bold]" if total_csv > 0 else "-",
      f"[bold]{total_json:,}[/bold]" if total_json > 0 else "-",
      f"[bold]{total_records:,}[/bold]",
      end_section=True,
    )

    console.print(table)

    # Show file locations
    console.print(
      f"\n[bold]{translator.t('cli.export.files_saved_to')}[/bold] {output_dir}"
    )
    console.print(
      f"[bold]{translator.t('cli.export.manifest_file')}[/bold] {output_dir}/manifest.json"
    )

    # Show generated files
    if output_dir.exists():
      files = list(output_dir.glob("*"))
      if files:
        console.print(f"\n[bold]{translator.t('cli.export.generated_files')}[/bold]")
        for file_path in sorted(files):
          if file_path.is_file():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            console.print(
              f"  • {file_path.name} "
              f"({translator.t('cli.common.file_size_mb', size=size_mb)})"
            )

  except Exception as e:
    logger.error(
      translator.t("log.cli.export_failed", error=_format_error(e, translator))
    )
    console.print(
      f"[bold red]{translator.t('cli.common.error')}:[/bold red] "
      f"{_format_error(e, translator)}"
    )
    sys.exit(1)


cli.add_command(report_command, name="report")
cli.add_command(visualize_command, name="visualize")


@cli.command(
  help=_t().t("cli.help.benchmark_command"),
  short_help=_t().t("cli.help.benchmark_command_short"),
  add_help_option=False,
)
@click.argument("xml_path", type=click.Path(exists=True), required=False)
@click.option(
  "--output",
  "-o",
  type=click.Path(),
  help=_t().t("cli.help.benchmark_output"),
)
@click.option(
  "--timeout",
  type=int,
  default=30,
  help=_t().t("cli.help.benchmark_timeout"),
)
@click.help_option("--help", "-h", help=_t().t("cli.help.help_option"))
def benchmark(xml_path: str, output: str | None, timeout: int):
  translator = _t()
  try:
    from src.processors.benchmark import run_benchmark

    xml_file = _resolve_xml_path(xml_path)
    output_dir = Path(output) if output else get_config().output_dir

    console.print(
      f"[bold blue]{translator.t('cli.benchmark.running')}[/bold blue] {xml_file}"
    )
    console.print(
      f"[bold blue]{translator.t('cli.common.output_dir')}[/bold blue] {output_dir}"
    )
    console.print(
      f"[bold blue]{translator.t('cli.benchmark.timeout_per_module')}[/bold blue] "
      f"{timeout} {translator.t('cli.benchmark.seconds')}"
    )

    # Run benchmark
    run_benchmark(str(xml_file), str(output_dir), timeout=timeout)

  except Exception as e:
    logger.error(
      translator.t("log.cli.benchmark_failed", error=_format_error(e, translator))
    )
    console.print(
      f"[bold red]{translator.t('cli.common.error')}:[/bold red] "
      f"{_format_error(e, translator)}"
    )
    sys.exit(1)


@cli.command(
  help=_t().t("cli.help.analyze_command"),
  short_help=_t().t("cli.help.analyze_command_short"),
  add_help_option=False,
)
@click.argument("xml_path", type=click.Path(exists=True), required=False)
@click.option(
  "--output",
  "-o",
  type=click.Path(),
  help=_t().t("cli.help.analyze_output"),
)
@click.option(
  "--age",
  type=int,
  help=_t().t("cli.help.analyze_age"),
)
@click.option(
  "--gender",
  type=click.Choice(["male", "female"]),
  help=_t().t("cli.help.analyze_gender"),
)
@click.option(
  "--date-range",
  help=_t().t("cli.help.analyze_date_range"),
)
@click.option(
  "--types",
  "-t",
  multiple=True,
  type=click.Choice(
    ["heart_rate", "sleep", "hrv", "resting_hr", "cardio_fitness", "all"]
  ),
  default=["all"],
  help=_t().t("cli.help.analyze_types"),
)
@click.option(
  "--format",
  "-f",
  type=click.Choice(["json", "text", "both"]),
  default="both",
  help=_t().t("cli.help.analyze_format"),
)
@click.option(
  "--generate-charts/--no-charts",
  default=True,
  help=_t().t("cli.help.analyze_charts"),
)
@click.help_option("--help", "-h", help=_t().t("cli.help.help_option"))
def analyze(
  xml_path: str,
  output: str | None,
  age: int | None,
  gender: str | None,
  date_range: str | None,
  types: list[str],
  format: str,
  generate_charts: bool,
):
  """Analyze heart rate and sleep data with comprehensive insights.

  XML_PATH: Path to the export.xml file

  Examples:\n
      # Basic analysis with age and gender for cardio fitness\n
      uv run python main.py analyze export.xml --age 30 --gender male

      # Analyze specific data types\n
      uv run python main.py analyze export.xml --types heart_rate --types sleep

      # Analyze specific date range\n
      uv run python main.py analyze export.xml --date-range 2024-01-01:2024-12-31

      # Analysis with custom output directory\n
      uv run python main.py analyze export.xml --output ./analysis_results
  """
  translator = _t()
  try:
    from src.analyzers.highlights import HighlightsGenerator
    from src.processors.heart_rate import HeartRateAnalyzer
    from src.processors.sleep import SleepAnalyzer

    xml_file = _resolve_xml_path(xml_path)
    output_dir = Path(output) if output else get_config().output_dir

    console.print(
      f"[bold blue]{translator.t('cli.analyze.analyzing_from')}[/bold blue] {xml_file}"
    )
    console.print(
      f"[bold blue]{translator.t('cli.common.output_dir')}[/bold blue] {output_dir}"
    )

    # Validate parameters
    if "cardio_fitness" in types or "all" in types:
      if not age or not gender:
        console.print(
          f"[bold red]{translator.t('cli.common.error')}:[/bold red] "
          f"{translator.t('cli.analyze.age_gender_required')}"
        )
        console.print(translator.t("cli.analyze.use_age_gender"))
        sys.exit(1)

    # Parse date range
    start_date = None
    end_date = None
    if date_range:
      try:
        start_str, end_str = date_range.split(":")
        start_date = datetime.fromisoformat(start_str)
        end_date = datetime.fromisoformat(end_str)
        console.print(
          f"[bold blue]{translator.t('cli.analyze.date_range')}[/bold blue] "
          f"{start_date.date()} {translator.t('cli.info.range_to')} {end_date.date()}"
        )
      except ValueError:
        console.print(
          f"[bold red]{translator.t('cli.common.error')}:[/bold red] "
          f"{translator.t('cli.analyze.invalid_date_range')}"
        )
        sys.exit(1)

    # Determine analysis types
    if "all" in types:
      analysis_types = [
        "heart_rate",
        "sleep",
        "hrv",
        "resting_hr",
        "cardio_fitness",
      ]
    else:
      analysis_types = types

    console.print(
      f"[bold blue]{translator.t('cli.analyze.types')}[/bold blue] "
      f"{', '.join(analysis_types)}"
    )

    # Initialize analyzers
    gender_value: Literal["male", "female"] | None
    if gender in ("male", "female"):
      gender_value = cast(Literal["male", "female"], gender)
    else:
      gender_value = None

    heart_rate_analyzer = HeartRateAnalyzer(
      age=age,
      gender=gender_value,
    )
    sleep_analyzer = SleepAnalyzer()
    highlights_generator = HighlightsGenerator()

    # Parse data with progress
    console.print(f"\n[bold]{translator.t('cli.analyze.step_parse')}[/bold]")
    with UnifiedProgress(
      translator.t("cli.progress.parsing_health_data"), total=None
    ) as progress:
      hr_records = []
      resting_hr_records = []
      hrv_records = []
      vo2_max_records = []
      sleep_records = []

      record_types = []
      if any(
        t in analysis_types
        for t in ["heart_rate", "resting_hr", "hrv", "cardio_fitness"]
      ):
        record_types.append(HEART_RATE_TYPE)
        if "resting_hr" in analysis_types:
          record_types.append(RESTING_HR_TYPE)
        if "hrv" in analysis_types:
          record_types.append(HRV_TYPE)
        if "cardio_fitness" in analysis_types:
          record_types.append(VO2_MAX_TYPE)

      if "sleep" in analysis_types:
        record_types.append(SLEEP_TYPE)

      if record_types:
        parser = StreamingXMLParser(xml_file)
        all_records = list(parser.parse_records(record_types=record_types))

        categorized = categorize_records(all_records)
        hr_records = categorized["heart_rate"]
        resting_hr_records = categorized["resting_hr"]
        hrv_records = categorized["hrv"]
        vo2_max_records = categorized["vo2_max"]
        sleep_records = categorized["sleep"]

      # Filter by date range if specified
      if start_date and end_date:

        def filter_by_date(records):
          return [r for r in records if start_date <= r.start_date <= end_date]

        hr_records = filter_by_date(hr_records)
        resting_hr_records = filter_by_date(resting_hr_records)
        hrv_records = filter_by_date(hrv_records)
        vo2_max_records = filter_by_date(vo2_max_records)
        sleep_records = filter_by_date(sleep_records)

      progress.update(
        1,
        translator.t(
          "cli.analyze.parsed_records",
          heart_rate=len(hr_records),
          sleep=len(sleep_records),
        ),
      )

    console.print(
      f"[green]✓ {translator.t('cli.analyze.parsed_records', heart_rate=len(hr_records), sleep=len(sleep_records))}[/green]"
    )

    # Perform analyses
    heart_rate_report = None
    sleep_report = None

    # Heart rate analysis
    if any(
      t in analysis_types for t in ["heart_rate", "resting_hr", "hrv", "cardio_fitness"]
    ):
      console.print(f"\n[bold]{translator.t('cli.analyze.step_hr')}[/bold]")
      with UnifiedProgress(
        translator.t("cli.progress.analyzing_hr_data"), total=None
      ) as progress:
        heart_rate_report = heart_rate_analyzer.analyze_comprehensive(
          heart_rate_records=hr_records,
          resting_hr_records=resting_hr_records
          if "resting_hr" in analysis_types
          else None,
          hrv_records=hrv_records if "hrv" in analysis_types else None,
          vo2_max_records=vo2_max_records
          if "cardio_fitness" in analysis_types
          else None,
        )
        progress.update(1, translator.t("cli.analyze.hr_completed"))
      console.print(f"[green]✓ {translator.t('cli.analyze.hr_completed')}[/green]")

    # Sleep analysis
    if "sleep" in analysis_types and sleep_records:
      console.print(f"\n[bold]{translator.t('cli.analyze.step_sleep')}[/bold]")
      with UnifiedProgress(
        translator.t("cli.progress.analyzing_sleep_data"), total=None
      ) as progress:
        sleep_report = sleep_analyzer.analyze_comprehensive(
          cast(list[HealthRecord], sleep_records)
        )
        progress.update(1, translator.t("cli.analyze.sleep_completed"))
      console.print(f"[green]✓ {translator.t('cli.analyze.sleep_completed')}[/green]")

    # Generate highlights
    console.print(f"\n[bold]{translator.t('cli.analyze.step_insights')}[/bold]")
    with UnifiedProgress(
      translator.t("cli.progress.generating_insights"), total=None
    ) as progress:
      highlights = highlights_generator.generate_comprehensive_highlights(
        heart_rate_report=heart_rate_report,
        sleep_report=sleep_report,
      )
      progress.update(
        1,
        translator.t(
          "cli.analyze.generated_insights",
          insights=len(highlights.insights),
          recommendations=len(highlights.recommendations),
        ),
      )
    console.print(
      f"[green]✓ {translator.t('cli.analyze.generated_insights', insights=len(highlights.insights), recommendations=len(highlights.recommendations))}[/green]"
    )

    # Display results
    console.print(
      f"\n[bold green]🎯 {translator.t('cli.analyze.results_title')}[/bold green]"
    )

    # Display heart rate results
    if heart_rate_report:
      _display_heart_rate_results(heart_rate_report)

    # Display sleep results
    if sleep_report:
      _display_sleep_results(sleep_report)

    # Display highlights
    _display_highlights(highlights)

    # Save results
    if format in ["json", "both"]:
      _save_analysis_results_json(
        output_dir, heart_rate_report, sleep_report, highlights
      )

    if format in ["text", "both"]:
      _save_analysis_results_text(
        output_dir, heart_rate_report, sleep_report, highlights
      )

    console.print(
      f"\n[bold green]✓ {translator.t('cli.analyze.completed', path=output_dir)}[/bold green]"
    )

  except Exception as e:
    logger.error(
      translator.t("log.cli.analyze_failed", error=_format_error(e, translator))
    )
    console.print(
      f"[bold red]{translator.t('cli.common.error')}:[/bold red] "
      f"{_format_error(e, translator)}"
    )
    sys.exit(1)


def _display_parsing_results(stats: dict):
  """Display parsing results in a formatted table."""
  translator = _t()
  table = Table(title=translator.t("cli.parse.results_title"))
  table.add_column(translator.t("cli.parse.column.metric"), style="cyan")
  table.add_column(
    translator.t("cli.parse.column.value"), style="magenta", justify="right"
  )

  table.add_row(translator.t("cli.parse.total_records"), f"{stats['total_records']:,}")
  table.add_row(
    translator.t("cli.parse.processed_records"), f"{stats['processed_records']:,}"
  )
  table.add_row(
    translator.t("cli.parse.skipped_records"), f"{stats['skipped_records']:,}"
  )
  table.add_row(
    translator.t("cli.parse.invalid_records"), f"{stats['invalid_records']:,}"
  )
  table.add_row(translator.t("cli.parse.success_rate"), f"{stats['success_rate']:.1%}")

  if stats["date_range"]["start"] and stats["date_range"]["end"]:
    table.add_row(
      translator.t("cli.parse.date_range"),
      f"{stats['date_range']['start']} {translator.t('cli.info.range_to')} {stats['date_range']['end']}",
    )

  console.print(table)

  # Show top record types
  if stats["record_types"]:
    console.print(f"\n[bold]{translator.t('cli.parse.top_record_types')}:[/bold]")
    sorted_types = sorted(
      stats["record_types"].items(), key=lambda x: x[1], reverse=True
    )
    for i, (record_type, count) in enumerate(sorted_types[:10]):
      console.print(f"  {i + 1:2d}. {record_type}: {count:,}")

  # Show top sources
  if stats["sources"]:
    console.print(f"\n[bold]{translator.t('cli.parse.top_sources')}:[/bold]")
    sorted_sources = sorted(stats["sources"].items(), key=lambda x: x[1], reverse=True)
    for i, (source, count) in enumerate(sorted_sources[:5]):
      console.print(f"  {i + 1:2d}. {source}: {count:,}")


def _display_records_preview(records: list):
  """Display a preview of parsed records."""
  if not records:
    return

  translator = _t()
  table = Table(title=translator.t("cli.parse.records_preview"))
  table.add_column(translator.t("cli.parse.column.type"), style="cyan")
  table.add_column(translator.t("cli.parse.column.source"), style="green")
  table.add_column(translator.t("cli.parse.column.start_date"), style="yellow")
  table.add_column(translator.t("cli.parse.column.value_label"), style="magenta")

  for record in records[:10]:
    value_str = ""
    if hasattr(record, "value"):
      if isinstance(record.value, float):
        value_str = f"{record.value:.2f}"
      else:
        value_str = str(record.value)
    elif hasattr(record, "sleep_stage"):
      value_str = record.sleep_stage.value

    table.add_row(
      record.type.split(".")[-1],  # Shorten type name
      record.source_name,
      record.start_date.strftime("%Y-%m-%d %H:%M"),
      value_str,
    )

  console.print(table)


def _save_parsed_data(records: list, stats: dict, output_dir: Path):
  """Save parsed data to output directory."""
  from src.processors.exporter import DataExporter

  output_dir.mkdir(parents=True, exist_ok=True)

  translator = _t()
  if not records:
    console.print(f"[yellow]{translator.t('cli.parse.no_records_to_save')}[/yellow]")
    return

  console.print(
    f"[bold blue]{translator.t('cli.parse.saving_records', count=len(records), path=output_dir)}[/bold blue]"
  )

  # Validate data quality before saving
  console.print(f"[bold]{translator.t('cli.parse.validating_data')}[/bold]")
  try:
    from src.processors.validator import validate_health_data

    validation_result = validate_health_data(records)
    validation_summary = validation_result.get_summary()

    console.print(f"[green]✓ {translator.t('cli.parse.validation_completed')}[/green]")
    console.print(
      f"  {translator.t('cli.parse.validation_quality')}: {validation_summary['quality_score']:.1%}"
    )
    console.print(
      f"  {translator.t('cli.parse.validation_warnings')}: {validation_summary['total_warnings']}"
    )
    console.print(
      f"  {translator.t('cli.parse.validation_outliers')}: {validation_summary['outliers_count']}"
    )

    # Save validation report
    validation_file = output_dir / "data_validation_report.json"
    with open(validation_file, "w", encoding="utf-8") as f:
      import json

      json.dump(
        {
          "validation_summary": validation_summary,
          "errors": validation_result.errors[:10],  # First 10 errors
          "warnings": validation_result.warnings[:10],  # First 10 warnings
          "outliers": validation_result.outliers_detected[:10],  # First 10 outliers
          "issues_by_type": validation_result.issues_by_type,
          "consistency_checks": validation_result.consistency_checks,
        },
        f,
        indent=2,
        ensure_ascii=False,
        default=str,
      )

    console.print(
      f"[green]✓ {translator.t('cli.parse.validation_report_saved', path=validation_file)}[/green]"
    )

  except Exception as e:
    logger.warning(
      translator.t("log.cli.validation_failed", error=_format_error(e, translator))
    )
    console.print(
      f"[yellow]⚠ {translator.t('cli.parse.validation_skipped', error=_format_error(e, translator))}[/yellow]"
    )

  # Group records by type for organized saving
  records_by_type = {}
  for record in records:
    record_type = getattr(record, "type", "Unknown")
    if record_type not in records_by_type:
      records_by_type[record_type] = []
    records_by_type[record_type].append(record)

  # Create exporter
  exporter = DataExporter(output_dir)

  total_saved = 0
  total_files = 0

  # Save each record type to separate files
  for record_type, type_records in records_by_type.items():
    if not type_records:
      continue

    # Clean record type name for filename
    clean_type = record_type.replace("HKQuantityTypeIdentifier", "").replace(
      "HKCategoryTypeIdentifier", ""
    )

    # Save to both CSV and JSON formats
    csv_path = output_dir / f"{clean_type}.csv"
    json_path = output_dir / f"{clean_type}.json"

    csv_count = exporter.export_to_csv(type_records, csv_path)
    json_count = exporter.export_to_json(type_records, json_path)

    if csv_count > 0:
      total_saved += csv_count
      total_files += 1
      csv_size = csv_path.stat().st_size
      console.print(
        f"  [green]✓ {clean_type}.csv:[/green] {csv_count:,} "
        f"{translator.t('cli.parse.records_unit')} ({csv_size:,} {translator.t('cli.parse.bytes_unit')})"
      )

    if json_count > 0:
      total_files += 1
      json_size = json_path.stat().st_size
      console.print(
        f"  [green]✓ {clean_type}.json:[/green] {json_count:,} "
        f"{translator.t('cli.parse.records_unit')} ({json_size:,} {translator.t('cli.parse.bytes_unit')})"
      )

  # Save parsing statistics
  stats_file = output_dir / "parsing_stats.json"
  with open(stats_file, "w", encoding="utf-8") as f:
    import json

    json.dump(stats, f, indent=2, ensure_ascii=False, default=str)

  console.print(f"  [green]✓ {translator.t('cli.parse.stats_saved')}[/green]")

  # Create a simple manifest
  manifest_file = output_dir / "data_manifest.txt"
  with open(manifest_file, "w", encoding="utf-8") as f:
    f.write(f"{translator.t('cli.parse.manifest_title')}\n")
    f.write("=" * 40 + "\n\n")
    f.write(
      f"{translator.t('cli.parse.manifest_export_date')}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    f.write(f"{translator.t('cli.parse.manifest_total_records')}: {total_saved:,}\n")
    f.write(
      f"{translator.t('cli.parse.manifest_total_files')}: {total_files + 1}\n"
    )  # +1 for stats file
    f.write(
      f"{translator.t('cli.parse.manifest_record_types')}: {len(records_by_type)}\n\n"
    )

    f.write(f"{translator.t('cli.parse.manifest_files_generated')}:\n")
    f.write("-" * 20 + "\n")
    for record_type, type_records in records_by_type.items():
      clean_type = record_type.replace("HKQuantityTypeIdentifier", "").replace(
        "HKCategoryTypeIdentifier", ""
      )
      f.write(
        f"• {clean_type}.csv ({len(type_records)} {translator.t('cli.parse.records_unit')})\n"
      )
      f.write(
        f"• {clean_type}.json ({len(type_records)} {translator.t('cli.parse.records_unit')})\n"
      )

    f.write(f"• {translator.t('cli.parse.manifest_statistics_file')}\n")
    f.write(f"• {translator.t('cli.parse.manifest_manifest_file')}\n")

  console.print(f"  [green]✓ {translator.t('cli.parse.manifest_saved')}[/green]")
  console.print(
    f"[bold green]✓ {translator.t('cli.parse.saved_records_summary', records=f'{total_saved:,}', files=total_files + 1)}[/bold green]"
  )


def _display_heart_rate_results(report):
  """Display heart rate analysis results."""
  translator = _t()
  console.print(f"\n[bold blue]❤️ {translator.t('cli.analyze.hr_section')}[/bold blue]")

  if report.resting_hr_analysis:
    resting = report.resting_hr_analysis
    console.print(
      f"  [cyan]{translator.t('cli.analyze.resting_hr')}:[/cyan] {resting.current_value:.1f} bpm"
    )
    console.print(
      f"  [cyan]{translator.t('cli.analyze.trend')}:[/cyan] {resting.trend_direction}"
    )
    console.print(
      f"  [cyan]{translator.t('cli.analyze.health_rating')}:[/cyan] {resting.health_rating}"
    )

  if report.hrv_analysis:
    hrv = report.hrv_analysis
    console.print(
      f"  [cyan]{translator.t('cli.analyze.hrv')}:[/cyan] {hrv.current_sdnn:.1f} ms"
    )
    console.print(
      f"  [cyan]{translator.t('cli.analyze.stress_level')}:[/cyan] {hrv.stress_level}"
    )
    console.print(
      f"  [cyan]{translator.t('cli.analyze.recovery_status')}:[/cyan] {hrv.recovery_status}"
    )

  if report.cardio_fitness:
    cardio = report.cardio_fitness
    console.print(
      f"  [cyan]{translator.t('cli.analyze.vo2_max')}:[/cyan] {cardio.current_vo2_max:.1f} ml/min·kg"
    )
    console.print(
      f"  [cyan]{translator.t('cli.analyze.fitness_rating')}:[/cyan] {cardio.age_adjusted_rating}"
    )

  console.print(
    f"  [cyan]{translator.t('cli.analyze.data_quality')}:[/cyan] {report.data_quality_score:.1%}"
  )
  console.print(
    f"  [cyan]{translator.t('cli.analyze.total_records')}:[/cyan] {report.record_count:,}"
  )


def _display_sleep_results(report):
  """Display sleep analysis results."""
  translator = _t()
  console.print(
    f"\n[bold blue]😴 {translator.t('cli.analyze.sleep_section')}[/bold blue]"
  )

  if report.quality_metrics:
    quality = report.quality_metrics
    console.print(
      f"  [cyan]{translator.t('cli.analyze.avg_duration')}:[/cyan] {quality.average_duration:.1f} hours"
    )
    console.print(
      f"  [cyan]{translator.t('cli.analyze.avg_efficiency')}:[/cyan] {quality.average_efficiency:.1%}"
    )
    console.print(
      f"  [cyan]{translator.t('cli.analyze.consistency_score')}:[/cyan] {quality.consistency_score:.1%}"
    )

  console.print(
    f"  [cyan]{translator.t('cli.analyze.data_quality')}:[/cyan] {report.data_quality_score:.1%}"
  )
  console.print(
    f"  [cyan]{translator.t('cli.analyze.total_records')}:[/cyan] {report.record_count:,}"
  )


def _display_highlights(highlights):
  """Display health highlights and recommendations."""
  translator = _t()
  console.print(
    f"\n[bold blue]💡 {translator.t('cli.analyze.insights_title')}[/bold blue]"
  )

  # Display insights
  if highlights.insights:
    console.print(f"\n[bold]{translator.t('cli.analyze.key_insights')}:[/bold]")
    for i, insight in enumerate(highlights.insights[:5], 1):  # Show top 5
      priority_colors = {"high": "red", "medium": "yellow", "low": "green"}
      color = priority_colors.get(insight.priority, "white")
      console.print(f"  {i}. [{color}]{insight.title}[/{color}]")
      console.print(f"     {insight.message}")

  # Display recommendations
  if highlights.recommendations:
    console.print(f"\n[bold]{translator.t('cli.analyze.recommendations')}:[/bold]")
    for i, rec in enumerate(highlights.recommendations[:3], 1):  # Show top 3
      console.print(f"  {i}. {rec}")


def _save_analysis_results_json(
  output_dir: Path, heart_rate_report, sleep_report, highlights
):
  """Save analysis results to JSON format."""
  import json

  output_dir.mkdir(parents=True, exist_ok=True)

  results = {
    "analysis_date": datetime.now().isoformat(),
    "heart_rate": _report_to_dict(heart_rate_report) if heart_rate_report else None,
    "sleep": _report_to_dict(sleep_report) if sleep_report else None,
    "highlights": {
      "insights": [
        {
          "category": i.category,
          "priority": i.priority,
          "title": i.title,
          "message": i.message,
          "confidence": i.confidence,
        }
        for i in highlights.insights
      ],
      "recommendations": highlights.recommendations,
      "summary": highlights.summary,
    },
  }

  output_file = output_dir / "analysis_results.json"
  with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)

  translator = _t()
  console.print(
    f"[green]✓ {translator.t('cli.analyze.json_saved', path=output_file)}[/green]"
  )


def _save_analysis_results_text(
  output_dir: Path, heart_rate_report, sleep_report, highlights
):
  """Save analysis results to text format."""
  output_dir.mkdir(parents=True, exist_ok=True)

  output_file = output_dir / "analysis_results.txt"

  with open(output_file, "w", encoding="utf-8") as f:
    translator = _t()
    f.write(f"{translator.t('cli.analyze.report_title')}\n")
    f.write("=" * 50 + "\n\n")
    f.write(
      f"{translator.t('cli.analyze.analysis_date')}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )

    if heart_rate_report:
      f.write(f"{translator.t('cli.analyze.heart_rate_section')}\n")
      f.write("-" * 20 + "\n")
      if heart_rate_report.resting_hr_analysis:
        resting = heart_rate_report.resting_hr_analysis
        f.write(
          f"{translator.t('cli.analyze.resting_hr')}: {resting.current_value:.1f} bpm\n"
        )
        f.write(f"{translator.t('cli.analyze.trend')}: {resting.trend_direction}\n")
        f.write(
          f"{translator.t('cli.analyze.health_rating')}: {resting.health_rating}\n"
        )
      if heart_rate_report.hrv_analysis:
        hrv = heart_rate_report.hrv_analysis
        f.write(f"{translator.t('cli.analyze.hrv')}: {hrv.current_sdnn:.1f} ms\n")
        f.write(f"{translator.t('cli.analyze.stress_level')}: {hrv.stress_level}\n")
      if heart_rate_report.cardio_fitness:
        cardio = heart_rate_report.cardio_fitness
        f.write(
          f"{translator.t('cli.analyze.vo2_max')}: {cardio.current_vo2_max:.1f} ml/min·kg\n"
        )
        f.write(
          f"{translator.t('cli.analyze.fitness_rating')}: {cardio.age_adjusted_rating}\n"
        )
      f.write(
        f"{translator.t('cli.analyze.data_quality')}: {heart_rate_report.data_quality_score:.1%}\n"
      )
      f.write(
        f"{translator.t('cli.analyze.total_records')}: {heart_rate_report.record_count:,}\n\n"
      )

    if sleep_report:
      f.write(f"{translator.t('cli.analyze.sleep_section_upper')}\n")
      f.write("-" * 15 + "\n")
      if sleep_report.quality_metrics:
        quality = sleep_report.quality_metrics
        f.write(
          f"{translator.t('cli.analyze.avg_duration')}: {quality.average_duration:.1f} hours\n"
        )
        f.write(
          f"{translator.t('cli.analyze.avg_efficiency')}: {quality.average_efficiency:.1%}\n"
        )
        f.write(
          f"{translator.t('cli.analyze.consistency_score')}: {quality.consistency_score:.1%}\n"
        )
      f.write(
        f"{translator.t('cli.analyze.data_quality')}: {sleep_report.data_quality_score:.1%}\n"
      )
      f.write(
        f"{translator.t('cli.analyze.total_records')}: {sleep_report.record_count:,}\n\n"
      )

    if highlights.insights:
      f.write(f"{translator.t('cli.analyze.key_insights_upper')}\n")
      f.write("-" * 12 + "\n")
      for i, insight in enumerate(highlights.insights[:5], 1):
        f.write(f"{i}. {insight.title}\n")
        f.write(f"   {insight.message}\n\n")

    if highlights.recommendations:
      f.write(f"{translator.t('cli.analyze.recommendations_upper')}\n")
      f.write("-" * 15 + "\n")
      for i, rec in enumerate(highlights.recommendations, 1):
        f.write(f"{i}. {rec}\n")

  translator = _t()
  console.print(
    f"[green]✓ {translator.t('cli.analyze.text_saved', path=output_file)}[/green]"
  )


def _report_to_dict(report):
  """Convert report object to dictionary for JSON serialization."""
  if report is None:
    return None

  # Simple conversion - in production, you'd want more sophisticated serialization
  analysis_date = getattr(report, "analysis_date", None)
  data_range = getattr(report, "data_range", None)
  return {
    "analysis_date": analysis_date.isoformat()
    if isinstance(analysis_date, datetime)
    else None,
    "data_range": [
      data_range[0].isoformat(),
      data_range[1].isoformat(),
    ]
    if isinstance(data_range, tuple)
    and len(data_range) == 2
    and all(isinstance(item, datetime) for item in data_range)
    else None,
    "record_count": getattr(report, "record_count", 0),
    "data_quality_score": getattr(report, "data_quality_score", 0.0),
  }


def _ensure_xml_file(xml_file: Path) -> None:
  """Validate XML file existence and size."""
  if not xml_file.exists():
    translator = _t()
    console.print(
      f"[bold red]{translator.t('cli.common.error')}:[/bold red] "
      f"{translator.t('cli.parse.file_not_found', path=xml_file)}"
    )
    console.print(
      f"[yellow]{translator.t('cli.common.tip')}:[/yellow] "
      f"{translator.t('cli.parse.check_file_path')}"
    )
    sys.exit(1)

  if xml_file.stat().st_size == 0:
    translator = _t()
    console.print(
      f"[bold red]{translator.t('cli.common.error')}:[/bold red] "
      f"{translator.t('cli.parse.file_empty', path=xml_file)}"
    )
    sys.exit(1)


def _display_file_info(xml_file: Path) -> None:
  """Display export file information with guardrails."""
  translator = _t()
  try:
    file_info = get_export_file_info(xml_file)
    if file_info:
      console.print(
        f"[green]{translator.t('cli.info.file_size')}:[/green] "
        f"{translator.t('cli.common.file_size_mb', size=file_info['file_size_mb'])}"
      )
      console.print(
        f"[green]{translator.t('cli.info.estimated_records')}:[/green] "
        f"{file_info['estimated_record_count']:,}"
      )
    else:
      console.print(
        f"[yellow]{translator.t('cli.common.warning')}:[/yellow] "
        f"{translator.t('cli.info.cannot_read_file_info')}"
      )
  except Exception as e:
    console.print(
      f"[yellow]{translator.t('cli.common.warning')}:[/yellow] "
      f"{translator.t('cli.info.cannot_analyze_file', error=_format_error(e, translator))}"
    )
    console.print(f"[yellow]{translator.t('cli.info.continuing_parse')}[/yellow]")


def _init_parser(xml_file: Path) -> StreamingXMLParser:
  """Initialize the streaming parser with error handling."""
  try:
    return StreamingXMLParser(xml_file)
  except Exception as e:
    translator = _t()
    console.print(
      f"[bold red]{translator.t('cli.common.error')}:[/bold red] "
      f"{translator.t('cli.parse.init_parser_failed', error=_format_error(e, translator))}"
    )
    console.print(
      f"[yellow]{translator.t('cli.common.tip')}:[/yellow] "
      f"{translator.t('cli.parse.invalid_xml_tip')}"
    )
    sys.exit(1)


def _parse_records_with_progress(
  parser: StreamingXMLParser, record_types: list[str] | None
) -> tuple[list, dict]:
  """Parse records with CLI progress feedback."""
  with UnifiedProgress(
    _t().t("cli.parse.parsing_records"),
    total=None,  # Always use indeterminate progress for parsing
    quiet=True,  # Disable logging for cleaner CLI output
  ) as progress:

    def update_progress(count: int) -> None:
      progress.update(
        count,
        _t().t("cli.parse.parsed_records", count=f"{count:,}"),
      )

    records_generator = parser.parse_records(
      record_types=record_types,
      progress_callback=update_progress,
      quiet=True,
    )
    records = list(records_generator)
    stats = parser.get_statistics()

    progress.update(
      len(records),
      _t().t("cli.parse.parsed_records", count=f"{len(records):,}"),
    )

  return records, stats


def _handle_parsing_exception(error: Exception) -> None:
  """Handle parsing errors with user-friendly messages."""
  translator = _t()
  logger.error(
    translator.t("log.cli.parse_failed", error=_format_error(error, translator))
  )

  error_str = str(error).lower()
  if "memory" in error_str:
    console.print(
      f"[bold red]{translator.t('cli.common.error')}:[/bold red] "
      f"{translator.t('cli.parse.memory_error')}"
    )
    console.print(
      f"[yellow]{translator.t('cli.common.tip')}:[/yellow] "
      f"{translator.t('cli.parse.memory_tip')}"
    )
  elif "permission" in error_str:
    console.print(
      f"[bold red]{translator.t('cli.common.error')}:[/bold red] "
      f"{translator.t('cli.parse.permission_error')}"
    )
    console.print(
      f"[yellow]{translator.t('cli.common.tip')}:[/yellow] "
      f"{translator.t('cli.parse.permission_tip')}"
    )
  elif "encoding" in error_str:
    console.print(
      f"[bold red]{translator.t('cli.common.error')}:[/bold red] "
      f"{translator.t('cli.parse.encoding_error')}"
    )
    console.print(
      f"[yellow]{translator.t('cli.common.tip')}:[/yellow] "
      f"{translator.t('cli.parse.encoding_tip')}"
    )
  else:
    console.print(
      f"[bold red]{translator.t('cli.common.error')}:[/bold red] "
      f"{translator.t('cli.parse.failed_to_parse', error=_format_error(error, translator))}"
    )
    console.print(
      f"[yellow]{translator.t('cli.common.tip')}:[/yellow] "
      f"{translator.t('cli.parse.check_xml_format')}"
    )

  sys.exit(1)


def _print_parsing_success(stats: dict) -> None:
  """Print parsing summary and success indicator."""
  success_rate = stats.get("success_rate", 0)
  if success_rate >= 0.95:
    translator = _t()
    console.print(
      f"[bold green]✓ {translator.t('cli.parse.completed_success')}[/bold green]"
    )
  elif success_rate >= 0.80:
    translator = _t()
    console.print(
      f"[bold yellow]⚠ {translator.t('cli.parse.completed_minor')}[/bold yellow]"
    )
  else:
    translator = _t()
    console.print(
      f"[bold yellow]⚠ {translator.t('cli.parse.completed_loss')}[/bold yellow]"
    )

  translator = _t()
  processed_count = f"{stats.get('processed_records', 0):,}"
  success_rate_str = f"{success_rate:.1%}"
  console.print(
    f"[green]{translator.t('cli.parse.processed_summary', processed=processed_count, success_rate=success_rate_str)}[/green]"
  )


def main():
  """Main entry point for the CLI."""
  try:
    cli()
  except KeyboardInterrupt:
    translator = _t()
    console.print(f"\n[bold yellow]{translator.t('cli.main.cancelled')}[/bold yellow]")
    sys.exit(1)
  except HealthAnalyzerError as e:
    translator = _t()
    console.print(
      f"[bold red]{translator.t('cli.main.health_error')}:[/bold red] "
      f"{_format_error(e, translator)}"
    )
    sys.exit(1)
  except Exception as e:
    translator = _t()
    logger.exception(translator.t("log.cli.unexpected_error"))
    console.print(
      f"[bold red]{translator.t('cli.common.unexpected_error')}:[/bold red] "
      f"{_format_error(e, translator)}"
    )
    sys.exit(1)


if __name__ == "__main__":
  main()
