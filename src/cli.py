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
from src.core.exceptions import HealthAnalyzerError
from src.core.data_models import HealthRecord
from src.core.xml_parser import StreamingXMLParser, get_export_file_info
from src.utils.logger import UnifiedProgress, get_logger
from src.utils.record_categorizer import (
  categorize_records,
  HEART_RATE_TYPE,
  RESTING_HR_TYPE,
  HRV_TYPE,
  VO2_MAX_TYPE,
  SLEEP_TYPE,
)

console = Console()
logger = get_logger(__name__)


@click.group()
@click.option(
  "--config",
  "config_path",
  type=click.Path(exists=True),
  help="Path to configuration file",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.version_option()
def cli(config_path: str | None, verbose: bool):
  """Apple Health Data Analyzer

  A comprehensive tool for parsing, analyzing, and visualizing Apple Health export data.
  """
  if config_path:
    import os

    os.environ["CONFIG_FILE"] = config_path
    reload_config()

  if verbose:
    import logging

    logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.argument("xml_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output directory for results")
@click.option(
  "--types",
  "-t",
  multiple=True,
  help="Record types to parse (can specify multiple)",
)
@click.option("--preview", is_flag=True, help="Show preview of parsed data")
def parse(xml_path: str, output: str | None, types: list[str], preview: bool):
  """Parse Apple Health export XML file.

  XML_PATH: Path to the export.xml file

  Examples:\n
      # Parse all records and save to default output directory\n
      uv run python main.py parse export.xml --output ./parsed_data

      # Parse only heart rate data with preview\n
      uv run python main.py parse export.xml --types HeartRate --preview

      # Parse specific record types\n
      uv run python main.py parse export.xml --types HeartRate --types Sleep
  """
  try:
    xml_file = Path(xml_path)

    # Validate input file
    _ensure_xml_file(xml_file)

    output_dir = Path(output) if output else get_config().output_dir

    console.print(f"[bold blue]Parsing Apple Health export:[/bold blue] {xml_file}")
    console.print(f"[bold blue]Output directory:[/bold blue] {output_dir}")

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
      console.print("[yellow]Warning:[/yellow] No records were parsed from the file")
      if record_types:
        console.print(
          f"[yellow]Tip:[/yellow] Check if record types {record_types} exist in the file"
        )
      else:
        console.print(
          "[yellow]Tip:[/yellow] Verify the file contains valid Apple Health data"
        )
      sys.exit(1)

    # Display results
    _display_parsing_results(stats)

    if preview and records:
      console.print("\n[bold]Data Preview:[/bold]")
      _display_records_preview(records[:10])

    # Save results if output specified
    if output:
      try:
        _save_parsed_data(records, stats, output_dir)
      except Exception as e:
        logger.error(f"Failed to save data: {e}")
        console.print(f"[bold red]Error:[/bold red] Failed to save data: {e}")
        console.print(
          f"[yellow]Tip:[/yellow] Check write permissions for directory: {output_dir}"
        )
        sys.exit(1)

    _print_parsing_success(stats)

  except KeyboardInterrupt:
    console.print("\n[yellow]Operation cancelled by user[/yellow]")
    sys.exit(1)
  except Exception as e:
    logger.error(f"Unexpected error during parsing: {e}")
    console.print(f"[bold red]Unexpected error:[/bold red] {e}")
    console.print(
      "[yellow]Tip:[/yellow] Please report this issue with the error details"
    )
    sys.exit(1)


@cli.command()
@click.argument("xml_path", type=click.Path(exists=True))
def info(xml_path: str):
  """Get information about an Apple Health export file.

  XML_PATH: Path to the export.xml file
  """
  try:
    xml_file = Path(xml_path)

    console.print(f"[bold blue]Analyzing file:[/bold blue] {xml_file}")

    # Get file information
    file_info = get_export_file_info(xml_file)

    if not file_info:
      console.print("[bold red]Failed to analyze file[/bold red]")
      return

    # Display file information
    table = Table(title="File Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("File Path", str(file_info["file_path"]))
    table.add_row("File Size", f"{file_info['file_size_mb']:.2f} MB")
    table.add_row("Estimated Records", f"{file_info['estimated_record_count']:,}")
    table.add_row("Last Modified", str(file_info["last_modified"]))

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
      logger.error(f"Error during parsing: {parse_error}")
      import traceback

      logger.error(f"Traceback: {traceback.format_exc()}")
      console.print(
        f"[yellow]Warning: Could not parse all records: {parse_error}[/yellow]"
      )
      console.print("[yellow]Showing partial results...[/yellow]")

    if sample_records:
      dates = [r.start_date.date() for r in sample_records]
      min_date = min(dates)
      max_date = max(dates)

      console.print(
        f"\n[green]Data date range (sample):[/green] {min_date} to {max_date}"
      )

      # Show record type distribution in sample
      from collections import Counter

      type_counts = Counter(r.type for r in sample_records)

      console.print("\n[bold]Record types in sample:[/bold]")
      for record_type, count in sorted(
        type_counts.items(), key=lambda x: x[1], reverse=True
      )[:10]:
        console.print(f"  {record_type}: {count}")
    else:
      console.print("[yellow]No records could be parsed from the file.[/yellow]")

  except Exception as e:
    logger.error(f"Info command failed: {e}")
    console.print(f"[bold red]Error:[/bold red] {e}")
    sys.exit(1)


@cli.command()
@click.argument("xml_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output directory")
@click.option(
  "--format",
  "-f",
  type=click.Choice(["csv", "json", "both"]),
  default="both",
  help="Output format (csv, json, or both)",
)
@click.option(
  "--types",
  "-t",
  multiple=True,
  help="Record types to export (can specify multiple)",
)
@click.option(
  "--clean/--no-clean",
  default=True,
  help="Enable/disable data cleaning (default: enabled)",
)
def export(
  xml_path: str, output: str | None, format: str, types: list[str], clean: bool
):
  """Export parsed data to CSV and/or JSON formats.

  XML_PATH: Path to the export.xml file

  Examples:\n
      # Export all data to both CSV and JSON (with data cleaning)\n
      uv run python main.py export export.xml

      # Export only heart rate data to CSV\n
      uv run python main.py export export.xml --format csv --types HeartRate

      # Export specific record types to custom output directory\n
      uv run python main.py export export.xml --output ./my_exports --types HeartRate --types Sleep

      # Export without data cleaning (raw data)\n
      uv run python main.py export export.xml --no-clean
  """
  try:
    from src.processors.exporter import DataExporter

    xml_file = Path(xml_path)
    output_dir = Path(output) if output else get_config().output_dir

    console.print(f"[bold blue]Exporting data from:[/bold blue] {xml_file}")
    console.print(f"[bold blue]Output directory:[/bold blue] {output_dir}")
    console.print(f"[bold blue]Export format:[/bold blue] {format}")

    if types:
      console.print(f"[bold blue]Record types:[/bold blue] {', '.join(types)}")

    # Determine export formats
    if format == "both":
      formats = ["csv", "json"]
    else:
      formats = [format]

    # Create exporter
    exporter = DataExporter(output_dir)

    # Perform export
    with UnifiedProgress(
      "Exporting data", total=None
    ) as progress:  # Total unknown for export
      export_stats = exporter.export_by_category(
        xml_file,
        formats=formats,
        record_types=list(types) if types else None,
        enable_cleaning=clean,
      )
      progress.update(1, "Export completed")

    # Validate export results
    total_exported = sum(
      sum(stats.get(fmt, 0) for fmt in ["csv", "json"])
      for stats in export_stats.values()
    )

    if total_exported == 0:
      console.print("[yellow]⚠ No records were exported![/yellow]")
      console.print("\nPossible reasons:")
      if types:
        console.print("• The specified record types may not exist in the data")
        console.print("• Use full type names like 'HKQuantityTypeIdentifierHeartRate'")
        console.print("• Run 'health-analyzer info <file>' to see available types")
      else:
        console.print("• The file may not contain valid Apple Health data")
        console.print("• Check if the XML file is corrupted")
      console.print("\n[yellow]Export completed but no data was saved.[/yellow]")
      sys.exit(1)

    # Display results
    console.print("\n[bold green]✓ Export completed successfully![/bold green]")

    # Show export summary
    table = Table(title="Export Summary")
    table.add_column("Record Type", style="cyan")
    table.add_column("CSV Records", style="green", justify="right")
    table.add_column("JSON Records", style="blue", justify="right")
    table.add_column("Total", style="magenta", justify="right")

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
      "[bold]TOTAL[/bold]",
      f"[bold]{total_csv:,}[/bold]" if total_csv > 0 else "-",
      f"[bold]{total_json:,}[/bold]" if total_json > 0 else "-",
      f"[bold]{total_records:,}[/bold]",
      end_section=True,
    )

    console.print(table)

    # Show file locations
    console.print(f"\n[bold]Files saved to:[/bold] {output_dir}")
    console.print(f"[bold]Manifest file:[/bold] {output_dir}/manifest.json")

    # Show generated files
    if output_dir.exists():
      files = list(output_dir.glob("*"))
      if files:
        console.print("\n[bold]Generated files:[/bold]")
        for file_path in sorted(files):
          if file_path.is_file():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            console.print(f"  • {file_path.name} ({size_mb:.2f} MB)")

  except Exception as e:
    logger.error(f"Export failed: {e}")
    console.print(f"[bold red]Error:[/bold red] {e}")
    sys.exit(1)


cli.add_command(report_command, name="report")
cli.add_command(visualize_command, name="visualize")


@cli.command()
@click.argument("xml_path", type=click.Path(exists=True))
@click.option(
  "--output",
  "-o",
  type=click.Path(),
  help="Output directory for benchmark results",
)
@click.option(
  "--timeout",
  type=int,
  default=30,
  help="Timeout in seconds for each test module (default: 30)",
)
def benchmark(xml_path: str, output: str | None, timeout: int):
  """Run performance benchmark tests on Apple Health data processing.

  XML_PATH: Path to the export.xml file

  Examples:\n
      # Run benchmark with default settings (30s timeout)\n
      uv run python main.py benchmark export.xml

      # Run benchmark with custom output directory\n
      uv run python main.py benchmark export.xml --output ./benchmark_results

      # Run benchmark with longer timeout\n
      uv run python main.py benchmark export.xml --timeout 60
  """
  try:
    from src.processors.benchmark import run_benchmark

    xml_file = Path(xml_path)
    output_dir = Path(output) if output else get_config().output_dir

    console.print(
      f"[bold blue]Running performance benchmark on:[/bold blue] {xml_file}"
    )
    console.print(f"[bold blue]Output directory:[/bold blue] {output_dir}")
    console.print(f"[bold blue]Timeout per module:[/bold blue] {timeout} seconds")

    # Run benchmark
    run_benchmark(str(xml_file), str(output_dir), timeout=timeout)

  except Exception as e:
    logger.error(f"Benchmark failed: {e}")
    console.print(f"[bold red]Error:[/bold red] {e}")
    sys.exit(1)


@cli.command()
@click.argument("xml_path", type=click.Path(exists=True))
@click.option(
  "--output",
  "-o",
  type=click.Path(),
  help="Output directory for analysis results",
)
@click.option("--age", type=int, help="User age (required for cardio fitness analysis)")
@click.option(
  "--gender",
  type=click.Choice(["male", "female"]),
  help="User gender (required for cardio fitness analysis)",
)
@click.option(
  "--date-range",
  help="Date range for analysis (format: YYYY-MM-DD:YYYY-MM-DD)",
)
@click.option(
  "--types",
  "-t",
  multiple=True,
  type=click.Choice(
    ["heart_rate", "sleep", "hrv", "resting_hr", "cardio_fitness", "all"]
  ),
  default=["all"],
  help="Analysis types to perform (can specify multiple)",
)
@click.option(
  "--format",
  "-f",
  type=click.Choice(["json", "text", "both"]),
  default="both",
  help="Output format for analysis results",
)
@click.option(
  "--generate-charts/--no-charts",
  default=True,
  help="Generate analysis charts (default: enabled)",
)
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
  try:
    from src.analyzers.highlights import HighlightsGenerator
    from src.processors.heart_rate import HeartRateAnalyzer
    from src.processors.sleep import SleepAnalyzer

    xml_file = Path(xml_path)
    output_dir = Path(output) if output else get_config().output_dir

    console.print(f"[bold blue]Analyzing data from:[/bold blue] {xml_file}")
    console.print(f"[bold blue]Output directory:[/bold blue] {output_dir}")

    # Validate parameters
    if "cardio_fitness" in types or "all" in types:
      if not age or not gender:
        console.print(
          "[bold red]Error:[/bold red] Age and gender are required for cardio fitness analysis"
        )
        console.print("Use --age and --gender options")
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
          f"[bold blue]Date range:[/bold blue] {start_date.date()} to {end_date.date()}"
        )
      except ValueError:
        console.print(
          "[bold red]Error:[/bold red] Invalid date range format. Use YYYY-MM-DD:YYYY-MM-DD"
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

    console.print(f"[bold blue]Analysis types:[/bold blue] {', '.join(analysis_types)}")

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
    console.print("\n[bold]Step 1: Parsing health data...[/bold]")
    with UnifiedProgress("Parsing health data", total=None) as progress:
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
        f"Parsed {len(hr_records)} heart rate, {len(sleep_records)} sleep records",
      )

    console.print(
      f"[green]✓ Parsed {len(hr_records)} heart rate, {len(sleep_records)} sleep records[/green]"
    )

    # Perform analyses
    heart_rate_report = None
    sleep_report = None

    # Heart rate analysis
    if any(
      t in analysis_types for t in ["heart_rate", "resting_hr", "hrv", "cardio_fitness"]
    ):
      console.print("\n[bold]Step 2: Analyzing heart rate data...[/bold]")
      with UnifiedProgress("Analyzing heart rate data", total=None) as progress:
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
        progress.update(1, "Heart rate analysis completed")
      console.print("[green]✓ Heart rate analysis completed[/green]")

    # Sleep analysis
    if "sleep" in analysis_types and sleep_records:
      console.print("\n[bold]Step 3: Analyzing sleep data...[/bold]")
      with UnifiedProgress("Analyzing sleep data", total=None) as progress:
        sleep_report = sleep_analyzer.analyze_comprehensive(
          cast(list[HealthRecord], sleep_records)
        )
        progress.update(1, "Sleep analysis completed")
      console.print("[green]✓ Sleep analysis completed[/green]")

    # Generate highlights
    console.print("\n[bold]Step 4: Generating health insights...[/bold]")
    with UnifiedProgress("Generating health insights", total=None) as progress:
      highlights = highlights_generator.generate_comprehensive_highlights(
        heart_rate_report=heart_rate_report,
        sleep_report=sleep_report,
      )
      progress.update(
        1,
        f"Generated {len(highlights.insights)} insights and {len(highlights.recommendations)} recommendations",
      )
    console.print(
      f"[green]✓ Generated {len(highlights.insights)} insights and {len(highlights.recommendations)} recommendations[/green]"
    )

    # Display results
    console.print("\n[bold green]🎯 Analysis Results[/bold green]")

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
      f"\n[bold green]✓ Analysis completed! Results saved to: {output_dir}[/bold green]"
    )

  except Exception as e:
    logger.error(f"Analysis failed: {e}")
    console.print(f"[bold red]Error:[/bold red] {e}")
    sys.exit(1)


def _display_parsing_results(stats: dict):
  """Display parsing results in a formatted table."""
  table = Table(title="Parsing Results")
  table.add_column("Metric", style="cyan")
  table.add_column("Value", style="magenta", justify="right")

  table.add_row("Total Records", f"{stats['total_records']:,}")
  table.add_row("Processed", f"{stats['processed_records']:,}")
  table.add_row("Skipped", f"{stats['skipped_records']:,}")
  table.add_row("Invalid", f"{stats['invalid_records']:,}")
  table.add_row("Success Rate", f"{stats['success_rate']:.1%}")

  if stats["date_range"]["start"] and stats["date_range"]["end"]:
    table.add_row(
      "Date Range",
      f"{stats['date_range']['start']} to {stats['date_range']['end']}",
    )

  console.print(table)

  # Show top record types
  if stats["record_types"]:
    console.print("\n[bold]Top Record Types:[/bold]")
    sorted_types = sorted(
      stats["record_types"].items(), key=lambda x: x[1], reverse=True
    )
    for i, (record_type, count) in enumerate(sorted_types[:10]):
      console.print(f"  {i + 1:2d}. {record_type}: {count:,}")

  # Show top sources
  if stats["sources"]:
    console.print("\n[bold]Top Data Sources:[/bold]")
    sorted_sources = sorted(stats["sources"].items(), key=lambda x: x[1], reverse=True)
    for i, (source, count) in enumerate(sorted_sources[:5]):
      console.print(f"  {i + 1:2d}. {source}: {count:,}")


def _display_records_preview(records: list):
  """Display a preview of parsed records."""
  if not records:
    return

  table = Table(title="Records Preview")
  table.add_column("Type", style="cyan")
  table.add_column("Source", style="green")
  table.add_column("Start Date", style="yellow")
  table.add_column("Value", style="magenta")

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

  if not records:
    console.print("[yellow]No records to save[/yellow]")
    return

  console.print(
    f"[bold blue]Saving {len(records)} records to: {output_dir}[/bold blue]"
  )

  # Validate data quality before saving
  console.print("[bold]Validating data quality...[/bold]")
  try:
    from src.processors.validator import validate_health_data

    validation_result = validate_health_data(records)
    validation_summary = validation_result.get_summary()

    console.print("[green]✓ Data validation completed[/green]")
    console.print(f"  Quality Score: {validation_summary['quality_score']:.1%}")
    console.print(f"  Warnings: {validation_summary['total_warnings']}")
    console.print(f"  Outliers: {validation_summary['outliers_count']}")

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

    console.print(f"[green]✓ Validation report saved to: {validation_file}[/green]")

  except Exception as e:
    logger.warning(f"Data validation failed: {e}")
    console.print(f"[yellow]⚠ Data validation skipped: {e}[/yellow]")

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
        f"  [green]✓ {clean_type}.csv:[/green] {csv_count:,} records ({csv_size:,} bytes)"
      )

    if json_count > 0:
      total_files += 1
      json_size = json_path.stat().st_size
      console.print(
        f"  [green]✓ {clean_type}.json:[/green] {json_count:,} records ({json_size:,} bytes)"
      )

  # Save parsing statistics
  stats_file = output_dir / "parsing_stats.json"
  with open(stats_file, "w", encoding="utf-8") as f:
    import json

    json.dump(stats, f, indent=2, ensure_ascii=False, default=str)

  console.print("  [green]✓ parsing_stats.json:[/green] parsing statistics")

  # Create a simple manifest
  manifest_file = output_dir / "data_manifest.txt"
  with open(manifest_file, "w", encoding="utf-8") as f:
    f.write("Apple Health Data Export Manifest\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total Records: {total_saved:,}\n")
    f.write(f"Total Files: {total_files + 1}\n")  # +1 for stats file
    f.write(f"Record Types: {len(records_by_type)}\n\n")

    f.write("Files Generated:\n")
    f.write("-" * 20 + "\n")
    for record_type, type_records in records_by_type.items():
      clean_type = record_type.replace("HKQuantityTypeIdentifier", "").replace(
        "HKCategoryTypeIdentifier", ""
      )
      f.write(f"• {clean_type}.csv ({len(type_records)} records)\n")
      f.write(f"• {clean_type}.json ({len(type_records)} records)\n")

    f.write("• parsing_stats.json (statistics)\n")
    f.write("• data_manifest.txt (this file)\n")

  console.print("  [green]✓ data_manifest.txt:[/green] export manifest")
  console.print(
    f"[bold green]✓ Successfully saved {total_saved:,} records to {total_files + 1} files[/bold green]"
  )


def _display_heart_rate_results(report):
  """Display heart rate analysis results."""
  console.print("\n[bold blue]❤️ Heart Rate Analysis[/bold blue]")

  if report.resting_hr_analysis:
    resting = report.resting_hr_analysis
    console.print(f"  [cyan]Resting HR:[/cyan] {resting.current_value:.1f} bpm")
    console.print(f"  [cyan]Trend:[/cyan] {resting.trend_direction}")
    console.print(f"  [cyan]Health Rating:[/cyan] {resting.health_rating}")

  if report.hrv_analysis:
    hrv = report.hrv_analysis
    console.print(f"  [cyan]HRV (SDNN):[/cyan] {hrv.current_sdnn:.1f} ms")
    console.print(f"  [cyan]Stress Level:[/cyan] {hrv.stress_level}")
    console.print(f"  [cyan]Recovery Status:[/cyan] {hrv.recovery_status}")

  if report.cardio_fitness:
    cardio = report.cardio_fitness
    console.print(f"  [cyan]VO2 Max:[/cyan] {cardio.current_vo2_max:.1f} ml/min·kg")
    console.print(f"  [cyan]Fitness Rating:[/cyan] {cardio.age_adjusted_rating}")

  console.print(f"  [cyan]Data Quality:[/cyan] {report.data_quality_score:.1%}")
  console.print(f"  [cyan]Total Records:[/cyan] {report.record_count:,}")


def _display_sleep_results(report):
  """Display sleep analysis results."""
  console.print("\n[bold blue]😴 Sleep Analysis[/bold blue]")

  if report.quality_metrics:
    quality = report.quality_metrics
    console.print(
      f"  [cyan]Average Duration:[/cyan] {quality.average_duration:.1f} hours"
    )
    console.print(
      f"  [cyan]Average Efficiency:[/cyan] {quality.average_efficiency:.1%}"
    )
    console.print(f"  [cyan]Consistency Score:[/cyan] {quality.consistency_score:.1%}")

  console.print(f"  [cyan]Data Quality:[/cyan] {report.data_quality_score:.1%}")
  console.print(f"  [cyan]Total Records:[/cyan] {report.record_count:,}")


def _display_highlights(highlights):
  """Display health highlights and recommendations."""
  console.print("\n[bold blue]💡 Health Insights[/bold blue]")

  # Display insights
  if highlights.insights:
    console.print("\n[bold]Key Insights:[/bold]")
    for i, insight in enumerate(highlights.insights[:5], 1):  # Show top 5
      priority_colors = {"high": "red", "medium": "yellow", "low": "green"}
      color = priority_colors.get(insight.priority, "white")
      console.print(f"  {i}. [{color}]{insight.title}[/{color}]")
      console.print(f"     {insight.message}")

  # Display recommendations
  if highlights.recommendations:
    console.print("\n[bold]Recommendations:[/bold]")
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

  console.print(f"[green]✓ JSON results saved to: {output_file}[/green]")


def _save_analysis_results_text(
  output_dir: Path, heart_rate_report, sleep_report, highlights
):
  """Save analysis results to text format."""
  output_dir.mkdir(parents=True, exist_ok=True)

  output_file = output_dir / "analysis_results.txt"

  with open(output_file, "w", encoding="utf-8") as f:
    f.write("Apple Health Analysis Report\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    if heart_rate_report:
      f.write("HEART RATE ANALYSIS\n")
      f.write("-" * 20 + "\n")
      if heart_rate_report.resting_hr_analysis:
        resting = heart_rate_report.resting_hr_analysis
        f.write(f"Resting HR: {resting.current_value:.1f} bpm\n")
        f.write(f"Trend: {resting.trend_direction}\n")
        f.write(f"Health Rating: {resting.health_rating}\n")
      if heart_rate_report.hrv_analysis:
        hrv = heart_rate_report.hrv_analysis
        f.write(f"HRV (SDNN): {hrv.current_sdnn:.1f} ms\n")
        f.write(f"Stress Level: {hrv.stress_level}\n")
      if heart_rate_report.cardio_fitness:
        cardio = heart_rate_report.cardio_fitness
        f.write(f"VO2 Max: {cardio.current_vo2_max:.1f} ml/min·kg\n")
        f.write(f"Fitness Rating: {cardio.age_adjusted_rating}\n")
      f.write(f"Data Quality: {heart_rate_report.data_quality_score:.1%}\n")
      f.write(f"Total Records: {heart_rate_report.record_count:,}\n\n")

    if sleep_report:
      f.write("SLEEP ANALYSIS\n")
      f.write("-" * 15 + "\n")
      if sleep_report.quality_metrics:
        quality = sleep_report.quality_metrics
        f.write(f"Average Duration: {quality.average_duration:.1f} hours\n")
        f.write(f"Average Efficiency: {quality.average_efficiency:.1%}\n")
        f.write(f"Consistency Score: {quality.consistency_score:.1%}\n")
      f.write(f"Data Quality: {sleep_report.data_quality_score:.1%}\n")
      f.write(f"Total Records: {sleep_report.record_count:,}\n\n")

    if highlights.insights:
      f.write("KEY INSIGHTS\n")
      f.write("-" * 12 + "\n")
      for i, insight in enumerate(highlights.insights[:5], 1):
        f.write(f"{i}. {insight.title}\n")
        f.write(f"   {insight.message}\n\n")

    if highlights.recommendations:
      f.write("RECOMMENDATIONS\n")
      f.write("-" * 15 + "\n")
      for i, rec in enumerate(highlights.recommendations, 1):
        f.write(f"{i}. {rec}\n")

  console.print(f"[green]✓ Text results saved to: {output_file}[/green]")


def _report_to_dict(report):
  """Convert report object to dictionary for JSON serialization."""
  if report is None:
    return None

  # Simple conversion - in production, you'd want more sophisticated serialization
  return {
    "analysis_date": report.analysis_date.isoformat()
    if hasattr(report, "analysis_date")
    else None,
    "data_range": [
      report.data_range[0].isoformat(),
      report.data_range[1].isoformat(),
    ]
    if hasattr(report, "data_range")
    else None,
    "record_count": getattr(report, "record_count", 0),
    "data_quality_score": getattr(report, "data_quality_score", 0.0),
  }


def _ensure_xml_file(xml_file: Path) -> None:
  """Validate XML file existence and size."""
  if not xml_file.exists():
    console.print(f"[bold red]Error:[/bold red] File not found: {xml_file}")
    console.print("[yellow]Tip:[/yellow] Check the file path and try again")
    sys.exit(1)

  if xml_file.stat().st_size == 0:
    console.print(f"[bold red]Error:[/bold red] File is empty: {xml_file}")
    sys.exit(1)


def _display_file_info(xml_file: Path) -> None:
  """Display export file information with guardrails."""
  try:
    file_info = get_export_file_info(xml_file)
    if file_info:
      console.print(f"[green]File size:[/green] {file_info['file_size_mb']:.1f} MB")
      console.print(
        f"[green]Estimated records:[/green] {file_info['estimated_record_count']:,}"
      )
    else:
      console.print("[yellow]Warning:[/yellow] Could not read file information")
  except Exception as e:
    console.print(f"[yellow]Warning:[/yellow] Could not analyze file: {e}")
    console.print("[yellow]Continuing with parsing...[/yellow]")


def _init_parser(xml_file: Path) -> StreamingXMLParser:
  """Initialize the streaming parser with error handling."""
  try:
    return StreamingXMLParser(xml_file)
  except Exception as e:
    console.print(f"[bold red]Error:[/bold red] Failed to initialize parser: {e}")
    console.print("[yellow]Tip:[/yellow] Check if the file is a valid XML file")
    sys.exit(1)


def _parse_records_with_progress(
  parser: StreamingXMLParser, record_types: list[str] | None
) -> tuple[list, dict]:
  """Parse records with CLI progress feedback."""
  with UnifiedProgress(
    "Parsing records",
    total=None,  # Always use indeterminate progress for parsing
    quiet=True,  # Disable logging for cleaner CLI output
  ) as progress:

    def update_progress(count: int) -> None:
      progress.update(count, f"Parsed {count:,} records")

    records_generator = parser.parse_records(
      record_types=record_types,
      progress_callback=update_progress,
      quiet=True,
    )
    records = list(records_generator)
    stats = parser.get_statistics()

    progress.update(len(records), f"Parsed {len(records):,} records")

  return records, stats


def _handle_parsing_exception(error: Exception) -> None:
  """Handle parsing errors with user-friendly messages."""
  logger.error(f"Parsing failed: {error}")

  error_str = str(error).lower()
  if "memory" in error_str:
    console.print("[bold red]Error:[/bold red] Insufficient memory for parsing")
    console.print(
      "[yellow]Tip:[/yellow] Try processing a smaller file or increase system memory"
    )
  elif "permission" in error_str:
    console.print("[bold red]Error:[/bold red] File permission denied")
    console.print("[yellow]Tip:[/yellow] Check file permissions")
  elif "encoding" in error_str:
    console.print("[bold red]Error:[/bold red] File encoding issue")
    console.print("[yellow]Tip:[/yellow] Ensure the file is UTF-8 encoded")
  else:
    console.print(f"[bold red]Error:[/bold red] Failed to parse records: {error}")
    console.print("[yellow]Tip:[/yellow] Check the XML file format and try again")

  sys.exit(1)


def _print_parsing_success(stats: dict) -> None:
  """Print parsing summary and success indicator."""
  success_rate = stats.get("success_rate", 0)
  if success_rate >= 0.95:
    console.print("[bold green]✓ Parsing completed successfully![/bold green]")
  elif success_rate >= 0.80:
    console.print("[bold yellow]⚠ Parsing completed with minor issues[/bold yellow]")
  else:
    console.print(
      "[bold yellow]⚠ Parsing completed but with significant data loss[/bold yellow]"
    )

  console.print(
    f"[green]Processed {stats.get('processed_records', 0):,} records with {success_rate:.1%} success rate[/green]"
  )


def main():
  """Main entry point for the CLI."""
  try:
    cli()
  except KeyboardInterrupt:
    console.print("\n[bold yellow]Operation cancelled by user[/bold yellow]")
    sys.exit(1)
  except HealthAnalyzerError as e:
    console.print(f"[bold red]Health Analyzer Error:[/bold red] {e}")
    sys.exit(1)
  except Exception as e:
    logger.exception("Unexpected error")
    console.print(f"[bold red]Unexpected error:[/bold red] {e}")
    sys.exit(1)


if __name__ == "__main__":
  main()
