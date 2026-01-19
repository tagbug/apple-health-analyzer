"""Command-line interface for Apple Health Analyzer.

Provides CLI commands for parsing, analyzing, and exporting health data.
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from src.config import get_config, reload_config
from src.core.exceptions import HealthAnalyzerError
from src.core.xml_parser import StreamingXMLParser, get_export_file_info
from src.utils.logger import get_logger

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
  """Apple Health Data Analyzer - 苹果健康数据分析工具

  A comprehensive tool for parsing, analyzing, and visualizing Apple Health export data.
  """
  if config_path:
    # Load custom config file
    import os

    os.environ["CONFIG_FILE"] = config_path
    reload_config()

  if verbose:
    import logging

    logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.argument("xml_path", type=click.Path(exists=True))
@click.option(
  "--output", "-o", type=click.Path(), help="Output directory for results"
)
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
  """
  try:
    xml_file = Path(xml_path)
    output_dir = Path(output) if output else get_config().output_dir

    console.print(
      f"[bold blue]Parsing Apple Health export:[/bold blue] {xml_file}"
    )
    console.print(f"[bold blue]Output directory:[/bold blue] {output_dir}")

    # Get file info
    file_info = get_export_file_info(xml_file)
    if file_info:
      console.print(
        f"[green]File size:[/green] {file_info['file_size_mb']:.1f} MB"
      )
      console.print(
        f"[green]Estimated records:[/green] {file_info['estimated_record_count']:,}"
      )

    # Initialize parser
    parser = StreamingXMLParser(xml_file)

    # Parse records
    record_types = list(types) if types else None
    records = []
    stats = {}

    with console.status("[bold green]Parsing records..."):
      records_generator = parser.parse_records(record_types)
      records = list(records_generator)
      stats = parser.get_statistics()

    # Display results
    _display_parsing_results(stats)

    if preview and records:
      console.print("\n[bold]Data Preview:[/bold]")
      _display_records_preview(records[:10])

    # Save results if output specified
    if output:
      _save_parsed_data(records, stats, output_dir)

    console.print("[bold green]✓ Parsing completed successfully![/bold green]")

  except Exception as e:
    logger.error(f"Parsing failed: {e}")
    console.print(f"[bold red]Error:[/bold red] {e}")
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
    table.add_row(
      "Estimated Records", f"{file_info['estimated_record_count']:,}"
    )
    table.add_row("Last Modified", str(file_info["last_modified"]))

    console.print(table)

    # Quick parsing to get more details
    # parser = StreamingXMLParser(xml_file)

    # Sample a few records to get data range
    sample_records = []
    try:
      # Parse without ProgressLogger to see actual errors
      context = ET.iterparse(xml_file, events=("start", "end"))
      context = iter(context)

      # Get root element
      event, root = next(context)

      count = 0
      for event, elem in context:
        if event == "start" and elem.tag == "Record":
          # Use the same parsing logic as the parser
          try:
            from src.core.data_models import (
              create_record_from_xml_element,
            )

            record, warnings = create_record_from_xml_element(elem)
            if record:
              sample_records.append(record)
              count += 1
              if count >= 100:  # Sample first 100 records
                break
          except Exception:
            # Skip invalid records in info command
            pass

          if count >= 100:
            break

        # Clear elements at end events to free memory
        if event == "end":
          elem.clear()

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
      console.print(
        "[yellow]No records could be parsed from the file.[/yellow]"
      )

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

  Examples:
      # Export all data to both CSV and JSON (with data cleaning)
      uv run python main.py export export.xml

      # Export only heart rate data to CSV
      uv run python main.py export export.xml --format csv --types HeartRate

      # Export specific record types to custom output directory
      uv run python main.py export export.xml --output ./my_exports --types HeartRate --types Sleep

      # Export without data cleaning (raw data)
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
    with console.status("[bold green]Exporting data..."):
      export_stats = exporter.export_by_category(
        xml_file,
        formats=formats,
        record_types=list(types) if types else None,
        enable_cleaning=clean,
      )

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


@cli.command()
@click.argument("xml_path", type=click.Path(exists=True))
@click.option(
  "--output", "-o", type=click.Path(), help="Output directory for charts"
)
def analyze(xml_path: str, output: str | None):
  """Analyze heart rate and sleep data.

  XML_PATH: Path to the export.xml file
  """
  try:
    xml_file = Path(xml_path)
    output_dir = Path(output) if output else get_config().output_dir

    console.print(f"[bold blue]Analyzing data from:[/bold blue] {xml_file}")
    console.print(f"[bold blue]Output directory:[/bold blue] {output_dir}")

    # TODO: Implement analysis functionality
    console.print("[yellow]Analysis functionality coming soon![/yellow]")

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
    sorted_sources = sorted(
      stats["sources"].items(), key=lambda x: x[1], reverse=True
    )
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
  output_dir.mkdir(parents=True, exist_ok=True)

  # TODO: Implement data saving
  console.print(f"[green]Data would be saved to: {output_dir}[/green]")


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
