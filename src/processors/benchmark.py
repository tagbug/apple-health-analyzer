"""Performance benchmark module.

Measures Apple Health analysis performance and generates CLI reports.
"""

import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import psutil
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..analyzers.highlights import HighlightsGenerator
from ..analyzers.statistical import StatisticalAnalyzer
from ..core.xml_parser import StreamingXMLParser
from ..utils.logger import get_logger
from .cleaner import DataCleaner
from .exporter import DataExporter

logger = get_logger(__name__)


class TimeoutError(Exception):
  """Timeout exception for benchmark runs."""


class BenchmarkModule:
  """Single benchmark module definition."""

  def __init__(self, name: str, test_func: Callable, description: str = ""):
    self.name = name
    self.test_func = test_func
    self.description = description or name


class BenchmarkResult:
  """Benchmark result payload."""

  def __init__(self, module_name: str):
    self.module_name = module_name
    self.status = "pending"  # pending, completed, timeout, error
    self.time_seconds = 0.0
    self.memory_mb = 0.0
    self.records_processed = 0
    self.throughput_records_per_sec = 0.0
    self.error_message = ""


class BenchmarkRunner:
  """Benchmark runner for performance tests."""

  def __init__(
    self,
    xml_path: Path,
    output_dir: Path | None = None,
    timeout_seconds: int = 30,
  ):
    self.xml_path = xml_path
    self.output_dir = output_dir if output_dir is not None else Path("output")
    self.output_dir.mkdir(exist_ok=True)
    self.timeout_seconds = timeout_seconds
    self.sample_records: list[Any] = []
    self.results: list[BenchmarkResult] = []
    self.start_time = time.time()

  def get_memory_usage(self) -> float:
    """Return current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

  def _run_with_timeout(self, func: Callable, *args, **kwargs) -> Any:
    """Run a function and stop it if it exceeds the timeout."""
    result = [None]
    exception: list[Exception | None] = [None]

    def target():
      try:
        result[0] = func(*args, **kwargs)
      except Exception as e:
        exception[0] = e

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(self.timeout_seconds)

    if thread.is_alive():
      logger.warning(f"Timeout ({self.timeout_seconds}s), stopping execution")
      raise TimeoutError(f"Timeout ({self.timeout_seconds}s)")

    if exception[0]:
      raise exception[0]

    return result[0]

  def load_sample_data(self, limit: int = 10000) -> tuple[list[Any], dict[str, Any]]:
    """Load a sample of records from XML and return parse metrics."""
    logger.info(f"Loading first {limit} XML records as benchmark sample...")

    sample_records = []
    parser = StreamingXMLParser(self.xml_path)
    count = 0

    # Record XML parsing metrics.
    start_time = time.time()
    start_mem = self.get_memory_usage()

    for record in parser.parse_records():
      sample_records.append(record)
      count += 1
      if count >= limit:
        break

    end_time = time.time()
    end_mem = self.get_memory_usage()

    # Compute parsing throughput metrics.
    parse_time = end_time - start_time
    parse_metrics = {
      "records_processed": len(sample_records),
      "throughput_records_per_sec": len(sample_records) / parse_time
      if parse_time > 0
      else 0,
      "memory_delta_mb": end_mem - start_mem,
      "parse_time_seconds": parse_time,
    }

    logger.info(f"Loaded {len(sample_records)} records for benchmarking")
    logger.info(
      "XML parse throughput: "
      f"{parse_metrics['throughput_records_per_sec']:.0f} records/sec"
    )

    self.sample_records = sample_records
    return sample_records, parse_metrics

  def run_module_with_timeout(
    self, module: BenchmarkModule, *args, **kwargs
  ) -> BenchmarkResult:
    """Run a module with timeout handling."""
    result = BenchmarkResult(module.name)

    try:
      logger.info(f"Starting benchmark module: {module.name}")

      start_time = time.time()
      start_mem = self.get_memory_usage()

      # Run the test function.
      test_result = self._run_with_timeout(module.test_func, *args, **kwargs)

      end_time = time.time()
      end_mem = self.get_memory_usage()

      # Record successful results.
      result.status = "completed"
      result.time_seconds = end_time - start_time
      result.memory_mb = end_mem - start_mem

      # Extract metrics from the test result.
      if isinstance(test_result, dict):
        result.records_processed = test_result.get("records_processed", 0)
        result.throughput_records_per_sec = test_result.get(
          "throughput_records_per_sec", 0.0
        )

      logger.info(
        f"Benchmark module {module.name} completed: {result.time_seconds:.2f}s"
      )

    except TimeoutError:
      result.status = "timeout"
      result.time_seconds = self.timeout_seconds
      logger.warning(f"Benchmark module {module.name} timed out")

    except Exception as e:
      result.status = "error"
      result.error_message = str(e)
      logger.error(f"Benchmark module {module.name} failed: {e}")

    return result

  def benchmark_data_cleaning(self, sample_records: list[Any]) -> dict[str, Any]:
    """Benchmark data cleaning performance."""
    start_time = time.time()
    start_mem = self.get_memory_usage()

    cleaner = DataCleaner()
    cleaned_records, dedup_result = cleaner.deduplicate_by_time_window(
      sample_records, window_seconds=300
    )

    end_time = time.time()
    end_mem = self.get_memory_usage()

    # Ensure a minimum elapsed time to avoid division by zero.
    elapsed_time = max(end_time - start_time, 0.001)  # Minimum 1 ms.

    return {
      "records_processed": len(sample_records),
      "throughput_records_per_sec": len(sample_records) / elapsed_time,
      "memory_delta_mb": end_mem - start_mem,
      "cleaned_records": len(cleaned_records),
      "duplicates_removed": dedup_result.removed_duplicates,
    }

  def benchmark_statistical_analysis(self, sample_records: list[Any]) -> dict[str, Any]:
    """Benchmark statistical analysis performance."""
    start_time = time.time()
    start_mem = self.get_memory_usage()

    analyzer = StatisticalAnalyzer()
    analyzer.generate_report(sample_records)

    end_time = time.time()
    end_mem = self.get_memory_usage()

    # Ensure a minimum elapsed time to avoid division by zero.
    elapsed_time = max(end_time - start_time, 0.001)  # Minimum 1 ms.

    return {
      "records_processed": len(sample_records),
      "throughput_records_per_sec": len(sample_records) / elapsed_time,
      "memory_delta_mb": end_mem - start_mem,
    }

  def benchmark_report_generation(self, sample_records: list[Any]) -> dict[str, Any]:
    """Benchmark report generation performance."""
    start_time = time.time()
    start_mem = self.get_memory_usage()

    # Simulate report generation.
    highlights_gen = HighlightsGenerator()
    _highlights = highlights_gen.generate_comprehensive_highlights()

    end_time = time.time()
    end_mem = self.get_memory_usage()

    # Ensure a minimum elapsed time to avoid division by zero.
    elapsed_time = max(end_time - start_time, 0.001)  # Minimum 1 ms.

    return {
      "records_processed": len(sample_records),
      "throughput_records_per_sec": len(sample_records) / elapsed_time,
      "memory_delta_mb": end_mem - start_mem,
    }

  def benchmark_data_export(self, sample_records: list[Any]) -> dict[str, Any]:
    """Benchmark data export performance."""
    start_time = time.time()
    start_mem = self.get_memory_usage()

    output_path = self.output_dir / "benchmark_export_sample.csv"
    exporter = DataExporter(self.output_dir)
    exporter.export_to_csv(sample_records, output_path)

    end_time = time.time()
    end_mem = self.get_memory_usage()

    # Ensure a minimum elapsed time to avoid division by zero.
    elapsed_time = max(end_time - start_time, 0.001)  # Minimum 1 ms.

    return {
      "records_processed": len(sample_records),
      "throughput_records_per_sec": len(sample_records) / elapsed_time,
      "memory_delta_mb": end_mem - start_mem,
      "output_file": str(output_path),
      "file_size_mb": output_path.stat().st_size / 1024 / 1024,
    }

  def run_all_benchmarks(self) -> list[BenchmarkResult]:
    """Run all benchmark modules."""
    logger.info("=== Starting full benchmark run ===")

    # 1. Load sample data and gather XML parse metrics.
    sample_records, xml_parse_metrics = self.load_sample_data(10000)
    if not sample_records:
      logger.error("Failed to load sample data, aborting benchmarks")
      return []

    # 2. Define benchmark modules.
    modules = [
      BenchmarkModule(
        "Data cleaning", self.benchmark_data_cleaning, "Benchmark deduplication"
      ),
      BenchmarkModule(
        "Statistical analysis",
        self.benchmark_statistical_analysis,
        "Benchmark statistical calculations",
      ),
      BenchmarkModule(
        "Report generation",
        self.benchmark_report_generation,
        "Benchmark report generation",
      ),
      BenchmarkModule("Data export", self.benchmark_data_export, "Benchmark exports"),
    ]

    # 3. Run all benchmark modules.
    results = []

    # Add XML parse results first (from load_sample_data).
    xml_result = BenchmarkResult("XML parsing")
    xml_result.status = "completed"
    xml_result.time_seconds = xml_parse_metrics["parse_time_seconds"]
    xml_result.memory_mb = xml_parse_metrics["memory_delta_mb"]
    xml_result.records_processed = xml_parse_metrics["records_processed"]
    xml_result.throughput_records_per_sec = xml_parse_metrics[
      "throughput_records_per_sec"
    ]
    results.append(xml_result)

    # Run the remaining modules.
    for module in modules:
      result = self.run_module_with_timeout(module, sample_records)
      results.append(result)

    self.results = results

    # 4. Compute overall statistics.
    completed_count = sum(1 for r in results if r.status == "completed")
    total_time = time.time() - self.start_time

    logger.info("=== Benchmark run completed ===")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Sample size: {len(sample_records)} records")
    logger.info(f"Modules completed: {completed_count}/{len(results)}")

    return results

  def print_report(self):
    """Print the benchmark report."""
    if not self.results:
      logger.error("No benchmark results available; run run_all_benchmarks() first")
      return

    console = Console()
    total_time = time.time() - self.start_time
    completed_count = sum(1 for r in self.results if r.status == "completed")

    # Title panel.
    title = Text("🍎 Apple Health Analyzer - Benchmark Report", style="bold blue")
    console.print(Panel(title, border_style="blue", padding=(1, 2)))

    # Summary table.
    info_table = Table(show_header=True, header_style="bold cyan", box=None)
    info_table.add_column("Metric", style="dim", width=14)
    info_table.add_column("Value", style="green")

    info_table.add_row("Start time", time.strftime("%Y-%m-%d %H:%M:%S"))
    info_table.add_row("Total time", f"{total_time:.2f} seconds")
    info_table.add_row("Sample size", f"{len(self.sample_records):,} records")
    info_table.add_row("Timeout", f"{self.timeout_seconds} seconds")
    info_table.add_row("Modules completed", f"{completed_count}/{len(self.results)}")

    console.print(info_table)
    console.print()

    # Performance table.
    perf_table = Table(
      title="🔍 Module Performance",
      title_style="bold yellow",
      show_header=True,
      header_style="bold magenta",
      border_style="blue",
    )

    perf_table.add_column("Module", style="cyan", min_width=12)
    perf_table.add_column("Status", style="green", min_width=6, justify="center")
    perf_table.add_column("Time (s)", style="yellow", min_width=10, justify="right")
    perf_table.add_column("Records", style="blue", min_width=10, justify="right")
    perf_table.add_column(
      "Throughput (records/s)", style="red", min_width=18, justify="right"
    )
    perf_table.add_column(
      "Memory delta (MB)", style="purple", min_width=14, justify="right"
    )

    for result in self.results:
      # Status icons and colors.
      status_config = {
        "completed": ("✅", "green"),
        "timeout": ("⏰", "yellow"),
        "error": ("❌", "red"),
        "pending": ("⏳", "blue"),
      }
      status_icon, status_color = status_config.get(result.status, ("❓", "white"))

      # Format throughput to avoid extreme values.
      throughput = result.throughput_records_per_sec
      if throughput > 1000000:  # Display "Instant" for extreme throughput.
        throughput_str = Text("Instant", style="bold green")
      else:
        throughput_str = f"{throughput:,.0f}"

      # Color memory delta (red for growth, green for drop).
      memory_color = "red" if result.memory_mb > 0 else "green"
      memory_str = f"{result.memory_mb:+.2f}"

      perf_table.add_row(
        result.module_name,
        Text(status_icon, style=status_color),
        f"{result.time_seconds:.2f}",
        f"{result.records_processed:,}",
        throughput_str,
        Text(memory_str, style=memory_color),
      )

    console.print(perf_table)

    # Bottleneck analysis.
    if completed_count > 0:
      console.print("\n💡 [bold cyan]Bottleneck analysis:[/bold cyan]")

      # Find the slowest module.
      sorted_by_time = sorted(
        [r for r in self.results if r.status == "completed"],
        key=lambda x: x.time_seconds,
        reverse=True,
      )
      if sorted_by_time:
        slowest = sorted_by_time[0]
        console.print(
          f"  ⚠️  [red]{slowest.module_name}[/red] is slowest "
          f"([bold]{slowest.time_seconds:.2f}s[/bold])"
        )

      # Find the lowest throughput module.
      sorted_by_throughput = sorted(
        [r for r in self.results if r.status == "completed"],
        key=lambda x: x.throughput_records_per_sec,
      )
      if (
        sorted_by_throughput and sorted_by_throughput[0].throughput_records_per_sec > 0
      ):
        lowest_throughput = sorted_by_throughput[0]
        console.print(
          f"  ⚠️  [red]{lowest_throughput.module_name}[/red] has lowest throughput "
          f"([bold]{lowest_throughput.throughput_records_per_sec:,.0f} records/s[/bold])"
        )

      # Find the highest memory usage module.
      sorted_by_memory = sorted(
        [r for r in self.results if r.status == "completed"],
        key=lambda x: x.memory_mb,
        reverse=True,
      )
      if sorted_by_memory and sorted_by_memory[0].memory_mb > 10:
        highest_memory = sorted_by_memory[0]
        memory_color = "red" if highest_memory.memory_mb > 0 else "green"
        console.print(
          f"  ⚠️  [red]{highest_memory.module_name}[/red] has highest memory delta "
          f"([bold {memory_color}]{highest_memory.memory_mb:.1f} MB[/bold {memory_color}])"
        )

    # Completion timestamp.
    console.print(
      f"\n✅ [green]Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}[/green]"
    )


def run_benchmark(xml_path: str, output_dir: str | None = None, timeout: int = 30):
  """Convenience runner for performance benchmarks."""
  xml_path_obj = Path(xml_path)
  if not xml_path_obj.exists():
    raise FileNotFoundError(f"XML file not found: {xml_path}")

  output_dir_obj = Path(output_dir) if output_dir else None

  runner = BenchmarkRunner(xml_path_obj, output_dir_obj, timeout)
  results = runner.run_all_benchmarks()
  runner.print_report()

  return results
