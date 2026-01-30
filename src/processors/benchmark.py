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
      logger.warning(f"测试超时 ({self.timeout_seconds}s)，强制停止")
      raise TimeoutError(f"测试超时 ({self.timeout_seconds}s)")

    if exception[0]:
      raise exception[0]

    return result[0]

  def load_sample_data(self, limit: int = 10000) -> tuple[list[Any], dict[str, Any]]:
    """Load a sample of records from XML and return parse metrics."""
    logger.info(f"从XML文件开头加载前{limit}条数据作为测试样本...")

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

    logger.info(f"已加载 {len(sample_records)} 条真实数据用于测试")
    logger.info(f"XML解析性能: {parse_metrics['throughput_records_per_sec']:.0f} 条/秒")

    self.sample_records = sample_records
    return sample_records, parse_metrics

  def run_module_with_timeout(
    self, module: BenchmarkModule, *args, **kwargs
  ) -> BenchmarkResult:
    """Run a module with timeout handling."""
    result = BenchmarkResult(module.name)

    try:
      logger.info(f"开始测试模块: {module.name}")

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

      logger.info(f"测试模块 {module.name} 完成: {result.time_seconds:.2f}s")

    except TimeoutError:
      result.status = "timeout"
      result.time_seconds = self.timeout_seconds
      logger.warning(f"测试模块 {module.name} 超时")

    except Exception as e:
      result.status = "error"
      result.error_message = str(e)
      logger.error(f"测试模块 {module.name} 出错: {e}")

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
    logger.info("=== 开始完整性能基准测试 ===")

    # 1. Load sample data and gather XML parse metrics.
    sample_records, xml_parse_metrics = self.load_sample_data(10000)
    if not sample_records:
      logger.error("无法加载样本数据，测试终止")
      return []

    # 2. Define benchmark modules.
    modules = [
      BenchmarkModule(
        "数据清洗", self.benchmark_data_cleaning, "测试数据去重和清洗性能"
      ),
      BenchmarkModule(
        "统计分析", self.benchmark_statistical_analysis, "测试统计分析计算性能"
      ),
      BenchmarkModule(
        "报告生成", self.benchmark_report_generation, "测试健康报告生成性能"
      ),
      BenchmarkModule("数据导出", self.benchmark_data_export, "测试数据导出到文件性能"),
    ]

    # 3. Run all benchmark modules.
    results = []

    # Add XML parse results first (from load_sample_data).
    xml_result = BenchmarkResult("XML 解析")
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

    logger.info("=== 性能基准测试完成 ===")
    logger.info(f"总测试时间: {total_time:.2f} 秒")
    logger.info(f"样本数据量: {len(sample_records)} 条记录")
    logger.info(f"完成模块: {completed_count}/{len(results)}")

    return results

  def print_report(self):
    """Print the benchmark report."""
    if not self.results:
      logger.error("没有可用的基准测试结果，请先运行 run_all_benchmarks()")
      return

    console = Console()
    total_time = time.time() - self.start_time
    completed_count = sum(1 for r in self.results if r.status == "completed")

    # Title panel.
    title = Text("🍎 Apple Health Analyzer - 性能基准测试报告", style="bold blue")
    console.print(Panel(title, border_style="blue", padding=(1, 2)))

    # Summary table.
    info_table = Table(show_header=True, header_style="bold cyan", box=None)
    info_table.add_column("指标", style="dim", width=10)
    info_table.add_column("数值", style="green")

    info_table.add_row("测试开始时间", time.strftime("%Y-%m-%d %H:%M:%S"))
    info_table.add_row("总测试时间", f"{total_time:.2f} 秒")
    info_table.add_row("样本数据量", f"{len(self.sample_records):,} 条记录")
    info_table.add_row("超时设置", f"{self.timeout_seconds} 秒")
    info_table.add_row("完成模块", f"{completed_count}/{len(self.results)}")

    console.print(info_table)
    console.print()

    # Performance table.
    perf_table = Table(
      title="🔍 各模块性能指标",
      title_style="bold yellow",
      show_header=True,
      header_style="bold magenta",
      border_style="blue",
    )

    perf_table.add_column("模块名称", style="cyan", min_width=12)
    perf_table.add_column("状态", style="green", min_width=6, justify="center")
    perf_table.add_column("时间(秒)", style="yellow", min_width=10, justify="right")
    perf_table.add_column("记录数", style="blue", min_width=10, justify="right")
    perf_table.add_column("吞吐量(条/秒)", style="red", min_width=15, justify="right")
    perf_table.add_column("内存增量(MB)", style="purple", min_width=12, justify="right")

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
      if throughput > 1000000:  # 如果吞吐量超过100万，显示为"瞬时"
        throughput_str = Text("瞬时", style="bold green")
      else:
        throughput_str = f"{throughput:,.0f}"

      # 内存增量颜色（正数红色表示增加，负数绿色表示减少）
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
      console.print("\n💡 [bold cyan]性能瓶颈分析:[/bold cyan]")

      # Find the slowest module.
      sorted_by_time = sorted(
        [r for r in self.results if r.status == "completed"],
        key=lambda x: x.time_seconds,
        reverse=True,
      )
      if sorted_by_time:
        slowest = sorted_by_time[0]
        console.print(
          f"  ⚠️  [red]{slowest.module_name}[/red]模块耗时最长（[bold]{slowest.time_seconds:.2f}秒[/bold]）"
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
          f"  ⚠️  [red]{lowest_throughput.module_name}[/red]模块吞吐量最低（[bold]{lowest_throughput.throughput_records_per_sec:,.0f}条/秒[/bold]）"
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
          f"  ⚠️  [red]{highest_memory.module_name}[/red]模块内存占用最高（[bold {memory_color}]{highest_memory.memory_mb:.1f} MB[/bold {memory_color}]）"
        )

    # Completion timestamp.
    console.print(
      f"\n✅ [green]测试完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}[/green]"
    )


def run_benchmark(xml_path: str, output_dir: str | None = None, timeout: int = 30):
  """Convenience runner for performance benchmarks."""
  xml_path_obj = Path(xml_path)
  if not xml_path_obj.exists():
    raise FileNotFoundError(f"XML文件不存在: {xml_path}")

  output_dir_obj = Path(output_dir) if output_dir else None

  runner = BenchmarkRunner(xml_path_obj, output_dir_obj, timeout)
  results = runner.run_all_benchmarks()
  runner.print_report()

  return results
