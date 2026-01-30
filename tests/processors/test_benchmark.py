"""Tests for benchmark module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.processors.benchmark import (
  BenchmarkModule,
  BenchmarkResult,
  BenchmarkRunner,
  TimeoutError,
  run_benchmark,
)


class TestBenchmarkModule:
  """Test BenchmarkModule class."""

  def test_initialization(self):
    """Test module initialization."""

    def test_func():
      return "test"

    module = BenchmarkModule("test_module", test_func, "Test description")

    assert module.name == "test_module"
    assert module.test_func == test_func
    assert module.description == "Test description"

  def test_initialization_default_description(self):
    """Test module initialization with default description."""

    def test_func():
      return "test"

    module = BenchmarkModule("test_module", test_func)

    assert module.name == "test_module"
    assert module.description == "test_module"


class TestBenchmarkResult:
  """Test BenchmarkResult class."""

  def test_initialization(self):
    """Test result initialization."""
    result = BenchmarkResult("test_module")

    assert result.module_name == "test_module"
    assert result.status == "pending"
    assert result.time_seconds == 0.0
    assert result.memory_mb == 0.0
    assert result.records_processed == 0
    assert result.throughput_records_per_sec == 0.0
    assert result.error_message == ""


class TestBenchmarkRunner:
  """Test BenchmarkRunner class."""

  def test_initialization(self):
    """Test runner initialization."""
    with tempfile.TemporaryDirectory() as temp_dir:
      xml_path = Path(temp_dir) / "test.xml"
      xml_path.write_text("<xml></xml>")

      runner = BenchmarkRunner(xml_path)

      assert runner.xml_path == xml_path
      assert runner.output_dir.exists()
      assert runner.timeout_seconds == 30
      assert runner.sample_records == []
      assert runner.results == []

  def test_initialization_with_output_dir(self):
    """Test runner initialization with custom output directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
      xml_path = Path(temp_dir) / "test.xml"
      xml_path.write_text("<xml></xml>")
      output_dir = Path(temp_dir) / "output"

      runner = BenchmarkRunner(xml_path, output_dir)

      assert runner.output_dir == output_dir

  @patch("src.processors.benchmark.psutil")
  def test_get_memory_usage(self, mock_psutil):
    """Test memory usage retrieval."""
    mock_process = MagicMock()
    mock_process.memory_info.return_value.rss = 1024 * 1024 * 50  # 50 MB
    mock_psutil.Process.return_value = mock_process

    with tempfile.TemporaryDirectory() as temp_dir:
      xml_path = Path(temp_dir) / "test.xml"
      xml_path.write_text("<xml></xml>")

      runner = BenchmarkRunner(xml_path)
      memory_mb = runner.get_memory_usage()

      assert memory_mb == 50.0

  def test_run_with_timeout_success(self):
    """Test running function with timeout - success case."""

    def test_func(x, y):
      return x + y

    with tempfile.TemporaryDirectory() as temp_dir:
      xml_path = Path(temp_dir) / "test.xml"
      xml_path.write_text("<xml></xml>")

      runner = BenchmarkRunner(xml_path)
      result = runner._run_with_timeout(test_func, 2, 3)

      assert result == 5

  def test_run_with_timeout_timeout(self):
    """Test running function with timeout - timeout case."""

    def slow_func():
      import time

      time.sleep(2)  # Sleep longer than timeout

    with tempfile.TemporaryDirectory() as temp_dir:
      xml_path = Path(temp_dir) / "test.xml"
      xml_path.write_text("<xml></xml>")

      runner = BenchmarkRunner(xml_path, timeout_seconds=1)

      with pytest.raises(TimeoutError):
        runner._run_with_timeout(slow_func)

  def test_run_with_timeout_exception(self):
    """Test running function with timeout - exception case."""

    def failing_func():
      raise ValueError("Test error")

    with tempfile.TemporaryDirectory() as temp_dir:
      xml_path = Path(temp_dir) / "test.xml"
      xml_path.write_text("<xml></xml>")

      runner = BenchmarkRunner(xml_path)

      with pytest.raises(ValueError, match="Test error"):
        runner._run_with_timeout(failing_func)

  @patch("src.processors.benchmark.StreamingXMLParser")
  def test_load_sample_data(self, mock_parser_class):
    """Test loading sample data."""
    # Mock parser
    mock_parser = MagicMock()
    mock_records = [{"type": "test", "value": 1}] * 100
    mock_parser.parse_records.return_value = iter(mock_records)
    mock_parser_class.return_value = mock_parser

    with tempfile.TemporaryDirectory() as temp_dir:
      xml_path = Path(temp_dir) / "test.xml"
      xml_path.write_text("<xml></xml>")

      runner = BenchmarkRunner(xml_path)
      records, metrics = runner.load_sample_data(limit=50)

      assert len(records) == 50
      assert metrics["records_processed"] == 50
      assert "throughput_records_per_sec" in metrics
      assert "memory_delta_mb" in metrics
      assert "parse_time_seconds" in metrics

  @patch("src.processors.benchmark.DataCleaner")
  def test_benchmark_data_cleaning(self, mock_cleaner_class):
    """Test data cleaning benchmark."""
    mock_cleaner = MagicMock()
    mock_cleaner.deduplicate_by_time_window.return_value = (
      [],
      MagicMock(removed_duplicates=5),
    )
    mock_cleaner_class.return_value = mock_cleaner

    sample_records = [{"type": "test"}] * 100

    with tempfile.TemporaryDirectory() as temp_dir:
      xml_path = Path(temp_dir) / "test.xml"
      xml_path.write_text("<xml></xml>")

      runner = BenchmarkRunner(xml_path)
      result = runner.benchmark_data_cleaning(sample_records)

      assert result["records_processed"] == 100
      assert "throughput_records_per_sec" in result
      assert "memory_delta_mb" in result
      assert result["cleaned_records"] == 0
      assert result["duplicates_removed"] == 5

  @patch("src.processors.benchmark.StatisticalAnalyzer")
  def test_benchmark_statistical_analysis(self, mock_analyzer_class):
    """Test statistical analysis benchmark."""
    mock_analyzer = MagicMock()
    mock_analyzer.generate_report.return_value = {}
    mock_analyzer_class.return_value = mock_analyzer

    sample_records = [{"type": "test"}] * 100

    with tempfile.TemporaryDirectory() as temp_dir:
      xml_path = Path(temp_dir) / "test.xml"
      xml_path.write_text("<xml></xml>")

      runner = BenchmarkRunner(xml_path)
      result = runner.benchmark_statistical_analysis(sample_records)

      assert result["records_processed"] == 100
      assert "throughput_records_per_sec" in result
      assert "memory_delta_mb" in result

  @patch("src.processors.benchmark.HighlightsGenerator")
  def test_benchmark_report_generation(self, mock_generator_class):
    """Test report generation benchmark."""
    mock_generator = MagicMock()
    mock_generator.generate_comprehensive_highlights.return_value = {}
    mock_generator_class.return_value = mock_generator

    sample_records = [{"type": "test"}] * 100

    with tempfile.TemporaryDirectory() as temp_dir:
      xml_path = Path(temp_dir) / "test.xml"
      xml_path.write_text("<xml></xml>")

      runner = BenchmarkRunner(xml_path)
      result = runner.benchmark_report_generation(sample_records)

      assert result["records_processed"] == 100
      assert "throughput_records_per_sec" in result
      assert "memory_delta_mb" in result

  def test_benchmark_data_export(self):
    """Test data export benchmark."""
    # Create mock records that can be converted to DataFrame
    from unittest.mock import MagicMock

    mock_record = MagicMock()
    mock_record.model_dump.return_value = {
      "type": "HKQuantityTypeIdentifierHeartRate",
      "value": 70.0,
      "unit": "count/min",
      "start_date": "2023-01-01T10:00:00Z",
      "end_date": "2023-01-01T10:01:00Z",
      "source_name": "Test Source",
    }
    sample_records = [mock_record] * 10

    with tempfile.TemporaryDirectory() as temp_dir:
      xml_path = Path(temp_dir) / "test.xml"
      xml_path.write_text("<xml></xml>")

      runner = BenchmarkRunner(xml_path)
      result = runner.benchmark_data_export(sample_records)

      assert result["records_processed"] == 10
      assert "throughput_records_per_sec" in result
      assert "memory_delta_mb" in result
      assert "output_file" in result
      assert "file_size_mb" in result

      # Check that output file was actually created
      output_file = Path(result["output_file"])
      assert output_file.exists()
      assert output_file.stat().st_size > 0

  @patch("src.processors.benchmark.StreamingXMLParser")
  def test_run_all_benchmarks(self, mock_parser_class):
    """Test running all benchmarks."""
    # Mock parser
    mock_parser = MagicMock()
    mock_records = [{"type": "test", "value": 1}] * 100
    mock_parser.parse_records.return_value = iter(mock_records)
    mock_parser_class.return_value = mock_parser

    with tempfile.TemporaryDirectory() as temp_dir:
      xml_path = Path(temp_dir) / "test.xml"
      xml_path.write_text("<xml></xml>")

      runner = BenchmarkRunner(xml_path)
      results = runner.run_all_benchmarks()

      assert len(results) >= 5  # XML parsing + 4 benchmark modules
      assert all(isinstance(r, BenchmarkResult) for r in results)

  def test_run_module_with_timeout_success(self):
    """Test running module with timeout - success case."""

    def test_func():
      import time

      time.sleep(0.01)  # Small delay to ensure measurable time
      return {"records_processed": 100, "throughput_records_per_sec": 1000.0}

    module = BenchmarkModule("test", test_func)

    with tempfile.TemporaryDirectory() as temp_dir:
      xml_path = Path(temp_dir) / "test.xml"
      xml_path.write_text("<xml></xml>")

      runner = BenchmarkRunner(xml_path)
      result = runner.run_module_with_timeout(module)

      assert result.status == "completed"
      assert result.time_seconds >= 0.01  # Should be at least the sleep time
      assert result.records_processed == 100
      assert result.throughput_records_per_sec == 1000.0

  def test_run_module_with_timeout_timeout(self):
    """Test running module with timeout - timeout case."""

    def slow_func():
      import time

      time.sleep(2)

    module = BenchmarkModule("slow_test", slow_func)

    with tempfile.TemporaryDirectory() as temp_dir:
      xml_path = Path(temp_dir) / "test.xml"
      xml_path.write_text("<xml></xml>")

      runner = BenchmarkRunner(xml_path, timeout_seconds=1)
      result = runner.run_module_with_timeout(module)

      assert result.status == "timeout"
      assert result.time_seconds == 1

  def test_run_module_with_timeout_error(self):
    """Test running module with timeout - error case."""

    def failing_func():
      raise ValueError("Test error")

    module = BenchmarkModule("failing_test", failing_func)

    with tempfile.TemporaryDirectory() as temp_dir:
      xml_path = Path(temp_dir) / "test.xml"
      xml_path.write_text("<xml></xml>")

      runner = BenchmarkRunner(xml_path)
      result = runner.run_module_with_timeout(module)

      assert result.status == "error"
      assert "Test error" in result.error_message


class TestRunBenchmark:
  """Test run_benchmark function."""

  def test_run_benchmark_file_not_found(self):
    """Test run_benchmark with non-existent file."""
    with pytest.raises(FileNotFoundError):
      run_benchmark("non_existent.xml")

  @patch("src.processors.benchmark.BenchmarkRunner")
  def test_run_benchmark_success(self, mock_runner_class):
    """Test run_benchmark success case."""
    mock_runner = MagicMock()
    mock_runner.run_all_benchmarks.return_value = []
    mock_runner_class.return_value = mock_runner

    with tempfile.TemporaryDirectory() as temp_dir:
      xml_path = Path(temp_dir) / "test.xml"
      xml_path.write_text("<xml></xml>")

      results = run_benchmark(str(xml_path))

      assert results == []
      mock_runner.run_all_benchmarks.assert_called_once()
      mock_runner.print_report.assert_called_once()
