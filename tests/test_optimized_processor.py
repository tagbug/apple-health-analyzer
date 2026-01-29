"""Tests for optimized processor module."""

import multiprocessing
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.core.data_models import QuantityRecord
from src.processors.optimized_processor import (
  MemoryOptimizer,
  OptimizedDataFrame,
  ParallelProcessor,
  PerformanceMonitor,
  StatisticalAggregator,
  aggregate_health_data_parallel,
  create_optimized_dataframe,
)


class TestOptimizedDataFrame:
  """Test OptimizedDataFrame class."""

  def test_initialization_valid(self):
    """Test valid initialization."""
    timestamps = np.array(
      [np.datetime64("2024-01-01"), np.datetime64("2024-01-02")]
    )
    values = np.array([70.0, 75.0])
    types = np.array(["HR", "HR"])
    sources = np.array(["Watch", "Watch"])
    units = np.array(["bpm", "bpm"])

    odf = OptimizedDataFrame(timestamps, values, types, sources, units)

    assert odf.record_count == 2
    assert len(odf.timestamps) == 2
    assert len(odf.values) == 2

  def test_initialization_invalid_lengths(self):
    """Test initialization with mismatched array lengths."""
    timestamps = np.array(
      [np.datetime64("2024-01-01"), np.datetime64("2024-01-02")]
    )
    values = np.array([70.0])  # Different length
    types = np.array(["HR", "HR"])
    sources = np.array(["Watch", "Watch"])
    units = np.array(["bpm", "bpm"])

    with pytest.raises(
      ValueError, match="All arrays must have the same length"
    ):
      OptimizedDataFrame(timestamps, values, types, sources, units)

  def test_to_pandas(self):
    """Test conversion to pandas DataFrame."""
    timestamps = np.array(
      [np.datetime64("2024-01-01"), np.datetime64("2024-01-02")]
    )
    values = np.array([70.0, 75.0])
    types = np.array(["HR", "HR"])
    sources = np.array(["Watch", "Watch"])
    units = np.array(["bpm", "bpm"])

    odf = OptimizedDataFrame(timestamps, values, types, sources, units)
    df = odf.to_pandas()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["timestamp", "value", "type", "source", "unit"]
    assert df.iloc[0]["value"] == 70.0

  def test_from_records_empty(self):
    """Test creating from empty records."""
    odf = OptimizedDataFrame.from_records([])

    assert odf.record_count == 0
    assert len(odf.timestamps) == 0
    assert len(odf.values) == 0

  def test_from_records_with_data(self):
    """Test creating from health records."""
    records = [
      QuantityRecord(
        type="HKQuantityTypeIdentifierHeartRate",
        source_name="TestWatch",
        source_version="1.0",
        device="TestDevice",
        unit="count/min",
        creation_date=datetime(2024, 1, 1, 10, 0),
        start_date=datetime(2024, 1, 1, 10, 0),
        end_date=datetime(2024, 1, 1, 10, 1),
        value=70,
      ),
      QuantityRecord(
        type="HKQuantityTypeIdentifierHeartRate",
        source_name="TestWatch",
        source_version="1.0",
        device="TestDevice",
        unit="count/min",
        creation_date=datetime(2024, 1, 1, 11, 0),
        start_date=datetime(2024, 1, 1, 11, 0),
        end_date=datetime(2024, 1, 1, 11, 1),
        value=75,
      ),
    ]

    odf = OptimizedDataFrame.from_records(records)

    assert odf.record_count == 2
    assert odf.values[0] == 70.0
    assert odf.values[1] == 75.0
    assert odf.types[0] == "HKQuantityTypeIdentifierHeartRate"
    assert odf.sources[0] == "TestWatch"
    assert odf.units[0] == "count/min"


class TestParallelProcessor:
  """Test ParallelProcessor class."""

  def test_initialization_default_workers(self):
    """Test initialization with default workers."""
    processor = ParallelProcessor()

    assert processor.max_workers == multiprocessing.cpu_count()

  def test_initialization_custom_workers(self):
    """Test initialization with custom workers."""
    processor = ParallelProcessor(max_workers=4)

    assert processor.max_workers == 4

  def test_process_records_parallel_small_dataset(self):
    """Test parallel processing with small dataset."""
    processor = ParallelProcessor(max_workers=2)

    records = [
      QuantityRecord(
        type="HKQuantityTypeIdentifierHeartRate",
        source_name="Test",
        source_version="1.0",
        device="TestDevice",
        unit="count/min",
        creation_date=datetime(2024, 1, 1, 10, i),
        start_date=datetime(2024, 1, 1, 10, i),
        end_date=datetime(2024, 1, 1, 10, i + 1),
        value=70 + i,
      )
      for i in range(5)
    ]

    def process_func(chunk):
      return len(chunk)

    results = processor.process_records_parallel(
      records, process_func, chunk_size=10
    )

    assert len(results) == 1  # Small dataset, no chunking
    assert results[0] == 5

  @patch("src.processors.optimized_processor.ProcessPoolExecutor")
  def test_process_records_parallel_large_dataset(self, mock_executor_class):
    """Test parallel processing with large dataset."""
    processor = ParallelProcessor(max_workers=2)

    # Mock executor
    mock_executor = MagicMock()
    mock_future1 = MagicMock()
    mock_future1.result.return_value = "result1"
    mock_future2 = MagicMock()
    mock_future2.result.return_value = "result2"
    mock_future3 = MagicMock()
    mock_future3.result.return_value = "result3"
    mock_executor.__enter__.return_value = mock_executor
    mock_executor.submit.side_effect = [
      mock_future1,
      mock_future2,
      mock_future3,
    ]
    mock_executor_class.return_value = mock_executor

    records = [
      QuantityRecord(
        type="HKQuantityTypeIdentifierHeartRate",
        source_name="Test",
        source_version="1.0",
        device="TestDevice",
        unit="count/min",
        creation_date=datetime(2024, 1, 1, 10, i),
        start_date=datetime(2024, 1, 1, 10, i),
        end_date=datetime(2024, 1, 1, 10, i + 1),
        value=70 + i,
      )
      for i in range(25)
    ]

    def process_func(chunk):
      return len(chunk)

    results = processor.process_records_parallel(
      records, process_func, chunk_size=10
    )

    assert len(results) == 3  # 25 records / 10 chunk_size = 3 chunks
    assert results == ["result1", "result2", "result3"]

  @patch("src.processors.optimized_processor.ProcessPoolExecutor")
  def test_process_records_parallel_with_error(self, mock_executor_class):
    """Test parallel processing with chunk error."""
    processor = ParallelProcessor(max_workers=2)

    # Mock executor with error
    mock_executor = MagicMock()
    mock_future1 = MagicMock()
    mock_future1.result.return_value = "result1"
    mock_future2 = MagicMock()
    mock_future2.result.side_effect = Exception("Processing error")
    mock_future3 = MagicMock()
    mock_future3.result.return_value = "result3"
    mock_executor.__enter__.return_value = mock_executor
    mock_executor.submit.side_effect = [
      mock_future1,
      mock_future2,
      mock_future3,
    ]
    mock_executor_class.return_value = mock_executor

    records = [
      QuantityRecord(
        type="HKQuantityTypeIdentifierHeartRate",
        source_name="Test",
        source_version="1.0",
        device="TestDevice",
        unit="count/min",
        creation_date=datetime(2024, 1, 1, 10, i),
        start_date=datetime(2024, 1, 1, 10, i),
        end_date=datetime(2024, 1, 1, 10, i + 1),
        value=70 + i,
      )
      for i in range(25)
    ]

    def process_func(chunk):
      return len(chunk)

    results = processor.process_records_parallel(
      records, process_func, chunk_size=10
    )

    assert len(results) == 2  # Two successful results (first and third chunks)
    assert results == ["result1", "result3"]


class TestStatisticalAggregator:
  """Test StatisticalAggregator class."""

  def test_aggregate_by_time_window_empty(self):
    """Test aggregation with empty data."""
    aggregator = StatisticalAggregator()

    # Create empty OptimizedDataFrame
    odf = OptimizedDataFrame.from_records([])
    result = aggregator.aggregate_by_time_window(odf, "1D")

    assert isinstance(result, pd.DataFrame)
    assert result.empty

  def test_aggregate_by_time_window_with_data(self):
    """Test aggregation with data."""
    aggregator = StatisticalAggregator()

    # Create test records
    records = []
    base_time = datetime(2024, 1, 1, 10, 0)
    for i in range(10):
      records.append(
        QuantityRecord(
          type="HKQuantityTypeIdentifierHeartRate",
          source_name="Test",
          source_version="1.0",
          device="TestDevice",
          unit="count/min",
          creation_date=base_time + timedelta(hours=i),
          start_date=base_time + timedelta(hours=i),
          end_date=base_time + timedelta(hours=i, minutes=1),
          value=70 + i,
        )
      )

    odf = OptimizedDataFrame.from_records(records)
    result = aggregator.aggregate_by_time_window(odf, "1D")

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert "timestamp" in result.columns
    assert any("value_" in col for col in result.columns)

  def test_calculate_rolling_stats_empty(self):
    """Test rolling stats with empty data."""
    aggregator = StatisticalAggregator()

    odf = OptimizedDataFrame.from_records([])
    result = aggregator.calculate_rolling_stats(odf)

    assert isinstance(result, pd.DataFrame)
    assert result.empty

  def test_calculate_rolling_stats_with_data(self):
    """Test rolling stats with data."""
    aggregator = StatisticalAggregator()

    # Create test records spanning multiple days
    records = []
    base_time = datetime(2024, 1, 1, 10, 0)
    for i in range(15):  # 15 days
      records.append(
        QuantityRecord(
          type="HKQuantityTypeIdentifierHeartRate",
          source_name="Test",
          source_version="1.0",
          device="TestDevice",
          unit="count/min",
          creation_date=base_time + timedelta(days=i),
          start_date=base_time + timedelta(days=i),
          end_date=base_time + timedelta(days=i, minutes=1),
          value=70 + (i % 10),  # Some variation
        )
      )

    odf = OptimizedDataFrame.from_records(records)
    result = aggregator.calculate_rolling_stats(odf, window=7)

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert "timestamp" in result.columns
    assert any("rolling_" in col for col in result.columns)


class TestMemoryOptimizer:
  """Test MemoryOptimizer class."""

  def test_optimize_dataframe_types_float64(self):
    """Test float64 optimization."""
    optimizer = MemoryOptimizer()

    df = pd.DataFrame({"value": [1.0, 2.0, 3.0], "other": ["a", "b", "c"]})

    optimized = optimizer.optimize_dataframe_types(df)

    assert optimized["value"].dtype == "float32"

  def test_optimize_dataframe_types_int64(self):
    """Test int64 optimization."""
    optimizer = MemoryOptimizer()

    df = pd.DataFrame({"count": [1, 2, 3], "other": ["a", "b", "c"]})

    optimized = optimizer.optimize_dataframe_types(df)

    # Should be optimized to smaller integer type
    assert optimized["count"].dtype in ["int8", "uint8"]

  def test_optimize_dataframe_types_object_low_cardinality(self):
    """Test object column optimization for low cardinality."""
    optimizer = MemoryOptimizer()

    df = pd.DataFrame(
      {
        "category": ["A", "A", "A", "B", "B", "B"] * 100,  # Low cardinality
        "value": [1, 2, 3] * 200,
      }
    )

    optimized = optimizer.optimize_dataframe_types(df)

    assert optimized["category"].dtype.name == "category"

  def test_chunked_processing(self):
    """Test chunked processing."""
    optimizer = MemoryOptimizer()

    data = list(range(100))

    def process_func(chunk):
      return sum(chunk)

    results = optimizer.chunked_processing(data, process_func, chunk_size=30)

    assert len(results) == 4  # 100 / 30 = 3.33, so 4 chunks
    assert sum(results) == sum(range(100))

  def test_chunked_processing_with_error(self):
    """Test chunked processing with error in one chunk."""
    optimizer = MemoryOptimizer()

    data = list(range(60))

    def process_func(chunk):
      if sum(chunk) > 800:  # Error in later chunks (only second chunk fails)
        raise ValueError("Processing error")
      return sum(chunk)

    results = optimizer.chunked_processing(data, process_func, chunk_size=30)

    # Should have some successful results and None for failed chunks
    assert len(results) == 2
    assert results[0] is not None  # First chunk (0-29, sum=435) succeeds
    assert results[1] is None  # Second chunk (30-59, sum=1035) fails


class TestPerformanceMonitor:
  """Test PerformanceMonitor class."""

  def test_start_end_operation(self):
    """Test basic operation timing."""
    monitor = PerformanceMonitor()

    monitor.start_operation("test_op")
    monitor.end_operation("test_op")

    stats = monitor.get_operation_stats("test_op")

    assert stats["count"] == 1
    assert "avg_time" in stats
    assert "total_time" in stats
    assert stats["avg_time"] >= 0

  def test_multiple_operations(self):
    """Test multiple operations."""
    monitor = PerformanceMonitor()

    monitor.start_operation("op1")
    monitor.end_operation("op1")

    monitor.start_operation("op2")
    monitor.end_operation("op2")

    stats1 = monitor.get_operation_stats("op1")
    stats2 = monitor.get_operation_stats("op2")

    assert stats1["count"] == 1
    assert stats2["count"] == 1

  def test_get_operation_stats_nonexistent(self):
    """Test getting stats for nonexistent operation."""
    monitor = PerformanceMonitor()

    stats = monitor.get_operation_stats("nonexistent")

    assert stats == {}

  def test_log_performance_summary(self):
    """Test performance summary logging."""
    monitor = PerformanceMonitor()

    monitor.start_operation("test_op")
    monitor.end_operation("test_op")

    # Should not raise any exceptions
    monitor.log_performance_summary()


class TestConvenienceFunctions:
  """Test convenience functions."""

  def test_create_optimized_dataframe(self):
    """Test create_optimized_dataframe function."""
    records = [
      QuantityRecord(
        type="HKQuantityTypeIdentifierHeartRate",
        source_name="Test",
        source_version="1.0",
        device="TestDevice",
        unit="count/min",
        creation_date=datetime(2024, 1, 1, 10, 0),
        start_date=datetime(2024, 1, 1, 10, 0),
        end_date=datetime(2024, 1, 1, 10, 1),
        value=70,
      )
    ]

    odf = create_optimized_dataframe(records)

    assert isinstance(odf, OptimizedDataFrame)
    assert odf.record_count == 1

  @patch("src.processors.optimized_processor.ParallelProcessor")
  @patch("src.processors.optimized_processor.StatisticalAggregator")
  def test_aggregate_health_data_parallel(
    self, mock_aggregator_class, mock_processor_class
  ):
    """Test parallel aggregation function."""
    # Mock processor
    mock_processor = MagicMock()
    mock_processor.process_records_parallel.return_value = [
      pd.DataFrame({"test": [1, 2, 3]})
    ]
    mock_processor_class.return_value = mock_processor

    # Mock aggregator
    mock_aggregator = MagicMock()
    mock_aggregator.aggregate_by_time_window.return_value = pd.DataFrame(
      {"test": [1, 2, 3]}
    )
    mock_aggregator_class.return_value = mock_aggregator

    records = [
      QuantityRecord(
        type="HKQuantityTypeIdentifierHeartRate",
        source_name="Test",
        source_version="1.0",
        device="TestDevice",
        unit="count/min",
        creation_date=datetime(2024, 1, 1, 10, 0),
        start_date=datetime(2024, 1, 1, 10, 0),
        end_date=datetime(2024, 1, 1, 10, 1),
        value=70,
      )
    ]

    result = aggregate_health_data_parallel(records, "1D", max_workers=2)

    assert isinstance(result, pd.DataFrame)
