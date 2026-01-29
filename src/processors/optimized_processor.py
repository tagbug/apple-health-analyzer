"""Optimized data processing module for improved performance.

Provides high-performance data processing capabilities including parallel processing,
memory-efficient operations, and optimized algorithms.
"""

import multiprocessing
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..core.data_models import HealthRecord
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OptimizedDataFrame:
  """Memory-optimized DataFrame structure for health data."""

  timestamps: np.ndarray
  values: np.ndarray
  types: np.ndarray
  sources: np.ndarray
  units: np.ndarray

  def __post_init__(self):
    """Validate data consistency."""
    n_records = len(self.timestamps)
    if not all(
      len(arr) == n_records
      for arr in [self.values, self.types, self.sources, self.units]
    ):
      raise ValueError("All arrays must have the same length")

  @property
  def record_count(self) -> int:
    """Get number of records."""
    return len(self.timestamps)

  def to_pandas(self) -> pd.DataFrame:
    """Convert to pandas DataFrame for compatibility."""
    return pd.DataFrame(
      {
        "timestamp": self.timestamps,
        "value": self.values,
        "type": self.types,
        "source": self.sources,
        "unit": self.units,
      }
    )

  @classmethod
  def from_records(
    cls, records: Sequence[HealthRecord]
  ) -> "OptimizedDataFrame":
    """Create OptimizedDataFrame from health records."""
    if not records:
      # Return empty arrays
      empty_array = np.array([], dtype=object)
      return cls(
        timestamps=np.array([], dtype="datetime64[ns]"),
        values=np.array([], dtype=float),
        types=empty_array,
        sources=empty_array,
        units=empty_array,
      )

    # Pre-allocate arrays for better performance
    n_records = len(records)
    timestamps = np.empty(n_records, dtype="datetime64[ns]")
    values = np.empty(n_records, dtype=float)
    types = np.empty(n_records, dtype=object)
    sources = np.empty(n_records, dtype=object)
    units = np.empty(n_records, dtype=object)

    for i, record in enumerate(records):
      timestamps[i] = np.datetime64(record.start_date)
      types[i] = getattr(record, "type", "Unknown")
      sources[i] = getattr(record, "source_name", "Unknown")
      units[i] = getattr(record, "unit", "Unknown")

      # Extract value based on record type
      if (
        hasattr(record, "value") and getattr(record, "value", None) is not None
      ):
        # Handle different value types
        value = record.value
        if isinstance(value, (int, float)):
          values[i] = float(value)
        elif isinstance(value, str):
          # For categorical values, store as NaN and handle separately
          values[i] = np.nan
        else:
          # Try to convert to float, fallback to NaN
          try:
            values[i] = float(value)
          except (ValueError, TypeError):
            values[i] = np.nan
      else:
        values[i] = np.nan

    return cls(timestamps, values, types, sources, units)


class ParallelProcessor:
  """Parallel processing utilities for health data analysis."""

  def __init__(self, max_workers: int | None = None):
    """Initialize parallel processor.

    Args:
        max_workers: Maximum number of worker processes (default: CPU count)
    """
    self.max_workers = max_workers or multiprocessing.cpu_count()
    self.logger = get_logger(__name__)

  def process_records_parallel(
    self,
    records: Sequence[HealthRecord],
    process_func: Callable[[Sequence[HealthRecord]], Any],
    chunk_size: int = 10000,
  ) -> list[Any]:
    """Process records in parallel chunks.

    Args:
        records: List of health records to process
        process_func: Function to apply to each chunk
        chunk_size: Size of each processing chunk

    Returns:
        List of processing results
    """
    if len(records) <= chunk_size:
      # No need for parallel processing
      return [process_func(records)]

    # Split records into chunks
    chunks = [
      records[i : i + chunk_size] for i in range(0, len(records), chunk_size)
    ]

    self.logger.info(
      f"Processing {len(records)} records in {len(chunks)} parallel chunks"
    )

    results = []
    with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
      # Submit all chunks for parallel processing
      futures = [executor.submit(process_func, chunk) for chunk in chunks]

      # Collect results
      for future in futures:
        try:
          result = future.result()
          results.append(result)
        except Exception as e:
          self.logger.error(f"Error processing chunk: {e}")
          results.append(None)  # Add None for failed chunks

    # Filter out None results and flatten if needed
    valid_results = [r for r in results if r is not None]
    return valid_results


class StatisticalAggregator:
  """High-performance statistical aggregation for health data."""

  def __init__(self):
    """Initialize statistical aggregator."""
    self.logger = get_logger(__name__)

  def aggregate_by_time_window(
    self,
    odf: OptimizedDataFrame,
    window: str = "1D",
    agg_funcs: list[str] | None = None,
  ) -> pd.DataFrame:
    """Aggregate data by time windows using optimized operations.

    Args:
        odf: OptimizedDataFrame to aggregate
        window: Time window (e.g., '1D', '1H', '1W')
        agg_funcs: Aggregation functions to apply

    Returns:
        Aggregated DataFrame
    """
    if odf.record_count == 0:
      return pd.DataFrame()

    if agg_funcs is None:
      agg_funcs = ["mean", "std", "min", "max", "count"]

    # Convert to pandas for time-based operations
    df = odf.to_pandas()

    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Set timestamp as index for resampling
    df = df.set_index("timestamp")

    # Group by time window and aggregate
    try:
      # Create aggregation dictionary - use proper typing
      agg_dict: dict[str, Any] = {"value": agg_funcs}
      # Only add non-numeric columns if they exist
      for col in ["type", "source", "unit"]:
        if col in df.columns:
          agg_dict[col] = "first"

      # Resample and aggregate
      result = df.resample(window).agg(agg_dict)  # type: ignore

      # Flatten column names
      result.columns = ["_".join(col).strip() for col in result.columns]

      # Reset index to get timestamp back as column
      result = result.reset_index()

      return result

    except Exception as e:
      self.logger.error(f"Error in time window aggregation: {e}")
      return pd.DataFrame()

  def calculate_rolling_stats(
    self, odf: OptimizedDataFrame, window: int = 7, center: bool = False
  ) -> pd.DataFrame:
    """Calculate rolling statistics efficiently.

    Args:
        odf: OptimizedDataFrame to process
        window: Rolling window size (in days for daily data)
        center: Whether to center the window

    Returns:
        DataFrame with rolling statistics
    """
    if odf.record_count == 0:
      return pd.DataFrame()

    df = odf.to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    # Calculate rolling statistics
    rolling_stats = (
      df["value"]
      .rolling(window=f"{window}D", center=center, min_periods=1)
      .agg(["mean", "std", "min", "max"])
    )

    # Rename columns
    rolling_stats.columns = [
      f"rolling_{col}_{window}d" for col in rolling_stats.columns
    ]

    return rolling_stats.reset_index()


class MemoryOptimizer:
  """Memory optimization utilities for large datasets."""

  def __init__(self):
    """Initialize memory optimizer."""
    self.logger = get_logger(__name__)

  def optimize_dataframe_types(self, df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame column types for memory efficiency.

    Args:
        df: DataFrame to optimize

    Returns:
        Memory-optimized DataFrame
    """
    optimized_df = df.copy()

    for col in optimized_df.columns:
      col_type = optimized_df[col].dtype

      # Optimize numeric columns
      if col_type == "float64":
        # Check if we can use float32
        if optimized_df[col].notna().any():
          min_val, max_val = optimized_df[col].min(), optimized_df[col].max()
          if abs(min_val) < 1e6 and abs(max_val) < 1e6:
            optimized_df[col] = optimized_df[col].astype("float32")
      elif col_type == "int64":
        # Check if we can use smaller integer types
        min_val, max_val = optimized_df[col].min(), optimized_df[col].max()
        if min_val >= 0:
          if max_val < 256:
            optimized_df[col] = optimized_df[col].astype("uint8")
          elif max_val < 65536:
            optimized_df[col] = optimized_df[col].astype("uint16")
          elif max_val < 4294967296:
            optimized_df[col] = optimized_df[col].astype("uint32")
        else:
          if min_val >= -128 and max_val < 128:
            optimized_df[col] = optimized_df[col].astype("int8")
          elif min_val >= -32768 and max_val < 32768:
            optimized_df[col] = optimized_df[col].astype("int16")
          elif min_val >= -2147483648 and max_val < 2147483648:
            optimized_df[col] = optimized_df[col].astype("int32")

      # Optimize object columns (strings)
      elif col_type == "object":
        # Check if column contains only strings
        if optimized_df[col].notna().any():
          try:
            # Try to convert to category if low cardinality
            unique_ratio = optimized_df[col].nunique() / len(optimized_df[col])
            if unique_ratio < 0.1:  # Less than 10% unique values
              optimized_df[col] = optimized_df[col].astype("category")
          except (TypeError, ValueError):
            pass  # Keep as object if conversion fails

    # Log memory usage
    memory_usage = optimized_df.memory_usage(deep=True).sum()
    self.logger.info(
      f"Optimized DataFrame memory usage: {memory_usage / 1024 / 1024:.2f} MB"
    )

    return optimized_df

  def chunked_processing(
    self,
    data: list[Any],
    process_func: Callable[[list[Any]], Any],
    chunk_size: int = 50000,
  ) -> list[Any]:
    """Process large datasets in chunks to manage memory.

    Args:
        data: Data to process
        process_func: Function to apply to each chunk
        chunk_size: Size of each chunk

    Returns:
        List of processing results
    """
    results = []

    for i in range(0, len(data), chunk_size):
      chunk = data[i : i + chunk_size]
      self.logger.debug(
        f"Processing chunk {i // chunk_size + 1} with {len(chunk)} items"
      )

      try:
        chunk_result = process_func(chunk)
        results.append(chunk_result)
      except Exception as e:
        self.logger.error(f"Error processing chunk {i // chunk_size + 1}: {e}")
        results.append(None)

    return results


class PerformanceMonitor:
  """Performance monitoring and profiling utilities."""

  def __init__(self):
    """Initialize performance monitor."""
    self.logger = get_logger(__name__)
    self.start_times = {}
    self.metrics = defaultdict(list)

  def start_operation(self, operation_name: str):
    """Start timing an operation.

    Args:
        operation_name: Name of the operation to time
    """
    self.start_times[operation_name] = datetime.now()

  def end_operation(self, operation_name: str):
    """End timing an operation and log duration.

    Args:
        operation_name: Name of the operation to end timing
    """
    if operation_name in self.start_times:
      duration = datetime.now() - self.start_times[operation_name]
      duration_seconds = duration.total_seconds()

      self.metrics[operation_name].append(duration_seconds)
      self.logger.info(
        f"Operation '{operation_name}' completed in {duration_seconds:.2f}s"
      )

      del self.start_times[operation_name]

  def get_operation_stats(self, operation_name: str) -> dict[str, Any]:
    """Get statistics for an operation.

    Args:
        operation_name: Name of the operation

    Returns:
        Dictionary with operation statistics
    """
    if operation_name not in self.metrics:
      return {}

    times = self.metrics[operation_name]
    return {
      "count": len(times),
      "total_time": sum(times),
      "avg_time": float(np.mean(times)),
      "min_time": float(min(times)),
      "max_time": float(max(times)),
      "std_time": float(np.std(times)),
    }

  def log_performance_summary(self):
    """Log a summary of all monitored operations."""
    if not self.metrics:
      return

    self.logger.info("=== Performance Summary ===")
    for operation, times in self.metrics.items():
      stats = self.get_operation_stats(operation)
      self.logger.info(
        f"{operation}: {stats['count']} runs, "
        f"avg {stats['avg_time']:.2f}s, "
        f"total {stats['total_time']:.2f}s"
      )


# Convenience functions for easy access
def create_optimized_dataframe(
  records: Sequence[HealthRecord],
) -> OptimizedDataFrame:
  """Create an optimized DataFrame from health records.

  Args:
      records: List of health records

  Returns:
      OptimizedDataFrame instance
  """
  return OptimizedDataFrame.from_records(records)


def aggregate_health_data_parallel(
  records: Sequence[HealthRecord],
  window: str = "1D",
  max_workers: int | None = None,
) -> pd.DataFrame:
  """Aggregate health data in parallel for better performance.

  Args:
      records: Health records to aggregate
      window: Time window for aggregation
      max_workers: Maximum number of worker processes

  Returns:
      Aggregated DataFrame
  """
  processor = ParallelProcessor(max_workers)
  aggregator = StatisticalAggregator()

  # Split records by type for parallel processing
  records_by_type = defaultdict(list)
  for record in records:
    record_type = getattr(record, "type", "Unknown")
    records_by_type[record_type].append(record)

  results = []
  for record_type, type_records in records_by_type.items():
    logger.info(f"Processing {len(type_records)} records of type {record_type}")

    def process_type_chunk(chunk):
      odf = OptimizedDataFrame.from_records(chunk)
      return aggregator.aggregate_by_time_window(odf, window)

    type_results = processor.process_records_parallel(
      type_records, process_type_chunk, chunk_size=50000
    )

    # Combine results for this type
    if type_results:
      combined = pd.concat(
        [r for r in type_results if r is not None and not r.empty]
      )
      if not combined.empty:
        combined["record_type"] = record_type
        results.append(combined)

  # Combine all types
  if results:
    final_result = pd.concat(results, ignore_index=True)
    return final_result

  return pd.DataFrame()
