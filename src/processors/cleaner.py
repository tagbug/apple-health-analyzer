"""Data cleaning and preprocessing module.

Provides data deduplication, merging, quality validation, and other functions.
"""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, cast

import pandas as pd
from pydantic import BaseModel, Field

from src.core.data_models import HealthRecord
from src.i18n import Translator, resolve_locale
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RecordRowData(BaseModel):
  """Type-safe wrapper for DataFrame row data.

  Used to convert between DataFrame rows and HealthRecord objects safely.
  """

  type: str
  source_name: str
  start_date: datetime
  end_date: datetime | None
  creation_date: datetime
  source_version: str = "1.0"
  device: str = "Unknown"
  unit: str | None = None
  value: float | str | None = None
  metadata: dict[str, Any] | None = None

  @classmethod
  def from_series(cls, row: pd.Series, record_type: str) -> "RecordRowData":
    """Create a typed RecordRowData from a pandas Series."""
    # Extract and convert fields manually to avoid pandas type inference issues.
    start_date_val = row["start_date"]
    if isinstance(start_date_val, str):
      start_date = pd.to_datetime(start_date_val).to_pydatetime()
    elif isinstance(start_date_val, pd.Timestamp):
      start_date = start_date_val.to_pydatetime()
    else:
      # Use cast to inform the type checker.
      start_date = cast(datetime, start_date_val)

    end_date_val = row.get("end_date")
    if end_date_val is not None and str(end_date_val).lower() not in (
      "",
      "nan",
      "none",
    ):
      if isinstance(end_date_val, str):
        end_date = pd.to_datetime(end_date_val).to_pydatetime()
      elif isinstance(end_date_val, pd.Timestamp):
        end_date = end_date_val.to_pydatetime()
      else:
        end_date = cast(datetime, end_date_val)
    else:
      end_date = None

    creation_date_val = row["creation_date"]
    if isinstance(creation_date_val, str):
      creation_date = pd.to_datetime(creation_date_val).to_pydatetime()
    elif isinstance(creation_date_val, pd.Timestamp):
      creation_date = creation_date_val.to_pydatetime()
    else:
      # Use cast to inform the type checker.
      creation_date = cast(datetime, creation_date_val)

    # Safely extract remaining fields.
    source_name = str(row.get("source_name", "Unknown"))
    source_version = str(row.get("source_version", "1.0"))
    device = str(row.get("device", "Unknown"))

    unit_val = row.get("unit")
    unit = (
      str(unit_val)
      if unit_val is not None and str(unit_val).lower() not in ("", "nan", "none")
      else None
    )

    value_val = row.get("value")
    # For category records value is string; for quantity records value is numeric.
    if value_val is not None and str(value_val).lower() not in (
      "",
      "nan",
      "none",
    ):
      # Check whether this is a sleep/category record.
      if "SleepAnalysis" in record_type or "Category" in record_type:
        # Keep category values as strings.
        value = str(value_val)
      else:
        # Convert quantity values to float.
        try:
          value = float(value_val)
        except (ValueError, TypeError):
          value = str(value_val)  # Keep string if conversion fails.
    else:
      value = None

    metadata = row.get("metadata", {})
    if metadata is None:
      metadata = {}

    return cls(
      type=record_type,
      source_name=source_name,
      start_date=start_date,
      end_date=end_date,
      creation_date=creation_date,
      source_version=source_version,
      device=device,
      unit=unit,
      value=value,
      metadata=metadata,
    )

  def to_health_record(self) -> HealthRecord:
    """Convert to a HealthRecord instance."""
    # Default end_date to start_date if missing.
    end_date = self.end_date if self.end_date is not None else self.start_date

    if self.value is not None:
      # Category records have string values.
      if isinstance(self.value, str):
        # Build CategoryRecord.
        from src.core.data_models import CategoryRecord

        return CategoryRecord(
          type=self.type,
          source_name=self.source_name,
          start_date=self.start_date,
          end_date=end_date,  # Guaranteed non-null.
          creation_date=self.creation_date,
          source_version=self.source_version,
          device=self.device,
          unit=None,  # Category records do not have units.
          value=self.value,
          metadata=self.metadata,
        )
      else:
        # Build QuantityRecord for numeric values.
        from src.core.data_models import QuantityRecord

        return QuantityRecord(
          type=self.type,
          source_name=self.source_name,
          start_date=self.start_date,
          end_date=end_date,  # Guaranteed non-null.
          creation_date=self.creation_date,
          source_version=self.source_version,
          device=self.device,
          unit=self.unit,
          value=self.value,
          metadata=self.metadata,
        )
    else:
      # Build a base HealthRecord.
      return HealthRecord(
        type=self.type,
        source_name=self.source_name,
        start_date=self.start_date,
        end_date=end_date,  # Guaranteed non-null.
        creation_date=self.creation_date,
        source_version=self.source_version,
        device=self.device,
        unit=self.unit,
        metadata=self.metadata,
      )


class DataQualityReport(BaseModel):
  """Data quality report."""

  total_records: int
  valid_records: int
  invalid_records: int
  duplicate_records: int
  cleaned_records: int
  quality_score: float  # 0-100

  # Detailed counts
  timestamp_issues: int = 0
  value_issues: int = 0
  metadata_issues: int = 0

  # Distribution data
  source_distribution: dict[str, int] = Field(default_factory=dict)
  type_distribution: dict[str, int] = Field(default_factory=dict)

  # Time range
  date_range: dict[str, datetime | None] = Field(
    default_factory=lambda: {"start": None, "end": None}
  )


class DeduplicationResult(BaseModel):
  """Deduplication result summary."""

  original_count: int
  deduplicated_count: int
  removed_duplicates: int
  strategy_used: str
  processing_time_seconds: float

  # Detailed counts
  duplicates_by_source: dict[str, int] = Field(default_factory=dict)
  time_windows_processed: int = 0


class DataCleaner:
  """Data cleaning core class.

  Provides data preparation utilities:
  - time-window deduplication
  - source priority handling
  - overlapping record merging
  - data quality validation
  """

  def __init__(
    self,
    source_priority: dict[str, int] | None = None,
    default_window_seconds: int = 60,
    locale: str | None = None,
  ):
    """Initialize the data cleaner.

    Args:
        source_priority: Source priority map (higher value is higher priority).
        default_window_seconds: Default deduplication window size in seconds.
    """
    # Default source priority map.
    self.source_priority = source_priority or {
      "🐙Watch": 3,  # Apple Watch highest priority
      "Apple Watch": 3,  # Alias
      "Xiaomi Health": 2,  # Xiaomi Health
      "Xiaomi Health": 2,  # Alias
      "Xiaomi Home": 2,  # Alias
      "🐙Phone": 1,  # iPhone lowest priority
      "iPhone": 1,  # Alias
    }

    self.default_window_seconds = default_window_seconds
    self.translator = Translator(resolve_locale(locale))
    self._fast_dedup_threshold = 50000
    logger.info(
      self.translator.t(
        "log.cleaner.initialized",
        count=len(self.source_priority),
      )
    )

  def deduplicate_by_time_window(
    self,
    records: list[HealthRecord],
    window_seconds: int | None = None,
    strategy: str = "priority",
  ) -> tuple[list[HealthRecord], DeduplicationResult]:
    """Deduplicate records within a time window.

    Args:
        records: Records to process.
        window_seconds: Window size in seconds (defaults to configured value).
        strategy: Deduplication strategy.
            - "priority": keep highest priority source
            - "latest": keep latest record
            - "average": average numeric values
            - "highest_quality": keep highest quality score

    Returns:
        (deduplicated records, deduplication summary)
    """
    if not records:
      return [], DeduplicationResult(
        original_count=0,
        deduplicated_count=0,
        removed_duplicates=0,
        strategy_used=strategy,
        processing_time_seconds=0.0,
      )

    start_time = datetime.now()
    window = window_seconds or self.default_window_seconds

    logger.info(
      self.translator.t(
        "log.cleaner.dedup_start",
        strategy=strategy,
        window=window,
      )
    )

    # Group records by record type.
    records_by_type = defaultdict(list)
    for record in records:
      records_by_type[record.type].append(record)

    deduplicated_records = []
    total_duplicates_removed = 0
    duplicates_by_source = defaultdict(int)

    for record_type, type_records in records_by_type.items():
      logger.debug(
        self.translator.t(
          "log.cleaner.processing_type",
          count=len(type_records),
          record_type=record_type,
        )
      )

      if self._should_use_fast_dedup(len(type_records), strategy):
        logger.info(
          self.translator.t(
            "log.cleaner.fast_path",
            record_type=record_type,
            count=len(type_records),
            strategy=strategy,
          )
        )
        (
          fast_deduped,
          removed_sources,
          removed_count,
        ) = self._deduplicate_records_fast(type_records, window, strategy)

        total_duplicates_removed += removed_count
        for source, count in removed_sources.items():
          duplicates_by_source[source] += count

        deduplicated_records.extend(fast_deduped)
        continue

      # Convert to DataFrame for processing.
      df = self._records_to_dataframe(type_records)

      # Ensure start_date is datetime.
      df["start_date"] = pd.to_datetime(df["start_date"])

      # Ensure creation_date is datetime.
      if "creation_date" in df.columns:
        df["creation_date"] = pd.to_datetime(df["creation_date"])

      # Compute time window using floor rounding.
      df["time_window"] = df["start_date"].dt.floor(f"{window}s")

      original_count = len(df)

      if strategy == "priority":
        # Calculate priority score (higher is higher).
        # Unknown sources default to lowest priority.
        df["priority_score"] = df["source_name"].map(self.source_priority).fillna(0)

        # Sort by time window (asc) and priority (desc).
        df.sort_values(
          by=["time_window", "priority_score"], ascending=[True, False], inplace=True
        )

        # Keep the first record per time window (highest priority).
        deduped_df = df.drop_duplicates(subset=["time_window"], keep="first")

      elif strategy == "latest":
        # Sort by time window (asc) and creation date (desc).
        df.sort_values(
          by=["time_window", "creation_date"], ascending=[True, False], inplace=True
        )

        # Keep the first record per time window (latest).
        deduped_df = df.drop_duplicates(subset=["time_window"], keep="first")

      elif strategy == "average" and self._is_numeric_type(record_type):
        # Ensure value column is numeric.
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        # Compute average values per time window.
        # This drops non-aggregated columns; we reuse the first row as template.

        # 1) Compute averages and counts.
        grouped = df.groupby("time_window")["value"]
        avg_values = grouped.mean()
        counts = grouped.size()

        # 2) Get the first record as a template.
        deduped_df = df.drop_duplicates(subset=["time_window"], keep="first").set_index(
          "time_window"
        )

        # 3) Update value and add counts.
        deduped_df["value"] = avg_values
        deduped_df["_count"] = counts
        deduped_df = deduped_df.reset_index()

        # 4) Mark metadata for averages; detailed metadata is handled later.

      elif strategy == "highest_quality":
        # Calculate quality score.
        # 1) Source priority score (0-40).
        df["priority_score"] = df["source_name"].map(self.source_priority).fillna(0)
        df["quality_score"] = (df["priority_score"] * 10).clip(lower=0, upper=40)

        # 2) Timestamp plausibility (0-30).
        time_diff = (df["creation_date"] - df["start_date"]).abs().dt.total_seconds()
        df.loc[time_diff < 86400, "quality_score"] += 30
        df.loc[(time_diff >= 86400) & (time_diff < 604800), "quality_score"] += 20

        # Sort by quality score (desc).
        df.sort_values(
          by=["time_window", "quality_score"], ascending=[True, False], inplace=True
        )

        # Deduplicate.
        deduped_df = df.drop_duplicates(subset=["time_window"], keep="first")

      else:
        # Default to priority strategy.
        df["priority_score"] = df["source_name"].map(self.source_priority).fillna(0)
        df.sort_values(
          by=["time_window", "priority_score"], ascending=[True, False], inplace=True
        )
        deduped_df = df.drop_duplicates(subset=["time_window"], keep="first")

      # Count removed duplicates.
      removed_count = original_count - len(deduped_df)
      total_duplicates_removed += removed_count

      # Track removed sources.
      if removed_count > 0:
        removed_mask = ~df.index.isin(deduped_df.index)
        removed_sources = df.loc[removed_mask, "source_name"].value_counts()
        for source, count in removed_sources.items():
          duplicates_by_source[source] += count

      # Convert rows back to HealthRecord instances.
      for _, row in deduped_df.iterrows():
        record = self._dataframe_row_to_record(row, record_type)
        if record:
          if strategy == "average" and self._is_numeric_type(record_type):
            # Add metadata marker for average strategy.
            if record.metadata is None:
              record.metadata = {}
            record.metadata["deduplication_method"] = "average"
            if "_count" in row:
              record.metadata["original_records_count"] = int(row["_count"])

          deduplicated_records.append(record)

    processing_time = (datetime.now() - start_time).total_seconds()

    result = DeduplicationResult(
      original_count=len(records),
      deduplicated_count=len(deduplicated_records),
      removed_duplicates=total_duplicates_removed,
      strategy_used=strategy,
      processing_time_seconds=processing_time,
      duplicates_by_source=dict(duplicates_by_source),
      time_windows_processed=len(deduplicated_records),
    )

    logger.info(
      self.translator.t(
        "log.cleaner.dedup_completed",
        original=result.original_count,
        deduped=result.deduplicated_count,
        duplicates=result.removed_duplicates,
      )
    )

    return deduplicated_records, result

  def _should_use_fast_dedup(self, record_count: int, strategy: str) -> bool:
    """Check if fast deduplication path should be used."""
    return record_count >= self._fast_dedup_threshold and strategy in {
      "priority",
      "latest",
    }

  def _deduplicate_records_fast(
    self, records: list[HealthRecord], window_seconds: int, strategy: str
  ) -> tuple[list[HealthRecord], dict[str, int], int]:
    """Fast deduplication using dict-based grouping."""
    selected: dict[int, tuple[HealthRecord, Any]] = {}
    removed_sources = defaultdict(int)

    for record in records:
      time_key = self._compute_time_window_key(record.start_date, window_seconds)

      if strategy == "latest":
        record_value = getattr(record, "creation_date", record.start_date)
      else:
        record_value = self.source_priority.get(record.source_name, 0)

      existing = selected.get(time_key)
      if existing is None:
        selected[time_key] = (record, record_value)
      else:
        _, existing_value = existing
        if record_value > existing_value and strategy == "priority":
          selected[time_key] = (record, record_value)
        elif record_value > existing_value and strategy == "latest":
          selected[time_key] = (record, record_value)

    deduped_records = [record for record, _value in selected.values()]
    selected_ids = {id(record) for record in deduped_records}

    for record in records:
      if id(record) not in selected_ids:
        removed_sources[record.source_name] += 1

    removed_count = len(records) - len(deduped_records)
    return deduped_records, dict(removed_sources), removed_count

  def _compute_time_window_key(self, start_date: datetime, window_seconds: int) -> int:
    """Compute time window key for deduplication."""
    if start_date.tzinfo is not None:
      epoch = datetime(1970, 1, 1, tzinfo=start_date.tzinfo)
    else:
      epoch = datetime(1970, 1, 1)

    seconds = int((start_date - epoch).total_seconds())
    return seconds - (seconds % window_seconds)

  def merge_overlapping_records(
    self, records: list[HealthRecord], merge_threshold_seconds: int = 5
  ) -> list[HealthRecord]:
    """
    Merge overlapping or adjacent records.

    This is mainly used for sleep and workout data that can be split
    into multiple consecutive records.

    Args:
        records: Records to merge.
        merge_threshold_seconds: Merge gap threshold in seconds.

    Returns:
        Merged records.
    """
    if not records or len(records) <= 1:
      return records

    logger.info(
      self.translator.t(
        "log.cleaner.merge_start",
        threshold=merge_threshold_seconds,
      )
    )

    # Group by record type.
    records_by_type = defaultdict(list)
    for record in records:
      records_by_type[record.type].append(record)

    merged_records = []

    for record_type, type_records in records_by_type.items():
      if not self._should_merge_type(record_type):
        # Skip types that should not be merged.
        merged_records.extend(type_records)
        continue

      # Sort then merge.
      sorted_records = sorted(type_records, key=lambda r: r.start_date)
      merged = self._merge_sorted_records(sorted_records, merge_threshold_seconds)
      merged_records.extend(merged)

    logger.info(
      self.translator.t(
        "log.cleaner.merge_completed",
        original=len(records),
        merged=len(merged_records),
      )
    )
    return merged_records

  def validate_data_quality(self, records: list[HealthRecord]) -> DataQualityReport:
    """
    Validate data quality and generate a report.

    Args:
        records: Records to validate.

    Returns:
        Data quality report.
    """
    if not records:
      return DataQualityReport(
        total_records=0,
        valid_records=0,
        invalid_records=0,
        duplicate_records=0,
        cleaned_records=0,
        quality_score=0.0,
      )

    logger.info(
      self.translator.t(
        "log.cleaner.quality_start",
        count=len(records),
      )
    )

    total_records = len(records)
    valid_records = 0
    invalid_records = 0

    # Detailed counts
    timestamp_issues = 0
    value_issues = 0
    metadata_issues = 0

    # Distribution stats
    source_distribution = defaultdict(int)
    type_distribution = defaultdict(int)

    # Time range
    dates = []

    for record in records:
      is_valid = True

      # Validate timestamps.
      if not self._validate_timestamp(record):
        timestamp_issues += 1
        is_valid = False

      # Validate numeric values.
      if not self._validate_value(record):
        value_issues += 1
        is_valid = False

      # Validate metadata.
      if not self._validate_metadata(record):
        metadata_issues += 1
        # Metadata issues do not invalidate records.

      if is_valid:
        valid_records += 1
      else:
        invalid_records += 1

      # Track distributions.
      source_distribution[record.source_name] += 1
      type_distribution[record.type] += 1

      # Collect date range.
      dates.append(record.start_date)

    # Compute quality score.
    quality_score = self._calculate_quality_score(
      total_records, valid_records, timestamp_issues, value_issues
    )

    # Time range.
    date_range = {
      "start": min(dates) if dates else None,
      "end": max(dates) if dates else None,
    }

    # Detect duplicates (simple match by time and value).
    duplicate_records = self._detect_duplicates(records)

    report = DataQualityReport(
      total_records=total_records,
      valid_records=valid_records,
      invalid_records=invalid_records,
      duplicate_records=duplicate_records,
      cleaned_records=valid_records,  # Assume valid records remain after cleaning.
      quality_score=quality_score,
      timestamp_issues=timestamp_issues,
      value_issues=value_issues,
      metadata_issues=metadata_issues,
      source_distribution=dict(source_distribution),
      type_distribution=dict(type_distribution),
      date_range=date_range,
    )

    logger.info(
      self.translator.t(
        "log.cleaner.quality_completed",
        valid=valid_records,
        total=total_records,
        score=quality_score,
      )
    )

    return report

  def _records_to_dataframe(self, records: list[HealthRecord]) -> pd.DataFrame:
    """Convert records to a DataFrame."""
    data = []
    for record in records:
      row = {
        "id": id(record),  # Use object id as a unique key.
        "type": record.type,
        "source_name": record.source_name,
        "start_date": record.start_date,
        "end_date": record.end_date,
        "creation_date": record.creation_date,
        "value": getattr(record, "value", None),
        "unit": getattr(record, "unit", None),
        "metadata": getattr(record, "metadata", None),
      }
      data.append(row)

    return pd.DataFrame(data)

  def _dataframe_row_to_record(
    self, row: pd.Series, record_type: str
  ) -> HealthRecord | None:
    """Convert a DataFrame row back to a record."""
    try:
      # Use RecordRowData for type-safe conversion.
      row_data = RecordRowData.from_series(row, record_type)
      return row_data.to_health_record()

    except Exception as e:
      logger.error(
        self.translator.t(
          "log.cleaner.reconstruct_failed",
          error=e,
        )
      )
      return None

  def _should_merge_type(self, record_type: str) -> bool:
    """Check whether a record type should be merged."""
    # Sleep and some workout records should be merged.
    merge_types = {
      "HKCategoryTypeIdentifierSleepAnalysis",
      "HKWorkoutTypeIdentifier",  # Workout records
    }
    return record_type in merge_types

  def _merge_sorted_records(
    self, records: list[HealthRecord], threshold_seconds: int
  ) -> list[HealthRecord]:
    """Merge a sorted list of records."""
    if not records:
      return []

    merged = [records[0]]

    for current in records[1:]:
      last = merged[-1]

      # Check if records can be merged.
      if self._can_merge_records(last, current, threshold_seconds):
        # Merge records.
        merged[-1] = self._merge_two_records(last, current)
      else:
        # Cannot merge; append as new record.
        merged.append(current)

    return merged

  def _can_merge_records(
    self, record1: HealthRecord, record2: HealthRecord, threshold_seconds: int
  ) -> bool:
    """Check whether two records can be merged."""
    # Allow contiguous or slightly overlapping records.
    time_gap = (record2.start_date - record1.end_date).total_seconds()
    return time_gap <= threshold_seconds

  def _merge_two_records(
    self, record1: HealthRecord, record2: HealthRecord
  ) -> HealthRecord:
    """Merge two records into a single record."""
    # Use the earliest start time and latest end time.
    merged_start = min(record1.start_date, record2.start_date)
    merged_end = max(record1.end_date, record2.end_date)

    # Merge values: average numeric values or keep the first value.
    merged_value = None
    value1 = getattr(record1, "value", None)
    value2 = getattr(record2, "value", None)
    if value1 is not None and value2 is not None:
      if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
        merged_value = (value1 + value2) / 2
      else:
        merged_value = value1

    # Merge metadata.
    merged_metadata = {}
    if record1.metadata:
      merged_metadata.update(record1.metadata)
    if record2.metadata:
      merged_metadata.update(record2.metadata)
    merged_metadata["merged_from"] = 2  # Marks merged result.

    # Build merged record using the first record as template.
    if merged_value is not None and isinstance(merged_value, (int, float)):
      # Numeric record.
      from src.core.data_models import QuantityRecord

      return QuantityRecord(
        type=record1.type,
        source_name=record1.source_name,
        start_date=merged_start,
        end_date=merged_end,
        creation_date=min(record1.creation_date, record2.creation_date),
        source_version=record1.source_version,
        device=record1.device,
        unit=getattr(record1, "unit", None),
        value=merged_value,
        metadata=merged_metadata,
      )
    elif value1 is not None:
      # Category record.
      from src.core.data_models import CategoryRecord

      return CategoryRecord(
        type=record1.type,
        source_name=record1.source_name,
        start_date=merged_start,
        end_date=merged_end,
        creation_date=min(record1.creation_date, record2.creation_date),
        source_version=record1.source_version,
        device=record1.device,
        value=cast(str, str(merged_value) if merged_value is not None else str(value1)),
        unit=None,
        metadata=merged_metadata,
      )
    else:
      # Base record.
      return HealthRecord(
        type=record1.type,
        source_name=record1.source_name,
        start_date=merged_start,
        end_date=merged_end,
        creation_date=min(record1.creation_date, record2.creation_date),
        source_version=record1.source_version,
        device=record1.device,
        unit=None,
        metadata=merged_metadata,
      )

  def _validate_timestamp(self, record: HealthRecord) -> bool:
    """Validate timestamp fields."""
    try:
      # Ensure timestamps exist.
      if not hasattr(record, "start_date") or not record.start_date:
        return False

      # Ensure timestamps are not too far in the future.
      now = datetime.now(record.start_date.tzinfo)
      if record.start_date > now + timedelta(days=1):
        return False

      # Ensure start time is not later than end time.
      if hasattr(record, "end_date") and record.end_date:
        if record.start_date > record.end_date:
          return False

      return True
    except Exception:
      return False

  def _validate_value(self, record: HealthRecord) -> bool:
    """Validate numeric values for quantitative records."""
    try:
      # Skip validation for non-numeric record types.
      if not self._is_numeric_type(record.type):
        return True

      # Ensure numeric records have a value field.
      if not hasattr(record, "value"):
        return False

      value = getattr(record, "value", None)
      if value is None:
        return False  # Numeric records must have a value.

      # Basic numeric checks.
      if not isinstance(value, (int, float)):
        return False

      # Type-specific checks.
      if record.type == "HKQuantityTypeIdentifierHeartRate":
        # Heart rate should be between 30-250 bpm.
        return 30 <= value <= 250
      elif record.type == "HKQuantityTypeIdentifierBodyMass":
        # Body mass should be between 20-300 kg.
        return 20 <= value <= 300

      # Default checks for other types.
      return abs(value) < 1e10  # Avoid extreme values.

    except Exception:
      return False

  def _validate_metadata(self, record: HealthRecord) -> bool:
    """Validate metadata payload."""
    try:
      if not hasattr(record, "metadata"):
        return True

      metadata = record.metadata
      if metadata is None:
        return True

      # Ensure metadata is a dict.
      if not isinstance(metadata, dict):
        return False

      # Basic metadata checks can be expanded here.

      return True
    except Exception:
      return False

  def _calculate_quality_score(
    self, total: int, valid: int, timestamp_issues: int, value_issues: int
  ) -> float:
    """Compute quality score (0-100)."""
    if total == 0:
      return 0.0

    # Validity score (60% weight).
    validity_score = (valid / total) * 60

    # Severity score (40% weight).
    issue_penalty = ((timestamp_issues + value_issues) / total) * 40

    return max(0.0, min(100.0, validity_score - issue_penalty))

  def _detect_duplicates(self, records: list[HealthRecord]) -> int:
    """Simple duplicate detection."""
    seen = set()
    duplicates = 0

    for record in records:
      # Build a record signature (type + time + value).
      signature = (
        record.type,
        record.start_date.isoformat(),
        getattr(record, "value", None),
        record.source_name,
      )

      if signature in seen:
        duplicates += 1
      else:
        seen.add(signature)

    return duplicates

  def _is_numeric_type(self, record_type: str) -> bool:
    """Check whether a record type is numeric."""
    numeric_types = {
      "HKQuantityTypeIdentifierHeartRate",
      "HKQuantityTypeIdentifierBodyMass",
      "HKQuantityTypeIdentifierBodyMassIndex",
      "HKQuantityTypeIdentifierHeight",
      "HKQuantityTypeIdentifierHeartRateVariabilitySDNN",
    }
    return record_type in numeric_types
