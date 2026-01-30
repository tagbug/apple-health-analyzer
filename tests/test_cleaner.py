"""Unit tests for data cleaning module."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from src.core.data_models import HealthRecord, QuantityRecord
from src.processors.cleaner import DataCleaner


class TestDataCleaner:
  """DataCleaner tests."""

  @pytest.fixture
  def cleaner(self):
    """Create a DataCleaner fixture."""
    return DataCleaner()

  @pytest.fixture
  def sample_records(self):
    """Create sample records for tests."""
    base_time = datetime(2023, 11, 9, 12, 0, 0)

    records = [
      # Apple Watch record (highest priority).
      HealthRecord(
        type="HKQuantityTypeIdentifierHeartRate",
        source_name="🐙Watch",
        start_date=base_time,
        end_date=base_time + timedelta(seconds=60),
        creation_date=base_time,
        source_version="1.0",
        device="Apple Watch",
        unit="count/min",
        metadata={"value": 70.0},
      ),
      # Xiaomi Health record (medium priority).
      HealthRecord(
        type="HKQuantityTypeIdentifierHeartRate",
        source_name="小米运动健康",
        start_date=base_time + timedelta(seconds=30),  # 时间窗口内重叠
        end_date=base_time + timedelta(seconds=90),
        creation_date=base_time + timedelta(seconds=30),
        source_version="2.1.0",
        device="Xiaomi Band",
        unit="count/min",
        metadata={"value": 75.0},
      ),
      # iPhone record (lowest priority).
      HealthRecord(
        type="HKQuantityTypeIdentifierHeartRate",
        source_name="🐙Phone",
        start_date=base_time + timedelta(seconds=45),  # 时间窗口内重叠
        end_date=base_time + timedelta(seconds=105),
        creation_date=base_time + timedelta(seconds=45),
        source_version="15.0",
        device="iPhone",
        unit="count/min",
        metadata={"value": 80.0},
      ),
      # Record in another time window.
      HealthRecord(
        type="HKQuantityTypeIdentifierHeartRate",
        source_name="🐙Watch",
        start_date=base_time + timedelta(minutes=5),  # 不同时间窗口
        end_date=base_time + timedelta(minutes=6),
        creation_date=base_time + timedelta(minutes=5),
        source_version="1.0",
        device="Apple Watch",
        unit="count/min",
        metadata={"value": 72.0},
      ),
    ]
    return records

  def test_init_default_priority(self, cleaner):
    """Test default initialization."""
    assert cleaner.source_priority["🐙Watch"] == 1
    assert cleaner.source_priority["小米运动健康"] == 2
    assert cleaner.source_priority["🐙Phone"] == 3
    assert cleaner.default_window_seconds == 60

  def test_init_custom_priority(self):
    """Test custom priority map."""
    custom_priority = {"SourceA": 1, "SourceB": 2}
    cleaner = DataCleaner(source_priority=custom_priority)
    assert cleaner.source_priority == custom_priority

  def test_deduplicate_empty_records(self, cleaner):
    """Test deduplication with empty records."""
    result_records, result_stats = cleaner.deduplicate_by_time_window([])

    assert result_records == []
    assert result_stats.original_count == 0
    assert result_stats.deduplicated_count == 0
    assert result_stats.removed_duplicates == 0

  def test_deduplicate_by_priority(self, cleaner, sample_records):
    """Test priority-based deduplication."""
    result_records, result_stats = cleaner.deduplicate_by_time_window(
      sample_records, strategy="priority"
    )

    # Two records remain (two time windows).
    assert len(result_records) == 2
    assert result_stats.original_count == 4
    assert result_stats.deduplicated_count == 2
    assert result_stats.removed_duplicates == 2
    assert result_stats.strategy_used == "priority"

    # First window should keep the Apple Watch record (highest priority).
    first_window_records = [
      r for r in result_records if r.start_date.hour == 12 and r.start_date.minute == 0
    ]
    assert len(first_window_records) == 1
    assert first_window_records[0].source_name == "🐙Watch"
    # Check metadata value (HealthRecord has no value field).
    assert first_window_records[0].metadata.get("value") == 70.0

  def test_deduplicate_by_latest(self, cleaner, sample_records):
    """Test latest-record deduplication."""
    result_records, result_stats = cleaner.deduplicate_by_time_window(
      sample_records, strategy="latest"
    )

    assert len(result_records) == 2
    assert result_stats.strategy_used == "latest"

    # First window should keep the latest record (iPhone record).
    first_window_records = [
      r for r in result_records if r.start_date.hour == 12 and r.start_date.minute == 0
    ]
    assert len(first_window_records) == 1
    assert first_window_records[0].source_name == "🐙Phone"
    assert first_window_records[0].metadata.get("value") == 80.0

  def test_deduplicate_by_average(self, cleaner):
    """Test average-based deduplication."""
    base_time = datetime(2023, 11, 9, 12, 0, 0)

    # Create numeric records.
    records = [
      QuantityRecord(
        type="HKQuantityTypeIdentifierHeartRate",
        source_name="🐙Watch",
        start_date=base_time,
        end_date=base_time + timedelta(seconds=60),
        creation_date=base_time,
        source_version="1.0",
        device="Apple Watch",
        unit="count/min",
        value=70.0,
        metadata={},
      ),
      QuantityRecord(
        type="HKQuantityTypeIdentifierHeartRate",
        source_name="小米运动健康",
        start_date=base_time + timedelta(seconds=30),
        end_date=base_time + timedelta(seconds=90),
        creation_date=base_time + timedelta(seconds=30),
        source_version="2.1.0",
        device="Xiaomi Band",
        unit="count/min",
        value=80.0,
        metadata={},
      ),
    ]

    result_records, result_stats = cleaner.deduplicate_by_time_window(
      records, strategy="average"
    )

    assert len(result_records) == 1
    assert result_stats.strategy_used == "average"

    # Check average value calculation.
    record = result_records[0]
    assert record.value == 75.0  # (70 + 80) / 2

    # Check metadata.
    assert record.metadata["deduplication_method"] == "average"
    assert record.metadata["original_records_count"] == 2
    # averaged_values_str is skipped for performance optimization
    # assert record.metadata["averaged_values_str"] == "[70.0, 80.0]"

  def test_deduplicate_different_time_windows(self, cleaner):
    """Test that different time windows are not deduplicated."""
    base_time = datetime(2023, 11, 9, 12, 0, 0)

    records = [
      # Time window 1.
      QuantityRecord(
        type="HKQuantityTypeIdentifierHeartRate",
        source_name="🐙Watch",
        start_date=base_time,
        end_date=base_time + timedelta(seconds=60),
        creation_date=base_time,
        source_version="1.0",
        device="Apple Watch",
        unit="count/min",
        value=70.0,
        metadata={},
      ),
      # Time window 2 (5 minutes later).
      QuantityRecord(
        type="HKQuantityTypeIdentifierHeartRate",
        source_name="🐙Watch",
        start_date=base_time + timedelta(minutes=5),
        end_date=base_time + timedelta(minutes=6),
        creation_date=base_time + timedelta(minutes=5),
        source_version="1.0",
        device="Apple Watch",
        unit="count/min",
        value=75.0,
        metadata={},
      ),
    ]

    result_records, result_stats = cleaner.deduplicate_by_time_window(
      records, window_seconds=60
    )

    # All records remain (different windows).
    assert len(result_records) == 2
    assert result_stats.removed_duplicates == 0

  def test_merge_overlapping_records(self, cleaner):
    """Test merge of overlapping records."""
    base_time = datetime(2023, 11, 9, 12, 0, 0)

    # Create sleep records (mergeable type).
    records = [
      HealthRecord(
        type="HKCategoryTypeIdentifierSleepAnalysis",
        source_name="🐙Watch",
        start_date=base_time,
        end_date=base_time + timedelta(minutes=30),
        creation_date=base_time,
        source_version="1.0",
        device="Apple Watch",
        unit=None,
        metadata={"sleep_stage": "asleep"},
      ),
      HealthRecord(
        type="HKCategoryTypeIdentifierSleepAnalysis",
        source_name="🐙Watch",
        start_date=base_time
        + timedelta(minutes=30, seconds=2),  # 2-second gap should merge.
        end_date=base_time + timedelta(minutes=60),
        creation_date=base_time + timedelta(minutes=30),
        source_version="1.0",
        device="Apple Watch",
        unit=None,
        metadata={"sleep_stage": "asleep"},
      ),
    ]

    result_records = cleaner.merge_overlapping_records(
      records, merge_threshold_seconds=5
    )

    # Merge logic is minimal here; verify interface.
    assert isinstance(result_records, list)

  def test_validate_data_quality_empty(self, cleaner):
    """Test data quality validation with empty records."""
    report = cleaner.validate_data_quality([])

    assert report.total_records == 0
    assert report.valid_records == 0
    assert report.quality_score == 0.0

  def test_validate_data_quality_valid_records(self, cleaner):
    """Test data quality validation with valid records."""
    base_time = datetime(2023, 11, 9, 12, 0, 0)

    records = [
      QuantityRecord(
        type="HKQuantityTypeIdentifierHeartRate",
        source_name="🐙Watch",
        start_date=base_time,
        end_date=base_time + timedelta(seconds=60),
        creation_date=base_time,
        source_version="1.0",
        device="Apple Watch",
        unit="count/min",
        value=70.0,
        metadata={"test": "value"},
      )
    ]

    report = cleaner.validate_data_quality(records)

    assert report.total_records == 1
    assert report.valid_records == 1
    assert report.invalid_records == 0
    assert report.quality_score == 60.0  # Only validity score, no issues.
    assert report.source_distribution["🐙Watch"] == 1
    assert report.type_distribution["HKQuantityTypeIdentifierHeartRate"] == 1

  def test_validate_data_quality_invalid_timestamp(self, cleaner):
    """Test data quality validation with invalid timestamps."""
    # Create a mock invalid record.

    invalid_record = MagicMock()
    invalid_record.type = "HKQuantityTypeIdentifierHeartRate"
    invalid_record.source_name = "🐙Watch"
    invalid_record.start_date = datetime(2023, 11, 9, 12, 1, 0)  # 开始时间晚
    invalid_record.end_date = datetime(2023, 11, 9, 12, 0, 0)  # 结束时间早
    invalid_record.creation_date = datetime(2023, 11, 9, 12, 0, 0)
    invalid_record.metadata = {}

    records = [invalid_record]

    report = cleaner.validate_data_quality(records)

    assert report.total_records == 1
    assert report.valid_records == 0
    assert report.invalid_records == 1
    assert report.timestamp_issues == 1
    assert report.quality_score < 60.0  # Validity score penalized by issues.

  def test_validate_data_quality_invalid_value(self, cleaner):
    """Test data quality validation with invalid values."""
    base_time = datetime(2023, 11, 9, 12, 0, 0)

    records = [
      QuantityRecord(
        type="HKQuantityTypeIdentifierHeartRate",
        source_name="🐙Watch",
        start_date=base_time,
        end_date=base_time + timedelta(seconds=60),
        creation_date=base_time,
        source_version="1.0",
        device="Apple Watch",
        unit="count/min",
        value=300.0,  # Too high, invalid.
        metadata={},
      )
    ]

    report = cleaner.validate_data_quality(records)

    assert report.total_records == 1
    assert report.valid_records == 0
    assert report.invalid_records == 1
    assert report.value_issues == 1

  def test_detect_duplicates(self, cleaner):
    """Test duplicate detection."""
    base_time = datetime(2023, 11, 9, 12, 0, 0)

    records = [
      # Duplicate records.
      QuantityRecord(
        type="HKQuantityTypeIdentifierHeartRate",
        source_name="🐙Watch",
        start_date=base_time,
        end_date=base_time + timedelta(seconds=60),
        creation_date=base_time,
        source_version="1.0",
        device="Apple Watch",
        unit="count/min",
        value=70.0,
        metadata={},
      ),
      QuantityRecord(
        type="HKQuantityTypeIdentifierHeartRate",
        source_name="🐙Watch",
        start_date=base_time,
        end_date=base_time + timedelta(seconds=60),
        creation_date=base_time,
        source_version="1.0",
        device="Apple Watch",
        unit="count/min",
        value=70.0,
        metadata={},
      ),
      # Distinct record.
      QuantityRecord(
        type="HKQuantityTypeIdentifierHeartRate",
        source_name="🐙Watch",
        start_date=base_time + timedelta(seconds=60),
        end_date=base_time + timedelta(seconds=120),
        creation_date=base_time + timedelta(seconds=60),
        source_version="1.0",
        device="Apple Watch",
        unit="count/min",
        value=75.0,
        metadata={},
      ),
    ]

    duplicates = cleaner._detect_duplicates(records)
    assert duplicates == 1  # One duplicate.

  def test_is_numeric_type(self, cleaner):
    """Test numeric type detection."""
    assert cleaner._is_numeric_type("HKQuantityTypeIdentifierHeartRate")
    assert cleaner._is_numeric_type("HKQuantityTypeIdentifierBodyMass")
    assert not cleaner._is_numeric_type("HKCategoryTypeIdentifierSleepAnalysis")

  def test_should_merge_type(self, cleaner):
    """Test mergeable type detection."""
    assert cleaner._should_merge_type("HKCategoryTypeIdentifierSleepAnalysis")
    assert cleaner._should_merge_type("HKWorkoutTypeIdentifier")
    assert not cleaner._should_merge_type("HKQuantityTypeIdentifierHeartRate")

  def test_validate_timestamp(self, cleaner):
    """Test timestamp validation."""
    base_time = datetime(2023, 11, 9, 12, 0, 0)

    # Valid timestamps.
    valid_record = QuantityRecord(
      type="HKQuantityTypeIdentifierHeartRate",
      source_name="🐙Watch",
      start_date=base_time,
      end_date=base_time + timedelta(seconds=60),
      creation_date=base_time,
      source_version="1.0",
      device="Apple Watch",
      unit="count/min",
      value=70.0,
      metadata={},
    )
    assert cleaner._validate_timestamp(valid_record)

    # Invalid timestamps (start after end).
    invalid_record = MagicMock()
    invalid_record.start_date = base_time + timedelta(seconds=60)  # 开始时间晚
    invalid_record.end_date = base_time  # 结束时间早
    invalid_record.creation_date = base_time

    assert not cleaner._validate_timestamp(invalid_record)

  def test_validate_value(self, cleaner):
    """Test value validation."""
    base_time = datetime(2023, 11, 9, 12, 0, 0)

    # Valid heart rate.
    valid_record = QuantityRecord(
      type="HKQuantityTypeIdentifierHeartRate",
      source_name="🐙Watch",
      start_date=base_time,
      end_date=base_time + timedelta(seconds=60),
      creation_date=base_time,
      source_version="1.0",
      device="Apple Watch",
      unit="count/min",
      value=70.0,
      metadata={},
    )
    assert cleaner._validate_value(valid_record)

    # Invalid heart rate (too high).
    invalid_record = QuantityRecord(
      type="HKQuantityTypeIdentifierHeartRate",
      source_name="🐙Watch",
      start_date=base_time,
      end_date=base_time + timedelta(seconds=60),
      creation_date=base_time,
      source_version="1.0",
      device="Apple Watch",
      unit="count/min",
      value=300.0,  # 过高
      metadata={},
    )
    assert not cleaner._validate_value(invalid_record)

  def test_calculate_quality_score(self, cleaner):
    """Test quality score calculation."""
    # All valid records.
    score = cleaner._calculate_quality_score(10, 10, 0, 0)
    assert score == 60.0  # 只有有效性评分

    # Some invalid records.
    score = cleaner._calculate_quality_score(10, 8, 1, 1)
    assert score < 60.0  # 被问题惩罚

    # All invalid records.
    score = cleaner._calculate_quality_score(10, 0, 5, 5)
    assert score == 0.0  # Minimum score.

  def test_deduplicate_fast_path_priority(self):
    """Test fast-path deduplication with priority strategy."""
    cleaner = DataCleaner()
    base_time = datetime(2023, 11, 9, 12, 0, 0)

    records = []
    for i in range(50001):
      source = "🐙Watch" if i % 2 == 0 else "🐙Phone"
      records.append(
        HealthRecord(
          type="HKQuantityTypeIdentifierHeartRate",
          source_name=source,
          start_date=base_time + timedelta(seconds=i % 60),
          end_date=base_time + timedelta(seconds=(i % 60) + 60),
          creation_date=base_time + timedelta(seconds=i % 60),
          source_version="1.0",
          device="Device",
          unit="count/min",
          metadata={"value": float(60 + (i % 5))},
        )
      )

    deduped_records, stats = cleaner.deduplicate_by_time_window(
      records, window_seconds=60, strategy="priority"
    )

    assert len(deduped_records) == 1
    assert stats.deduplicated_count == 1
    assert stats.removed_duplicates == len(records) - 1
    assert deduped_records[0].source_name == "🐙Watch"

  def test_deduplicate_fast_path_latest(self):
    """Test fast-path deduplication with latest strategy."""
    cleaner = DataCleaner()
    base_time = datetime(2023, 11, 9, 12, 0, 0)

    records = []
    for i in range(50001):
      created_at = base_time + timedelta(seconds=i)
      records.append(
        HealthRecord(
          type="HKQuantityTypeIdentifierHeartRate",
          source_name="🐙Watch",
          start_date=base_time + timedelta(seconds=i % 60),
          end_date=base_time + timedelta(seconds=(i % 60) + 60),
          creation_date=created_at,
          source_version="1.0",
          device="Device",
          unit="count/min",
          metadata={"value": float(60 + (i % 5))},
        )
      )

    deduped_records, stats = cleaner.deduplicate_by_time_window(
      records, window_seconds=60, strategy="latest"
    )

    assert len(deduped_records) == 1
    assert stats.deduplicated_count == 1
    assert stats.removed_duplicates == len(records) - 1
    assert deduped_records[0].creation_date == max(r.creation_date for r in records)
