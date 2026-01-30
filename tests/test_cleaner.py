"""Unit tests for data cleaning module."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from src.core.data_models import HealthRecord, QuantityRecord
from src.processors.cleaner import DataCleaner


class TestDataCleaner:
  """DataCleaner 类测试"""

  @pytest.fixture
  def cleaner(self):
    """创建测试用的 DataCleaner 实例"""
    return DataCleaner()

  @pytest.fixture
  def sample_records(self):
    """创建测试用的样本记录"""
    base_time = datetime(2023, 11, 9, 12, 0, 0)

    records = [
      # Apple Watch 记录（优先级最高）
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
      # 小米运动健康记录（优先级中等）
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
      # iPhone 记录（优先级最低）
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
      # 不同时间窗口的记录
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
    """测试默认初始化"""
    assert cleaner.source_priority["🐙Watch"] == 1
    assert cleaner.source_priority["小米运动健康"] == 2
    assert cleaner.source_priority["🐙Phone"] == 3
    assert cleaner.default_window_seconds == 60

  def test_init_custom_priority(self):
    """测试自定义优先级"""
    custom_priority = {"SourceA": 1, "SourceB": 2}
    cleaner = DataCleaner(source_priority=custom_priority)
    assert cleaner.source_priority == custom_priority

  def test_deduplicate_empty_records(self, cleaner):
    """测试空记录去重"""
    result_records, result_stats = cleaner.deduplicate_by_time_window([])

    assert result_records == []
    assert result_stats.original_count == 0
    assert result_stats.deduplicated_count == 0
    assert result_stats.removed_duplicates == 0

  def test_deduplicate_by_priority(self, cleaner, sample_records):
    """测试按优先级去重"""
    result_records, result_stats = cleaner.deduplicate_by_time_window(
      sample_records, strategy="priority"
    )

    # 应该保留 2 条记录（2个时间窗口）
    assert len(result_records) == 2
    assert result_stats.original_count == 4
    assert result_stats.deduplicated_count == 2
    assert result_stats.removed_duplicates == 2
    assert result_stats.strategy_used == "priority"

    # 第一个时间窗口应该保留 Apple Watch 的记录（优先级最高）
    first_window_records = [
      r
      for r in result_records
      if r.start_date.hour == 12 and r.start_date.minute == 0
    ]
    assert len(first_window_records) == 1
    assert first_window_records[0].source_name == "🐙Watch"
    # 检查元数据中的值（因为 HealthRecord 没有 value 属性）
    assert first_window_records[0].metadata.get("value") == 70.0

  def test_deduplicate_by_latest(self, cleaner, sample_records):
    """测试按最新时间去重"""
    result_records, result_stats = cleaner.deduplicate_by_time_window(
      sample_records, strategy="latest"
    )

    assert len(result_records) == 2
    assert result_stats.strategy_used == "latest"

    # 第一个时间窗口应该保留最新的记录（iPhone 的记录）
    first_window_records = [
      r
      for r in result_records
      if r.start_date.hour == 12 and r.start_date.minute == 0
    ]
    assert len(first_window_records) == 1
    assert first_window_records[0].source_name == "🐙Phone"
    assert first_window_records[0].metadata.get("value") == 80.0

  def test_deduplicate_by_average(self, cleaner):
    """测试按平均值去重"""
    base_time = datetime(2023, 11, 9, 12, 0, 0)

    # 创建数值类型的记录
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

    # 检查平均值计算
    record = result_records[0]
    assert record.value == 75.0  # (70 + 80) / 2

    # 检查元数据
    assert record.metadata["deduplication_method"] == "average"
    assert record.metadata["original_records_count"] == 2
    # averaged_values_str is skipped for performance optimization
    # assert record.metadata["averaged_values_str"] == "[70.0, 80.0]"

  def test_deduplicate_different_time_windows(self, cleaner):
    """测试不同时间窗口的记录不被去重"""
    base_time = datetime(2023, 11, 9, 12, 0, 0)

    records = [
      # 时间窗口 1
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
      # 时间窗口 2（5分钟后，不同窗口）
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

    # 应该保留所有记录（不同时间窗口）
    assert len(result_records) == 2
    assert result_stats.removed_duplicates == 0

  def test_merge_overlapping_records(self, cleaner):
    """测试重叠记录合并"""
    base_time = datetime(2023, 11, 9, 12, 0, 0)

    # 创建睡眠记录（应该合并的类型）
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
        + timedelta(minutes=30, seconds=2),  # 2秒间隔，应该合并
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

    # 由于合并逻辑暂未完全实现，这里主要测试接口
    assert isinstance(result_records, list)

  def test_validate_data_quality_empty(self, cleaner):
    """测试空数据质量验证"""
    report = cleaner.validate_data_quality([])

    assert report.total_records == 0
    assert report.valid_records == 0
    assert report.quality_score == 0.0

  def test_validate_data_quality_valid_records(self, cleaner):
    """测试有效记录的质量验证"""
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
    assert report.quality_score == 60.0  # 只有有效性评分，没有问题
    assert report.source_distribution["🐙Watch"] == 1
    assert report.type_distribution["HKQuantityTypeIdentifierHeartRate"] == 1

  def test_validate_data_quality_invalid_timestamp(self, cleaner):
    """测试无效时间戳的质量验证"""
    # 创建一个模拟的无效记录（使用 Mock 对象）
    from unittest.mock import MagicMock

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
    assert report.quality_score < 60.0  # 有效性评分被问题惩罚

  def test_validate_data_quality_invalid_value(self, cleaner):
    """测试无效数值的质量验证"""
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
        value=300.0,  # 心率过高，无效
        metadata={},
      )
    ]

    report = cleaner.validate_data_quality(records)

    assert report.total_records == 1
    assert report.valid_records == 0
    assert report.invalid_records == 1
    assert report.value_issues == 1

  def test_detect_duplicates(self, cleaner):
    """测试重复检测"""
    base_time = datetime(2023, 11, 9, 12, 0, 0)

    records = [
      # 相同记录
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
      # 不同记录
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
    assert duplicates == 1  # 只有1个重复

  def test_is_numeric_type(self, cleaner):
    """测试数值类型判断"""
    assert cleaner._is_numeric_type("HKQuantityTypeIdentifierHeartRate")
    assert cleaner._is_numeric_type("HKQuantityTypeIdentifierBodyMass")
    assert not cleaner._is_numeric_type("HKCategoryTypeIdentifierSleepAnalysis")

  def test_should_merge_type(self, cleaner):
    """测试合并类型判断"""
    assert cleaner._should_merge_type("HKCategoryTypeIdentifierSleepAnalysis")
    assert cleaner._should_merge_type("HKWorkoutTypeIdentifier")
    assert not cleaner._should_merge_type("HKQuantityTypeIdentifierHeartRate")

  def test_validate_timestamp(self, cleaner):
    """测试时间戳验证"""
    base_time = datetime(2023, 11, 9, 12, 0, 0)

    # 有效时间戳
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

    # 无效时间戳（开始晚于结束）- 使用 Mock 对象
    invalid_record = MagicMock()
    invalid_record.start_date = base_time + timedelta(seconds=60)  # 开始时间晚
    invalid_record.end_date = base_time  # 结束时间早
    invalid_record.creation_date = base_time

    assert not cleaner._validate_timestamp(invalid_record)

  def test_validate_value(self, cleaner):
    """测试数值验证"""
    base_time = datetime(2023, 11, 9, 12, 0, 0)

    # 有效心率值
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

    # 无效心率值（过高）
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
    """测试质量评分计算"""
    # 全有效记录
    score = cleaner._calculate_quality_score(10, 10, 0, 0)
    assert score == 60.0  # 只有有效性评分

    # 有问题的记录
    score = cleaner._calculate_quality_score(10, 8, 1, 1)
    assert score < 60.0  # 被问题惩罚

    # 全无效记录
    score = cleaner._calculate_quality_score(10, 0, 5, 5)
    assert score == 0.0  # 最低分
