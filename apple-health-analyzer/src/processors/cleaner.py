"""Data cleaning and preprocessing module.

Provides data deduplication, merging, quality validation, and other functions.
"""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, cast

import pandas as pd
from pydantic import BaseModel, Field

from src.core.data_models import HealthRecord
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RecordRowData(BaseModel):
  """DataFrame 行数据的类型安全包装

  用于在 DataFrame 和 HealthRecord 对象之间进行类型安全的转换。
  """

  type: str
  source_name: str
  start_date: datetime
  end_date: datetime | None
  creation_date: datetime
  source_version: str = "1.0"
  device: str = "Unknown"
  unit: str | None = None
  value: float | None = None
  metadata: dict[str, Any] | None = None

  @classmethod
  def from_series(cls, row: pd.Series, record_type: str) -> "RecordRowData":
    """从 pandas Series 创建类型安全的数据对象"""
    # 手动提取并转换字段，避免 pandas 类型推断问题
    start_date_val = row["start_date"]
    if isinstance(start_date_val, str):
      start_date = pd.to_datetime(start_date_val).to_pydatetime()
    elif isinstance(start_date_val, pd.Timestamp):
      start_date = start_date_val.to_pydatetime()
    else:
      # 使用 cast 明确告诉类型检查器这是 datetime
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
      # 使用 cast 明确告诉类型检查器这是 datetime
      creation_date = cast(datetime, creation_date_val)

    # 安全地提取其他字段
    source_name = str(row.get("source_name", "Unknown"))
    source_version = str(row.get("source_version", "1.0"))
    device = str(row.get("device", "Unknown"))

    unit_val = row.get("unit")
    unit = (
      str(unit_val)
      if unit_val is not None
      and str(unit_val).lower() not in ("", "nan", "none")
      else None
    )

    value_val = row.get("value")
    value = (
      float(value_val)
      if value_val is not None
      and str(value_val).lower() not in ("", "nan", "none")
      else None
    )

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
    """转换为 HealthRecord"""
    # 如果 end_date 为 None，使用 start_date 作为默认值
    end_date = self.end_date if self.end_date is not None else self.start_date

    if self.value is not None:
      # 创建 QuantityRecord
      from src.core.data_models import QuantityRecord

      return QuantityRecord(
        type=self.type,
        source_name=self.source_name,
        start_date=self.start_date,
        end_date=end_date,  # 现在保证不为 None
        creation_date=self.creation_date,
        source_version=self.source_version,
        device=self.device,
        unit=self.unit,
        value=self.value,
        metadata=self.metadata,
      )
    else:
      # 创建基础 HealthRecord
      return HealthRecord(
        type=self.type,
        source_name=self.source_name,
        start_date=self.start_date,
        end_date=end_date,  # 现在保证不为 None
        creation_date=self.creation_date,
        source_version=self.source_version,
        device=self.device,
        unit=self.unit,
        metadata=self.metadata,
      )


class DataQualityReport(BaseModel):
  """数据质量报告"""

  total_records: int
  valid_records: int
  invalid_records: int
  duplicate_records: int
  cleaned_records: int
  quality_score: float  # 0-100

  # 详细统计
  timestamp_issues: int = 0
  value_issues: int = 0
  metadata_issues: int = 0

  # 数据分布
  source_distribution: dict[str, int] = Field(default_factory=dict)
  type_distribution: dict[str, int] = Field(default_factory=dict)

  # 时间范围
  date_range: dict[str, datetime | None] = Field(
    default_factory=lambda: {"start": None, "end": None}
  )


class DeduplicationResult(BaseModel):
  """去重结果"""

  original_count: int
  deduplicated_count: int
  removed_duplicates: int
  strategy_used: str
  processing_time_seconds: float

  # 详细统计
  duplicates_by_source: dict[str, int] = Field(default_factory=dict)
  time_windows_processed: int = 0


class DataCleaner:
  """数据清洗核心类

  提供多种数据清洗和预处理功能：
  - 时间窗口去重
  - 数据源优先级处理
  - 叠加数据合并
  - 数据质量验证
  """

  def __init__(
    self,
    source_priority: dict[str, int] | None = None,
    default_window_seconds: int = 60,
  ):
    """
    初始化数据清洗器

    Args:
        source_priority: 数据源优先级映射，越小优先级越高
            例如: {"🐙Watch": 1, "小米运动健康": 2, "🐙Phone": 3}
        default_window_seconds: 默认时间窗口（秒）
    """
    # 默认数据源优先级（根据用户需求）
    self.source_priority = source_priority or {
      "🐙Watch": 1,  # Apple Watch 最高优先级
      "Apple Watch": 1,  # 别名
      "小米运动健康": 2,  # 小米运动健康
      "Xiaomi Home": 2,  # 别名
      "🐙Phone": 3,  # iPhone 最低优先级
      "iPhone": 3,  # 别名
    }

    self.default_window_seconds = default_window_seconds
    logger.info(
      f"DataCleaner initialized with {len(self.source_priority)} source priorities"
    )

  def deduplicate_by_time_window(
    self,
    records: list[HealthRecord],
    window_seconds: int | None = None,
    strategy: str = "priority",
  ) -> tuple[list[HealthRecord], DeduplicationResult]:
    """
    基于时间窗口的去重处理

    Args:
        records: 待处理的记录列表
        window_seconds: 时间窗口大小（秒），None 使用默认值
        strategy: 去重策略
            - "priority": 按数据源优先级保留
            - "latest": 保留最新的记录
            - "average": 计算平均值（仅数值类型）
            - "highest_quality": 基于质量评分保留

    Returns:
        (去重后的记录列表, 去重结果统计)
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
      f"Starting deduplication with strategy '{strategy}', window {window}s"
    )

    # 按记录类型分组处理
    records_by_type = defaultdict(list)
    for record in records:
      records_by_type[record.type].append(record)

    deduplicated_records = []
    total_duplicates_removed = 0
    duplicates_by_source = defaultdict(int)

    for record_type, type_records in records_by_type.items():
      logger.debug(
        f"Processing {len(type_records)} records of type {record_type}"
      )

      # 转换为 DataFrame 便于处理
      df = self._records_to_dataframe(type_records)

      # 确保 start_date 是 datetime 类型
      df["start_date"] = pd.to_datetime(df["start_date"])

      # 按时间窗口分组
      df["time_window"] = df["start_date"].dt.floor(f"{window}s").astype(str)

      # 对每个时间窗口内的记录进行去重
      cleaned_groups = []
      groupby_result = list(df.groupby("time_window"))
      windows_processed = len(groupby_result)  # 确保变量总是被赋值

      for _window_start, group in groupby_result:
        if len(group) == 1:
          # 只有一个记录，无需去重
          cleaned_groups.append(group.iloc[0])
          continue

        # 多个记录，需要去重
        cleaned_record, duplicates_removed = self._deduplicate_group(
          group, strategy, record_type
        )

        if cleaned_record is not None:
          cleaned_groups.append(cleaned_record)

        total_duplicates_removed += duplicates_removed
        # 统计被移除的记录按数据源分布
        if cleaned_record is not None:
          cleaned_id = cleaned_record.name  # DataFrame index
          for _, row in group.iterrows():
            if row.name != cleaned_id:
              duplicates_by_source[row["source_name"]] += 1

      # 转换回记录对象
      for cleaned_row in cleaned_groups:
        record = self._dataframe_row_to_record(cleaned_row, record_type)
        if record:
          deduplicated_records.append(record)

    processing_time = (datetime.now() - start_time).total_seconds()

    result = DeduplicationResult(
      original_count=len(records),
      deduplicated_count=len(deduplicated_records),
      removed_duplicates=total_duplicates_removed,
      strategy_used=strategy,
      processing_time_seconds=processing_time,
      duplicates_by_source=dict(duplicates_by_source),
      time_windows_processed=windows_processed,
    )

    logger.info(
      f"Deduplication completed: {result.original_count} -> "
      f"{result.deduplicated_count} records "
      f"({result.removed_duplicates} duplicates removed)"
    )

    return deduplicated_records, result

  def merge_overlapping_records(
    self, records: list[HealthRecord], merge_threshold_seconds: int = 5
  ) -> list[HealthRecord]:
    """
    合并重叠或相邻的记录

    主要用于睡眠数据和运动数据，这些数据可能被分割成多个连续的记录。

    Args:
        records: 待合并的记录列表
        merge_threshold_seconds: 合并阈值（秒），记录间隔小于此值则合并

    Returns:
        合并后的记录列表
    """
    if not records or len(records) <= 1:
      return records

    logger.info(
      f"Merging overlapping records, threshold: {merge_threshold_seconds}s"
    )

    # 按记录类型分组
    records_by_type = defaultdict(list)
    for record in records:
      records_by_type[record.type].append(record)

    merged_records = []

    for record_type, type_records in records_by_type.items():
      if not self._should_merge_type(record_type):
        # 该类型不需要合并
        merged_records.extend(type_records)
        continue

      # 排序并合并
      sorted_records = sorted(type_records, key=lambda r: r.start_date)
      merged = self._merge_sorted_records(
        sorted_records, merge_threshold_seconds
      )
      merged_records.extend(merged)

    logger.info(
      f"Merge completed: {len(records)} -> {len(merged_records)} records"
    )
    return merged_records

  def validate_data_quality(
    self, records: list[HealthRecord]
  ) -> DataQualityReport:
    """
    验证数据质量并生成报告

    Args:
        records: 待验证的记录列表

    Returns:
        数据质量报告
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

    logger.info(f"Validating data quality for {len(records)} records")

    total_records = len(records)
    valid_records = 0
    invalid_records = 0

    # 详细统计
    timestamp_issues = 0
    value_issues = 0
    metadata_issues = 0

    # 分布统计
    source_distribution = defaultdict(int)
    type_distribution = defaultdict(int)

    # 时间范围
    dates = []

    for record in records:
      is_valid = True

      # 检查时间戳
      if not self._validate_timestamp(record):
        timestamp_issues += 1
        is_valid = False

      # 检查数值
      if not self._validate_value(record):
        value_issues += 1
        is_valid = False

      # 检查元数据
      if not self._validate_metadata(record):
        metadata_issues += 1
        # 元数据问题不影响记录有效性，只记录统计

      if is_valid:
        valid_records += 1
      else:
        invalid_records += 1

      # 统计分布
      source_distribution[record.source_name] += 1
      type_distribution[record.type] += 1

      # 收集日期
      dates.append(record.start_date)

    # 计算质量评分
    quality_score = self._calculate_quality_score(
      total_records, valid_records, timestamp_issues, value_issues
    )

    # 时间范围
    date_range = {
      "start": min(dates) if dates else None,
      "end": max(dates) if dates else None,
    }

    # 检测重复（简单检测，基于时间和值完全相同）
    duplicate_records = self._detect_duplicates(records)

    report = DataQualityReport(
      total_records=total_records,
      valid_records=valid_records,
      invalid_records=invalid_records,
      duplicate_records=duplicate_records,
      cleaned_records=valid_records,  # 假设清理后保留有效记录
      quality_score=quality_score,
      timestamp_issues=timestamp_issues,
      value_issues=value_issues,
      metadata_issues=metadata_issues,
      source_distribution=dict(source_distribution),
      type_distribution=dict(type_distribution),
      date_range=date_range,
    )

    logger.info(
      f"Quality validation completed: {valid_records}/{total_records} valid "
      f"(score: {quality_score:.1f})"
    )

    return report

  def _records_to_dataframe(self, records: list[HealthRecord]) -> pd.DataFrame:
    """将记录列表转换为 DataFrame"""
    data = []
    for record in records:
      row = {
        "id": id(record),  # 使用对象ID作为唯一标识
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

  def _deduplicate_group(
    self, group: pd.DataFrame, strategy: str, record_type: str
  ) -> tuple[pd.Series | None, int]:
    """
    对单个时间窗口内的记录组进行去重

    Returns:
        (保留的记录行, 移除的重复记录数)
    """
    if len(group) <= 1:
      return group.iloc[0] if len(group) == 1 else None, 0

    if strategy == "priority":
      return self._deduplicate_by_priority(group), len(group) - 1
    elif strategy == "latest":
      return self._deduplicate_by_latest(group), len(group) - 1
    elif strategy == "average" and self._is_numeric_type(record_type):
      return self._deduplicate_by_average(group), len(group) - 1
    elif strategy == "highest_quality":
      return self._deduplicate_by_quality(group), len(group) - 1
    else:
      # 默认使用优先级策略
      return self._deduplicate_by_priority(group), len(group) - 1

  def _deduplicate_by_priority(self, group: pd.DataFrame) -> pd.Series:
    """按数据源优先级去重"""

    # 为每条记录计算优先级分数
    def get_priority_score(source_name: str) -> int:
      return self.source_priority.get(source_name, 999)  # 默认最低优先级

    group = group.copy()
    group["priority_score"] = group["source_name"].apply(get_priority_score)

    # 选择优先级最高的记录（分数最小）
    idx = group["priority_score"].idxmin()
    if idx is not None:
      result = group.loc[idx]
      assert isinstance(result, pd.Series)  # 类型细化
      return result
    else:
      result = group.iloc[0]
      assert isinstance(result, pd.Series)  # 类型细化
      return result

  def _deduplicate_by_latest(self, group: pd.DataFrame) -> pd.Series:
    """保留最新的记录"""
    idx = group["creation_date"].idxmax()
    if idx is not None:
      result = group.loc[idx]
      assert isinstance(result, pd.Series)  # 类型细化
      return result
    else:
      result = group.iloc[0]
      assert isinstance(result, pd.Series)  # 类型细化
      return result

  def _deduplicate_by_average(self, group: pd.DataFrame) -> pd.Series:
    """计算平均值（仅数值类型）"""
    result = group.iloc[0].copy()  # 使用第一条记录作为模板

    # 计算数值平均值
    numeric_values = group["value"].dropna()
    if not numeric_values.empty:
      result["value"] = numeric_values.mean()

    # 更新元数据表示这是平均值
    result["metadata"] = {
      **(result["metadata"] or {}),
      "deduplication_method": "average",
      "original_records_count": len(group),
      "averaged_values_str": str(
        list(numeric_values)
      ),  # 转换为字符串避免序列化问题
    }

    return result

  def _deduplicate_by_quality(self, group: pd.DataFrame) -> pd.Series:
    """基于质量评分去重"""

    def calculate_quality_score(row) -> float:
      score = 0.0

      # 元数据完整性 (0-30分)
      metadata = row.get("metadata", {}) or {}
      completeness = len(metadata) / 10  # 假设最多10个元数据字段
      score += min(completeness * 30, 30)

      # 数据源优先级 (0-40分)
      priority = self.source_priority.get(row["source_name"], 999)
      priority_score = max(
        0, 40 - priority * 10
      )  # 优先级1得40分，优先级2得30分等
      score += priority_score

      # 时间戳合理性 (0-30分)
      if pd.notna(row.get("creation_date")) and pd.notna(row.get("start_date")):
        time_diff = abs(
          (row["creation_date"] - row["start_date"]).total_seconds()
        )
        if time_diff < 86400:  # 24小时内
          score += 30
        elif time_diff < 604800:  # 7天内
          score += 20

      return score

    group = group.copy()
    group["quality_score"] = group.apply(calculate_quality_score, axis=1)

    # 选择质量评分最高的记录
    idx = group["quality_score"].idxmax()
    if idx is not None:
      result = group.loc[idx]
      assert isinstance(result, pd.Series)  # 类型细化
      return result
    else:
      result = group.iloc[0]
      assert isinstance(result, pd.Series)  # 类型细化
      return result

  def _dataframe_row_to_record(
    self, row: pd.Series, record_type: str
  ) -> HealthRecord | None:
    """将 DataFrame 行转换回记录对象"""
    try:
      # 使用 RecordRowData 中间类进行类型安全的转换
      row_data = RecordRowData.from_series(row, record_type)
      return row_data.to_health_record()

    except Exception as e:
      logger.error(f"Failed to reconstruct record: {e}")
      return None

  def _should_merge_type(self, record_type: str) -> bool:
    """判断记录类型是否需要合并"""
    # 睡眠记录和某些运动记录需要合并
    merge_types = {
      "HKCategoryTypeIdentifierSleepAnalysis",
      "HKWorkoutTypeIdentifier",  # 运动记录
    }
    return record_type in merge_types

  def _merge_sorted_records(
    self, records: list[HealthRecord], threshold_seconds: int
  ) -> list[HealthRecord]:
    """合并已排序的记录列表"""
    if not records:
      return []

    merged = [records[0]]

    for current in records[1:]:
      last = merged[-1]

      # 检查是否可以合并
      if self._can_merge_records(last, current, threshold_seconds):
        # 合并记录
        merged[-1] = self._merge_two_records(last, current)
      else:
        # 不能合并，添加为新记录
        merged.append(current)

    return merged

  def _can_merge_records(
    self, record1: HealthRecord, record2: HealthRecord, threshold_seconds: int
  ) -> bool:
    """判断两条记录是否可以合并"""
    # 时间上连续或轻微重叠
    time_gap = (record2.start_date - record1.end_date).total_seconds()
    return time_gap <= threshold_seconds

  def _merge_two_records(
    self, record1: HealthRecord, record2: HealthRecord
  ) -> HealthRecord:
    """合并两条记录"""
    # 创建合并后的记录
    # 这里需要根据具体记录类型实现合并逻辑
    # 暂时返回第一条记录
    logger.warning("Record merging not fully implemented")
    return record1

  def _validate_timestamp(self, record: HealthRecord) -> bool:
    """验证时间戳有效性"""
    try:
      # 检查时间戳是否存在
      if not hasattr(record, "start_date") or not record.start_date:
        return False

      # 检查时间戳合理性（不能是未来太远的日期）
      now = datetime.now(record.start_date.tzinfo)
      if record.start_date > now + timedelta(days=1):
        return False

      # 检查开始时间不能晚于结束时间
      if hasattr(record, "end_date") and record.end_date:
        if record.start_date > record.end_date:
          return False

      return True
    except Exception:
      return False

  def _validate_value(self, record: HealthRecord) -> bool:
    """验证数值有效性"""
    try:
      # 检查是否为数值类型记录
      if not self._is_numeric_type(record.type):
        return True  # 非数值类型记录不需要数值验证

      # 检查是否有 value 属性
      if not hasattr(record, "value"):
        return False  # 数值类型记录必须有 value

      value = getattr(record, "value", None)
      if value is None:
        return False  # 数值类型记录的 value 不能为 None

      # 基本数值检查
      if not isinstance(value, (int, float)):
        return False

      # 针对不同记录类型的特殊检查
      if record.type == "HKQuantityTypeIdentifierHeartRate":
        # 心率应该在 30-250 bpm 之间
        return 30 <= value <= 250
      elif record.type == "HKQuantityTypeIdentifierBodyMass":
        # 体重应该在 20-300 kg 之间
        return 20 <= value <= 300

      # 其他类型使用通用检查
      return abs(value) < 1e10  # 避免极端值

    except Exception:
      return False

  def _validate_metadata(self, record: HealthRecord) -> bool:
    """验证元数据有效性"""
    try:
      if not hasattr(record, "metadata"):
        return True

      metadata = record.metadata
      if metadata is None:
        return True

      # 检查元数据是否为字典
      if not isinstance(metadata, dict):
        return False

      # 检查是否有基本的元数据字段
      # 这里可以添加更详细的元数据验证逻辑

      return True
    except Exception:
      return False

  def _calculate_quality_score(
    self, total: int, valid: int, timestamp_issues: int, value_issues: int
  ) -> float:
    """计算质量评分 (0-100)"""
    if total == 0:
      return 0.0

    # 有效性评分 (60% 权重)
    validity_score = (valid / total) * 60

    # 问题严重程度评分 (40% 权重)
    issue_penalty = ((timestamp_issues + value_issues) / total) * 40

    return max(0.0, min(100.0, validity_score - issue_penalty))

  def _detect_duplicates(self, records: list[HealthRecord]) -> int:
    """简单重复检测"""
    seen = set()
    duplicates = 0

    for record in records:
      # 创建记录的签名（类型 + 时间 + 值）
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
    """判断记录类型是否为数值类型"""
    numeric_types = {
      "HKQuantityTypeIdentifierHeartRate",
      "HKQuantityTypeIdentifierBodyMass",
      "HKQuantityTypeIdentifierBodyMassIndex",
      "HKQuantityTypeIdentifierHeight",
      "HKQuantityTypeIdentifierHeartRateVariabilitySDNN",
    }
    return record_type in numeric_types
