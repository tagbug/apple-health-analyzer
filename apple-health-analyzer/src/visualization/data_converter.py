"""数据转换模块 - 将健康记录转换为可视化所需的DataFrame格式"""

from typing import Any, List

import pandas as pd

from ..core.data_models import CategoryRecord, HealthRecord, QuantityRecord
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataConverter:
  """健康数据转换器

  将 HealthRecord 和 SleepSession 对象转换为适合可视化的 DataFrame 格式。
  """

  @staticmethod
  def heart_rate_to_df(records: List[HealthRecord]) -> pd.DataFrame:
    """将心率记录转换为DataFrame

    Args:
        records: 心率健康记录列表

    Returns:
        包含时间戳和心率值的DataFrame
    """
    if not records:
      return pd.DataFrame(columns=["timestamp", "value", "source"])

    data = []
    for record in records:
      if isinstance(record, (QuantityRecord, CategoryRecord)) and hasattr(
        record, "value"
      ):
        data.append(
          {
            "timestamp": record.start_date,
            "value": float(record.value),
            "source": getattr(record, "source_name", "Unknown"),
            "unit": getattr(record, "unit", "bpm"),
          }
        )

    df = pd.DataFrame(data)
    if not df.empty:
      df = df.sort_values("timestamp").reset_index(drop=True)

    logger.debug(
      f"Converted {len(records)} heart rate records to DataFrame with {len(df)} rows"
    )
    return df

  @staticmethod
  def resting_hr_to_df(records: List[HealthRecord]) -> pd.DataFrame:
    """将静息心率记录转换为DataFrame

    Args:
        records: 静息心率记录列表

    Returns:
        包含日期和静息心率值的DataFrame
    """
    if not records:
      return pd.DataFrame(columns=["date", "value", "source"])

    data = []
    for record in records:
      if isinstance(record, (QuantityRecord, CategoryRecord)) and hasattr(
        record, "value"
      ):
        data.append(
          {
            "date": record.start_date.date(),
            "timestamp": record.start_date,
            "value": float(record.value),
            "source": getattr(record, "source_name", "Unknown"),
            "unit": getattr(record, "unit", "bpm"),
          }
        )

    df = pd.DataFrame(data)
    if not df.empty:
      # 按日期去重，保留最新的记录
      df = df.sort_values("timestamp").drop_duplicates("date", keep="last")
      df = df.sort_values("date").reset_index(drop=True)

    logger.debug(
      f"Converted {len(records)} resting HR records to DataFrame with {len(df)} rows"
    )
    return df

  @staticmethod
  def hrv_to_df(records: List[HealthRecord]) -> pd.DataFrame:
    """将HRV记录转换为DataFrame

    Args:
        records: HRV记录列表

    Returns:
        包含日期和HRV值的DataFrame
    """
    if not records:
      return pd.DataFrame(columns=["date", "value", "source"])

    data = []
    for record in records:
      if isinstance(record, (QuantityRecord, CategoryRecord)) and hasattr(
        record, "value"
      ):
        data.append(
          {
            "date": record.start_date.date(),
            "timestamp": record.start_date,
            "value": float(record.value),
            "source": getattr(record, "source_name", "Unknown"),
            "unit": getattr(record, "unit", "ms"),
          }
        )

    df = pd.DataFrame(data)
    if not df.empty:
      # 按日期去重，保留最新的记录
      df = df.sort_values("timestamp").drop_duplicates("date", keep="last")
      df = df.sort_values("date").reset_index(drop=True)

    logger.debug(
      f"Converted {len(records)} HRV records to DataFrame with {len(df)} rows"
    )
    return df

  @staticmethod
  def sleep_sessions_to_df(
    sessions: Any,
  ) -> pd.DataFrame:
    """将睡眠会话转换为DataFrame

    Args:
        sessions: 睡眠会话列表

    Returns:
        包含睡眠会话详细信息的DataFrame
    """
    if not sessions:
      return pd.DataFrame(
        columns=[
          "date",
          "start_time",
          "end_time",
          "total_duration",
          "sleep_duration",
          "efficiency",
          "deep_sleep",
          "rem_sleep",
          "awakenings",
        ]
      )

    data = []
    for session in sessions:
      data.append(
        {
          "date": session.start_date.date(),
          "start_time": session.start_date,
          "end_time": session.end_date,
          "total_duration": session.total_duration,  # 分钟
          "sleep_duration": session.sleep_duration,  # 分钟
          "efficiency": session.efficiency,  # 0-1
          "deep_sleep": session.deep_sleep,  # 分钟
          "rem_sleep": session.rem_sleep,  # 分钟
          "light_sleep": session.light_sleep,  # 分钟
          "awakenings": session.awakenings_count,
          "latency": session.sleep_latency,  # 分钟
          "wake_after_onset": session.wake_after_onset,  # 分钟
        }
      )

    df = pd.DataFrame(data)
    if not df.empty:
      df = df.sort_values("date").reset_index(drop=True)

    logger.debug(
      f"Converted {len(sessions)} sleep sessions to DataFrame with {len(df)} rows"
    )
    return df

  @staticmethod
  def aggregate_heart_rate_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    """按小时聚合心率数据

    Args:
        df: 心率DataFrame（包含timestamp和value列）

    Returns:
        按小时聚合的DataFrame
    """
    if df.empty:
      return pd.DataFrame(
        columns=["hour", "mean_hr", "min_hr", "max_hr", "count"]
      )

    # 确保timestamp列存在
    if "timestamp" not in df.columns:
      logger.warning("No timestamp column in heart rate DataFrame")
      return pd.DataFrame()

    # 创建小时索引
    df_copy = df.copy()
    df_copy["hour"] = df_copy["timestamp"].dt.floor("H")

    # 按小时聚合
    hourly_stats = (
      df_copy.groupby("hour")
      .agg({"value": ["mean", "min", "max", "count"]})
      .round(1)
    )

    # 重新整理列名
    hourly_stats.columns = ["mean_hr", "min_hr", "max_hr", "count"]
    hourly_stats = hourly_stats.reset_index()

    logger.debug(
      f"Aggregated heart rate data to {len(hourly_stats)} hourly records"
    )
    return hourly_stats

  @staticmethod
  def aggregate_heart_rate_by_day(df: pd.DataFrame) -> pd.DataFrame:
    """按天聚合心率数据

    Args:
        df: 心率DataFrame（包含timestamp和value列）

    Returns:
        按天聚合的DataFrame
    """
    if df.empty:
      return pd.DataFrame(
        columns=["date", "mean_hr", "min_hr", "max_hr", "count"]
      )

    # 确保timestamp列存在
    if "timestamp" not in df.columns:
      logger.warning("No timestamp column in heart rate DataFrame")
      return pd.DataFrame()

    # 创建日期索引
    df_copy = df.copy()
    df_copy["date"] = df_copy["timestamp"].dt.date

    # 按日期聚合
    daily_stats = (
      df_copy.groupby("date")
      .agg({"value": ["mean", "min", "max", "count"]})
      .round(1)
    )

    # 重新整理列名
    daily_stats.columns = ["mean_hr", "min_hr", "max_hr", "count"]
    daily_stats = daily_stats.reset_index()

    logger.debug(
      f"Aggregated heart rate data to {len(daily_stats)} daily records"
    )
    return daily_stats

  @staticmethod
  def aggregate_sleep_by_day(df: pd.DataFrame) -> pd.DataFrame:
    """按天聚合睡眠数据

    Args:
        df: 睡眠DataFrame

    Returns:
        按天聚合的DataFrame
    """
    if df.empty:
      return pd.DataFrame(
        columns=[
          "date",
          "total_duration",
          "sleep_duration",
          "efficiency",
          "deep_sleep",
          "rem_sleep",
          "awakenings",
        ]
      )

    # 按日期聚合（如果一天有多个会话，取总和）
    daily_stats = (
      df.groupby("date")
      .agg(
        {
          "total_duration": "sum",
          "sleep_duration": "sum",
          "efficiency": "mean",  # 平均效率
          "deep_sleep": "sum",
          "rem_sleep": "sum",
          "light_sleep": "sum",
          "awakenings": "sum",
          "latency": "mean",
          "wake_after_onset": "sum",
        }
      )
      .round(1)
      .reset_index()
    )

    logger.debug(f"Aggregated sleep data to {len(daily_stats)} daily records")
    return daily_stats

  @staticmethod
  def prepare_heart_rate_zones(
    df: pd.DataFrame, age: int | None = None
  ) -> pd.DataFrame:
    """准备心率区间分析数据

    Args:
        df: 心率DataFrame
        age: 年龄（用于计算最大心率）

    Returns:
        包含心率区间统计的DataFrame
    """
    if df.empty:
      return pd.DataFrame()

    # 计算最大心率（简单估算）
    if age:
      max_hr = 220 - age
    else:
      max_hr = 200  # 默认值

    # 定义心率区间
    zones = {
      "zone1": (0, max_hr * 0.6),  # 恢复区间
      "zone2": (max_hr * 0.6, max_hr * 0.7),  # 脂肪燃烧
      "zone3": (max_hr * 0.7, max_hr * 0.8),  # 有氧耐力
      "zone4": (max_hr * 0.8, max_hr * 0.9),  # 无氧耐力
      "zone5": (max_hr * 0.9, max_hr * 1.0),  # 最大努力
    }

    # 统计每个区间的时间占比
    zone_counts = {}
    total_count = len(df)

    for zone_name, (min_hr, max_hr) in zones.items():
      count = len(df[(df["value"] >= min_hr) & (df["value"] < max_hr)])
      zone_counts[zone_name] = {
        "count": count,
        "percentage": (count / total_count * 100) if total_count > 0 else 0,
        "min_hr": min_hr,
        "max_hr": max_hr,
      }

    # 转换为DataFrame
    zone_df = pd.DataFrame.from_dict(zone_counts, orient="index")
    zone_df = zone_df.reset_index().rename(columns={"index": "zone"})

    logger.debug(f"Prepared heart rate zones data with {len(zone_df)} zones")
    return zone_df

  @staticmethod
  def prepare_sleep_stages_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """准备睡眠阶段分布数据

    Args:
        df: 睡眠DataFrame

    Returns:
        睡眠阶段分布DataFrame
    """
    if df.empty:
      return pd.DataFrame(columns=["stage", "duration", "percentage"])

    # 计算各阶段总时长
    total_deep = df["deep_sleep"].sum()
    total_rem = df["rem_sleep"].sum()
    total_light = df["light_sleep"].sum()
    total_sleep = total_deep + total_rem + total_light

    if total_sleep == 0:
      return pd.DataFrame()

    stages_data = [
      {
        "stage": "Deep Sleep",
        "duration": total_deep,
        "percentage": total_deep / total_sleep * 100,
        "color": "#1f77b4",  # 蓝色
      },
      {
        "stage": "REM Sleep",
        "duration": total_rem,
        "percentage": total_rem / total_sleep * 100,
        "color": "#ff7f0e",  # 橙色
      },
      {
        "stage": "Light Sleep",
        "duration": total_light,
        "percentage": total_light / total_sleep * 100,
        "color": "#2ca02c",  # 绿色
      },
    ]

    stages_df = pd.DataFrame(stages_data)

    logger.debug(
      f"Prepared sleep stages distribution with {len(stages_df)} stages"
    )
    return stages_df

  @staticmethod
  def sample_data_for_performance(
    df: pd.DataFrame, max_points: int = 10000
  ) -> pd.DataFrame:
    """对大数据集进行采样以提高性能

    Args:
        df: 原始DataFrame
        max_points: 最大数据点数

    Returns:
        采样后的DataFrame
    """
    if len(df) <= max_points:
      return df

    # 简单随机采样
    sampled_df = df.sample(n=max_points, random_state=42).sort_index()

    logger.debug(
      f"Sampled data from {len(df)} to {len(sampled_df)} points for performance"
    )
    return sampled_df
