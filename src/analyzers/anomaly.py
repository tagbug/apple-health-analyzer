"""异常检测模块 - 提供多种异常检测算法"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, TypedDict

import numpy as np
import pandas as pd

from ..core.data_models import CategoryRecord, HealthRecord, QuantityRecord
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SeverityThresholds(TypedDict):
  """严重程度阈值配置"""

  low: float
  medium: float
  high: float


class AnomalyConfig(TypedDict, total=False):
  """异常检测配置"""

  zscore_threshold: float
  iqr_multiplier: float
  ma_threshold: float
  context_threshold: float
  severity_thresholds: SeverityThresholds


@dataclass
class AnomalyRecord:
  """异常记录数据类"""

  timestamp: datetime
  value: float
  expected_value: float  # 预期值
  deviation: float  # 偏差程度
  severity: Literal["low", "medium", "high"]  # 严重程度
  method: str  # 检测方法
  confidence: float  # 置信度 (0-1)
  context: dict[str, str | float | int]  # 上下文信息


@dataclass
class AnomalyReport:
  """异常检测报告"""

  total_records: int
  anomaly_count: int
  anomaly_rate: float
  anomalies_by_severity: dict[str, int]
  anomalies_by_method: dict[str, int]
  time_distribution: dict[str, dict[str, int]]  # 异常的时间分布
  recommendations: list[str]  # 改进建议


class AnomalyDetector:
  """异常检测核心类"""

  def __init__(self, config: AnomalyConfig | None = None):
    """初始化异常检测器

    Args:
        config: 检测配置参数
    """
    default_config: AnomalyConfig = {
      "zscore_threshold": 3.0,  # Z-Score 阈值
      "iqr_multiplier": 1.5,  # IQR 倍数
      "ma_threshold": 2.0,  # 移动平均阈值
      "context_threshold": 2.5,  # 上下文异常阈值
      "severity_thresholds": {  # 严重程度阈值
        "low": 1.5,
        "medium": 2.5,
        "high": 3.5,
      },
    }

    self.config = default_config
    if config:
      self.config.update(config)

    logger.info("AnomalyDetector initialized")

  def detect_anomalies(
    self,
    records: Sequence[HealthRecord],
    methods: list[Literal["zscore", "iqr", "moving_average", "contextual"]]
    | None = None,
    context: Literal[
      "time_of_day", "day_of_week", "sleep_wake"
    ] = "time_of_day",
  ) -> list[AnomalyRecord]:
    """检测异常值

    Args:
        records: 健康记录列表
        methods: 检测方法列表
        context: 上下文类型 (用于上下文异常检测)

    Returns:
        异常记录列表
    """
    if not records:
      logger.warning("No records provided for anomaly detection")
      return []

    if methods is None:
      methods = ["zscore", "iqr"]

    logger.info(
      f"Detecting anomalies in {len(records)} records using methods: {methods}"
    )

    # 转换为DataFrame
    df = self._records_to_dataframe(records)

    if df.empty or "value" not in df.columns:
      logger.warning("No valid data for anomaly detection")
      return []

    all_anomalies = []

    # 使用不同方法检测异常
    for method in methods:
      try:
        if method == "zscore":
          anomalies = self._detect_zscore(df)
        elif method == "iqr":
          anomalies = self._detect_iqr(df)
        elif method == "moving_average":
          anomalies = self._detect_moving_average(df)
        elif method == "contextual":
          anomalies = self._detect_contextual(df, context)
        else:
          logger.warning(f"Unknown detection method: {method}")
          continue

        all_anomalies.extend(anomalies)
        logger.debug(f"Method {method} found {len(anomalies)} anomalies")

      except Exception as e:
        logger.error(f"Error in {method} detection: {e}")
        continue

    # 去重 (同一个时间点的异常只保留最严重的)
    unique_anomalies = self._deduplicate_anomalies(all_anomalies)

    logger.info(f"Total unique anomalies detected: {len(unique_anomalies)}")
    return unique_anomalies

  def generate_report(
    self, anomalies: Sequence[AnomalyRecord], total_records: int
  ) -> AnomalyReport:
    """生成异常检测报告

    Args:
        anomalies: 异常记录列表
        total_records: 总记录数

    Returns:
        异常检测报告
    """
    anomaly_count = len(anomalies)
    anomaly_rate = anomaly_count / total_records if total_records > 0 else 0

    # 按严重程度分类
    by_severity = {
      "low": sum(1 for a in anomalies if a.severity == "low"),
      "medium": sum(1 for a in anomalies if a.severity == "medium"),
      "high": sum(1 for a in anomalies if a.severity == "high"),
    }

    # 按方法分类
    by_method = {}
    for anomaly in anomalies:
      by_method[anomaly.method] = by_method.get(anomaly.method, 0) + 1

    # 时间分布分析
    time_distribution = self._analyze_time_distribution(anomalies)

    # 生成建议
    recommendations = self._generate_recommendations(anomalies, anomaly_rate)

    return AnomalyReport(
      total_records=total_records,
      anomaly_count=anomaly_count,
      anomaly_rate=round(anomaly_rate, 4),
      anomalies_by_severity=by_severity,
      anomalies_by_method=by_method,
      time_distribution=time_distribution,
      recommendations=recommendations,
    )

  def _detect_zscore(self, df: pd.DataFrame) -> list[AnomalyRecord]:
    """Z-Score 异常检测

    原理: (x - μ) / σ > threshold
    适用: 数据近似正态分布时效果最好
    """
    values = df["value"].dropna()
    if len(values) < 3:
      return []

    mean_val = values.mean()
    std_val = values.std()

    if std_val == 0:
      return []

    threshold = self.config["zscore_threshold"]
    anomalies = []

    for _idx, row in df.iterrows():
      if pd.isna(row["value"]):
        continue

      z_score = abs(row["value"] - mean_val) / std_val

      if z_score > threshold:
        severity = self._calculate_severity(z_score)
        confidence = min(1.0, z_score / 5.0)  # 基于Z-Score计算置信度

        anomalies.append(
          AnomalyRecord(
            timestamp=row["start_date"],
            value=row["value"],
            expected_value=mean_val,
            deviation=z_score,
            severity=severity,
            method="zscore",
            confidence=round(confidence, 3),
            context={
              "mean": round(mean_val, 2),
              "std": round(std_val, 2),
              "z_score": round(z_score, 2),
            },
          )
        )

    return anomalies

  def _detect_iqr(self, df: pd.DataFrame) -> list[AnomalyRecord]:
    """IQR 四分位距异常检测

    原理: Q1 - k*IQR < x < Q3 + k*IQR
    优势: 对极端值不敏感，更鲁棒
    """
    values = df["value"].dropna()
    if len(values) < 4:  # 需要至少4个值计算四分位数
      return []

    Q1 = values.quantile(0.25)
    Q3 = values.quantile(0.75)
    IQR = Q3 - Q1

    if IQR == 0:
      return []

    multiplier = self.config["iqr_multiplier"]
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    anomalies = []

    for _idx, row in df.iterrows():
      if pd.isna(row["value"]):
        continue

      value = row["value"]

      if value < lower_bound or value > upper_bound:
        # 计算偏差程度 (相对于IQR的倍数)
        if value < lower_bound:
          deviation = (lower_bound - value) / IQR if IQR > 0 else 0
        else:
          deviation = (value - upper_bound) / IQR if IQR > 0 else 0

        severity = self._calculate_severity(deviation)
        confidence = min(1.0, deviation / 3.0)  # 基于IQR倍数计算置信度

        anomalies.append(
          AnomalyRecord(
            timestamp=row["start_date"],
            value=row["value"],
            expected_value=float(np.asarray((Q1 + Q3) / 2)),  # 中位数作为预期值
            deviation=deviation,
            severity=severity,
            method="iqr",
            confidence=round(confidence, 3),
            context={
              "Q1": round(float(np.asarray(Q1)), 2),
              "Q3": round(float(np.asarray(Q3)), 2),
              "IQR": round(float(np.asarray(IQR)), 2),
              "lower_bound": round(float(np.asarray(lower_bound)), 2),
              "upper_bound": round(float(np.asarray(upper_bound)), 2),
            },
          )
        )

    return anomalies

  def _detect_moving_average(
    self, df: pd.DataFrame, window: int = 7
  ) -> list[AnomalyRecord]:
    """移动平均异常检测

    原理: 当前值与移动平均值偏差 > threshold * std
    优势: 捕捉短期异常波动
    """
    if len(df) < window:
      return []

    # 计算移动平均和移动标准差
    df = df.copy().sort_values("start_date")
    df["ma"] = df["value"].rolling(window=window, center=True).mean()
    df["ma_std"] = df["value"].rolling(window=window, center=True).std()

    threshold = self.config["ma_threshold"]
    anomalies = []

    for _idx, row in df.iterrows():
      if pd.isna(row["ma"]) or pd.isna(row["ma_std"]) or row["ma_std"] == 0:
        continue

      deviation = abs(row["value"] - row["ma"])
      threshold_value = threshold * row["ma_std"]

      if deviation > threshold_value:
        severity = self._calculate_severity(deviation / row["ma_std"])
        confidence = min(1.0, deviation / (3 * row["ma_std"]))

        anomalies.append(
          AnomalyRecord(
            timestamp=row["start_date"],
            value=row["value"],
            expected_value=float(np.asarray(row["ma"])),
            deviation=deviation / float(np.asarray(row["ma_std"])),
            severity=severity,
            method="moving_average",
            confidence=round(confidence, 3),
            context={
              "moving_average": round(float(np.asarray(row["ma"])), 2),
              "ma_std": round(float(np.asarray(row["ma_std"])), 2),
              "window": window,
            },
          )
        )

    return anomalies

  def _detect_contextual(
    self,
    df: pd.DataFrame,
    context: Literal["time_of_day", "day_of_week", "sleep_wake"],
  ) -> list[AnomalyRecord]:
    """上下文异常检测

    基于时间模式的异常检测
    """
    if context == "time_of_day":
      return self._detect_time_of_day_anomalies(df)
    elif context == "day_of_week":
      return self._detect_day_of_week_anomalies(df)
    elif context == "sleep_wake":
      return self._detect_sleep_wake_anomalies(df)
    else:
      logger.warning(f"Unknown context type: {context}")
      return []

  def _detect_time_of_day_anomalies(
    self, df: pd.DataFrame
  ) -> list[AnomalyRecord]:
    """按小时的异常检测"""
    df = df.copy()
    df["hour"] = df["start_date"].dt.hour

    # 计算每个小时的统计值
    hourly_stats = df.groupby("hour")["value"].agg(["mean", "std"]).dropna()

    threshold = self.config["context_threshold"]
    anomalies = []

    for _idx, row in df.iterrows():
      hour = row["hour"]

      if hour not in hourly_stats.index:
        continue

      mean_val = hourly_stats.loc[hour, "mean"]
      std_val = hourly_stats.loc[hour, "std"]

      if std_val == 0:
        continue

      z_score = abs(row["value"] - mean_val) / std_val

      if z_score > threshold:
        severity = self._calculate_severity(z_score)
        confidence = min(1.0, z_score / 4.0)

        # 安全地处理pandas Scalar类型
        mean_val_float = float(np.asarray(mean_val))
        std_val_float = float(np.asarray(std_val))

        anomalies.append(
          AnomalyRecord(
            timestamp=row["start_date"],
            value=row["value"],
            expected_value=mean_val_float,
            deviation=z_score,
            severity=severity,
            method="contextual_time_of_day",
            confidence=round(confidence, 3),
            context={
              "hour": hour,
              "hourly_mean": round(mean_val_float, 2),
              "hourly_std": round(std_val_float, 2),
            },
          )
        )

    return anomalies

  def _detect_day_of_week_anomalies(
    self, df: pd.DataFrame
  ) -> list[AnomalyRecord]:
    """按星期的异常检测"""
    df = df.copy()
    df["day_of_week"] = df["start_date"].dt.dayofweek  # 0=Monday, 6=Sunday

    # 计算每周每一天的统计值
    daily_stats = (
      df.groupby("day_of_week")["value"].agg(["mean", "std"]).dropna()
    )

    threshold = self.config["context_threshold"]
    anomalies = []

    for _idx, row in df.iterrows():
      day = row["day_of_week"]

      if day not in daily_stats.index:
        continue

      mean_val = daily_stats.loc[day, "mean"]
      std_val = daily_stats.loc[day, "std"]

      if std_val == 0:
        continue

      z_score = abs(row["value"] - mean_val) / std_val

      if z_score > threshold:
        severity = self._calculate_severity(z_score)
        confidence = min(1.0, z_score / 4.0)

        day_names = [
          "Monday",
          "Tuesday",
          "Wednesday",
          "Thursday",
          "Friday",
          "Saturday",
          "Sunday",
        ]

        anomalies.append(
          AnomalyRecord(
            timestamp=row["start_date"],
            value=row["value"],
            expected_value=float(np.asarray(mean_val)),
            deviation=z_score,
            severity=severity,
            method="contextual_day_of_week",
            confidence=round(confidence, 3),
            context={
              "day_of_week": day,
              "day_name": day_names[day],
              "daily_mean": round(float(np.asarray(mean_val)), 2),
              "daily_std": round(float(np.asarray(std_val)), 2),
            },
          )
        )

    return anomalies

  def _detect_sleep_wake_anomalies(
    self, df: pd.DataFrame
  ) -> list[AnomalyRecord]:
    """睡眠/清醒状态异常检测"""
    # 这需要睡眠数据，目前简化实现
    # 实际实现需要结合睡眠记录
    logger.info("Sleep/wake anomaly detection not yet implemented")
    return []

  def _calculate_severity(
    self, deviation: float
  ) -> Literal["low", "medium", "high"]:
    """根据偏差程度计算严重性"""
    thresholds = self.config["severity_thresholds"]

    if deviation >= thresholds["high"]:
      return "high"
    elif deviation >= thresholds["medium"]:
      return "medium"
    else:
      return "low"

  def _deduplicate_anomalies(
    self, anomalies: Sequence[AnomalyRecord]
  ) -> list[AnomalyRecord]:
    """去重异常记录，保留最严重的"""
    if not anomalies:
      return []

    # 按时间戳分组
    by_timestamp = {}
    for anomaly in anomalies:
      timestamp = anomaly.timestamp
      if timestamp not in by_timestamp:
        by_timestamp[timestamp] = []
      by_timestamp[timestamp].append(anomaly)

    # 对每个时间戳保留最严重的异常
    unique_anomalies = []
    severity_order = {"low": 1, "medium": 2, "high": 3}

    for _timestamp, anomaly_list in by_timestamp.items():
      # 按严重程度排序，取最严重的
      most_severe = max(anomaly_list, key=lambda x: severity_order[x.severity])
      unique_anomalies.append(most_severe)

    return unique_anomalies

  def _analyze_time_distribution(
    self, anomalies: Sequence[AnomalyRecord]
  ) -> dict[str, dict[str, int]]:
    """分析异常的时间分布"""
    if not anomalies:
      return {}

    distribution: dict[str, dict[str, int]] = {
      "by_hour": {},
      "by_day_of_week": {},
      "by_month": {},
    }

    for anomaly in anomalies:
      # 按小时分布
      hour = anomaly.timestamp.hour
      distribution["by_hour"][str(hour)] = (
        distribution["by_hour"].get(str(hour), 0) + 1
      )

      # 按星期分布
      day_of_week = anomaly.timestamp.weekday()
      day_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
      ]
      day_name = day_names[day_of_week]
      distribution["by_day_of_week"][day_name] = (
        distribution["by_day_of_week"].get(day_name, 0) + 1
      )

      # 按月份分布
      month = anomaly.timestamp.month
      distribution["by_month"][str(month)] = (
        distribution["by_month"].get(str(month), 0) + 1
      )

    return distribution

  def _generate_recommendations(
    self, anomalies: Sequence[AnomalyRecord], anomaly_rate: float
  ) -> list[str]:
    """生成异常检测建议"""
    recommendations = []

    if anomaly_rate > 0.1:  # 异常率超过10%
      recommendations.append("⚠️ 异常率较高，建议检查数据质量或调整检测阈值")

    if anomaly_rate < 0.001:  # 异常率过低
      recommendations.append("ℹ️ 检测到的异常较少，可能阈值设置过高")

    # 分析严重程度分布
    high_severity = sum(1 for a in anomalies if a.severity == "high")
    if high_severity > len(anomalies) * 0.3:
      recommendations.append("🚨 高严重程度异常较多，建议重点关注")

    # 分析时间分布
    if anomalies:
      time_dist = self._analyze_time_distribution(anomalies)

      # 检查是否集中在特定时间
      hour_counts = time_dist.get("by_hour", {})
      max_hour_count = max(hour_counts.values()) if hour_counts else 0
      if max_hour_count > len(anomalies) * 0.5:
        recommendations.append("📊 异常主要集中在特定小时，可能是正常模式")

    return recommendations

  def _records_to_dataframe(
    self, records: Sequence[HealthRecord]
  ) -> pd.DataFrame:
    """将健康记录转换为DataFrame"""
    data = []
    for record in records:
      # 获取数值 (只处理有数值的记录)
      value = None
      # 检查是否是QuantityRecord或CategoryRecord子类，这些类有value属性
      if isinstance(record, (QuantityRecord, CategoryRecord)):
        value = record.value

      data.append(
        {
          "type": record.type,
          "source_name": record.source_name,
          "start_date": record.start_date,
          "end_date": record.end_date,
          "value": value,
          "unit": record.unit,
        }
      )

    return pd.DataFrame(data)
