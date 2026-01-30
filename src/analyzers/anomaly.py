"""Anomaly detection module with multiple detection methods."""

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
  """Severity threshold configuration."""

  low: float
  medium: float
  high: float


class AnomalyConfig(TypedDict, total=False):
  """Anomaly detection configuration."""

  zscore_threshold: float
  iqr_multiplier: float
  ma_threshold: float
  context_threshold: float
  severity_thresholds: SeverityThresholds


@dataclass
class AnomalyRecord:
  """Anomaly record data model."""

  timestamp: datetime
  value: float
  expected_value: float  # Expected value
  deviation: float  # Deviation magnitude
  severity: Literal["low", "medium", "high"]  # Severity
  method: str  # Detection method
  confidence: float  # Confidence (0-1)
  context: dict[str, str | float | int]  # Context metadata


@dataclass
class AnomalyReport:
  """Anomaly detection report."""

  total_records: int
  anomaly_count: int
  anomaly_rate: float
  anomalies_by_severity: dict[str, int]
  anomalies_by_method: dict[str, int]
  time_distribution: dict[str, dict[str, int]]  # Time distribution of anomalies
  recommendations: list[str]  # Recommendations


class AnomalyDetector:
  """Core anomaly detection class."""

  def __init__(self, config: AnomalyConfig | None = None):
    """Initialize the anomaly detector.

    Args:
        config: Optional detection configuration overrides.
    """
    default_config: AnomalyConfig = {
      "zscore_threshold": 3.0,  # Z-score threshold
      "iqr_multiplier": 1.5,  # IQR multiplier
      "ma_threshold": 2.0,  # Moving average threshold
      "context_threshold": 2.5,  # Contextual anomaly threshold
      "severity_thresholds": {  # Severity thresholds
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
    context: Literal["time_of_day", "day_of_week", "sleep_wake"] = "time_of_day",
  ) -> list[AnomalyRecord]:
    """Detect anomalies in a set of health records.

    Args:
        records: Health record list.
        methods: Detection methods to apply.
        context: Context mode for contextual detection.

    Returns:
        List of anomaly records.
    """
    if not records:
      logger.warning("No records provided for anomaly detection")
      return []

    if methods is None:
      methods = ["zscore", "iqr"]

    logger.info(
      f"Detecting anomalies in {len(records)} records using methods: {methods}"
    )

    # Convert to DataFrame for vectorized operations.
    df = self._records_to_dataframe(records)

    if df.empty or "value" not in df.columns:
      logger.warning("No valid data for anomaly detection")
      return []

    all_anomalies = []

    # Run each detection method and collect anomalies.
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

    # Deduplicate anomalies by timestamp (keep the most severe one).
    unique_anomalies = self._deduplicate_anomalies(all_anomalies)

    logger.info(f"Total unique anomalies detected: {len(unique_anomalies)}")
    return unique_anomalies

  def generate_report(
    self, anomalies: Sequence[AnomalyRecord], total_records: int
  ) -> AnomalyReport:
    """Generate an anomaly detection report.

    Args:
        anomalies: Anomaly records.
        total_records: Total record count.

    Returns:
        Anomaly report.
    """
    anomaly_count = len(anomalies)
    anomaly_rate = anomaly_count / total_records if total_records > 0 else 0

    # Group by severity.
    by_severity = {
      "low": sum(1 for a in anomalies if a.severity == "low"),
      "medium": sum(1 for a in anomalies if a.severity == "medium"),
      "high": sum(1 for a in anomalies if a.severity == "high"),
    }

    # Group by method.
    by_method = {}
    for anomaly in anomalies:
      by_method[anomaly.method] = by_method.get(anomaly.method, 0) + 1

    # Analyze time distribution.
    time_distribution = self._analyze_time_distribution(anomalies)

    # Generate recommendations.
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
    """Z-score anomaly detection.

    Principle: (x - mu) / sigma > threshold
    Best for: roughly normal distributions
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
        confidence = min(1.0, z_score / 5.0)  # Confidence from z-score magnitude.

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
    """IQR-based anomaly detection.

    Principle: Q1 - k*IQR < x < Q3 + k*IQR
    Best for: robust detection without assuming normality
    """
    values = df["value"].dropna()
    if len(values) < 4:  # Need at least 4 values for quartiles.
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
        # Deviation relative to IQR.
        if value < lower_bound:
          deviation = (lower_bound - value) / IQR if IQR > 0 else 0
        else:
          deviation = (value - upper_bound) / IQR if IQR > 0 else 0

        severity = self._calculate_severity(deviation)
        confidence = min(1.0, deviation / 3.0)  # Confidence based on IQR multiples.

        anomalies.append(
          AnomalyRecord(
            timestamp=row["start_date"],
            value=row["value"],
            expected_value=float(
              np.asarray((Q1 + Q3) / 2)
            ),  # Midpoint as expected value.
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
    """Moving average anomaly detection.

    Principle: deviation from moving average exceeds threshold * std
    Best for: short-term fluctuations in time-series data
    """
    if len(df) < window:
      return []

    # Compute rolling mean and standard deviation.
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
    """Contextual anomaly detection based on temporal patterns."""
    if context == "time_of_day":
      return self._detect_time_of_day_anomalies(df)
    elif context == "day_of_week":
      return self._detect_day_of_week_anomalies(df)
    elif context == "sleep_wake":
      return self._detect_sleep_wake_anomalies(df)
    else:
      logger.warning(f"Unknown context type: {context}")
      return []

  def _detect_time_of_day_anomalies(self, df: pd.DataFrame) -> list[AnomalyRecord]:
    """Detect anomalies by hour of day."""
    df = df.copy()
    df["hour"] = df["start_date"].dt.hour

    # Compute hourly statistics.
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

        # Safely handle pandas scalar types.
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

  def _detect_day_of_week_anomalies(self, df: pd.DataFrame) -> list[AnomalyRecord]:
    """Detect anomalies by day of week."""
    df = df.copy()
    df["day_of_week"] = df["start_date"].dt.dayofweek  # 0=Monday, 6=Sunday

    # Compute daily statistics.
    daily_stats = df.groupby("day_of_week")["value"].agg(["mean", "std"]).dropna()

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

  def _detect_sleep_wake_anomalies(self, df: pd.DataFrame) -> list[AnomalyRecord]:
    """Detect anomalies by sleep vs wake time windows."""
    if df.empty or "start_date" not in df.columns or "value" not in df.columns:
      return []

    df = df.copy()
    df["hour"] = df["start_date"].dt.hour
    df["is_sleep_hour"] = (df["hour"] >= 22) | (df["hour"] < 6)

    sleep_stats = df.groupby("is_sleep_hour")["value"].agg(["mean", "std"]).dropna()

    threshold = self.config["context_threshold"]
    anomalies = []

    for _idx, row in df.iterrows():
      is_sleep_hour = row["is_sleep_hour"]

      if is_sleep_hour not in sleep_stats.index:
        continue

      mean_val = sleep_stats.loc[is_sleep_hour, "mean"]
      std_val = sleep_stats.loc[is_sleep_hour, "std"]

      if std_val == 0:
        continue

      z_score = abs(row["value"] - mean_val) / std_val

      if z_score > threshold:
        severity = self._calculate_severity(z_score)
        confidence = min(1.0, z_score / 4.0)

        anomalies.append(
          AnomalyRecord(
            timestamp=row["start_date"],
            value=row["value"],
            expected_value=float(np.asarray(mean_val)),
            deviation=z_score,
            severity=severity,
            method="contextual_sleep_wake",
            confidence=round(confidence, 3),
            context={
              "is_sleep_hour": bool(is_sleep_hour),
              "sleep_mean": round(float(np.asarray(mean_val)), 2),
              "sleep_std": round(float(np.asarray(std_val)), 2),
            },
          )
        )

    return anomalies

  def _calculate_severity(self, deviation: float) -> Literal["low", "medium", "high"]:
    """Compute severity from deviation magnitude."""
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
    """Deduplicate anomalies by timestamp, keeping the most severe."""
    if not anomalies:
      return []

    # Group by timestamp.
    by_timestamp = {}
    for anomaly in anomalies:
      timestamp = anomaly.timestamp
      if timestamp not in by_timestamp:
        by_timestamp[timestamp] = []
      by_timestamp[timestamp].append(anomaly)

    # Keep the most severe anomaly per timestamp.
    unique_anomalies = []
    severity_order = {"low": 1, "medium": 2, "high": 3}

    for _timestamp, anomaly_list in by_timestamp.items():
      # Pick the most severe anomaly.
      most_severe = max(anomaly_list, key=lambda x: severity_order[x.severity])
      unique_anomalies.append(most_severe)

    return unique_anomalies

  def _analyze_time_distribution(
    self, anomalies: Sequence[AnomalyRecord]
  ) -> dict[str, dict[str, int]]:
    """Analyze the time distribution of anomalies."""
    if not anomalies:
      return {}

    distribution: dict[str, dict[str, int]] = {
      "by_hour": {},
      "by_day_of_week": {},
      "by_month": {},
    }

    for anomaly in anomalies:
      # By hour.
      hour = anomaly.timestamp.hour
      distribution["by_hour"][str(hour)] = distribution["by_hour"].get(str(hour), 0) + 1

      # By weekday.
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

      # By month.
      month = anomaly.timestamp.month
      distribution["by_month"][str(month)] = (
        distribution["by_month"].get(str(month), 0) + 1
      )

    return distribution

  def _generate_recommendations(
    self, anomalies: Sequence[AnomalyRecord], anomaly_rate: float
  ) -> list[str]:
    """Generate recommendations based on anomaly findings."""
    recommendations = []

    if anomaly_rate > 0.1:  # Anomaly rate exceeds 10%.
      recommendations.append(
        "⚠️ Anomaly rate is high; review data quality or adjust thresholds."
      )

    if anomaly_rate < 0.001:  # Anomaly rate is very low.
      recommendations.append("ℹ️ Few anomalies detected; thresholds may be too strict.")

    # Analyze severity distribution.
    high_severity = sum(1 for a in anomalies if a.severity == "high")
    if high_severity > len(anomalies) * 0.3:
      recommendations.append(
        "🚨 High-severity anomalies are frequent; prioritize investigation."
      )

    # Analyze time distribution.
    if anomalies:
      time_dist = self._analyze_time_distribution(anomalies)

      # Check if anomalies cluster at specific times.
      hour_counts = time_dist.get("by_hour", {})
      max_hour_count = max(hour_counts.values()) if hour_counts else 0
      if max_hour_count > len(anomalies) * 0.5:
        recommendations.append(
          "📊 Anomalies cluster at specific hours; this may reflect normal patterns."
        )

    return recommendations

  def _records_to_dataframe(self, records: Sequence[HealthRecord]) -> pd.DataFrame:
    """Convert health records to a DataFrame."""
    data = []
    for record in records:
      # Use numeric value fields when present.
      value = None
      # Handle QuantityRecord/CategoryRecord subclasses with a value field.
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
