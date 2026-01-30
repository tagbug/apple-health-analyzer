"""Data conversion module - converts health records to DataFrame format for visualization"""

from collections.abc import Sequence
from typing import Any

import pandas as pd

from ..core.data_models import CategoryRecord, HealthRecord, QuantityRecord
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataConverter:
  """Health data converter.

  Converts HealthRecord and SleepSession objects into DataFrames for visualization.
  """

  @staticmethod
  def _build_value_dataframe(
    records: Sequence[HealthRecord],
    *,
    include_date: bool,
    default_unit: str,
    empty_columns: list[str],
    dedupe_by_date: bool = False,
  ) -> pd.DataFrame:
    """Build a value DataFrame from HealthRecord items."""
    if not records:
      return pd.DataFrame(columns=empty_columns)

    data = []
    for record in records:
      if isinstance(record, (QuantityRecord, CategoryRecord)) and hasattr(
        record, "value"
      ):
        row = {
          "timestamp": record.start_date,
          "value": float(record.value),
          "source": getattr(record, "source_name", "Unknown"),
          "unit": getattr(record, "unit", default_unit),
        }
        if include_date:
          row["date"] = record.start_date.date()
        data.append(row)

    df = pd.DataFrame(data)
    if df.empty:
      return df

    if include_date:
      df = df.sort_values("timestamp")
      if dedupe_by_date:
        df = df.drop_duplicates("date", keep="last")
        df = df.sort_values("date").reset_index(drop=True)
      else:
        df = df.reset_index(drop=True)
    else:
      df = df.sort_values("timestamp").reset_index(drop=True)

    if include_date:
      ordered_columns = ["date", "timestamp", "value", "source", "unit"]
      df = df[[col for col in ordered_columns if col in df.columns]]
    else:
      ordered_columns = ["timestamp", "value", "source", "unit"]
      df = df[[col for col in ordered_columns if col in df.columns]]

    return df

  @staticmethod
  def heart_rate_to_df(records: Sequence[HealthRecord]) -> pd.DataFrame:
    """Convert heart rate records into a DataFrame.

    Args:
        records: List of heart rate health records.

    Returns:
        DataFrame with timestamps and heart rate values.
    """
    df = DataConverter._build_value_dataframe(
      records,
      include_date=False,
      default_unit="bpm",
      empty_columns=["timestamp", "value", "source"],
    )

    logger.debug(
      f"Converted {len(records)} heart rate records to DataFrame with {len(df)} rows"
    )
    return df

  @staticmethod
  def resting_hr_to_df(records: Sequence[HealthRecord]) -> pd.DataFrame:
    """Convert resting heart rate records into a DataFrame.

    Args:
        records: List of resting heart rate records.

    Returns:
        DataFrame with dates and resting heart rate values.
    """
    df = DataConverter._build_value_dataframe(
      records,
      include_date=True,
      default_unit="bpm",
      empty_columns=["date", "value", "source"],
      dedupe_by_date=True,
    )

    logger.debug(
      f"Converted {len(records)} resting HR records to DataFrame with {len(df)} rows"
    )
    return df

  @staticmethod
  def hrv_to_df(records: Sequence[HealthRecord]) -> pd.DataFrame:
    """Convert HRV records into a DataFrame.

    Args:
        records: List of HRV records.

    Returns:
        DataFrame with dates and HRV values.
    """
    df = DataConverter._build_value_dataframe(
      records,
      include_date=True,
      default_unit="ms",
      empty_columns=["date", "value", "source"],
      dedupe_by_date=True,
    )

    logger.debug(
      f"Converted {len(records)} HRV records to DataFrame with {len(df)} rows"
    )
    return df

  @staticmethod
  def sleep_sessions_to_df(
    sessions: Any,
  ) -> pd.DataFrame:
    """Convert sleep sessions into a DataFrame.

    Args:
        sessions: List of sleep sessions.

    Returns:
        DataFrame with detailed sleep session information.
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
          "total_duration": session.total_duration,  # Minutes.
          "sleep_duration": session.sleep_duration,  # Minutes.
          "efficiency": session.efficiency,  # 0-1
          "deep_sleep": session.deep_sleep,  # Minutes.
          "rem_sleep": session.rem_sleep,  # Minutes.
          "light_sleep": session.light_sleep,  # Minutes.
          "awakenings": session.awakenings_count,
          "latency": session.sleep_latency,  # Minutes.
          "wake_after_onset": session.wake_after_onset,  # Minutes.
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
    """Aggregate heart rate data by hour.

    Args:
        df: Heart rate DataFrame (timestamp and value columns).

    Returns:
        Hourly aggregated DataFrame.
    """
    if df.empty:
      return pd.DataFrame(columns=["hour", "mean_hr", "min_hr", "max_hr", "count"])

    # Ensure timestamp column exists.
    if "timestamp" not in df.columns:
      logger.warning("No timestamp column in heart rate DataFrame")
      return pd.DataFrame()

    # Create hourly index.
    df_copy = df.copy()
    df_copy["hour"] = df_copy["timestamp"].dt.floor("h")

    # Aggregate by hour.
    hourly_stats = (
      df_copy.groupby("hour").agg({"value": ["mean", "min", "max", "count"]}).round(1)
    )

    # Normalize column names.
    hourly_stats.columns = ["mean_hr", "min_hr", "max_hr", "count"]
    hourly_stats = hourly_stats.reset_index()

    logger.debug(f"Aggregated heart rate data to {len(hourly_stats)} hourly records")
    return hourly_stats

  @staticmethod
  def aggregate_heart_rate_by_day(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate heart rate data by day.

    Args:
        df: Heart rate DataFrame (timestamp and value columns).

    Returns:
        Daily aggregated DataFrame.
    """
    if df.empty:
      return pd.DataFrame(columns=["date", "mean_hr", "min_hr", "max_hr", "count"])

    # Ensure timestamp column exists.
    if "timestamp" not in df.columns:
      logger.warning("No timestamp column in heart rate DataFrame")
      return pd.DataFrame()

    # Create date index.
    df_copy = df.copy()
    df_copy["date"] = df_copy["timestamp"].dt.date

    # Aggregate by date.
    daily_stats = (
      df_copy.groupby("date").agg({"value": ["mean", "min", "max", "count"]}).round(1)
    )

    # Normalize column names.
    daily_stats.columns = ["mean_hr", "min_hr", "max_hr", "count"]
    daily_stats = daily_stats.reset_index()

    logger.debug(f"Aggregated heart rate data to {len(daily_stats)} daily records")
    return daily_stats

  @staticmethod
  def aggregate_sleep_by_day(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sleep data by day.

    Args:
        df: Sleep DataFrame.

    Returns:
        Daily aggregated DataFrame.
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

    # Aggregate by date (sum multiple sessions in a day).
    daily_stats = (
      df.groupby("date")
      .agg(
        {
          "total_duration": "sum",
          "sleep_duration": "sum",
          "efficiency": "mean",  # Average efficiency.
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
    """Prepare heart rate zone analysis data.

    Args:
        df: Heart rate DataFrame.
        age: Age (used to estimate max heart rate).

    Returns:
        DataFrame with heart rate zone stats.
    """
    if df.empty:
      return pd.DataFrame()

    # Estimate maximum heart rate.
    if age:
      max_hr = 220 - age
    else:
      max_hr = 200  # Default value.

    # Define heart rate zones.
    zones = {
      "zone1": (0, max_hr * 0.6),  # Recovery zone.
      "zone2": (max_hr * 0.6, max_hr * 0.7),  # Fat burn zone.
      "zone3": (max_hr * 0.7, max_hr * 0.8),  # Aerobic endurance.
      "zone4": (max_hr * 0.8, max_hr * 0.9),  # Anaerobic endurance.
      "zone5": (max_hr * 0.9, max_hr * 1.0),  # Max effort.
    }

    # Compute time share per zone.
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

    # Convert to DataFrame.
    zone_df = pd.DataFrame.from_dict(zone_counts, orient="index")
    zone_df = zone_df.reset_index().rename(columns={"index": "zone"})

    logger.debug(f"Prepared heart rate zones data with {len(zone_df)} zones")
    return zone_df

  @staticmethod
  def prepare_sleep_stages_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare sleep stage distribution data.

    Args:
        df: Sleep DataFrame.

    Returns:
        Sleep stage distribution DataFrame.
    """
    if df.empty:
      return pd.DataFrame(columns=["stage", "duration", "percentage"])

    # Compute total duration per stage.
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
        "color": "#1f77b4",  # Blue.
      },
      {
        "stage": "REM Sleep",
        "duration": total_rem,
        "percentage": total_rem / total_sleep * 100,
        "color": "#ff7f0e",  # Orange.
      },
      {
        "stage": "Light Sleep",
        "duration": total_light,
        "percentage": total_light / total_sleep * 100,
        "color": "#2ca02c",  # Green.
      },
    ]

    stages_df = pd.DataFrame(stages_data)

    logger.debug(f"Prepared sleep stages distribution with {len(stages_df)} stages")
    return stages_df

  @staticmethod
  def sample_data_for_performance(
    df: pd.DataFrame, max_points: int = 10000
  ) -> pd.DataFrame:
    """Sample large datasets to improve performance.

    Args:
        df: Source DataFrame.
        max_points: Maximum number of data points.

    Returns:
        Sampled DataFrame.
    """
    if len(df) <= max_points:
      return df

    # Simple random sampling.
    sampled_df = df.sample(n=max_points, random_state=42).sort_index()

    logger.debug(
      f"Sampled data from {len(df)} to {len(sampled_df)} points for performance"
    )
    return sampled_df
