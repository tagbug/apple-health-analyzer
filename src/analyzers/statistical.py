"""Statistical analysis module with multi-dimensional metrics."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..core.data_models import CategoryRecord, HealthRecord, QuantityRecord
from ..utils.logger import get_logger
from ..utils.type_conversion import safe_float

logger = get_logger(__name__)


@dataclass
class StatisticsReport:
  """Statistics report data model."""

  record_type: str
  time_interval: str
  total_records: int
  date_range: tuple[datetime, datetime]

  # Basic statistics
  min_value: float
  max_value: float
  mean_value: float
  median_value: float
  std_deviation: float

  # Percentiles
  percentile_25: float
  percentile_75: float
  percentile_95: float

  # Data quality
  data_quality_score: float
  missing_values: int

  # Time distribution
  records_per_day: float
  active_days: int
  total_days: int

  # Trend metrics
  trend_slope: float | None = None
  trend_r_squared: float | None = None


@dataclass
class TrendAnalysis:
  """Trend analysis result."""

  method: str
  slope: float
  intercept: float
  r_squared: float
  p_value: float
  trend_direction: Literal["increasing", "decreasing", "stable"]
  confidence_level: float


class StatisticalAnalyzer:
  """Core statistical analyzer."""

  def __init__(self):
    """Initialize the statistical analyzer."""
    logger.info("StatisticalAnalyzer initialized")

  def aggregate_by_interval(
    self,
    records: list[HealthRecord],
    interval: Literal["hour", "day", "week", "month", "6month", "year"],
  ) -> pd.DataFrame:
    """Aggregate records by time interval.

    Args:
        records: Health record list.
        interval: Time interval ("hour", "day", "week", "month", "6month", "year").

    Returns:
        Aggregated DataFrame with interval boundaries and metrics.
    """
    if not records:
      logger.warning("No records provided for aggregation")
      return pd.DataFrame()

    logger.info(f"Aggregating {len(records)} records by {interval} interval")

    # Convert to DataFrame.
    df = self._records_to_dataframe(records)

    # Map interval to pandas frequency.
    freq_map = {
      "hour": "h",
      "day": "D",
      "week": "W",
      "month": "ME",
      "6month": "6ME",
      "year": "YE",
    }

    freq = freq_map.get(interval, "D")

    # Aggregate by time interval.
    try:
      # Group by time and compute aggregates.
      grouped = df.groupby(pd.Grouper(key="start_date", freq=freq))

      aggregated = (
        grouped["value"].agg(["count", "min", "max", "mean", "median", "std"]).round(4)
      )

      # Rename columns.
      aggregated.columns = [
        "record_count",
        "min_value",
        "max_value",
        "mean_value",
        "median_value",
        "std_deviation",
      ]

      # Add interval boundaries.
      aggregated["interval_start"] = aggregated.index
      # Compute interval end times, keeping types safe.
      try:
        if freq == "ME":
          # Month end: next month start minus one second.
          aggregated["interval_end"] = aggregated.index + pd.offsets.MonthEnd(1)
          aggregated["interval_end"] = aggregated["interval_end"] - pd.Timedelta(
            seconds=1
          )
        elif freq == "6ME":
          aggregated["interval_end"] = aggregated.index + pd.offsets.MonthEnd(6)
          aggregated["interval_end"] = aggregated["interval_end"] - pd.Timedelta(
            seconds=1
          )
        elif freq == "YE":
          aggregated["interval_end"] = aggregated.index + pd.offsets.YearEnd(1)
          aggregated["interval_end"] = aggregated["interval_end"] - pd.Timedelta(
            seconds=1
          )
        else:
          # Other intervals use fixed deltas.
          if freq == "h":
            delta = pd.Timedelta(hours=1) - pd.Timedelta(seconds=1)
          elif freq == "D":
            delta = pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
          elif freq == "W":
            delta = pd.Timedelta(weeks=1) - pd.Timedelta(seconds=1)
          else:
            delta = pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

          aggregated["interval_end"] = aggregated.index + delta

      except Exception:
        # Fall back to index if end-time calculation fails.
        aggregated["interval_end"] = aggregated.index

      # Reorder columns.
      cols = [
        "interval_start",
        "interval_end",
        "record_count",
        "min_value",
        "max_value",
        "mean_value",
        "median_value",
        "std_deviation",
      ]
      aggregated = aggregated[cols]

      logger.info(f"Aggregated into {len(aggregated)} {interval} intervals")
      return aggregated

    except Exception as e:
      logger.error(f"Error aggregating data by {interval}: {e}")
      return pd.DataFrame()

  def calculate_statistics(
    self, data: pd.DataFrame, value_column: str = "value"
  ) -> StatisticsReport | None:
    """Calculate detailed statistics.

    Args:
        data: DataFrame with numeric values.
        value_column: Column name for values.

    Returns:
        Statistics report object.
    """
    if data.empty or value_column not in data.columns:
      logger.warning(
        f"No data or missing column '{value_column}' for statistics calculation"
      )
      return None

    logger.info(f"Calculating statistics for {len(data)} records")

    values = data[value_column].dropna()

    if values.empty:
      logger.warning("No valid values found for statistics calculation")
      return None

    # Basic statistics
    min_val = safe_float(values.min())
    max_val = safe_float(values.max())
    mean_val = safe_float(values.mean())
    median_val = safe_float(values.median())
    std_val = safe_float(values.std()) if len(values) > 1 else 0.0

    # Percentiles
    p25 = safe_float(values.quantile(0.25))
    p75 = safe_float(values.quantile(0.75))
    p95 = safe_float(values.quantile(0.95))

    # Time distribution analysis
    if "start_date" in data.columns:
      date_range = (data["start_date"].min(), data["start_date"].max())
      total_days = (date_range[1] - date_range[0]).days + 1
      active_days = data["start_date"].dt.date.nunique()
      records_per_day = len(data) / total_days if total_days > 0 else 0
    else:
      date_range = (datetime.now(), datetime.now())
      total_days = 1
      active_days = 1
      records_per_day = len(data)

    # Infer record type from the first row.
    record_type = "Unknown"
    if hasattr(data, "record_type") and not data.empty:
      record_type = getattr(data.iloc[0], "record_type", "Unknown")
    elif len(data) > 0 and hasattr(data.iloc[0], "type"):
      record_type = data.iloc[0].type

    # Data quality score (multi-factor).
    data_quality = self._calculate_data_quality_score(values, data, record_type)

    return StatisticsReport(
      record_type=record_type,
      time_interval="overall",
      total_records=len(data),
      date_range=date_range,
      min_value=min_val,
      max_value=max_val,
      mean_value=mean_val,
      median_value=median_val,
      std_deviation=std_val,
      percentile_25=p25,
      percentile_75=p75,
      percentile_95=p95,
      data_quality_score=round(data_quality, 3),
      missing_values=len(data) - len(values),
      records_per_day=round(records_per_day, 2),
      active_days=active_days,
      total_days=total_days,
    )

  def analyze_trend(
    self,
    data: pd.DataFrame,
    time_column: str = "start_date",
    value_column: str = "value",
    method: Literal["linear", "polynomial", "moving_average"] = "linear",
    window: int = 7,
  ) -> TrendAnalysis | None:
    """Analyze trends in time-series data.

    Args:
        data: Time-series DataFrame.
        time_column: Timestamp column name.
        value_column: Value column name.
        method: Trend analysis method.
        window: Moving average window size (moving_average only).

    Returns:
        Trend analysis result.
    """
    if (
      data.empty or time_column not in data.columns or value_column not in data.columns
    ):
      logger.warning("Insufficient data for trend analysis")
      return None

    logger.info(f"Analyzing trend using {method} method for {len(data)} records")

    try:
      # Prepare data.
      df = data.copy()
      df = df.dropna(subset=[time_column, value_column])
      df = df.sort_values(time_column)

      if len(df) < 3:
        logger.warning("Need at least 3 data points for trend analysis")
        return None

      # Convert to numeric timestamps and normalize.
      df["timestamp"] = (
        pd.to_datetime(df[time_column]).astype(int) / 10**9
      )  # Seconds since epoch.
      # Normalize to avoid numeric issues.
      df["timestamp"] = df["timestamp"] - df["timestamp"].min()
      X = np.asarray(df["timestamp"].values, dtype=np.float64).reshape(-1, 1)
      y = np.asarray(df[value_column].values, dtype=np.float64)

      if method == "linear":
        return self._linear_trend_analysis(X, y)
      elif method == "polynomial":
        return self._polynomial_trend_analysis(X, y)
      elif method == "moving_average":
        return self._moving_average_trend_analysis(df, value_column, window)
      else:
        logger.error(f"Unknown trend analysis method: {method}")
        return None

    except Exception as e:
      logger.error(f"Error in trend analysis: {e}")
      return None

  def generate_report(
    self,
    records: list[HealthRecord],
    intervals: list[Literal["hour", "day", "week", "month", "6month", "year"]]
    | None = None,
    output_format: Literal["dict", "dataframe"] = "dict",
  ) -> dict[str, Any] | pd.DataFrame:
    """Generate a full statistical analysis report.

    Args:
        records: Health record list.
        intervals: Time intervals to analyze.
        output_format: Output format.

    Returns:
        Statistical analysis report.
    """
    if not records:
      logger.warning("No records provided for report generation")
      return {} if output_format == "dict" else pd.DataFrame()

    if intervals is None:
      intervals = ["day", "week", "month"]

    logger.info(f"Generating statistical report for {len(records)} records")

    report: dict[str, Any] = {
      "summary": self.calculate_statistics(self._records_to_dataframe(records)),
      "interval_analyses": {},
    }

    for interval in intervals:
      try:
        aggregated_data = self.aggregate_by_interval(records, interval)
        if not aggregated_data.empty:
          stats = self.calculate_statistics(aggregated_data, "mean_value")
          trend = self.analyze_trend(aggregated_data, "interval_start", "mean_value")

          interval_analyses = report["interval_analyses"]
          if isinstance(interval_analyses, dict):
            interval_analyses[interval] = {
              "statistics": stats,
              "trend": trend,
              "data_points": len(aggregated_data),
            }
      except Exception as e:
        logger.error(f"Error analyzing {interval} interval: {e}")
        continue

    if output_format == "dataframe":
      # Convert to DataFrame format (simplified).
      return self._report_to_dataframe(report)
    else:
      return report

  def _records_to_dataframe(self, records: list[HealthRecord]) -> pd.DataFrame:
    """Convert health records to a DataFrame."""
    data = []
    for record in records:
      # Extract numeric values when present.
      value = None
      # Quantity/Category records expose a value field.
      if isinstance(record, (QuantityRecord, CategoryRecord)):
        value = record.value
        # Ensure value is numeric; skip string values.
        if isinstance(value, str):
          value = None
        elif value is not None:
          try:
            value = float(value)
          except (ValueError, TypeError):
            value = None

      data.append(
        {
          "type": record.type,
          "source_name": record.source_name,
          "start_date": record.start_date,
          "end_date": record.end_date,
          "value": value,
          "unit": record.unit,
          "record_type": record.type,
        }
      )

    df = pd.DataFrame(data)
    # Ensure numeric column types.
    if "value" in df.columns:
      df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df

  def _linear_trend_analysis(self, X: np.ndarray, y: np.ndarray) -> TrendAnalysis:
    """Linear trend analysis using numpy.polyfit."""
    # Use numpy.polyfit for linear regression (degree=1).
    coeffs = np.polyfit(X.flatten(), y, 1)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])

    # Compute R-squared.
    y_pred = slope * X.flatten() + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    # Compute p-value using t-statistics.
    n = len(y)
    dof = n - 2  # Degrees of freedom.
    if dof > 0:
      se = np.sqrt(ss_res / dof)
      t_stat = abs(slope) / (se / np.sqrt(np.sum((X.flatten() - X.mean()) ** 2)))
      from scipy import stats

      p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), dof)))
    else:
      p_value = 1.0

    # Determine trend direction.
    # Threshold 0.00001 ~= 0.864 units/day for second-level timestamps.
    # This aligns with ~1 bpm/day changes for heart rate data.
    if abs(slope) < 0.00001:
      direction = "stable"
    elif slope > 0:
      direction = "increasing"
    else:
      direction = "decreasing"

    # Compute confidence level from p-value.
    confidence = (1 - p_value) * 100

    return TrendAnalysis(
      method="linear",
      slope=slope,
      intercept=intercept,
      r_squared=r_squared,
      p_value=p_value,
      trend_direction=direction,
      confidence_level=round(confidence, 2),
    )

  def _polynomial_trend_analysis(
    self, X: np.ndarray, y: np.ndarray, degree: int = 2
  ) -> TrendAnalysis | None:
    """Polynomial trend analysis."""
    try:
      # Polynomial fit using numpy.
      coeffs = np.polyfit(X.flatten(), y, degree)
      poly = np.poly1d(coeffs)

      # Compute R-squared.
      y_pred = poly(X.flatten())
      ss_res = np.sum((y - y_pred) ** 2)
      ss_tot = np.sum((y - np.mean(y)) ** 2)
      r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

      # Simplified output with linear components.
      return TrendAnalysis(
        method="polynomial",
        slope=float(coeffs[-2]),  # Linear term as slope.
        intercept=float(coeffs[-1]),  # Constant term.
        r_squared=float(r_squared),
        p_value=0.0,  # Polynomial fit does not compute p-values.
        trend_direction="stable",  # Requires more complex analysis.
        confidence_level=round(r_squared * 100, 2),
      )
    except Exception as e:
      logger.error(f"Error in polynomial trend analysis: {e}")
      return None

  def _moving_average_trend_analysis(
    self, data: pd.DataFrame, value_column: str, window: int
  ) -> TrendAnalysis | None:
    """Moving average trend analysis."""
    try:
      # Compute moving average.
      ma = data[value_column].rolling(window=window, center=True).mean()

      # Compute trend of the moving average.
      valid_ma = ma.dropna()
      if len(valid_ma) < 3:
        return None

      # Use moving average deltas to infer trend.
      diff = valid_ma.diff().dropna()
      avg_change = float(diff.mean())

      # Determine trend direction.
      if abs(avg_change) < 0.001:
        direction = "stable"
      elif avg_change > 0:
        direction = "increasing"
      else:
        direction = "decreasing"

      # Compute R-squared (simplified).
      std_val = float(data[value_column].std())
      r_squared = min(0.5, abs(avg_change) / (std_val + 0.001))

      # Calculate intercept.
      intercept = float(valid_ma.iloc[0])

      return TrendAnalysis(
        method="moving_average",
        slope=avg_change,  # Already float.
        intercept=intercept,  # Already float.
        r_squared=float(r_squared),
        p_value=0.0,
        trend_direction=direction,
        confidence_level=round(r_squared * 100, 2),
      )
    except Exception as e:
      logger.error(f"Error in moving average trend analysis: {e}")
      return None

  def _calculate_data_quality_score(
    self, values: pd.Series, data: pd.DataFrame, record_type: str = "Unknown"
  ) -> float:
    """Compute data quality score from multiple factors.

    Factors:
    1) Completeness (40%)
    2) Reasonableness (30%)
    3) Consistency (30%)
    """

    # Return 0 for empty data.
    if values.empty or data.empty:
      return 0.0

    # 1) Completeness score (40%).
    completeness = len(values) / len(data) if len(data) > 0 else 0

    # 2) Reasonableness score (30%) based on type ranges.
    reasonable_ranges = {
      "HKQuantityTypeIdentifierHeartRate": (40, 200),
      "HKQuantityTypeIdentifierRestingHeartRate": (40, 100),
      "HKQuantityTypeIdentifierStepCount": (0, 50000),
      "HKCategoryTypeIdentifierSleepAnalysis": (0, 3),
    }

    # Get reasonable bounds with type safety.
    default_min = safe_float(values.min()) if not values.empty else 0.0
    default_max = safe_float(values.max()) if not values.empty else 100.0
    range_tuple = reasonable_ranges.get(record_type, (default_min, default_max))
    min_val, max_val = safe_float(range_tuple[0]), safe_float(range_tuple[1])

    # Compute proportion within bounds.
    reasonable_mask = (values >= min_val) & (values <= max_val)
    reasonable_count = int(np.asarray(reasonable_mask.sum()))
    reasonability = reasonable_count / len(values) if len(values) > 0 else 0

    # 3) Consistency score (30%) based on coefficient of variation.
    # CV = std / mean; lower indicates higher consistency.
    mean_val = safe_float(values.mean())
    if mean_val > 0:
      std_val = safe_float(values.std())
      cv = std_val / mean_val
      # CV in [0,1] is acceptable; above 1 indicates high variance.
      consistency = max(0, min(1, 1 - cv / 2))
    else:
      consistency = 0.5

    # Weighted total score.
    quality = 0.4 * completeness + 0.3 * reasonability + 0.3 * consistency

    logger.debug(
      f"Quality score breakdown: completeness={completeness:.3f}, "
      f"reasonability={reasonability:.3f}, consistency={consistency:.3f}"
    )

    return quality

  def _calculate_normality_score(self, values: pd.Series) -> float:
    """Compute normality score (0-1)."""
    try:
      # For small samples, return a neutral score.
      if len(values) < 10:
        logger.debug(
          f"Small sample size ({len(values)}), returning default normality score"
        )
        return 0.5

      # Sample large datasets to limit cost.
      if len(values) > 5000:
        # Sample 5,000 points for normality testing.
        sample = values.sample(n=5000, random_state=42)
        logger.debug(
          f"Large dataset detected ({len(values)} records), using sampling (n=5000)"
        )
      else:
        sample = values

      from scipy import stats

      # Ensure sample is a numpy array.
      sample_array = np.asarray(sample.values, dtype=np.float64)
      stat, p_value = stats.shapiro(sample_array)

      # Higher p-values imply closer-to-normal distribution.
      normality_score = min(1.0, float(p_value) * 2)  # Scale p-value impact.

      return normality_score
    except Exception as e:
      logger.warning(f"Normality test failed: {e}, returning default score")
      # Fall back to neutral score on failure.
      return 0.5

  def _report_to_dataframe(self, report: dict[str, Any]) -> pd.DataFrame:
    """Convert a report to a DataFrame."""
    rows = []

    # Overall summary.
    if report.get("summary"):
      summary = report["summary"]
      rows.append(
        {
          "interval": "overall",
          "record_count": summary.total_records,
          "mean": summary.mean_value,
          "median": summary.median_value,
          "std": summary.std_deviation,
          "min": summary.min_value,
          "max": summary.max_value,
          "data_quality": summary.data_quality_score,
        }
      )

    # Interval summaries.
    for interval, analysis in report.get("interval_analyses", {}).items():
      if analysis.get("statistics"):
        stats = analysis["statistics"]
        rows.append(
          {
            "interval": interval,
            "record_count": stats.total_records,
            "mean": stats.mean_value,
            "median": stats.median_value,
            "std": stats.std_deviation,
            "min": stats.min_value,
            "max": stats.max_value,
            "data_quality": stats.data_quality_score,
          }
        )

    return pd.DataFrame(rows)
