"""统计分析模块 - 提供多维度统计分析功能"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..core.data_models import CategoryRecord, HealthRecord, QuantityRecord
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class StatisticsReport:
  """统计报告数据类"""

  record_type: str
  time_interval: str
  total_records: int
  date_range: tuple[datetime, datetime]

  # 基础统计
  min_value: float
  max_value: float
  mean_value: float
  median_value: float
  std_deviation: float

  # 分位数
  percentile_25: float
  percentile_75: float
  percentile_95: float

  # 数据质量
  data_quality_score: float
  missing_values: int

  # 时间分布
  records_per_day: float
  active_days: int
  total_days: int

  # 趋势指标
  trend_slope: float | None = None
  trend_r_squared: float | None = None


@dataclass
class TrendAnalysis:
  """趋势分析结果"""

  method: str
  slope: float
  intercept: float
  r_squared: float
  p_value: float
  trend_direction: Literal["increasing", "decreasing", "stable"]
  confidence_level: float


class StatisticalAnalyzer:
  """统计分析核心类"""

  def __init__(self):
    """初始化统计分析器"""
    logger.info("StatisticalAnalyzer initialized")

  def aggregate_by_interval(
    self,
    records: list[HealthRecord],
    interval: Literal["hour", "day", "week", "month", "6month", "year"],
  ) -> pd.DataFrame:
    """
    按时间区间聚合数据

    Args:
        records: 健康记录列表
        interval: 时间区间 ("hour", "day", "week", "month", "6month", "year")

    Returns:
        聚合后的DataFrame，包含时间区间和统计值
    """
    if not records:
      logger.warning("No records provided for aggregation")
      return pd.DataFrame()

    logger.info(f"Aggregating {len(records)} records by {interval} interval")

    # 转换为DataFrame
    df = self._records_to_dataframe(records)

    # 根据区间类型设置频率
    freq_map = {
      "hour": "h",
      "day": "D",
      "week": "W",
      "month": "ME",
      "6month": "6ME",
      "year": "YE",
    }

    freq = freq_map.get(interval, "D")

    # 按时间区间聚合
    try:
      # 按时间分组并计算统计值
      grouped = df.groupby(pd.Grouper(key="start_date", freq=freq))

      aggregated = (
        grouped["value"]
        .agg(["count", "min", "max", "mean", "median", "std"])
        .round(4)
      )

      # 重命名列
      aggregated.columns = [
        "record_count",
        "min_value",
        "max_value",
        "mean_value",
        "median_value",
        "std_deviation",
      ]

      # 添加时间区间标识
      aggregated["interval_start"] = aggregated.index
      # 计算区间结束时间 - 分离复杂操作避免类型错误
      try:
        if freq == "ME":
          # 月份比较复杂，使用下一个月的第一天减去1秒
          aggregated["interval_end"] = aggregated.index + pd.offsets.MonthEnd(1)
          aggregated["interval_end"] = aggregated[
            "interval_end"
          ] - pd.Timedelta(seconds=1)
        elif freq == "6ME":
          aggregated["interval_end"] = aggregated.index + pd.offsets.MonthEnd(6)
          aggregated["interval_end"] = aggregated[
            "interval_end"
          ] - pd.Timedelta(seconds=1)
        elif freq == "YE":
          aggregated["interval_end"] = aggregated.index + pd.offsets.YearEnd(1)
          aggregated["interval_end"] = aggregated[
            "interval_end"
          ] - pd.Timedelta(seconds=1)
        else:
          # 对于其他情况，直接计算delta
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
        # 如果计算失败，至少设置interval_end为index
        aggregated["interval_end"] = aggregated.index

      # 重新排序列
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
    """
    计算详细统计指标

    Args:
        data: 包含数值的DataFrame
        value_column: 数值列名

    Returns:
        统计报告对象
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

    # 基础统计
    min_val = float(values.min())
    max_val = float(values.max())
    mean_val = float(values.mean())
    median_val = float(values.median())
    std_val = float(values.std()) if len(values) > 1 else 0.0

    # 分位数
    p25 = float(values.quantile(0.25))
    p75 = float(values.quantile(0.75))
    p95 = float(values.quantile(0.95))

    # 时间分布分析
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

    # 确定记录类型 (从第一条记录推断)
    record_type = "Unknown"
    if hasattr(data, "record_type") and not data.empty:
      record_type = getattr(data.iloc[0], "record_type", "Unknown")
    elif len(data) > 0 and hasattr(data.iloc[0], "type"):
      record_type = data.iloc[0].type

    # 数据质量评分 (多维度评估)
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
    """
    分析数据趋势

    Args:
        data: 时间序列数据
        time_column: 时间列名
        value_column: 数值列名
        method: 趋势分析方法
        window: 移动平均窗口大小 (仅用于moving_average方法)

    Returns:
        趋势分析结果
    """
    if (
      data.empty
      or time_column not in data.columns
      or value_column not in data.columns
    ):
      logger.warning("Insufficient data for trend analysis")
      return None

    logger.info(
      f"Analyzing trend using {method} method for {len(data)} records"
    )

    try:
      # 准备数据
      df = data.copy()
      df = df.dropna(subset=[time_column, value_column])
      df = df.sort_values(time_column)

      if len(df) < 3:
        logger.warning("Need at least 3 data points for trend analysis")
        return None

      # 转换为数值时间戳并归一化 - 确保类型安全
      df["timestamp"] = (
        pd.to_datetime(df[time_column]).astype(int) / 10**9
      )  # 秒级时间戳
      # 归一化时间戳以避免数值问题
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
    """
    生成完整的统计分析报告

    Args:
        records: 健康记录列表
        intervals: 要分析的时间区间列表
        output_format: 输出格式

    Returns:
        统计分析报告
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
          trend = self.analyze_trend(
            aggregated_data, "interval_start", "mean_value"
          )

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
      # 转换为DataFrame格式 (简化版)
      return self._report_to_dataframe(report)
    else:
      return report

  def _records_to_dataframe(self, records: list[HealthRecord]) -> pd.DataFrame:
    """将健康记录转换为DataFrame"""
    data = []
    for record in records:
      # 获取数值 (只处理有数值的记录)
      value = None
      # 检查是否是QuantityRecord或CategoryRecord子类，这些类有value属性
      if isinstance(record, (QuantityRecord, CategoryRecord)):
        value = record.value
        # 确保value是数值类型，如果是字符串则跳过
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
    # 确保数值列的类型正确
    if "value" in df.columns:
      df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df

  def _linear_trend_analysis(
    self, X: np.ndarray, y: np.ndarray
  ) -> TrendAnalysis:
    """线性趋势分析 - 使用 numpy.polyfit 避免 scipy 类型问题"""
    # 使用 numpy.polyfit 进行线性回归（degree=1）
    coeffs = np.polyfit(X.flatten(), y, 1)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])

    # 计算 R² 值
    y_pred = slope * X.flatten() + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    # 计算 p-value (使用 t 统计量)
    n = len(y)
    dof = n - 2  # 自由度
    if dof > 0:
      se = np.sqrt(ss_res / dof)
      t_stat = abs(slope) / (
        se / np.sqrt(np.sum((X.flatten() - X.mean()) ** 2))
      )
      from scipy import stats

      p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), dof)))
    else:
      p_value = 1.0

    # 确定趋势方向
    # 阈值 0.00001 相当于每天变化约 0.864 个单位（对于秒级时间戳）
    # 这对于心率数据来说是合适的（每天变化 1 bpm 左右）
    if abs(slope) < 0.00001:
      direction = "stable"
    elif slope > 0:
      direction = "increasing"
    else:
      direction = "decreasing"

    # 计算置信水平 (基于p值)
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
    """多项式趋势分析"""
    try:
      # 使用numpy进行多项式拟合
      coeffs = np.polyfit(X.flatten(), y, degree)
      poly = np.poly1d(coeffs)

      # 计算R²值
      y_pred = poly(X.flatten())
      ss_res = np.sum((y - y_pred) ** 2)
      ss_tot = np.sum((y - np.mean(y)) ** 2)
      r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

      # 简化：只返回线性趋势的主要信息
      return TrendAnalysis(
        method="polynomial",
        slope=float(coeffs[-2]),  # 一次项系数作为slope
        intercept=float(coeffs[-1]),  # 常数项
        r_squared=float(r_squared),
        p_value=0.0,  # 多项式拟合不计算p值
        trend_direction="stable",  # 需要更复杂的分析
        confidence_level=round(r_squared * 100, 2),
      )
    except Exception as e:
      logger.error(f"Error in polynomial trend analysis: {e}")
      return None

  def _moving_average_trend_analysis(
    self, data: pd.DataFrame, value_column: str, window: int
  ) -> TrendAnalysis | None:
    """移动平均趋势分析"""
    try:
      # 计算移动平均
      ma = data[value_column].rolling(window=window, center=True).mean()

      # 计算移动平均的趋势
      valid_ma = ma.dropna()
      if len(valid_ma) < 3:
        return None

      # 使用移动平均的差分来判断趋势
      diff = valid_ma.diff().dropna()
      avg_change = float(diff.mean())

      # 确定趋势方向
      if abs(avg_change) < 0.001:
        direction = "stable"
      elif avg_change > 0:
        direction = "increasing"
      else:
        direction = "decreasing"

      # 计算R² (简化版)
      std_val = float(data[value_column].std())
      r_squared = min(0.5, abs(avg_change) / (std_val + 0.001))

      # 获取intercept
      intercept = float(valid_ma.iloc[0])

      return TrendAnalysis(
        method="moving_average",
        slope=avg_change,  # 已是float
        intercept=intercept,  # 已是float
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
    """
    多维度数据质量评分

    考虑因素:
    1. 数据完整性 (40%)
    2. 数值合理性 (30%)
    3. 数据一致性 (30%)
    """

    # 空数据直接返回0
    if values.empty or data.empty:
      return 0.0

    # 1. 完整性得分 (40%)
    completeness = len(values) / len(data) if len(data) > 0 else 0

    # 2. 合理性得分 (30%) - 基于记录类型的合理范围
    reasonable_ranges = {
      "HKQuantityTypeIdentifierHeartRate": (40, 200),
      "HKQuantityTypeIdentifierRestingHeartRate": (40, 100),
      "HKQuantityTypeIdentifierStepCount": (0, 50000),
      "HKCategoryTypeIdentifierSleepAnalysis": (0, 3),
    }

    # 获取合理范围 - 确保类型安全
    default_min = float(values.min()) if not values.empty else 0.0
    default_max = float(values.max()) if not values.empty else 100.0
    range_tuple = reasonable_ranges.get(record_type, (default_min, default_max))
    min_val, max_val = float(range_tuple[0]), float(range_tuple[1])

    # 计算在合理范围内的数据比例
    reasonable_mask = (values >= min_val) & (values <= max_val)
    reasonable_count = int(np.asarray(reasonable_mask.sum()))
    reasonability = reasonable_count / len(values) if len(values) > 0 else 0

    # 3. 一致性得分 (30%) - 基于变异系数
    # CV (变异系数) = std / mean，越小说明数据越一致
    mean_val = float(np.asarray(values.mean()))
    if mean_val > 0:
      std_val = float(np.asarray(values.std()))
      cv = std_val / mean_val
      # CV在0-1之间认为是好的，超过1则认为变异过大
      consistency = max(0, min(1, 1 - cv / 2))
    else:
      consistency = 0.5

    # 综合评分
    quality = 0.4 * completeness + 0.3 * reasonability + 0.3 * consistency

    logger.debug(
      f"Quality score breakdown: completeness={completeness:.3f}, "
      f"reasonability={reasonability:.3f}, consistency={consistency:.3f}"
    )

    return quality

  def _calculate_normality_score(self, values: pd.Series) -> float:
    """计算数据正态性评分 (0-1)"""
    try:
      # 对于小样本，正态性检验不可靠，直接返回中等评分
      if len(values) < 10:
        logger.debug(
          f"Small sample size ({len(values)}), returning default normality score"
        )
        return 0.5

      # 对大数据集采样，避免计算成本过高
      if len(values) > 5000:
        # 随机采样5000个数据点进行正态性检验
        sample = values.sample(n=5000, random_state=42)
        logger.debug(
          f"Large dataset detected ({len(values)} records), using sampling (n=5000)"
        )
      else:
        sample = values

      from scipy import stats

      # 确保sample是numpy数组
      sample_array = np.asarray(sample.values, dtype=np.float64)
      stat, p_value = stats.shapiro(sample_array)

      # p值越大，越接近正态分布
      normality_score = min(1.0, float(p_value) * 2)  # 放大p值影响

      return normality_score
    except Exception as e:
      logger.warning(f"Normality test failed: {e}, returning default score")
      # 如果检验失败，返回中等评分
      return 0.5

  def _report_to_dataframe(self, report: dict[str, Any]) -> pd.DataFrame:
    """将报告转换为DataFrame格式"""
    rows = []

    # 总体统计
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

    # 区间统计
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
