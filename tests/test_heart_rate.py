"""Unit tests for heart rate processor."""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from src.analyzers.anomaly import AnomalyDetector
from src.analyzers.statistical import StatisticalAnalyzer
from src.core.data_models import QuantityRecord
from src.processors.heart_rate import (
  CardioFitnessAnalysis,
  HeartRateAnalysisReport,
  HeartRateAnalyzer,
  HRVAnalysis,
  RestingHRAnalysis,
)


class TestHeartRateAnalyzer:
  """HeartRateAnalyzer 测试类"""

  @pytest.fixture
  def analyzer(self):
    """创建测试用的HeartRateAnalyzer实例"""
    return HeartRateAnalyzer()

  @pytest.fixture
  def sample_hr_records(self):
    """创建示例心率记录"""
    base_time = datetime(2024, 1, 1, 8, 0, 0)
    records = []

    # 生成30天的模拟心率数据
    for day in range(30):
      for hour in range(24):
        # 模拟一天中的心率变化
        base_hr = 70
        if 6 <= hour <= 8:  # 早晨锻炼
          hr_variation = 20
        elif 22 <= hour <= 24:  # 晚上休息
          hr_variation = -10
        else:
          hr_variation = 0

        hr_value = base_hr + hr_variation + (day % 5)  # 添加一些随机性

        record_time = base_time + timedelta(days=day, hours=hour)
        records.append(
          QuantityRecord(
            type="HKQuantityTypeIdentifierHeartRate",
            source_name="Apple Watch",
            start_date=record_time,
            end_date=record_time + timedelta(minutes=1),
            creation_date=record_time,
            value=float(hr_value),
            unit="count/min",
            source_version="1.0",
            device="Apple Watch Series 8",
          )
        )

    return records

  @pytest.fixture
  def sample_resting_hr_records(self):
    """创建示例静息心率记录"""
    base_time = datetime(2024, 1, 1, 8, 0, 0)
    records = []

    # 生成30天的静息心率数据，逐渐下降
    for day in range(30):
      resting_hr = 72 - (day * 0.1)  # 逐渐下降
      record_time = base_time + timedelta(days=day)

      records.append(
        QuantityRecord(
          type="HKQuantityTypeIdentifierRestingHeartRate",
          source_name="Apple Watch",
          start_date=record_time,
          end_date=record_time + timedelta(days=1),
          creation_date=record_time,
          value=float(resting_hr),
          unit="count/min",
          source_version="1.0",
          device="Apple Watch Series 8",
        )
      )

    return records

  @pytest.fixture
  def sample_hrv_records(self):
    """创建示例HRV记录"""
    base_time = datetime(2024, 1, 1, 8, 0, 0)
    records = []

    # 生成30天的HRV数据，逐渐改善
    for day in range(30):
      hrv_value = 35 + (day * 0.2)  # 逐渐改善
      record_time = base_time + timedelta(days=day)

      records.append(
        QuantityRecord(
          type="HKQuantityTypeIdentifierHeartRateVariabilitySDNN",
          source_name="Apple Watch",
          start_date=record_time,
          end_date=record_time + timedelta(days=1),
          creation_date=record_time,
          value=float(hrv_value),
          unit="ms",
          source_version="1.0",
          device="Apple Watch Series 8",
        )
      )

    return records

  def test_initialization(self, analyzer):
    """测试初始化"""
    assert isinstance(analyzer, HeartRateAnalyzer)
    assert isinstance(analyzer.stat_analyzer, StatisticalAnalyzer)
    assert isinstance(analyzer.anomaly_detector, AnomalyDetector)

  def test_analyze_resting_heart_rate(
    self, analyzer, sample_resting_hr_records
  ):
    """测试静息心率分析"""
    analysis = analyzer.analyze_resting_heart_rate(sample_resting_hr_records)

    assert isinstance(analysis, RestingHRAnalysis)
    assert analysis.current_value < 72.0  # 应该下降
    assert (
      analysis.baseline_value >= analysis.current_value
    )  # 基线值应该大于等于当前值（因为是下降趋势）
    assert analysis.change_from_baseline <= 0  # 变化应该是负数或零
    assert analysis.trend_direction in ["increasing", "decreasing", "stable"]
    assert analysis.health_rating in ["excellent", "good", "fair", "poor"]

  def test_analyze_hrv(self, analyzer, sample_hrv_records):
    """测试HRV分析"""
    analysis = analyzer.analyze_hrv(sample_hrv_records)

    assert isinstance(analysis, HRVAnalysis)
    assert analysis.current_sdnn > 35.0  # 应该改善
    assert (
      analysis.baseline_sdnn <= analysis.current_sdnn
    )  # 基线值应该小于等于当前值
    assert analysis.change_from_baseline >= 0  # 变化应该是正数或零
    assert analysis.trend_direction in ["improving", "declining", "stable"]
    assert analysis.stress_level in ["low", "moderate", "high", "very_high"]
    assert analysis.recovery_status in ["poor", "fair", "good", "excellent"]

  def test_analyze_cardio_fitness_no_vo2_data(self, analyzer):
    """测试心肺适能分析 - 无VO2数据"""
    analysis = analyzer.analyze_cardio_fitness([])

    assert analysis is None

  def test_analyze_cardio_fitness_with_vo2_data(self, analyzer):
    """测试心肺适能分析 - 有VO2数据"""
    # 创建带年龄和性别的分析器
    analyzer_with_age = HeartRateAnalyzer(age=30, gender="male")

    vo2_records = [
      QuantityRecord(
        type="HKQuantityTypeIdentifierVO2Max",
        source_name="Apple Watch",
        start_date=datetime(2024, 1, 15),
        end_date=datetime(2024, 1, 16),
        creation_date=datetime(2024, 1, 15),
        value=42.0,
        unit="mL/min·kg",
        source_version="1.0",
        device="Apple Watch Series 8",
      )
    ]

    analysis = analyzer_with_age.analyze_cardio_fitness(vo2_records)

    assert isinstance(analysis, CardioFitnessAnalysis)
    assert analysis.current_vo2_max == 42.0
    assert analysis.age_adjusted_rating in [
      "superior",
      "excellent",
      "good",
      "fair",
      "poor",
    ]
    assert isinstance(analysis.fitness_percentile, (int, float))
    assert isinstance(analysis.improvement_potential, (int, float))
    assert isinstance(analysis.training_recommendations, list)

  @patch.object(StatisticalAnalyzer, "analyze_trend")
  def test_analyze_comprehensive(self, mock_trend, analyzer, sample_hr_records):
    """测试综合分析"""
    # Mock趋势分析结果
    mock_trend.return_value = Mock(
      slope=-0.1, r_squared=0.8, trend_direction="decreasing"
    )

    report = analyzer.analyze_comprehensive(sample_hr_records)

    assert isinstance(report, HeartRateAnalysisReport)
    assert report.record_count == len(sample_hr_records)
    assert isinstance(report.data_quality_score, float)
    assert 0 <= report.data_quality_score <= 1

  def test_analyze_comprehensive_empty_records(self, analyzer):
    """测试综合分析 - 空记录"""
    report = analyzer.analyze_comprehensive([])

    assert isinstance(report, HeartRateAnalysisReport)
    assert report.record_count == 0

  def test_data_quality_assessment(self, analyzer, sample_hr_records):
    """测试数据质量评估"""
    quality = analyzer._assess_data_quality(sample_hr_records)

    assert isinstance(quality, float)
    assert 0 <= quality <= 1

  def test_data_quality_assessment_empty(self, analyzer):
    """测试数据质量评估 - 空数据"""
    quality = analyzer._assess_data_quality([])

    assert quality == 0.0
