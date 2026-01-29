"""
端到端集成测试
测试Apple Health数据分析器的完整工作流程
"""

import sys
from datetime import datetime
from pathlib import Path

import pytest

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.analyzers.highlights import HighlightsGenerator
from src.analyzers.statistical import StatisticalAnalyzer
from src.core.xml_parser import parse_export_file
from src.processors.cleaner import DataCleaner
from src.processors.exporter import DataExporter
from src.visualization.reports import ReportGenerator


@pytest.fixture
def sample_xml_path():
  """获取测试用的XML文件路径"""
  xml_path = Path(__file__).parent.parent / "export_data" / "export.xml"
  if not xml_path.exists():
    pytest.skip(f"测试数据文件不存在: {xml_path}")
  return xml_path


@pytest.fixture
def output_dir():
  """创建输出目录"""
  output_path = Path(__file__).parent.parent / "output"
  output_path.mkdir(exist_ok=True)
  return output_path


@pytest.fixture
def reports_dir():
  """创建报告目录"""
  reports_path = Path(__file__).parent.parent / "reports"
  reports_path.mkdir(exist_ok=True)
  return reports_path


@pytest.mark.slow
@pytest.mark.integration
def test_full_workflow(sample_xml_path, output_dir, reports_dir):
  """测试完整工作流程"""
  # 1. 数据解析
  records, stats = parse_export_file(sample_xml_path)
  assert len(records) > 0
  assert stats is not None

  # 2. 数据清洗
  cleaner = DataCleaner()
  cleaned_records, dedup_result = cleaner.deduplicate_by_time_window(records)
  assert len(cleaned_records) > 0
  assert len(cleaned_records) <= len(records)

  # 3. 数据质量验证
  quality_report = cleaner.validate_data_quality(cleaned_records)
  assert quality_report.quality_score >= 0.0
  assert quality_report.quality_score <= 1.0

  # 4. 统计分析
  analyzer = StatisticalAnalyzer()
  stats_report = analyzer.generate_report(cleaned_records)
  assert stats_report is not None

  # 5. 亮点生成
  highlights_gen = HighlightsGenerator()
  highlights = highlights_gen.generate_comprehensive_highlights()
  assert highlights is not None

  # 6. 数据导出
  exporter = DataExporter(output_dir)
  export_stats = exporter.export_by_category(sample_xml_path)
  assert len(export_stats) > 0

  # 7. 报告生成
  report_gen = ReportGenerator()

  # 创建一个简单的报告对象
  class SimpleReport:
    def __init__(self, records, stats_report, quality_report, highlights):
      self.records = records
      self.stats_report = stats_report
      self.quality_report = quality_report
      self.highlights = highlights
      # 添加一些必需的属性
      self.overall_wellness_score = 0.75
      self.data_range = (datetime.now(), datetime.now())
      self.data_completeness_score = 0.85
      self.analysis_confidence = 0.8

  simple_report = SimpleReport(
    cleaned_records, stats_report, quality_report, highlights
  )
  html_report = report_gen.generate_comprehensive_report(simple_report)
  assert html_report is not None


@pytest.mark.integration
def test_error_handling():
  """测试错误处理"""
  # 测试不存在的文件
  with pytest.raises(FileNotFoundError):
    parse_export_file(Path("nonexistent.xml"))

  # 测试空记录列表
  cleaner = DataCleaner()
  result, dedup_result = cleaner.deduplicate_by_time_window([])
  assert len(result) == 0

  # 测试无效数据
  analyzer = StatisticalAnalyzer()
  report = analyzer.generate_report([])
  assert report is not None  # 应该能处理空数据


@pytest.mark.integration
def test_edge_cases(sample_xml_path):
  """测试边缘情况"""
  # 测试单条记录
  from src.core.data_models import QuantityRecord

  single_record = QuantityRecord(
    type="HKQuantityTypeIdentifierHeartRate",
    source_name="Test",
    start_date=datetime.now(),
    end_date=datetime.now(),
    value=70.0,
    unit="count/min",
    source_version="1.0",
    device="Test Device",
    creation_date=datetime.now(),
  )

  cleaner = DataCleaner()
  result, dedup_result = cleaner.deduplicate_by_time_window([single_record])
  assert len(result) == 1

  # 测试大数据集子集
  records, _ = parse_export_file(sample_xml_path)
  subset = records[:1000]  # 只处理前1000条

  cleaner = DataCleaner()
  cleaned_subset, _ = cleaner.deduplicate_by_time_window(subset)

  analyzer = StatisticalAnalyzer()
  report = analyzer.generate_report(cleaned_subset)
  assert report is not None


@pytest.mark.integration
def test_data_consistency(sample_xml_path):
  """测试数据一致性"""
  # 多次解析同一文件应该得到相同的结果
  records1, stats1 = parse_export_file(sample_xml_path)
  records2, stats2 = parse_export_file(sample_xml_path)

  assert len(records1) == len(records2)
  assert stats1["total_records"] == stats2["total_records"]

  # 数据清洗应该是确定性的
  cleaner = DataCleaner()
  cleaned1, _ = cleaner.deduplicate_by_time_window(records1)
  cleaned2, _ = cleaner.deduplicate_by_time_window(records2)

  assert len(cleaned1) == len(cleaned2)


@pytest.mark.integration
def test_export_formats(sample_xml_path, output_dir):
  """测试各种导出格式"""
  records, _ = parse_export_file(sample_xml_path)
  cleaner = DataCleaner()
  cleaned_records, _ = cleaner.deduplicate_by_time_window(records)

  exporter = DataExporter(output_dir)

  # 测试CSV导出
  csv_path = output_dir / "test_export.csv"
  csv_count = exporter.export_to_csv(cleaned_records, csv_path)
  assert csv_path.exists()
  assert csv_path.stat().st_size > 0
  assert csv_count > 0

  # 测试JSON导出
  json_path = output_dir / "test_export.json"
  json_count = exporter.export_to_json(cleaned_records, json_path)
  assert json_path.exists()
  assert json_path.stat().st_size > 0
  assert json_count > 0

  # 记录数量应该一致
  assert csv_count == json_count


@pytest.mark.integration
def test_analysis_completeness(sample_xml_path):
  """测试分析的完整性"""
  records, _ = parse_export_file(sample_xml_path)
  cleaner = DataCleaner()
  cleaned_records, _ = cleaner.deduplicate_by_time_window(records)

  # 统计分析
  analyzer = StatisticalAnalyzer()
  stats_report = analyzer.generate_report(
    cleaned_records, output_format="dataframe"
  )

  # 检查报告包含必要的信息
  assert hasattr(stats_report, "record_count")
  assert stats_report.record_count > 0  # type: ignore

  # 亮点生成
  highlights_gen = HighlightsGenerator()
  highlights = highlights_gen.generate_comprehensive_highlights()

  # 检查亮点包含必要的信息
  assert hasattr(highlights, "insights")
  assert hasattr(highlights, "recommendations")


@pytest.mark.integration
def test_memory_cleanup():
  """测试内存清理"""
  import gc

  initial_objects = len(gc.get_objects())

  # 执行完整流程
  xml_path = Path(__file__).parent.parent / "export_data" / "export.xml"
  if xml_path.exists():
    records, _ = parse_export_file(xml_path)
    cleaner = DataCleaner()
    cleaned_records, _ = cleaner.deduplicate_by_time_window(records[:1000])

    analyzer = StatisticalAnalyzer()
    analyzer.generate_report(cleaned_records)

    highlights_gen = HighlightsGenerator()
    highlights_gen.generate_comprehensive_highlights()

  # 强制垃圾回收
  gc.collect()

  # 检查对象数量没有过度增长
  final_objects = len(gc.get_objects())
  growth_ratio = final_objects / initial_objects

  # 对象增长不应超过50%
  assert growth_ratio < 1.5, f"对象数量增长过大: {growth_ratio:.2f}"


@pytest.mark.integration
def test_workflow_robustness(sample_xml_path, output_dir):
  """测试工作流程的鲁棒性"""
  # 测试部分失败不影响整体流程
  try:
    # 1. 解析
    records, stats = parse_export_file(sample_xml_path)

    # 2. 清洗 - 即使失败也继续
    cleaner = DataCleaner()
    try:
      cleaned_records, _ = cleaner.deduplicate_by_time_window(records)
    except Exception:
      cleaned_records = records  # 使用原始数据继续

    # 3. 分析 - 即使失败也继续
    analyzer = StatisticalAnalyzer()
    try:
      stats_report = analyzer.generate_report(cleaned_records)
    except Exception:
      stats_report = None

    # 4. 亮点生成 - 即使失败也继续
    highlights_gen = HighlightsGenerator()
    try:
      highlights = highlights_gen.generate_comprehensive_highlights()
    except Exception:
      highlights = None

    # 5. 导出 - 即使失败也继续
    exporter = DataExporter(output_dir)
    try:
      export_stats = exporter.export_by_category(sample_xml_path)
    except Exception:
      export_stats = {}

    # 至少解析应该成功
    assert len(records) > 0

  except Exception as e:
    pytest.fail(f"工作流程过于脆弱: {e}")
