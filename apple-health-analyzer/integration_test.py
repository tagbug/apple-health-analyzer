#!/usr/bin/env python3
"""
端到端集成测试脚本
测试Apple Health数据分析器的完整工作流程
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.analyzers.highlights import HighlightsGenerator
from src.analyzers.statistical import StatisticalAnalyzer
from src.core.xml_parser import parse_export_file
from src.processors.cleaner import DataCleaner
from src.processors.exporter import DataExporter
from src.visualization.reports import ReportGenerator


def get_memory_usage():
  """获取当前进程的内存使用量（MB）"""
  process = psutil.Process(os.getpid())
  return process.memory_info().rss / 1024 / 1024  # MB


def test_full_workflow():
  """测试完整工作流程"""
  print("=== 端到端集成测试 ===")
  print(f"初始内存使用: {get_memory_usage():.2f} MB")

  start_time = time.time()
  start_mem = get_memory_usage()

  try:
    # 1. 数据解析
    print("\n1. 解析XML数据...")
    xml_path = Path("../export_data/export.xml")
    records, stats = parse_export_file(xml_path)
    print(f"✅ 解析完成: {len(records)} 条记录")

    # 2. 数据清洗
    print("\n2. 数据清洗...")
    cleaner = DataCleaner()
    cleaned_records, dedup_result = cleaner.deduplicate_by_time_window(records)
    print(
      f"✅ 清洗完成: {len(cleaned_records)} 条记录 (移除 {dedup_result.removed_duplicates} 条重复)"
    )

    # 3. 数据质量验证
    print("\n3. 数据质量验证...")
    quality_report = cleaner.validate_data_quality(cleaned_records)
    print(f"✅ 质量验证完成: 质量评分 {quality_report.quality_score:.2f}")

    # 4. 统计分析
    print("\n4. 统计分析...")
    analyzer = StatisticalAnalyzer()
    stats_report = analyzer.generate_report(cleaned_records)
    print("✅ 统计分析完成")

    # 5. 亮点生成
    print("\n5. 亮点生成...")
    highlights_gen = HighlightsGenerator()
    highlights = highlights_gen.generate_comprehensive_highlights()
    print(f"✅ 亮点生成完成: {len(highlights.insights)} 个洞察")

    # 6. 数据导出
    print("\n6. 数据导出...")
    exporter = DataExporter(Path("output"))
    export_stats = exporter.export_by_category(xml_path)
    print(
      f"✅ 数据导出完成: {sum(len(files) for files in export_stats.values())} 个文件"
    )

    # 7. 报告生成
    print("\n7. 报告生成...")
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
    print("✅ 报告生成完成")

    # 计算总时间和内存
    total_time = time.time() - start_time
    total_mem = get_memory_usage() - start_mem

    print("\n=== 集成测试结果 ===")
    print(f"总处理时间: {total_time:.2f} 秒")
    print(f"内存使用: {total_mem:.2f} MB")
    print(f"处理速度: {len(records) / total_time:.0f} 条/秒")
    print("✅ 端到端集成测试通过")
    return True

  except Exception as e:
    print(f"❌ 集成测试失败: {e}")
    import traceback

    traceback.print_exc()
    return False


def test_error_handling():
  """测试错误处理"""
  print("\n=== 错误处理测试 ===")

  # 测试不存在的文件
  try:
    parse_export_file(Path("nonexistent.xml"))
    print("❌ 应该抛出文件不存在错误")
    return False
  except FileNotFoundError:
    print("✅ 正确处理文件不存在错误")

  # 测试空记录列表
  try:
    cleaner = DataCleaner()
    result = cleaner.deduplicate_by_time_window([])
    print("✅ 正确处理空记录列表")
  except Exception as e:
    print(f"❌ 处理空记录列表失败: {e}")
    return False

  # 测试无效数据
  try:
    analyzer = StatisticalAnalyzer()
    result = analyzer.generate_report([])
    print("✅ 正确处理空统计分析")
  except Exception as e:
    print(f"❌ 处理空统计分析失败: {e}")
    return False

  print("✅ 错误处理测试通过")
  return True


def test_edge_cases():
  """测试边缘情况"""
  print("\n=== 边缘情况测试 ===")

  # 测试单条记录
  try:
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
    print("✅ 单条记录处理正确")
  except Exception as e:
    print(f"❌ 单条记录处理失败: {e}")
    return False

  # 测试大数据集子集
  try:
    xml_path = Path("../export_data/export.xml")
    records, stats = parse_export_file(xml_path)

    # 只处理前1000条记录
    subset = records[:1000]
    analyzer = StatisticalAnalyzer()
    report = analyzer.generate_report(subset)
    print("✅ 大数据集子集处理正确")
  except Exception as e:
    print(f"❌ 大数据集子集处理失败: {e}")
    return False

  print("✅ 边缘情况测试通过")
  return True


def main():
  """主函数"""
  print("开始系统集成测试...")

  # 确保输出目录存在
  Path("output").mkdir(exist_ok=True)
  Path("reports").mkdir(exist_ok=True)

  results = []

  # 运行各项测试
  results.append(("完整工作流程", test_full_workflow()))
  results.append(("错误处理", test_error_handling()))
  results.append(("边缘情况", test_edge_cases()))

  # 输出总结
  print("\n=== 测试总结 ===")
  passed = 0
  total = len(results)

  for test_name, success in results:
    status = "✅ 通过" if success else "❌ 失败"
    print(f"{test_name}: {status}")
    if success:
      passed += 1

  print(f"\n总体结果: {passed}/{total} 项测试通过")

  if passed == total:
    print("🎉 所有集成测试通过！")
    return 0
  else:
    print("⚠️  部分测试失败，需要检查")
    return 1


if __name__ == "__main__":
  sys.exit(main())
