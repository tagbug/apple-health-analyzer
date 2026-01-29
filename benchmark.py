#!/usr/bin/env python3
"""
性能基准测试脚本
测试Apple Health数据分析器的各项性能指标
"""

import os
import sys
import time
from pathlib import Path

import psutil

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.analyzers.highlights import HighlightsGenerator
from src.analyzers.statistical import StatisticalAnalyzer
from src.core.xml_parser import parse_export_file
from src.processors.cleaner import DataCleaner
from src.processors.exporter import DataExporter


def get_memory_usage():
  """获取当前进程的内存使用量（MB）"""
  process = psutil.Process(os.getpid())
  return process.memory_info().rss / 1024 / 1024  # MB


def benchmark_parsing(xml_path: str):
  """测试XML解析性能"""
  print("\n1. 测试XML解析性能...")
  start_time = time.time()
  start_mem = get_memory_usage()

  records, stats = parse_export_file(Path(xml_path))

  parse_time = time.time() - start_time
  parse_mem = get_memory_usage() - start_mem

  print(f"解析时间: {parse_time:.2f} 秒")
  print(f"解析内存使用: {parse_mem:.2f} MB")
  print(f"记录数量: {len(records)}")
  print(f"解析速度: {len(records) / parse_time:.0f} 条/秒")

  return records, parse_time, parse_mem


def benchmark_cleaning(records):
  """测试数据清洗性能"""
  print("\n2. 测试数据清洗性能...")
  start_time = time.time()
  start_mem = get_memory_usage()

  cleaner = DataCleaner()
  cleaned_records, _ = cleaner.deduplicate_by_time_window(
    records,
    window_seconds=300,  # 5分钟 = 300秒
  )

  clean_time = time.time() - start_time
  clean_mem = get_memory_usage() - start_mem

  print(f"清洗时间: {clean_time:.2f} 秒")
  print(f"清洗内存使用: {clean_mem:.2f} MB")
  print(f"原始记录: {len(records)}, 清洗后: {len(cleaned_records)}")

  return cleaned_records, clean_time, clean_mem


def benchmark_analysis(cleaned_records):
  """测试统计分析性能"""
  print("\n3. 测试统计分析性能...")
  start_time = time.time()
  start_mem = get_memory_usage()

  analyzer = StatisticalAnalyzer()
  stats_report = analyzer.generate_report(cleaned_records)

  stats_time = time.time() - start_time
  stats_mem = get_memory_usage() - start_mem

  print(f"统计分析时间: {stats_time:.2f} 秒")
  print(f"统计分析内存使用: {stats_mem:.2f} MB")

  return stats_report, stats_time, stats_mem


def benchmark_highlights(cleaned_records, stats_report):
  """测试亮点生成性能"""
  print("\n4. 测试亮点生成性能...")
  start_time = time.time()
  start_mem = get_memory_usage()

  highlights_gen = HighlightsGenerator()
  highlights = highlights_gen.generate_comprehensive_highlights()

  highlights_time = time.time() - start_time
  highlights_mem = get_memory_usage() - start_mem

  print(f"亮点生成时间: {highlights_time:.2f} 秒")
  print(f"亮点生成内存使用: {highlights_mem:.2f} MB")

  return highlights, highlights_time, highlights_mem


def benchmark_export(cleaned_records, output_path: str):
  """测试数据导出性能"""
  print("\n5. 测试数据导出性能...")
  start_time = time.time()
  start_mem = get_memory_usage()

  exporter = DataExporter(Path("output"))
  exporter.export_to_csv(cleaned_records, Path(output_path))

  export_time = time.time() - start_time
  export_mem = get_memory_usage() - start_mem

  print(f"导出时间: {export_time:.2f} 秒")
  print(f"导出内存使用: {export_mem:.2f} MB")

  return export_time, export_mem


def main():
  """主函数"""
  print("=== 性能基准测试 ===")
  print(f"初始内存使用: {get_memory_usage():.2f} MB")

  # 设置路径
  xml_path = "../export_data/export.xml"
  output_path = "output/benchmark_test.csv"

  # 确保输出目录存在
  Path("output").mkdir(exist_ok=True)

  try:
    # 执行各项基准测试
    records, parse_time, parse_mem = benchmark_parsing(xml_path)
    cleaned_records, clean_time, clean_mem = benchmark_cleaning(records)
    stats_report, stats_time, stats_mem = benchmark_analysis(cleaned_records)
    highlights, highlights_time, highlights_mem = benchmark_highlights(
      cleaned_records, stats_report
    )
    export_time, export_mem = benchmark_export(cleaned_records, output_path)

    # 计算总计
    total_time = (
      parse_time + clean_time + stats_time + highlights_time + export_time
    )
    total_mem = parse_mem + clean_mem + stats_mem + highlights_mem + export_mem

    print("\n=== 性能汇总 ===")
    print(f"总处理时间: {total_time:.2f} 秒")
    print(f"总内存使用: {total_mem:.2f} MB")
    print(f"平均处理速度: {len(records) / total_time:.0f} 条/秒")
    print(f"最终内存使用: {get_memory_usage():.2f} MB")

    # 输出详细性能指标
    print("\n=== 详细性能指标 ===")
    print(f"XML解析: {parse_time:.3f}s ({parse_mem:.2f}MB)")
    print(f"数据清洗: {clean_time:.3f}s ({clean_mem:.2f}MB)")
    print(f"统计分析: {stats_time:.3f}s ({stats_mem:.2f}MB)")
    print(f"亮点生成: {highlights_time:.3f}s ({highlights_mem:.2f}MB)")
    print(f"数据导出: {export_time:.3f}s ({export_mem:.2f}MB)")

    print("\n✅ 性能基准测试完成")

  except Exception as e:
    print(f"❌ 性能基准测试失败: {e}")
    sys.exit(1)


if __name__ == "__main__":
  main()
