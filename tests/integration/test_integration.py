"""End-to-end integration tests for Apple Health Analyzer."""

import sys
from datetime import datetime
from pathlib import Path

import pytest

# Add src directory to Python path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from src.analyzers.extended_analyzer import ExtendedHealthAnalyzer
from src.analyzers.highlights import HighlightsGenerator
from src.analyzers.statistical import StatisticalAnalyzer
from src.core.xml_parser import parse_export_file
from src.processors.cleaner import DataCleaner
from src.processors.exporter import DataExporter
from src.processors.heart_rate import HeartRateAnalyzer
from src.processors.sleep import SleepAnalyzer
from src.visualization.reports import ReportGenerator


@pytest.fixture
def sample_xml_path():
  """Get XML path for integration tests."""
  xml_path = Path(__file__).resolve().parents[2] / "example" / "example.xml"
  if not xml_path.exists():
    pytest.fail(
      f"Test data file missing: {xml_path}\n"
      f"Generate test data with:\n"
      f"  python example/create_example_xml.py"
    )
  return xml_path


@pytest.fixture
def output_dir():
  """Create output directory."""
  output_path = Path(__file__).resolve().parents[2] / "output" / "test"
  output_path.mkdir(parents=True, exist_ok=True)
  return output_path


@pytest.fixture
def reports_dir():
  """Create reports directory."""
  reports_path = Path(__file__).resolve().parents[2] / "output" / "test" / "reports"
  reports_path.mkdir(parents=True, exist_ok=True)
  return reports_path


@pytest.mark.slow
@pytest.mark.integration
def test_full_workflow(sample_xml_path, output_dir, reports_dir):
  """Test full workflow."""
  # 1. Parse records.
  records, stats = parse_export_file(sample_xml_path)
  assert len(records) > 0
  assert stats is not None

  # 2. Clean records.
  cleaner = DataCleaner()
  cleaned_records, dedup_result = cleaner.deduplicate_by_time_window(records)
  assert len(cleaned_records) > 0
  assert len(cleaned_records) <= len(records)

  # 3. Validate data quality.
  quality_report = cleaner.validate_data_quality(cleaned_records)
  assert quality_report.quality_score >= 0.0
  assert quality_report.quality_score <= 100.0

  # 4. Statistical analysis.
  analyzer = StatisticalAnalyzer()
  stats_report = analyzer.generate_report(cleaned_records)
  assert stats_report is not None

  # 5. Highlights generation.
  highlights_gen = HighlightsGenerator()
  highlights = highlights_gen.generate_comprehensive_highlights()
  assert highlights is not None

  # 6. Export data.
  exporter = DataExporter(output_dir)
  export_stats = exporter.export_by_category(sample_xml_path)
  assert len(export_stats) > 0

  # 7. Report generation.
  report_gen = ReportGenerator(reports_dir)

  # Generate analyzer reports to exercise report paths.
  sleep_report = SleepAnalyzer().analyze_comprehensive(cleaned_records)
  heart_rate_report = HeartRateAnalyzer().analyze_comprehensive(cleaned_records)

  # Create a minimal report object.
  class SimpleReport:
    def __init__(self, records, stats_report, quality_report, highlights):
      self.records = records
      self.stats_report = stats_report
      self.quality_report = quality_report
      self.highlights = highlights
      # Include required attributes.
      self.overall_wellness_score = 0.75
      self.data_range = (datetime.now(), datetime.now())
      self.data_completeness_score = 0.85
      self.analysis_confidence = 0.8

  simple_report = SimpleReport(
    cleaned_records, stats_report, quality_report, highlights
  )
  html_report = report_gen.generate_comprehensive_report(simple_report)
  assert html_report is not None

  detailed_report = report_gen.generate_html_report(
    title="Integration Report",
    heart_rate_report=heart_rate_report,
    sleep_report=sleep_report,
    highlights=highlights,
    include_charts=False,
  )
  assert detailed_report is not None


@pytest.mark.integration
def test_error_handling():
  """Test error handling."""
  # Missing file.
  with pytest.raises(FileNotFoundError):
    parse_export_file(Path("nonexistent.xml"))

  # Empty record list.
  cleaner = DataCleaner()
  result, dedup_result = cleaner.deduplicate_by_time_window([])
  assert len(result) == 0

  # Invalid data.
  analyzer = StatisticalAnalyzer()
  report = analyzer.generate_report([])
  assert report is not None  # Should handle empty data.


@pytest.mark.integration
def test_edge_cases(sample_xml_path):
  """Test edge cases."""
  # Single record.
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

  # Large dataset subset.
  records, _ = parse_export_file(sample_xml_path)
  subset = records[:1000]  # Only process first 1000 records.

  cleaner = DataCleaner()
  cleaned_subset, _ = cleaner.deduplicate_by_time_window(subset)

  analyzer = StatisticalAnalyzer()
  report = analyzer.generate_report(cleaned_subset)
  assert report is not None


@pytest.mark.integration
def test_data_consistency(sample_xml_path):
  """Test data consistency."""
  # Parse the same file twice.
  records1, stats1 = parse_export_file(sample_xml_path)
  records2, stats2 = parse_export_file(sample_xml_path)

  assert len(records1) == len(records2)
  assert stats1["total_records"] == stats2["total_records"]

  # Cleaning should be deterministic.
  cleaner = DataCleaner()
  cleaned1, _ = cleaner.deduplicate_by_time_window(records1)
  cleaned2, _ = cleaner.deduplicate_by_time_window(records2)

  assert len(cleaned1) == len(cleaned2)


@pytest.mark.integration
def test_export_formats(sample_xml_path, output_dir):
  """Test export formats."""
  records, _ = parse_export_file(sample_xml_path)
  cleaner = DataCleaner()
  cleaned_records, _ = cleaner.deduplicate_by_time_window(records)

  exporter = DataExporter(output_dir)

  # CSV export.
  csv_path = output_dir / "test_export.csv"
  csv_count = exporter.export_to_csv(cleaned_records, csv_path)
  assert csv_path.exists()
  assert csv_path.stat().st_size > 0
  assert csv_count > 0

  # JSON export.
  json_path = output_dir / "test_export.json"
  json_count = exporter.export_to_json(cleaned_records, json_path)
  assert json_path.exists()
  assert json_path.stat().st_size > 0
  assert json_count > 0

  # Record counts should match.
  assert csv_count == json_count


@pytest.mark.integration
def test_analysis_completeness(sample_xml_path):
  """Test analysis completeness."""
  records, _ = parse_export_file(sample_xml_path)
  cleaner = DataCleaner()
  cleaned_records, _ = cleaner.deduplicate_by_time_window(records)

  # Statistical analysis.
  analyzer = StatisticalAnalyzer()
  stats_report = analyzer.generate_report(cleaned_records, output_format="dataframe")

  # Report should include required info.
  assert hasattr(stats_report, "record_count")

  # Highlights generation.
  highlights_gen = HighlightsGenerator()
  highlights = highlights_gen.generate_comprehensive_highlights()

  # Highlights should include required info.
  assert hasattr(highlights, "insights")
  assert hasattr(highlights, "recommendations")

  # Specialized processors should return report objects.
  sleep_analyzer = SleepAnalyzer()
  sleep_report = sleep_analyzer.analyze_comprehensive(cleaned_records)
  assert sleep_report is not None
  assert sleep_report.record_count >= 0

  heart_rate_analyzer = HeartRateAnalyzer()
  hr_report = heart_rate_analyzer.analyze_comprehensive(cleaned_records)
  assert hr_report is not None
  assert hr_report.record_count >= 0

  # Extended analyzer should produce a report with scores.
  extended_analyzer = ExtendedHealthAnalyzer()
  comprehensive_report = extended_analyzer.analyze_comprehensive_health(cleaned_records)
  assert comprehensive_report is not None
  assert comprehensive_report.overall_wellness_score >= 0
  assert comprehensive_report.analysis_confidence >= 0


@pytest.mark.integration
def test_memory_cleanup():
  """Test memory cleanup."""
  import gc

  initial_objects = len(gc.get_objects())

  # Run a workflow subset.
  xml_path = Path(__file__).parent.parent / "example" / "example.xml"
  if xml_path.exists():
    records, _ = parse_export_file(xml_path)
    cleaner = DataCleaner()
    cleaned_records, _ = cleaner.deduplicate_by_time_window(records[:1000])

    analyzer = StatisticalAnalyzer()
    analyzer.generate_report(cleaned_records)

    highlights_gen = HighlightsGenerator()
    highlights_gen.generate_comprehensive_highlights()

  # Force garbage collection.
  gc.collect()

  # Ensure object count does not grow too much.
  final_objects = len(gc.get_objects())
  growth_ratio = final_objects / initial_objects

  # Object growth should not exceed 50%.
  assert growth_ratio < 1.5, f"Object growth too large: {growth_ratio:.2f}"


@pytest.mark.integration
def test_workflow_robustness(sample_xml_path, output_dir):
  """Test workflow robustness."""
  # Partial failures should not break the entire flow.
  try:
    # 1. Parse.
    records, stats = parse_export_file(sample_xml_path)

    # 2. Clean - continue on failure.
    cleaner = DataCleaner()
    try:
      cleaned_records, _ = cleaner.deduplicate_by_time_window(records)
    except Exception:
      cleaned_records = records  # Continue with raw data.

    # 3. Analyze - continue on failure.
    analyzer = StatisticalAnalyzer()
    try:
      stats_report = analyzer.generate_report(cleaned_records)
    except Exception:
      stats_report = None

    # 4. Highlights - continue on failure.
    highlights_gen = HighlightsGenerator()
    try:
      highlights = highlights_gen.generate_comprehensive_highlights()
    except Exception:
      highlights = None

    # 5. Export - continue on failure.
    exporter = DataExporter(output_dir)
    try:
      export_stats = exporter.export_by_category(sample_xml_path)
    except Exception:
      export_stats = {}

    # Parsing should succeed at minimum.
    assert len(records) > 0

  except Exception as e:
    pytest.fail(f"Workflow is too fragile: {e}")
