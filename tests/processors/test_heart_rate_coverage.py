"""Additional coverage for heart rate analysis branches."""

from datetime import datetime

from src.processors.heart_rate import HeartRateAnalyzer
from src.core.data_models import QuantityRecord


def test_analyze_comprehensive_with_only_base_records():
  """Ensure analysis handles missing optional record types."""
  analyzer = HeartRateAnalyzer(age=30, gender="male")

  record = QuantityRecord(
    type="HKQuantityTypeIdentifierHeartRate",
    source_name="Apple Watch",
    start_date=datetime(2024, 1, 1, 8, 0, 0),
    end_date=datetime(2024, 1, 1, 8, 1, 0),
    creation_date=datetime(2024, 1, 1, 8, 1, 0),
    value=70.0,
    unit="count/min",
    source_version="1.0",
    device="Apple Watch",
  )

  report = analyzer.analyze_comprehensive([record])

  assert report.record_count == 1
  assert report.resting_hr_analysis is None
  assert report.hrv_analysis is None
  assert report.cardio_fitness is None
  assert report.data_quality_score >= 0
