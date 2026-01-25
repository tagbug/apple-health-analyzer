"""Tests for XML parser functionality."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.core.data_models import HeartRateRecord
from src.core.xml_parser import (
  StreamingXMLParser,
  get_export_file_info,
  parse_export_file,
)


class TestStreamingXMLParser:
  """Test StreamingXMLParser class."""

  @pytest.fixture
  def sample_xml_content(self):
    """Create sample XML content for testing."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<HealthData locale="en_US">
    <ExportDate>2023-01-01 12:00:00 -0800</ExportDate>
    <Me HKCharacteristicTypeIdentifierDateOfBirth="1990-01-01" HKCharacteristicTypeIdentifierBiologicalSex="HKBiologicalSexMale"/>
    <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Apple Watch" sourceVersion="8.0" device="Watch6,1" unit="count/min" creationDate="2023-01-01 10:00:00 -0800" startDate="2023-01-01 10:00:00 -0800" endDate="2023-01-01 10:00:00 -0800" value="75"/>
    <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Apple Watch" sourceVersion="8.0" device="Watch6,1" unit="count/min" creationDate="2023-01-01 10:01:00 -0800" startDate="2023-01-01 10:01:00 -0800" endDate="2023-01-01 10:01:00 -0800" value="80"/>
    <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Apple Watch" sourceVersion="8.0" device="Watch6,1" creationDate="2023-01-01 22:00:00 -0800" startDate="2023-01-01 22:00:00 -0800" endDate="2023-01-01 23:00:00 -0800" value="HKCategoryValueSleepAnalysisAsleepCore"/>
    <Workout workoutActivityType="HKWorkoutActivityTypeRunning" duration="1800.0" durationUnit="s" sourceName="Apple Watch" sourceVersion="8.0" device="Watch6,1" creationDate="2023-01-01 18:00:00 -0800" startDate="2023-01-01 18:00:00 -0800" endDate="2023-01-01 19:30:00 -0800">
        <WorkoutStatistics type="HKQuantityTypeIdentifierActiveEnergyBurned" startDate="2023-01-01 18:00:00 -0800" endDate="2023-01-01 19:30:00 -0800" sum="300.0" unit="kcal"/>
        <WorkoutStatistics type="HKQuantityTypeIdentifierDistanceWalkingRunning" startDate="2023-01-01 18:00:00 -0800" endDate="2023-01-01 19:30:00 -0800" sum="5.0" unit="km"/>
        <MetadataEntry key="HKWeatherTemperature" value="20.5"/>
    </Workout>
    <ActivitySummary dateComponents="2023-01-01" activeEnergyBurned="800.0" appleExerciseTime="45.0" appleStandHours="12.0" activeEnergyBurnedGoal="600.0" appleExerciseTimeGoal="30.0" appleStandHoursGoal="12.0"/>
</HealthData>"""

  @pytest.fixture
  def temp_xml_file(self, sample_xml_content):
    """Create a temporary XML file for testing."""
    with tempfile.NamedTemporaryFile(
      mode="w", suffix=".xml", delete=False
    ) as f:
      f.write(sample_xml_content)
      temp_path = Path(f.name)
    yield temp_path
    temp_path.unlink()  # Cleanup

  def test_parser_initialization(self, temp_xml_file):
    """Test parser initialization."""
    parser = StreamingXMLParser(temp_xml_file)

    assert parser.xml_path == temp_xml_file
    assert parser.stats["total_records"] == 0
    assert parser.stats["processed_records"] == 0

  def test_parse_records_all_types(self, temp_xml_file):
    """Test parsing all record types."""
    parser = StreamingXMLParser(temp_xml_file)

    records = list(parser.parse_records())

    # Should parse 3 records: 2 heart rate + 1 sleep
    assert len(records) == 3

    # Check record types
    record_types = [r.type for r in records]  # type: ignore
    assert "HKQuantityTypeIdentifierHeartRate" in record_types
    assert "HKCategoryTypeIdentifierSleepAnalysis" in record_types

    # Check values
    hr_records = [r for r in records if isinstance(r, HeartRateRecord)]
    assert len(hr_records) == 2
    assert hr_records[0].value == 75.0
    assert hr_records[1].value == 80.0

  def test_parse_records_filtered_types(self, temp_xml_file):
    """Test parsing with record type filter."""
    parser = StreamingXMLParser(temp_xml_file)

    # Only parse heart rate records
    records = list(
      parser.parse_records(record_types=["HKQuantityTypeIdentifierHeartRate"])
    )

    assert len(records) == 2
    assert all(r.type == "HKQuantityTypeIdentifierHeartRate" for r in records)  # type: ignore

  def test_parse_workouts(self, temp_xml_file):
    """Test parsing workout records."""
    parser = StreamingXMLParser(temp_xml_file)

    workouts = list(parser.parse_workouts())

    assert len(workouts) == 1
    workout = workouts[0]
    assert workout.activity_type == "Running"
    assert workout.workout_duration_seconds == 1800.0
    assert workout.calories == 300.0
    assert workout.distance_km == 5.0
    assert workout.metadata["HKWeatherTemperature"] == "20.5"  # type: ignore

  def test_parse_activity_summaries(self, temp_xml_file):
    """Test parsing activity summary records."""
    parser = StreamingXMLParser(temp_xml_file)

    summaries = list(parser.parse_activity_summaries())

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.move_calories == 800.0
    assert summary.exercise_minutes == 45.0
    assert summary.stand_hours == 12.0
    assert summary.move_achieved is True  # 800 >= 600
    assert summary.exercise_achieved is True  # 45 >= 30
    assert summary.stand_achieved is True  # 12 >= 12

  def test_get_statistics(self, temp_xml_file):
    """Test getting parsing statistics."""
    parser = StreamingXMLParser(temp_xml_file)

    # Parse all records first
    records = list(parser.parse_records())  # noqa: F841
    stats = parser.get_statistics()

    assert stats["total_records"] == 3
    assert stats["processed_records"] == 3
    assert stats["skipped_records"] == 0
    assert stats["invalid_records"] == 0
    assert stats["success_rate"] == 1.0

    # Check record types
    assert stats["record_types"]["HKQuantityTypeIdentifierHeartRate"] == 2
    assert stats["record_types"]["HKCategoryTypeIdentifierSleepAnalysis"] == 1

    # Check sources
    assert stats["sources"]["Apple Watch"] == 3

  def test_parse_records_with_filter(self, temp_xml_file):
    """Test parsing with record type filter and statistics."""
    parser = StreamingXMLParser(temp_xml_file)

    # Parse only heart rate records
    records = list(
      parser.parse_records(record_types=["HKQuantityTypeIdentifierHeartRate"])
    )
    stats = parser.get_statistics()

    assert len(records) == 2
    assert stats["processed_records"] == 2
    assert stats["skipped_records"] == 1  # Sleep record was skipped

  def test_empty_xml_file(self):
    """Test parsing empty XML file."""
    with tempfile.NamedTemporaryFile(
      mode="w", suffix=".xml", delete=False
    ) as f:
      f.write('<?xml version="1.0"?><HealthData></HealthData>')
      temp_path = Path(f.name)

    try:
      parser = StreamingXMLParser(temp_path)
      records = list(parser.parse_records())

      assert len(records) == 0

      stats = parser.get_statistics()
      assert stats["total_records"] == 0
      assert stats["processed_records"] == 0
    finally:
      temp_path.unlink()

  def test_malformed_xml_handling(self):
    """Test handling of malformed XML."""
    with tempfile.NamedTemporaryFile(
      mode="w", suffix=".xml", delete=False
    ) as f:
      f.write("<invalid xml content>")
      temp_path = Path(f.name)

    try:
      parser = StreamingXMLParser(temp_path)

      # Should raise an exception for malformed XML
      with pytest.raises((Exception, ValueError, OSError)):
        list(parser.parse_records())
    finally:
      temp_path.unlink()

  def test_memory_cleanup(self, temp_xml_file):
    """Test that memory cleanup works properly."""
    parser = StreamingXMLParser(temp_xml_file)

    # Parse records with small batch size to trigger cleanup
    records = list(parser.parse_records(batch_size=1))

    assert len(records) == 3

    # Parser should still be functional after cleanup
    stats = parser.get_statistics()
    assert stats["processed_records"] == 3

  @patch("src.core.xml_parser.ProgressLogger")
  def test_progress_logging(self, mock_progress_logger, temp_xml_file):
    """Test that progress logging is called."""
    parser = StreamingXMLParser(temp_xml_file)

    records = list(parser.parse_records(batch_size=1))

    # ProgressLogger should have been called
    assert mock_progress_logger.called
    assert len(records) == 3


class TestParseExportFile:
  """Test the parse_export_file convenience function."""

  @pytest.fixture
  def sample_xml_content(self):
    """Create sample XML content for testing."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<HealthData locale="en_US">
    <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Apple Watch" sourceVersion="8.0" device="Watch6,1" unit="count/min" creationDate="2023-01-01 10:00:00 -0800" startDate="2023-01-01 10:00:00 -0800" endDate="2023-01-01 10:00:00 -0800" value="75"/>
    <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Apple Watch" sourceVersion="8.0" device="Watch6,1" unit="count/min" creationDate="2023-01-01 10:01:00 -0800" startDate="2023-01-01 10:01:00 -0800" endDate="2023-01-01 10:01:00 -0800" value="80"/>
</HealthData>"""

  @pytest.fixture
  def temp_xml_file(self, sample_xml_content):
    """Create a temporary XML file for testing."""
    with tempfile.NamedTemporaryFile(
      mode="w", suffix=".xml", delete=False
    ) as f:
      f.write(sample_xml_content)
      temp_path = Path(f.name)
    yield temp_path
    temp_path.unlink()

  def test_parse_export_file_all_records(self, temp_xml_file):
    """Test parse_export_file function."""
    records, stats = parse_export_file(temp_xml_file)

    assert len(records) == 2
    assert isinstance(records[0], HeartRateRecord)
    assert records[0].value == 75.0
    assert records[1].value == 80.0

    assert stats["total_records"] == 2
    assert stats["processed_records"] == 2
    assert stats["success_rate"] == 1.0

  def test_parse_export_file_filtered_types(self, temp_xml_file):
    """Test parse_export_file with record type filter."""
    records, stats = parse_export_file(
      temp_xml_file, record_types=["HKQuantityTypeIdentifierHeartRate"]
    )

    assert len(records) == 2
    assert stats["processed_records"] == 2


class TestGetExportFileInfo:
  """Test get_export_file_info function."""

  @pytest.fixture
  def sample_xml_content(self):
    """Create sample XML content for testing."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<HealthData locale="en_US">
    <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Apple Watch" value="75"/>
    <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Apple Watch" value="80"/>
    <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Apple Watch" value="HKCategoryValueSleepAnalysisAsleepCore"/>
</HealthData>"""

  @pytest.fixture
  def temp_xml_file(self, sample_xml_content):
    """Create a temporary XML file for testing."""
    with tempfile.NamedTemporaryFile(
      mode="w", suffix=".xml", delete=False
    ) as f:
      f.write(sample_xml_content)
      temp_path = Path(f.name)
    yield temp_path
    temp_path.unlink()

  def test_get_export_file_info(self, temp_xml_file):
    """Test getting file information."""
    info = get_export_file_info(temp_xml_file)

    assert "file_path" in info
    assert "file_size_bytes" in info
    assert "file_size_mb" in info
    assert "estimated_record_count" in info
    assert "last_modified" in info

    assert info["estimated_record_count"] == 3  # 3 Record elements
    assert info["file_size_bytes"] > 0
    assert info["file_size_mb"] >= 0  # Allow very small files

  def test_get_export_file_info_nonexistent_file(self):
    """Test getting info for nonexistent file."""
    info = get_export_file_info(Path("nonexistent.xml"))

    assert info == {}  # Should return empty dict on error


class TestXMLParserEdgeCases:
  """Test edge cases and error handling."""

  @pytest.fixture
  def temp_xml_file(self):
    """Create temporary XML file for testing."""
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<HealthData locale="en_US">
    <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Apple Watch" sourceVersion="8.0" device="Watch6,1" unit="count/min" creationDate="2023-01-01 10:00:00 -0800" startDate="2023-01-01 10:00:00 -0800" endDate="2023-01-01 10:00:00 -0800" value="75"/>
    <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Apple Watch" sourceVersion="8.0" device="Watch6,1" unit="count/min" creationDate="2023-01-01 10:01:00 -0800" startDate="2023-01-01 10:01:00 -0800" endDate="2023-01-01 10:01:00 -0800" value="80"/>
    <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Apple Watch" sourceVersion="8.0" device="Watch6,1" creationDate="2023-01-01 22:00:00 -0800" startDate="2023-01-01 22:00:00 -0800" endDate="2023-01-01 23:00:00 -0800" value="HKCategoryValueSleepAnalysisAsleepCore"/>
</HealthData>"""

    with tempfile.NamedTemporaryFile(
      mode="w", suffix=".xml", delete=False
    ) as f:
      f.write(xml_content)
      temp_path = Path(f.name)
    yield temp_path
    temp_path.unlink()

  def test_record_with_missing_attributes(self):
    """Test parsing record with missing attributes."""
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<HealthData>
    <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Apple Watch" value="75"/>
</HealthData>"""

    with tempfile.NamedTemporaryFile(
      mode="w", suffix=".xml", delete=False
    ) as f:
      f.write(xml_content)
      temp_path = Path(f.name)

    try:
      parser = StreamingXMLParser(temp_path)
      records = list(parser.parse_records())

      # Should still parse successfully even with missing attributes
      assert len(records) == 1
      # Check that it's a HeartRateRecord with the correct value
      from src.core.data_models import HeartRateRecord

      assert isinstance(records[0], HeartRateRecord)
      assert records[0].value == 75.0

      # Check that warnings were recorded
      stats = parser.get_statistics()
      assert (
        stats["warning_records"] == 1
      )  # Should have warnings for missing fields
      assert len(parser.stats["warnings"]) == 1
    finally:
      temp_path.unlink()

  def test_workout_with_missing_statistics(self):
    """Test parsing workout with missing statistics."""
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<HealthData>
    <Workout workoutActivityType="HKWorkoutActivityTypeRunning" duration="1800.0" sourceName="Apple Watch" startDate="2023-01-01 18:00:00 -0800" endDate="2023-01-01 19:30:00 -0800"/>
</HealthData>"""

    with tempfile.NamedTemporaryFile(
      mode="w", suffix=".xml", delete=False
    ) as f:
      f.write(xml_content)
      temp_path = Path(f.name)

    try:
      parser = StreamingXMLParser(temp_path)
      workouts = list(parser.parse_workouts())

      assert len(workouts) == 1
      workout = workouts[0]
      assert workout.activity_type == "Running"
      assert workout.calories is None  # Missing statistics
      assert workout.distance_km is None
    finally:
      temp_path.unlink()

  def test_activity_summary_with_missing_goals(self):
    """Test parsing activity summary with missing goals."""
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<HealthData>
    <ActivitySummary dateComponents="2023-01-01" activeEnergyBurned="500.0" appleExerciseTime="30.0" appleStandHours="8.0"/>
</HealthData>"""

    with tempfile.NamedTemporaryFile(
      mode="w", suffix=".xml", delete=False
    ) as f:
      f.write(xml_content)
      temp_path = Path(f.name)

    try:
      parser = StreamingXMLParser(temp_path)
      summaries = list(parser.parse_activity_summaries())

      assert len(summaries) == 1
      summary = summaries[0]
      assert summary.move_calories == 500.0
      assert summary.move_goal is None  # Missing goal
      assert (
        summary.move_achieved is False
      )  # Cannot determine achievement without goal
    finally:
      temp_path.unlink()

  def test_large_batch_size(self, temp_xml_file):
    """Test parsing with large batch size."""
    parser = StreamingXMLParser(temp_xml_file)

    # Use large batch size (larger than record count)
    records = list(parser.parse_records(batch_size=1000))

    assert len(records) == 3  # Should still parse all records

  def test_zero_batch_size(self, temp_xml_file):
    """Test parsing with zero batch size."""
    parser = StreamingXMLParser(temp_xml_file)

    # Should handle zero batch size gracefully
    records = list(parser.parse_records(batch_size=0))

    assert len(records) == 3
