"""Streaming XML parser for Apple Health export data.

Provides memory-efficient parsing of large XML files using iterative parsing.
"""

import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

from src.config import Config, get_config
from src.core.data_models import (
  ActivitySummaryRecord,
  AnyRecord,
  WorkoutRecord,
  create_record_from_xml_element,
)
from src.utils.logger import get_logger, performance_logger

logger = get_logger(__name__)


class StreamingXMLParser:
  """Memory-efficient XML parser for Apple Health export files.

  Uses iterative parsing to handle large files without loading everything into memory.
  """

  def __init__(self, xml_path: Path, config: Config | None = None):
    """Initialize the parser.

    Args:
        xml_path: Path to the export.xml file
        config: Configuration instance (uses global config if None)
    """
    self.xml_path = xml_path
    self.config = config or get_config()
    self.logger = get_logger(__name__)

    # Statistics tracking
    self.stats: dict[str, Any] = {
      "total_records": 0,
      "processed_records": 0,
      "skipped_records": 0,
      "invalid_records": 0,
      "warning_records": 0,  # Records parsed with warnings/defaults
      "warnings": [],  # Warning details
      "record_types": defaultdict(int),
      "sources": defaultdict(int),
      "date_range": {"start": None, "end": None},
    }

  def parse_records(
    self,
    record_types: list[str] | None = None,
    batch_size: int = 1000,
    progress_callback: Callable[[int], None] | None = None,
    quiet: bool = False,
  ) -> Generator[AnyRecord, None, None]:
    """Parse Record elements from the XML file.

    Args:
        record_types: List of record types to parse (None for all)
        batch_size: Number of records to process before logging progress

    Yields:
        Parsed health record objects
    """
    if not quiet:
      self.logger.info(f"Starting to parse records from {self.xml_path}")
      self.logger.info(f"Record types filter: {record_types or 'all'}")

    # Reset statistics
    self._reset_stats()

    try:
      # Use iterative parsing to avoid loading the entire file
      context = ET.iterparse(self.xml_path, events=("start", "end"))
      context = iter(context)

      # Get root element
      event, root = next(context)

      processed_in_batch = 0

      for event, elem in context:
        if event == "start" and elem.tag == "Record":
          self.stats["total_records"] += 1

          # Check if we should process this record type
          record_type = elem.get("type")
          if record_types and record_type not in record_types:
            self.stats["skipped_records"] += 1
            # For start events, we need to clear when we reach the end
            continue

          # Parse the record
          record = self._parse_record_element(elem)
          if record:
            # Update statistics
            self._update_record_stats(record)

            # Yield the record
            yield record
            self.stats["processed_records"] += 1
            processed_in_batch += 1

            # Call progress callback if provided
            if progress_callback:
              progress_callback(self.stats["processed_records"])

            # Periodic memory cleanup
            if processed_in_batch >= batch_size:
              processed_in_batch = 0
              # Clear processed elements to free memory
              root.clear()
          else:
            self.stats["invalid_records"] += 1

        # Clear elements at end events to free memory
        if event == "end":
          elem.clear()

      # Final memory cleanup
      root.clear()

    except Exception as e:
      self.logger.error(f"Error during XML parsing: {e}")
      raise

    self.logger.info(
      f"Completed parsing {self.stats['processed_records']} records"
    )
    self._log_parsing_summary()

  def parse_workouts(
    self, progress_callback: Callable[[int], None] | None = None
  ) -> Generator[WorkoutRecord, None, None]:
    """Parse Workout elements from the XML file.

    Yields:
        WorkoutRecord instances
    """
    self.logger.info("Starting to parse workouts")

    try:
      context = ET.iterparse(self.xml_path, events=("start", "end"))
      context = iter(context)
      event, root = next(context)

      workout_count = 0
      for event, elem in context:
        if event == "end" and elem.tag == "Workout":
          workout = self._parse_workout_element(elem)
          if workout:
            yield workout
            workout_count += 1
            if progress_callback:
              progress_callback(workout_count)

          elem.clear()

      root.clear()

      self.logger.info(f"Completed parsing {workout_count} workouts")

    except Exception as e:
      self.logger.error(f"Error parsing workouts: {e}")
      raise

  def parse_activity_summaries(
    self, progress_callback: Callable[[int], None] | None = None
  ) -> Generator[ActivitySummaryRecord, None, None]:
    """Parse ActivitySummary elements from the XML file.

    Yields:
        ActivitySummaryRecord instances
    """
    self.logger.info("Starting to parse activity summaries")

    try:
      context = ET.iterparse(self.xml_path, events=("start", "end"))
      context = iter(context)
      event, root = next(context)

      summary_count = 0
      for event, elem in context:
        if event == "end" and elem.tag == "ActivitySummary":
          summary = self._parse_activity_summary_element(elem)
          if summary:
            yield summary
            summary_count += 1
            if progress_callback:
              progress_callback(summary_count)

          elem.clear()

      root.clear()

      self.logger.info(f"Completed parsing {summary_count} activity summaries")

    except Exception as e:
      self.logger.error(f"Error parsing activity summaries: {e}")
      raise

  def get_statistics(self) -> dict[str, Any]:
    """Get parsing statistics.

    Returns:
        Dictionary containing parsing statistics
    """
    return {
      "total_records": self.stats["total_records"],
      "processed_records": self.stats["processed_records"],
      "skipped_records": self.stats["skipped_records"],
      "invalid_records": self.stats["invalid_records"],
      "warning_records": self.stats["warning_records"],
      "warnings": self.stats["warnings"],
      "record_types": dict(self.stats["record_types"]),
      "sources": dict(self.stats["sources"]),
      "date_range": self.stats["date_range"],
      "success_rate": (
        self.stats["processed_records"] / self.stats["total_records"]
        if self.stats["total_records"] > 0
        else 0
      ),
    }

  def _reset_stats(self) -> None:
    """Reset parsing statistics."""
    self.stats = {
      "total_records": 0,
      "processed_records": 0,
      "skipped_records": 0,
      "invalid_records": 0,
      "warning_records": 0,  # Records parsed with warnings/defaults
      "warnings": [],  # Warning details
      "record_types": defaultdict(int),
      "sources": defaultdict(int),
      "date_range": {"start": None, "end": None},
    }

  def _parse_record_element(self, elem: ET.Element) -> AnyRecord | None:
    """Parse a single Record XML element.

    Args:
        elem: XML element to parse

    Returns:
        Parsed record object or None if parsing fails
    """
    try:
      record, warnings = create_record_from_xml_element(elem)

      # Handle warnings
      if warnings:
        self.stats["warning_records"] += 1
        # Record warning details
        self.stats["warnings"].append(
          {
            "record_type": elem.get("type", "Unknown"),
            "warnings": warnings,
            "source": elem.get("sourceName", "Unknown"),
          }
        )
        # Log warnings
        # self.logger.warning(
        #   f"Record parsed with warnings: {elem.get('type', 'Unknown')} - {', '.join(warnings)}"
        # )

      return record
    except Exception as e:
      self.logger.debug(f"Failed to parse record element: {e}")
      return None

  def _parse_workout_element(self, elem: ET.Element) -> WorkoutRecord | None:
    """Parse a single Workout XML element.

    Args:
        elem: Workout XML element

    Returns:
        WorkoutRecord instance or None if parsing fails
    """
    try:
      from datetime import datetime

      # Parse basic workout data
      activity_type = elem.get("workoutActivityType", "").replace(
        "HKWorkoutActivityType", ""
      )
      workout_duration_seconds = float(elem.get("duration", 0))
      source_name = elem.get("sourceName", "")
      start_date = datetime.strptime(
        elem.get("startDate", ""), "%Y-%m-%d %H:%M:%S %z"
      )
      end_date = datetime.strptime(
        elem.get("endDate", ""), "%Y-%m-%d %H:%M:%S %z"
      )

      # Parse workout statistics
      calories = None
      distance_km = None
      average_heart_rate = None

      for stat in elem.findall(".//WorkoutStatistics"):
        stat_type = stat.get("type", "")
        value = stat.get("sum")
        unit = stat.get("unit", "")

        if value:
          if "ActiveEnergyBurned" in stat_type:
            calories = float(value)
          elif "DistanceWalkingRunning" in stat_type:
            if unit == "km":
              distance_km = float(value)
            elif unit == "m":
              distance_km = float(value) / 1000
          elif "HeartRate" in stat_type and "Average" in stat_type:
            average_heart_rate = float(value)

      # Parse metadata
      metadata = {}
      for meta in elem.findall(".//MetadataEntry"):
        key = meta.get("key")
        value = meta.get("value")
        if key and value:
          metadata[key] = value

      # Create WorkoutRecord instance
      return WorkoutRecord(
        source_name=source_name,
        start_date=start_date,
        end_date=end_date,
        activity_type=activity_type,
        workout_duration_seconds=workout_duration_seconds,
        calories=calories,
        distance_km=distance_km,
        average_heart_rate=average_heart_rate,
        metadata=metadata if metadata else None,
      )

    except Exception as e:
      self.logger.debug(f"Failed to parse workout element: {e}")
      return None

  def _parse_activity_summary_element(
    self, elem: ET.Element
  ) -> ActivitySummaryRecord | None:
    """Parse a single ActivitySummary XML element.

    Args:
        elem: ActivitySummary XML element

    Returns:
        ActivitySummaryRecord instance or None if parsing fails
    """
    try:
      from datetime import datetime

      # Parse basic data
      date = datetime.strptime(elem.get("dateComponents", ""), "%Y-%m-%d")
      source_name = elem.get(
        "sourceName", "Apple Health"
      )  # Activity summaries typically from Apple Health

      # Parse activity rings
      move_calories = (
        float(elem.get("activeEnergyBurned", 0))
        if elem.get("activeEnergyBurned")
        else None
      )
      exercise_minutes = (
        float(elem.get("appleExerciseTime", 0))
        if elem.get("appleExerciseTime")
        else None
      )
      stand_hours = (
        float(elem.get("appleStandHours", 0))
        if elem.get("appleStandHours")
        else None
      )

      # Parse goals
      move_goal = (
        float(elem.get("activeEnergyBurnedGoal", 0))
        if elem.get("activeEnergyBurnedGoal")
        else None
      )
      exercise_goal = (
        float(elem.get("appleExerciseTimeGoal", 0))
        if elem.get("appleExerciseTimeGoal")
        else None
      )
      stand_goal = (
        float(elem.get("appleStandHoursGoal", 0))
        if elem.get("appleStandHoursGoal")
        else None
      )

      # Determine achievements
      move_achieved = (
        move_calories is not None
        and move_goal is not None
        and isinstance(move_calories, (int, float))
        and isinstance(move_goal, (int, float))
        and move_calories >= move_goal
      )

      exercise_achieved = (
        exercise_minutes is not None
        and exercise_goal is not None
        and isinstance(exercise_minutes, (int, float))
        and isinstance(exercise_goal, (int, float))
        and exercise_minutes >= exercise_goal
      )

      stand_achieved = (
        stand_hours is not None
        and stand_goal is not None
        and isinstance(stand_hours, (int, float))
        and isinstance(stand_goal, (int, float))
        and stand_hours >= stand_goal
      )

      # Create ActivitySummaryRecord instance
      return ActivitySummaryRecord(
        source_name=source_name,
        date=date,
        move_calories=move_calories,
        exercise_minutes=exercise_minutes,
        stand_hours=stand_hours,
        move_goal=move_goal,
        exercise_goal=exercise_goal,
        stand_goal=stand_goal,
        move_achieved=move_achieved,
        exercise_achieved=exercise_achieved,
        stand_achieved=stand_achieved,
      )

    except Exception as e:
      self.logger.debug(f"Failed to parse activity summary element: {e}")
      return None

  def _update_record_stats(self, record: AnyRecord) -> None:
    """Update parsing statistics with a new record.

    Args:
        record: Parsed record to include in statistics
    """
    # Update type counts using unified record_type property
    record_type = record.record_type
    self.stats["record_types"][record_type] += 1

    # Update source counts
    if hasattr(record, "source_name") and isinstance(
      getattr(record, "source_name", None), str
    ):
      source_name = record.source_name
      self.stats["sources"][source_name] += 1

    # Update date range
    if hasattr(record, "start_date"):
      start_date = getattr(record, "start_date", None)
      if start_date is not None and hasattr(start_date, "date"):
        record_date = start_date.date()
        if (
          self.stats["date_range"]["start"] is None
          or record_date < self.stats["date_range"]["start"]
        ):
          self.stats["date_range"]["start"] = record_date
        if (
          self.stats["date_range"]["end"] is None
          or record_date > self.stats["date_range"]["end"]
        ):
          self.stats["date_range"]["end"] = record_date

  def _log_parsing_summary(self) -> None:
    """Log a summary of the parsing results."""
    stats = self.get_statistics()

    self.logger.info("=== XML Parsing Summary ===")
    self.logger.info(f"Total records in file: {stats['total_records']:,}")
    self.logger.info(f"Successfully processed: {stats['processed_records']:,}")
    self.logger.info(f"Skipped (filtered): {stats['skipped_records']:,}")
    self.logger.info(f"Invalid/malformed: {stats['invalid_records']:,}")
    self.logger.info(f"Success rate: {stats['success_rate']:.1%}")

    # Log warnings if any
    if self.stats["warning_records"] > 0:
      self.logger.warning(
        f"⚠️  {self.stats['warning_records']:,} records parsed with warnings"
      )
      self.logger.warning("   (Records used default values for missing fields)")

      # Show first few warnings
      for _i, warning_info in enumerate(self.stats["warnings"][:3]):
        self.logger.warning(
          f"   - {warning_info['record_type']}: {', '.join(warning_info['warnings'])}"
        )

      if len(self.stats["warnings"]) > 3:
        self.logger.warning(
          f"   ... and {len(self.stats['warnings']) - 3} more warnings"
        )

    if stats["date_range"]["start"] and stats["date_range"]["end"]:
      self.logger.info(
        f"Date range: {stats['date_range']['start']} to {stats['date_range']['end']}"
      )

    # Log top record types
    if stats["record_types"]:
      self.logger.info("Top record types:")
      sorted_types = sorted(
        stats["record_types"].items(), key=lambda x: x[1], reverse=True
      )
      for record_type, count in sorted_types[:5]:
        self.logger.info(f"  {record_type}: {count:,}")

    # Log top sources
    if stats["sources"]:
      self.logger.info("Top data sources:")
      sorted_sources = sorted(
        stats["sources"].items(), key=lambda x: x[1], reverse=True
      )
      for source, count in sorted_sources[:5]:
        self.logger.info(f"  {source}: {count:,}")


@performance_logger
def parse_export_file(
  xml_path: Path,
  record_types: list[str] | None = None,
  config: Config | None = None,
) -> tuple[list[AnyRecord], dict[str, Any]]:
  """Convenience function to parse an entire export file.

  Args:
      xml_path: Path to the export.xml file
      record_types: List of record types to parse (None for all)
      config: Configuration instance

  Returns:
      Tuple of (records_list, statistics_dict)
  """
  config = config or get_config()
  parser = StreamingXMLParser(xml_path, config)

  records = list(parser.parse_records(record_types, config.batch_size))
  stats = parser.get_statistics()

  return records, stats


def get_export_file_info(xml_path: Path) -> dict[str, Any]:
  """Get basic information about an export file without full parsing.

  Args:
      xml_path: Path to the export.xml file

  Returns:
      Dictionary with file information
  """
  try:
    file_size = xml_path.stat().st_size
    file_size_mb = file_size / (1024 * 1024)

    # Quick scan for record count (approximate)
    record_count = 0
    with open(xml_path, encoding="utf-8") as f:
      for line in f:
        if "<Record " in line:
          record_count += 1

    return {
      "file_path": str(xml_path),
      "file_size_bytes": file_size,
      "file_size_mb": round(file_size_mb, 2),
      "estimated_record_count": record_count,
      "last_modified": xml_path.stat().st_mtime,
    }

  except Exception as e:
    logger.error(f"Error getting file info: {e}")
    return {}
