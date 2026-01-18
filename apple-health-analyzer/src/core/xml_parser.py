"""Streaming XML parser for Apple Health export data.

Provides memory-efficient parsing of large XML files using iterative parsing.
"""

import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Generator
from pathlib import Path
from typing import Any

from src.config import Config, get_config
from src.core.data_models import AnyRecord, create_record_from_xml_element
from src.utils.logger import ProgressLogger, get_logger, performance_logger

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
            'total_records': 0,
            'processed_records': 0,
            'skipped_records': 0,
            'invalid_records': 0,
            'record_types': defaultdict(int),
            'sources': defaultdict(int),
            'date_range': {'start': None, 'end': None},
        }

    def parse_records(
        self,
        record_types: list[str] | None = None,
        batch_size: int = 1000
    ) -> Generator[AnyRecord, None, None]:
        """Parse Record elements from the XML file.

        Args:
            record_types: List of record types to parse (None for all)
            batch_size: Number of records to process before logging progress

        Yields:
            Parsed health record objects
        """
        self.logger.info(f"Starting to parse records from {self.xml_path}")
        self.logger.info(f"Record types filter: {record_types or 'all'}")

        # Reset statistics
        self._reset_stats()

        try:
            # Use iterative parsing to avoid loading the entire file
            context = ET.iterparse(self.xml_path, events=('start', 'end'))
            context = iter(context)

            # Get root element
            event, root = next(context)

            processed_in_batch = 0

            with ProgressLogger("XML record parsing", log_interval=batch_size) as progress:
                for event, elem in context:
                    if event == 'start' and elem.tag == 'Record':
                        self.stats['total_records'] += 1

                        # Check if we should process this record type
                        record_type = elem.get('type')
                        if record_types and record_type not in record_types:
                            self.stats['skipped_records'] += 1
                            # For start events, we need to clear when we reach the end
                            continue

                        # Parse the record
                        record = self._parse_record_element(elem)
                        if record:
                            # Update statistics
                            self._update_record_stats(record)

                            # Yield the record
                            yield record
                            self.stats['processed_records'] += 1
                            processed_in_batch += 1

                            # Log progress
                            progress.update()

                            # Periodic memory cleanup
                            if processed_in_batch >= batch_size:
                                processed_in_batch = 0
                                # Clear processed elements to free memory
                                root.clear()
                        else:
                            self.stats['invalid_records'] += 1

                    # Clear elements at end events to free memory
                    if event == 'end':
                        elem.clear()

                # Final memory cleanup
                root.clear()

        except Exception as e:
            self.logger.error(f"Error during XML parsing: {e}")
            raise

        self.logger.info(f"Completed parsing {self.stats['processed_records']} records")
        self._log_parsing_summary()

    def parse_workouts(self) -> Generator[dict[str, Any], None, None]:
        """Parse Workout elements from the XML file.

        Yields:
            Dictionary containing workout data
        """
        self.logger.info("Starting to parse workouts")

        try:
            context = ET.iterparse(self.xml_path, events=('start', 'end'))
            context = iter(context)
            event, root = next(context)

            workout_count = 0
            with ProgressLogger("XML workout parsing") as progress:
                for event, elem in context:
                    if event == 'end' and elem.tag == 'Workout':
                        workout_data = self._parse_workout_element(elem)
                        if workout_data:
                            yield workout_data
                            workout_count += 1
                            progress.update()

                        elem.clear()

                root.clear()

            self.logger.info(f"Completed parsing {workout_count} workouts")

        except Exception as e:
            self.logger.error(f"Error parsing workouts: {e}")
            raise

    def parse_activity_summaries(self) -> Generator[dict[str, Any], None, None]:
        """Parse ActivitySummary elements from the XML file.

        Yields:
            Dictionary containing activity summary data
        """
        self.logger.info("Starting to parse activity summaries")

        try:
            context = ET.iterparse(self.xml_path, events=('start', 'end'))
            context = iter(context)
            event, root = next(context)

            summary_count = 0
            with ProgressLogger("XML activity summary parsing") as progress:
                for event, elem in context:
                    if event == 'end' and elem.tag == 'ActivitySummary':
                        summary_data = self._parse_activity_summary_element(elem)
                        if summary_data:
                            yield summary_data
                            summary_count += 1
                            progress.update()

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
            'total_records': self.stats['total_records'],
            'processed_records': self.stats['processed_records'],
            'skipped_records': self.stats['skipped_records'],
            'invalid_records': self.stats['invalid_records'],
            'record_types': dict(self.stats['record_types']),
            'sources': dict(self.stats['sources']),
            'date_range': self.stats['date_range'],
            'success_rate': (
                self.stats['processed_records'] / self.stats['total_records']
                if self.stats['total_records'] > 0 else 0
            ),
        }

    def _reset_stats(self) -> None:
        """Reset parsing statistics."""
        self.stats = {
            'total_records': 0,
            'processed_records': 0,
            'skipped_records': 0,
            'invalid_records': 0,
            'record_types': defaultdict(int),
            'sources': defaultdict(int),
            'date_range': {'start': None, 'end': None},
        }

    def _parse_record_element(self, elem: ET.Element) -> AnyRecord | None:
        """Parse a single Record XML element.

        Args:
            elem: XML element to parse

        Returns:
            Parsed record object or None if parsing fails
        """
        try:
            return create_record_from_xml_element(elem)
        except Exception as e:
            self.logger.debug(f"Failed to parse record element: {e}")
            return None

    def _parse_workout_element(self, elem: ET.Element) -> dict[str, Any] | None:
        """Parse a single Workout XML element.

        Args:
            elem: Workout XML element

        Returns:
            Dictionary with workout data or None if parsing fails
        """
        try:
            from datetime import datetime

            workout_data: dict[str, Any] = {
                'activity_type': elem.get('workoutActivityType', '').replace('HKWorkoutActivityType', ''),
                'duration_seconds': float(elem.get('duration', 0)),
                'source_name': elem.get('sourceName', ''),
                'start_date': datetime.strptime(elem.get('startDate', ''), '%Y-%m-%d %H:%M:%S %z'),
                'end_date': datetime.strptime(elem.get('endDate', ''), '%Y-%m-%d %H:%M:%S %z'),
                'metadata': {},
            }

            # Parse workout statistics
            for stat in elem.findall('.//WorkoutStatistics'):
                stat_type = stat.get('type', '')
                value = stat.get('sum')
                unit = stat.get('unit', '')

                if value:
                    if 'ActiveEnergyBurned' in stat_type:
                        workout_data['calories'] = float(value)
                    elif 'DistanceWalkingRunning' in stat_type:
                        if unit == 'km':
                            workout_data['distance_km'] = float(value)
                        elif unit == 'm':
                            workout_data['distance_km'] = float(value) / 1000

            # Parse metadata
            metadata = workout_data['metadata']
            if isinstance(metadata, dict):
                for meta in elem.findall('.//MetadataEntry'):
                    key = meta.get('key')
                    value = meta.get('value')
                    if key and value:
                        metadata[key] = value

            return workout_data

        except Exception as e:
            self.logger.debug(f"Failed to parse workout element: {e}")
            return None

    def _parse_activity_summary_element(self, elem: ET.Element) -> dict[str, Any] | None:
        """Parse a single ActivitySummary XML element.

        Args:
            elem: ActivitySummary XML element

        Returns:
            Dictionary with activity summary data or None if parsing fails
        """
        try:
            from datetime import datetime

            summary_data: dict[str, Any] = {
                'date': datetime.strptime(elem.get('dateComponents', ''), '%Y-%m-%d'),
                'move_calories': float(elem.get('activeEnergyBurned', 0)) if elem.get('activeEnergyBurned') else None,
                'exercise_minutes': float(elem.get('appleExerciseTime', 0)) if elem.get('appleExerciseTime') else None,
                'stand_hours': float(elem.get('appleStandHours', 0)) if elem.get('appleStandHours') else None,
                'move_goal': float(elem.get('activeEnergyBurnedGoal', 0)) if elem.get('activeEnergyBurnedGoal') else None,
                'exercise_goal': float(elem.get('appleExerciseTimeGoal', 0)) if elem.get('appleExerciseTimeGoal') else None,
                'stand_goal': float(elem.get('appleStandHoursGoal', 0)) if elem.get('appleStandHoursGoal') else None,
            }

            # Determine achievements
            move_calories = summary_data['move_calories']
            move_goal = summary_data['move_goal']
            summary_data['move_achieved'] = (
                move_calories is not None and
                move_goal is not None and
                isinstance(move_calories, (int, float)) and
                isinstance(move_goal, (int, float)) and
                move_calories >= move_goal
            )

            exercise_minutes = summary_data['exercise_minutes']
            exercise_goal = summary_data['exercise_goal']
            summary_data['exercise_achieved'] = (
                exercise_minutes is not None and
                exercise_goal is not None and
                isinstance(exercise_minutes, (int, float)) and
                isinstance(exercise_goal, (int, float)) and
                exercise_minutes >= exercise_goal
            )

            stand_hours = summary_data['stand_hours']
            stand_goal = summary_data['stand_goal']
            summary_data['stand_achieved'] = (
                stand_hours is not None and
                stand_goal is not None and
                isinstance(stand_hours, (int, float)) and
                isinstance(stand_goal, (int, float)) and
                stand_hours >= stand_goal
            )

            return summary_data

        except Exception as e:
            self.logger.debug(f"Failed to parse activity summary element: {e}")
            return None

    def _update_record_stats(self, record: AnyRecord) -> None:
        """Update parsing statistics with a new record.

        Args:
            record: Parsed record to include in statistics
        """
        # Update type counts
        if hasattr(record, 'type') and isinstance(getattr(record, 'type', None), str):
            record_type = getattr(record, 'type')
            self.stats['record_types'][record_type] += 1

        # Update source counts
        if hasattr(record, 'source_name') and isinstance(getattr(record, 'source_name', None), str):
            source_name = getattr(record, 'source_name')
            self.stats['sources'][source_name] += 1

        # Update date range
        if hasattr(record, 'start_date'):
            start_date = getattr(record, 'start_date', None)
            if start_date is not None and hasattr(start_date, 'date'):
                record_date = start_date.date()
                if self.stats['date_range']['start'] is None or record_date < self.stats['date_range']['start']:
                    self.stats['date_range']['start'] = record_date
                if self.stats['date_range']['end'] is None or record_date > self.stats['date_range']['end']:
                    self.stats['date_range']['end'] = record_date

    def _log_parsing_summary(self) -> None:
        """Log a summary of the parsing results."""
        stats = self.get_statistics()

        self.logger.info("=== XML Parsing Summary ===")
        self.logger.info(f"Total records in file: {stats['total_records']:,}")
        self.logger.info(f"Successfully processed: {stats['processed_records']:,}")
        self.logger.info(f"Skipped (filtered): {stats['skipped_records']:,}")
        self.logger.info(f"Invalid/malformed: {stats['invalid_records']:,}")
        self.logger.info(f"Success rate: {stats['success_rate']:.1%}")

        if stats['date_range']['start'] and stats['date_range']['end']:
            self.logger.info(f"Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")

        # Log top record types
        if stats['record_types']:
            self.logger.info("Top record types:")
            sorted_types = sorted(stats['record_types'].items(), key=lambda x: x[1], reverse=True)
            for record_type, count in sorted_types[:5]:
                self.logger.info(f"  {record_type}: {count:,}")

        # Log top sources
        if stats['sources']:
            self.logger.info("Top data sources:")
            sorted_sources = sorted(stats['sources'].items(), key=lambda x: x[1], reverse=True)
            for source, count in sorted_sources[:5]:
                self.logger.info(f"  {source}: {count:,}")

@performance_logger
def parse_export_file(
    xml_path: Path,
    record_types: list[str] | None = None,
    config: Config | None = None
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
        with open(xml_path, encoding='utf-8') as f:
            for line in f:
                if '<Record ' in line:
                    record_count += 1

        return {
            'file_path': str(xml_path),
            'file_size_bytes': file_size,
            'file_size_mb': round(file_size_mb, 2),
            'estimated_record_count': record_count,
            'last_modified': xml_path.stat().st_mtime,
        }

    except Exception as e:
        logger.error(f"Error getting file info: {e}")
        return {}
