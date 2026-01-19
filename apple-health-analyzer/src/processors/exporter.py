"""Data export functionality for Apple Health records.

Provides functionality to export parsed health records to CSV and JSON formats,
with support for categorized exports and manifest generation.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..core.data_models import AnyRecord
from ..core.xml_parser import StreamingXMLParser
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ExportManifest:
    """Export manifest containing metadata about exported files."""

    def __init__(self, export_dir: Path):
        self.export_dir = export_dir
        self.manifest_path = export_dir / "manifest.json"
        self.manifest_data: dict[str, Any] = {
            "export_timestamp": pd.Timestamp.now().isoformat(),
            "export_format": "apple_health_analyzer_v1",
            "files": {},
            "summary": {}
        }

    def add_file(self, record_type: str, file_path: Path, record_count: int, file_size: int):
        """Add a file entry to the manifest."""
        self.manifest_data["files"][record_type] = {
            "path": str(file_path.relative_to(self.export_dir)),
            "record_count": record_count,
            "file_size_bytes": file_size,
            "format": file_path.suffix[1:]  # Remove the dot
        }

    def set_summary(self, total_records: int, total_files: int, total_size: int):
        """Set summary statistics."""
        self.manifest_data["summary"] = {
            "total_records": total_records,
            "total_files": total_files,
            "total_size_bytes": total_size,
            "export_duration_seconds": None  # Will be set when saving
        }

    def save(self, export_duration: float | None = None):
        """Save the manifest to disk."""
        if export_duration is not None:
            self.manifest_data["summary"]["export_duration_seconds"] = export_duration

        with open(self.manifest_path, 'w', encoding='utf-8') as f:
            json.dump(self.manifest_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Export manifest saved to {self.manifest_path}")


class DataExporter:
    """Exports Apple Health records to CSV and JSON formats."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = ExportManifest(self.output_dir)

    def export_to_csv(self, records: list[AnyRecord], output_path: Path) -> int:
        """Export records to CSV format.

        Args:
            records: List of health records to export
            output_path: Path to save the CSV file

        Returns:
            Number of records exported
        """
        if not records:
            logger.warning(f"No records to export to {output_path}")
            return 0

        # Convert records to DataFrame
        df = self._records_to_dataframe(records)

        # Export to CSV with proper encoding and formatting
        df.to_csv(
            output_path,
            index=False,
            encoding='utf-8-sig',  # BOM for Excel compatibility
            date_format='%Y-%m-%d %H:%M:%S%z',
            float_format='%.6f'
        )

        record_count = len(records)

        logger.info(f"Exported {record_count} records to CSV: {output_path}")
        return record_count

    def export_to_json(self, records: list[AnyRecord], output_path: Path) -> int:
        """Export records to JSON format.

        Args:
            records: List of health records to export
            output_path: Path to save the JSON file

        Returns:
            Number of records exported
        """
        if not records:
            logger.warning(f"No records to export to {output_path}")
            return 0

        # Convert records to dictionaries
        records_data = [record.model_dump() for record in records]

        # Export to JSON with proper formatting
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(records_data, f, indent=2, ensure_ascii=False, default=str)

        record_count = len(records)

        logger.info(f"Exported {record_count} records to JSON: {output_path}")
        return record_count

    def export_by_category(
        self,
        xml_path: Path,
        formats: list[str] | None = None,
        record_types: list[str] | None = None
    ) -> dict[str, dict[str, int]]:
        """Export records by category from XML file.

        Args:
            xml_path: Path to the Apple Health export XML file
            formats: List of export formats ('csv', 'json'). Defaults to both.
            record_types: List of record types to export. If None, exports all.

        Returns:
            Dictionary with export statistics by record type and format
        """
        if formats is None:
            formats = ['csv', 'json']

        formats = [fmt.lower() for fmt in formats]
        if not all(fmt in ['csv', 'json'] for fmt in formats):
            raise ValueError("Supported formats are 'csv' and 'json'")

        logger.info(f"Starting categorized export from {xml_path}")
        logger.info(f"Export formats: {formats}")
        if record_types:
            logger.info(f"Record types to export: {record_types}")

        # Parse XML and group records by type
        parser = StreamingXMLParser(xml_path)
        records_by_type = {}

        total_records = 0
        start_time = pd.Timestamp.now()

        for record in parser.parse_records(record_types=record_types):
            # Use unified record_type property
            record_type = record.record_type

            if record_types and record_type not in record_types:
                continue

            if record_type not in records_by_type:
                records_by_type[record_type] = []

            records_by_type[record_type].append(record)
            total_records += 1

            # Log progress every 10,000 records
            if total_records % 10000 == 0:
                logger.info(f"Processed {total_records} records...")

        logger.info(f"Parsed {total_records} records from {len(records_by_type)} record types")

        # Export each record type
        export_stats = {}
        total_files = 0
        total_size = 0

        for record_type, records in records_by_type.items():
            if not records:
                continue

            export_stats[record_type] = {}

            # Clean record type name for filename
            clean_type = record_type.replace('HKQuantityTypeIdentifier', '').replace('HKCategoryTypeIdentifier', '')

            for fmt in formats:
                filename = f"{clean_type}.{fmt}"
                output_path = self.output_dir / filename

                count = 0
                if fmt == 'csv':
                    count = self.export_to_csv(records, output_path)
                elif fmt == 'json':
                    count = self.export_to_json(records, output_path)

                if count > 0:
                    file_size = output_path.stat().st_size
                    self.manifest.add_file(record_type, output_path, count, file_size)
                    export_stats[record_type][fmt] = count
                    total_files += 1
                    total_size += file_size

        # Save manifest
        end_time = pd.Timestamp.now()
        export_duration = (end_time - start_time).total_seconds()

        self.manifest.set_summary(total_records, total_files, total_size)
        self.manifest.save(export_duration)

        logger.info(f"Export completed in {export_duration:.2f} seconds")
        logger.info(f"Total files: {total_files}, Total size: {total_size:,} bytes")

        return export_stats

    def _records_to_dataframe(self, records: list[AnyRecord]) -> pd.DataFrame:
        """Convert records to pandas DataFrame.

        Args:
            records: List of health records

        Returns:
            DataFrame with record data
        """
        if not records:
            return pd.DataFrame()

        # Convert records to dictionaries
        records_data = []
        for record in records:
            record_dict = record.model_dump()

            # Flatten metadata if present
            if 'metadata' in record_dict and record_dict['metadata']:
                for key, value in record_dict['metadata'].items():
                    record_dict[f"metadata_{key}"] = value
                del record_dict['metadata']

            records_data.append(record_dict)

        # Create DataFrame
        df = pd.DataFrame(records_data)

        # Ensure consistent column ordering
        priority_columns = [
            'type', 'source_name', 'source_version', 'device',
            'creation_date', 'start_date', 'end_date'
        ]

        other_columns = [col for col in df.columns if col not in priority_columns]
        column_order = priority_columns + sorted(other_columns)

        # Only include columns that exist
        final_columns = [col for col in column_order if col in df.columns]
        df = df[final_columns]

        # Ensure we return a DataFrame (type hint for pyright)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
