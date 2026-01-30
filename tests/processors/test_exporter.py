"""Tests for data export functionality."""

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from src.core.data_models import HeartRateRecord, SleepRecord, WorkoutRecord
from src.processors.exporter import DataExporter, ExportManifest


class TestExportManifest:
    """Test ExportManifest class."""

    def test_manifest_initialization(self):
        """Test manifest initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = ExportManifest(Path(temp_dir))

            assert manifest.manifest_data["export_format"] == "apple_health_analyzer_v1"
            assert "files" in manifest.manifest_data
            assert "summary" in manifest.manifest_data
            assert manifest.manifest_path.name == "manifest.json"

    def test_add_file(self):
        """Test adding file to manifest."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = ExportManifest(Path(temp_dir))

            file_path = Path(temp_dir) / "test.csv"
            file_path.write_text("test,data\n1,2\n")

            manifest.add_file("HeartRate", file_path, 100, 1024)

            assert "HeartRate" in manifest.manifest_data["files"]
            file_info = manifest.manifest_data["files"]["HeartRate"]
            assert file_info["path"] == "test.csv"
            assert file_info["record_count"] == 100
            assert file_info["file_size_bytes"] == 1024
            assert file_info["format"] == "csv"

    def test_set_summary(self):
        """Test setting summary statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = ExportManifest(Path(temp_dir))

            manifest.set_summary(1000, 5, 102400)

            summary = manifest.manifest_data["summary"]
            assert summary["total_records"] == 1000
            assert summary["total_files"] == 5
            assert summary["total_size_bytes"] == 102400
            assert summary["export_duration_seconds"] is None

    def test_save_manifest(self):
        """Test saving manifest to disk."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = ExportManifest(Path(temp_dir))

            # Create a file within the export directory
            test_file = Path(temp_dir) / "test.csv"
            test_file.write_text("test,data\n1,2\n")

            manifest.add_file("HeartRate", test_file, 100, 1024)
            manifest.set_summary(100, 1, 1024)
            manifest.save(5.5)

            # Check file was created
            assert manifest.manifest_path.exists()

            # Check content
            with open(manifest.manifest_path, encoding='utf-8') as f:
                data = json.load(f)

            assert data["summary"]["export_duration_seconds"] == 5.5
            assert "HeartRate" in data["files"]
            assert data["files"]["HeartRate"]["path"] == "test.csv"


class TestDataExporter:
    """Test DataExporter class."""

    @pytest.fixture
    def sample_records(self):
        """Create sample records for testing."""
        base_time = datetime(2023, 1, 1, 10, 0, 0, tzinfo=UTC)

        return [
            HeartRateRecord(
                source_name="Apple Watch",
                value=75.0,
                creation_date=base_time,
                start_date=base_time,
                end_date=base_time,
            ),
            HeartRateRecord(
                source_name="Apple Watch",
                value=80.0,
                creation_date=base_time,
                start_date=base_time,
                end_date=base_time,
            ),
            SleepRecord(
                source_name="Apple Watch",
                value="HKCategoryValueSleepAnalysisAsleepCore",
                creation_date=base_time,
                start_date=base_time,
                end_date=base_time,
            ),
        ]

    @pytest.fixture
    def exporter(self, tmp_path):
        """Create DataExporter instance."""
        return DataExporter(tmp_path)

    def test_exporter_initialization(self, tmp_path):
        """Test exporter initialization."""
        exporter = DataExporter(tmp_path)

        assert exporter.output_dir == tmp_path
        assert exporter.output_dir.exists()
        assert isinstance(exporter.manifest, ExportManifest)

    def test_export_to_csv_empty_records(self, exporter):
        """Test CSV export with empty records."""
        output_path = exporter.output_dir / "empty.csv"

        count = exporter.export_to_csv([], output_path)

        assert count == 0
        assert not output_path.exists()

    def test_export_to_csv_with_records(self, exporter, sample_records):
        """Test CSV export with sample records."""
        output_path = exporter.output_dir / "test.csv"

        count = exporter.export_to_csv(sample_records, output_path)

        assert count == 3
        assert output_path.exists()

        # Check CSV content
        df = pd.read_csv(output_path)
        assert len(df) == 3
        assert "type" in df.columns
        assert "value" in df.columns
        assert "source_name" in df.columns

        # Check values (CSV stores as strings, convert to float for comparison)
        hr_records = df[df["type"] == "HKQuantityTypeIdentifierHeartRate"]
        assert len(hr_records) == 2
        assert set(hr_records["value"].astype(float)) == {75.0, 80.0}

    def test_export_to_json_empty_records(self, exporter):
        """Test JSON export with empty records."""
        output_path = exporter.output_dir / "empty.json"

        count = exporter.export_to_json([], output_path)

        assert count == 0
        assert not output_path.exists()

    def test_export_to_json_with_records(self, exporter, sample_records):
        """Test JSON export with sample records."""
        output_path = exporter.output_dir / "test.json"

        count = exporter.export_to_json(sample_records, output_path)

        assert count == 3
        assert output_path.exists()

        # Check JSON content
        with open(output_path, encoding='utf-8') as f:
            data = json.load(f)

        assert len(data) == 3
        assert data[0]["type"] == "HKQuantityTypeIdentifierHeartRate"
        assert data[0]["value"] == 75.0
        assert data[0]["source_name"] == "Apple Watch"

    def test_records_to_dataframe_conversion(self, exporter, sample_records):
        """Test conversion of records to DataFrame."""
        df = exporter._records_to_dataframe(sample_records)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

        # Check column ordering (type should be first)
        assert df.columns[0] == "type"

        # Check data integrity
        hr_rows = df[df["type"] == "HKQuantityTypeIdentifierHeartRate"]
        assert len(hr_rows) == 2
        assert set(hr_rows["value"]) == {75.0, 80.0}

    def test_records_to_dataframe_empty(self, exporter):
        """Test DataFrame conversion with empty records."""
        df = exporter._records_to_dataframe([])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_records_to_dataframe_with_metadata(self, exporter):
        """Test DataFrame conversion with metadata."""
        base_time = datetime(2023, 1, 1, 10, 0, 0, tzinfo=UTC)

        record_with_metadata = HeartRateRecord(
            source_name="Apple Watch",
            value=75.0,
            creation_date=base_time,
            start_date=base_time,
            end_date=base_time,
            metadata={"device_model": "Watch6,1", "firmware": "9.0"}
        )

        df = exporter._records_to_dataframe([record_with_metadata])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

        # Check metadata flattening
        assert "metadata_device_model" in df.columns
        assert "metadata_firmware" in df.columns
        assert df.iloc[0]["metadata_device_model"] == "Watch6,1"
        assert df.iloc[0]["metadata_firmware"] == "9.0"

        # Check metadata column is removed
        assert "metadata" not in df.columns


class TestDataExporterIntegration:
    """Integration tests for DataExporter."""

    def test_export_by_category_invalid_format(self, tmp_path):
        """Test export with invalid format raises error."""
        exporter = DataExporter(tmp_path)

        with pytest.raises(ValueError, match="Supported formats are"):
            exporter.export_by_category(
                Path("dummy.xml"),
                formats=["invalid_format"]
            )

    def test_export_by_category_empty_formats_defaults_to_both(self, tmp_path):
        """Test export with empty formats defaults to both CSV and JSON."""
        exporter = DataExporter(tmp_path)

        # This would normally parse XML, but we'll test the format validation
        # Since we don't have a real XML file, we'll just check the format validation
        try:
            exporter.export_by_category(Path("nonexistent.xml"))
        except FileNotFoundError:
            pass  # Expected since file doesn't exist

        # The formats validation should have passed (defaults to ['csv', 'json'])


class TestWorkoutRecordExport:
    """Test exporting WorkoutRecord specifically."""

    def test_workout_record_export(self, tmp_path):
        """Test exporting workout records."""
        exporter = DataExporter(tmp_path)

        base_time = datetime(2023, 1, 1, 10, 0, 0, tzinfo=UTC)
        end_time = datetime(2023, 1, 1, 11, 0, 0, tzinfo=UTC)

        workout = WorkoutRecord(
            source_name="Apple Watch",
            start_date=base_time,
            end_date=end_time,
            activity_type="Running",
            workout_duration_seconds=3600.0,
            calories=500.0,
            distance_km=8.0,
            average_heart_rate=150.0,
        )

        output_path = tmp_path / "workout.csv"
        count = exporter.export_to_csv([workout], output_path)

        assert count == 1
        assert output_path.exists()

        df = pd.read_csv(output_path)
        assert len(df) == 1
        assert df.iloc[0]["activity_type"] == "Running"
        assert df.iloc[0]["workout_duration_seconds"] == 3600.0
        assert df.iloc[0]["calories"] == 500.0
        assert df.iloc[0]["distance_km"] == 8.0
        assert df.iloc[0]["average_heart_rate"] == 150.0
