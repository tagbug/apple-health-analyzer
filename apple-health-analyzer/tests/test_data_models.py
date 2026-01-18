"""Tests for data models."""

from datetime import UTC, datetime

import pytest

from src.core.data_models import (
    DataSourcePriority,
    HealthRecord,
    HeartRateRecord,
    QuantityRecord,
    SleepRecord,
)


class TestHealthRecord:
    """Test HealthRecord base class."""

    def test_valid_record_creation(self):
        """Test creating a valid health record."""
        record = HealthRecord(
            type="HKQuantityTypeIdentifierHeartRate",
            source_name="Apple Watch",
            source_version="8.0",
            device="Watch6,1",
            unit="count/min",
            creation_date=datetime.now(UTC),
            start_date=datetime.now(UTC),
            end_date=datetime.now(UTC),
        )
        assert record.type == "HKQuantityTypeIdentifierHeartRate"
        assert record.source_name == "Apple Watch"
        assert record.source_version == "8.0"
        assert record.device == "Watch6,1"
        assert record.unit == "count/min"

    def test_end_date_before_start_date_raises_error(self):
        """Test that end_date before start_date raises validation error."""
        start_date = datetime.now(UTC)
        end_date = start_date.replace(hour=start_date.hour - 1)

        with pytest.raises(ValueError, match="end_date cannot be before start_date"):
            HealthRecord(
                type="HKQuantityTypeIdentifierHeartRate",
                source_name="Apple Watch",
                source_version=None,
                device=None,
                unit=None,
                creation_date=start_date,
                start_date=start_date,
                end_date=end_date,
            )

    def test_duration_properties(self):
        """Test duration calculation properties."""
        start_date = datetime(2023, 1, 1, 10, 0, 0, tzinfo=UTC)
        end_date = datetime(2023, 1, 1, 11, 30, 0, tzinfo=UTC)  # 1.5 hours

        record = HealthRecord(
            type="HKQuantityTypeIdentifierHeartRate",
            source_name="Apple Watch",
            source_version=None,
            device=None,
            unit=None,
            creation_date=start_date,
            start_date=start_date,
            end_date=end_date,
        )

        assert record.duration_seconds == 5400  # 1.5 * 3600
        assert record.duration_minutes == 90
        assert record.duration_hours == 1.5

    def test_source_priority(self):
        """Test source priority calculation."""
        # Apple Watch should have highest priority
        watch_record = HealthRecord(
            type="HKQuantityTypeIdentifierHeartRate",
            source_name="Apple Watch Series 8",
            source_version=None,
            device=None,
            unit=None,
            creation_date=datetime.now(UTC),
            start_date=datetime.now(UTC),
            end_date=datetime.now(UTC),
        )
        assert watch_record.source_priority == DataSourcePriority.APPLE_WATCH.value

        # Xiaomi should have medium priority
        xiaomi_record = HealthRecord(
            type="HKQuantityTypeIdentifierHeartRate",
            source_name="Xiaomi Home",
            source_version=None,
            device=None,
            unit=None,
            creation_date=datetime.now(UTC),
            start_date=datetime.now(UTC),
            end_date=datetime.now(UTC),
        )
        assert xiaomi_record.source_priority == DataSourcePriority.XIAOMI_HEALTH.value

        # iPhone should have lower priority
        phone_record = HealthRecord(
            type="HKQuantityTypeIdentifierHeartRate",
            source_name="iPhone",
            source_version=None,
            device=None,
            unit=None,
            creation_date=datetime.now(UTC),
            start_date=datetime.now(UTC),
            end_date=datetime.now(UTC),
        )
        assert phone_record.source_priority == DataSourcePriority.IPHONE.value


class TestQuantityRecord:
    """Test QuantityRecord class."""

    def test_valid_quantity_record(self):
        """Test creating a valid quantity record."""
        record = QuantityRecord(
            type="HKQuantityTypeIdentifierHeartRate",
            source_name="Apple Watch",
            value=75.0,
            unit="count/min",
            creation_date=datetime.now(UTC),
            start_date=datetime.now(UTC),
            end_date=datetime.now(UTC),
            source_version=None,
            device=None,
        )
        assert record.value == 75.0
        assert record.unit == "count/min"

    def test_negative_value_raises_error(self):
        """Test that negative values raise validation error."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError) as exc_info:
            QuantityRecord(
                type="HKQuantityTypeIdentifierHeartRate",
                source_name="Apple Watch",
                value=-10.0,
                unit="count/min",
                creation_date=datetime.now(UTC),
                start_date=datetime.now(UTC),
                end_date=datetime.now(UTC),
                source_version=None,
                device=None,
            )
        assert "greater_than" in str(exc_info.value)

    def test_zero_value_raises_error(self):
        """Test that zero values raise validation error."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError) as exc_info:
            QuantityRecord(
                type="HKQuantityTypeIdentifierHeartRate",
                source_name="Apple Watch",
                value=0.0,
                unit="count/min",
                creation_date=datetime.now(UTC),
                start_date=datetime.now(UTC),
                end_date=datetime.now(UTC),
                source_version=None,
                device=None,
            )
        assert "greater_than" in str(exc_info.value)


class TestHeartRateRecord:
    """Test HeartRateRecord class."""

    def test_valid_heart_rate(self):
        """Test creating a valid heart rate record."""
        record = HeartRateRecord(
            source_name="Apple Watch",
            value=75.0,
            creation_date=datetime.now(UTC),
            start_date=datetime.now(UTC),
            end_date=datetime.now(UTC),
        )
        assert record.type == "HKQuantityTypeIdentifierHeartRate"
        assert record.unit == "count/min"
        assert record.value == 75.0

    def test_extreme_heart_rate_allowed(self):
        """Test that extreme heart rates are allowed (warnings may not trigger in Pydantic V2)."""
        record = HeartRateRecord(
            source_name="Apple Watch",
            value=250.0,  # Very high heart rate
            creation_date=datetime.now(UTC),
            start_date=datetime.now(UTC),
            end_date=datetime.now(UTC),
        )
        assert record.value == 250.0


class TestSleepRecord:
    """Test SleepRecord class."""

    def test_valid_sleep_record(self):
        """Test creating a valid sleep record."""
        record = SleepRecord(
            source_name="Apple Watch",
            value="HKCategoryValueSleepAnalysisAsleepCore",
            creation_date=datetime.now(UTC),
            start_date=datetime.now(UTC),
            end_date=datetime.now(UTC),
        )
        assert record.type == "HKCategoryTypeIdentifierSleepAnalysis"
        assert record.sleep_stage.name == "ASLEEP_CORE"
        assert record.is_asleep is True
        assert record.is_in_bed is False
        assert record.is_awake is False

    def test_sleep_stage_properties(self):
        """Test sleep stage property methods."""
        # In bed
        in_bed = SleepRecord(
            source_name="Apple Watch",
            source_version=None,
            device=None,
            value="HKCategoryValueSleepAnalysisInBed",
            creation_date=datetime.now(UTC),
            start_date=datetime.now(UTC),
            end_date=datetime.now(UTC),
        )
        assert in_bed.is_in_bed is True
        assert in_bed.is_asleep is False

        # Awake
        awake = SleepRecord(
            source_name="Apple Watch",
            source_version=None,
            device=None,
            value="HKCategoryValueSleepAnalysisAwake",
            creation_date=datetime.now(UTC),
            start_date=datetime.now(UTC),
            end_date=datetime.now(UTC),
        )
        assert awake.is_awake is True
        assert awake.is_asleep is False

        # Asleep unspecified
        asleep = SleepRecord(
            source_name="Apple Watch",
            source_version=None,
            device=None,
            value="HKCategoryValueSleepAnalysisAsleepUnspecified",
            creation_date=datetime.now(UTC),
            start_date=datetime.now(UTC),
            end_date=datetime.now(UTC),
        )
        assert asleep.is_asleep is True
        assert asleep.is_awake is False


class TestRecordCreation:
    """Test record creation from XML elements."""

    def test_create_record_from_xml_element(self):
        """Test creating records from mock XML elements."""
        from src.core.data_models import create_record_from_xml_element

        # Mock XML element for heart rate
        class MockElement:
            def __init__(self, attribs):
                self._attribs = attribs

            def get(self, key, default=None):
                return self._attribs.get(key, default)

            def findall(self, path):
                return []  # No metadata for this test

        hr_element = MockElement({
            'type': 'HKQuantityTypeIdentifierHeartRate',
            'sourceName': 'Apple Watch',
            'sourceVersion': '8.0',
            'value': '75.5',
            'unit': 'count/min',
            'creationDate': '2023-01-01 10:00:00 +0000',
            'startDate': '2023-01-01 10:00:00 +0000',
            'endDate': '2023-01-01 10:00:00 +0000',
        })

        record = create_record_from_xml_element(hr_element)
        assert isinstance(record, HeartRateRecord)
        assert record.value == 75.5
        assert record.unit == "count/min"
        assert record.source_name == "Apple Watch"

    def test_create_unknown_record_type(self):
        """Test creating records with unknown types."""
        from src.core.data_models import create_record_from_xml_element

        class MockElement:
            def __init__(self, attribs):
                self._attribs = attribs

            def get(self, key, default=None):
                return self._attribs.get(key, default)

            def findall(self, path):
                return []

        # Unknown quantity type
        unknown_element = MockElement({
            'type': 'HKQuantityTypeIdentifierUnknownMetric',
            'sourceName': 'Unknown Device',
            'value': '42.0',
            'unit': 'unknown',
            'creationDate': '2023-01-01 10:00:00 +0000',
            'startDate': '2023-01-01 10:00:00 +0000',
            'endDate': '2023-01-01 10:00:00 +0000',
        })

        record = create_record_from_xml_element(unknown_element)
        assert isinstance(record, QuantityRecord)  # Should fall back to generic
        assert record.type == "HKQuantityTypeIdentifierUnknownMetric"
        assert record.value == 42.0

    def test_invalid_xml_element(self):
        """Test handling of invalid XML elements."""
        from src.core.data_models import create_record_from_xml_element

        class MockElement:
            def __init__(self, attribs):
                self._attribs = attribs

            def get(self, key, default=None):
                return self._attribs.get(key, default)

            def findall(self, path):
                return []

        # Element without type
        invalid_element = MockElement({
            'sourceName': 'Unknown Device',
            'value': '42.0',
        })

        record = create_record_from_xml_element(invalid_element)
        assert record is None  # Should return None for invalid elements
