"""Data models for Apple Health data using Pydantic.

Provides type-safe data structures for health records with validation and serialization.
"""

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, computed_field


class SleepStage(Enum):
    """Sleep stage enumeration based on Apple Health categories."""

    IN_BED = "HKCategoryValueSleepAnalysisInBed"
    ASLEEP_UNSPECIFIED = "HKCategoryValueSleepAnalysisAsleepUnspecified"
    ASLEEP_CORE = "HKCategoryValueSleepAnalysisAsleepCore"
    ASLEEP_DEEP = "HKCategoryValueSleepAnalysisAsleepDeep"
    ASLEEP_REM = "HKCategoryValueSleepAnalysisAsleepREM"
    AWAKE = "HKCategoryValueSleepAnalysisAwake"

class DataSourcePriority(Enum):
    """Data source priority levels (higher number = higher priority)."""

    IPHONE = 1
    XIAOMI_HEALTH = 2
    APPLE_WATCH = 3

class HealthRecord(BaseModel):
    """Base class for all health records from Apple Health export."""

    # Core attributes
    type: str = Field(..., description="Health data type identifier")
    source_name: str = Field(..., description="Data source device/app name")
    source_version: str | None = Field(None, description="Source version")
    device: str | None = Field(None, description="Device identifier")
    unit: str | None = Field(None, description="Measurement unit")

    # Timestamps
    creation_date: datetime = Field(..., description="Record creation timestamp")
    start_date: datetime = Field(..., description="Measurement start time")
    end_date: datetime = Field(..., description="Measurement end time")

    # Metadata
    metadata: dict[str, str | float | int] | None = Field(
        default_factory=dict, description="Additional metadata key-value pairs"
    )

    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
        }
    )

    @field_validator('end_date')
    @classmethod
    def validate_end_date(cls, v, info):
        """Ensure end_date is not before start_date."""
        if info.data and 'start_date' in info.data and v < info.data['start_date']:
            raise ValueError('end_date cannot be before start_date')
        return v

    @property
    def duration_seconds(self) -> float:
        """Calculate duration in seconds."""
        return (self.end_date - self.start_date).total_seconds()

    @property
    def duration_minutes(self) -> float:
        """Calculate duration in minutes."""
        return self.duration_seconds / 60

    @property
    def duration_hours(self) -> float:
        """Calculate duration in hours."""
        return self.duration_seconds / 3600

    @property
    def source_priority(self) -> int:
        """Get priority level for this data source."""
        source_lower = self.source_name.lower()

        if "watch" in source_lower:
            return DataSourcePriority.APPLE_WATCH.value
        elif "xiaomi" in source_lower:
            return DataSourcePriority.XIAOMI_HEALTH.value
        elif "phone" in source_lower or "iphone" in source_lower:
            return DataSourcePriority.IPHONE.value

        return 0

class QuantityRecord(HealthRecord):
    """Quantity-based health record (e.g., heart rate, steps)."""

    value: float = Field(..., description="Measured quantity value", gt=0)

    @field_validator('value')
    @classmethod
    def validate_value(cls, v):
        """Validate quantity value is positive."""
        if v <= 0:
            raise ValueError('Quantity value must be positive')
        return v

class CategoryRecord(HealthRecord):
    """Category-based health record (e.g., sleep stages, menstrual flow)."""

    value: str = Field(..., description="Category value identifier")

# Specialized record types

class HeartRateRecord(QuantityRecord):
    """Heart rate measurement record."""

    @computed_field
    @property
    def record_type(self) -> Literal["HKQuantityTypeIdentifierHeartRate"]:
        """Heart rate type identifier."""
        return "HKQuantityTypeIdentifierHeartRate"

    @computed_field
    @property
    def record_unit(self) -> Literal["count/min"]:
        """BPM unit."""
        return "count/min"

    def __init__(self, **data):
        # Set defaults if not provided
        if 'type' not in data:
            data['type'] = "HKQuantityTypeIdentifierHeartRate"
        if 'unit' not in data:
            data['unit'] = "count/min"
        super().__init__(**data)

    @field_validator('value')
    @classmethod
    def validate_heart_rate(cls, v: float) -> float:
        """Validate heart rate is within reasonable bounds."""
        if not (30 <= v <= 250):
            # Allow but warn about unusual values
            import warnings
            warnings.warn(f"Unusual heart rate value: {v} BPM", stacklevel=2)
        return v

class RestingHeartRateRecord(QuantityRecord):
    """Resting heart rate measurement record."""

    def __init__(self, **data):
        super().__init__(
            type="HKQuantityTypeIdentifierRestingHeartRate",
            unit="count/min",
            **data
        )

class HeartRateVariabilityRecord(QuantityRecord):
    """Heart rate variability measurement record."""

    def __init__(self, **data):
        super().__init__(
            type="HKQuantityTypeIdentifierHeartRateVariabilitySDNN",
            unit="ms",
            **data
        )

class WalkingHeartRateAverageRecord(QuantityRecord):
    """Walking heart rate average measurement record."""

    def __init__(self, **data):
        super().__init__(
            type="HKQuantityTypeIdentifierWalkingHeartRateAverage",
            unit="count/min",
            **data
        )

class VO2MaxRecord(QuantityRecord):
    """VO2 Max (cardio fitness) measurement record."""

    def __init__(self, **data):
        super().__init__(
            type="HKQuantityTypeIdentifierVO2Max",
            unit="mL/minkg",
            **data
        )

class StepCountRecord(QuantityRecord):
    """Step count measurement record."""

    def __init__(self, **data):
        super().__init__(
            type="HKQuantityTypeIdentifierStepCount",
            unit="count",
            **data
        )

class DistanceRecord(QuantityRecord):
    """Walking/running distance measurement record."""

    def __init__(self, **data):
        super().__init__(
            type="HKQuantityTypeIdentifierDistanceWalkingRunning",
            unit="km",
            **data
        )

class SleepRecord(CategoryRecord):
    """Sleep analysis record."""

    def __init__(self, **data):
        super().__init__(
            type="HKCategoryTypeIdentifierSleepAnalysis",
            **data
        )

    @property
    def sleep_stage(self) -> SleepStage:
        """Get the sleep stage enum from the value."""
        try:
            return SleepStage(self.value)
        except ValueError:
            return SleepStage.ASLEEP_UNSPECIFIED

    @property
    def is_asleep(self) -> bool:
        """Check if this record represents sleep (not just in bed)."""
        return self.sleep_stage in {
            SleepStage.ASLEEP_UNSPECIFIED,
            SleepStage.ASLEEP_CORE,
            SleepStage.ASLEEP_DEEP,
            SleepStage.ASLEEP_REM,
        }

    @property
    def is_in_bed(self) -> bool:
        """Check if this record represents time in bed."""
        return self.sleep_stage == SleepStage.IN_BED

    @property
    def is_awake(self) -> bool:
        """Check if this record represents awake time."""
        return self.sleep_stage == SleepStage.AWAKE

class WorkoutRecord(BaseModel):
    """Workout record from Apple Health export."""

    # Basic info
    activity_type: str = Field(..., description="Workout activity type")
    duration_seconds: float = Field(..., description="Workout duration in seconds")
    source_name: str = Field(..., description="Data source name")

    # Timestamps
    start_date: datetime = Field(..., description="Workout start time")
    end_date: datetime = Field(..., description="Workout end time")

    # Metrics
    calories: float | None = Field(None, description="Calories burned")
    distance_km: float | None = Field(None, description="Distance in kilometers")
    average_heart_rate: float | None = Field(None, description="Average heart rate")

    # Metadata
    metadata: dict[str, str | float | int] | None = Field(
        default_factory=dict, description="Additional workout metadata"
    )

    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
        }
    )

    @property
    def duration_minutes(self) -> float:
        """Get duration in minutes."""
        return self.duration_seconds / 60

    @property
    def duration_hours(self) -> float:
        """Get duration in hours."""
        return self.duration_seconds / 3600

class ActivitySummaryRecord(BaseModel):
    """Daily activity summary record."""

    date: datetime = Field(..., description="Summary date")

    # Activity rings
    move_calories: float | None = Field(None, description="Move ring calories")
    exercise_minutes: float | None = Field(None, description="Exercise ring minutes")
    stand_hours: float | None = Field(None, description="Stand ring hours")

    # Goals
    move_goal: float | None = Field(None, description="Move goal calories")
    exercise_goal: float | None = Field(None, description="Exercise goal minutes")
    stand_goal: float | None = Field(None, description="Stand goal hours")

    # Achievements
    move_achieved: bool = Field(default=False, description="Move goal achieved")
    exercise_achieved: bool = Field(default=False, description="Exercise goal achieved")
    stand_achieved: bool = Field(default=False, description="Stand goal achieved")

    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
        }
    )

# Type unions for easier handling
AnyRecord = (
    QuantityRecord
    | CategoryRecord
    | HeartRateRecord
    | SleepRecord
    | WorkoutRecord
    | ActivitySummaryRecord
)

# Record type mapping for parsing
RECORD_TYPE_MAPPING = {
    "HKQuantityTypeIdentifierHeartRate": HeartRateRecord,
    "HKQuantityTypeIdentifierRestingHeartRate": RestingHeartRateRecord,
    "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": HeartRateVariabilityRecord,
    "HKQuantityTypeIdentifierWalkingHeartRateAverage": WalkingHeartRateAverageRecord,
    "HKQuantityTypeIdentifierVO2Max": VO2MaxRecord,
    "HKQuantityTypeIdentifierStepCount": StepCountRecord,
    "HKQuantityTypeIdentifierDistanceWalkingRunning": DistanceRecord,
    "HKCategoryTypeIdentifierSleepAnalysis": SleepRecord,
}

def create_record_from_xml_element(element) -> AnyRecord | None:
    """Create appropriate record instance from XML element.

    Args:
        element: XML element from Apple Health export

    Returns:
        Record instance or None if parsing fails
    """
    try:
        record_type = element.get('type')
        if not record_type:
            return None

        # Get the appropriate record class
        record_class = RECORD_TYPE_MAPPING.get(record_type)
        if not record_class:
            # Fallback to generic record types
            if record_type.startswith('HKQuantityTypeIdentifier'):
                record_class = QuantityRecord
            elif record_type.startswith('HKCategoryTypeIdentifier'):
                record_class = CategoryRecord
            else:
                return None

        # Parse common attributes
        data: dict[str, str | float | int | datetime | dict[str, str | float | int] | None] = {
            'type': record_type,
            'source_name': element.get('sourceName', ''),
            'source_version': element.get('sourceVersion'),
            'device': element.get('device'),
            'unit': element.get('unit'),
            'creation_date': datetime.strptime(
                element.get('creationDate', ''), '%Y-%m-%d %H:%M:%S %z'
            ),
            'start_date': datetime.strptime(
                element.get('startDate', ''), '%Y-%m-%d %H:%M:%S %z'
            ),
            'end_date': datetime.strptime(
                element.get('endDate', ''), '%Y-%m-%d %H:%M:%S %z'
            ),
        }

        # Add value for quantity/category records
        value = element.get('value')
        if value is not None:
            if record_class == QuantityRecord or issubclass(record_class, QuantityRecord):
                data['value'] = float(value)
            else:
                data['value'] = value

        # Parse metadata
        metadata: dict[str, str | float | int] = {}
        for meta in element.findall('.//MetadataEntry'):
            key = meta.get('key')
            val = meta.get('value')
            if key and val is not None:
                # Try to convert to appropriate type
                try:
                    # Try float first
                    metadata[key] = float(val)
                except ValueError:
                    # Fall back to string
                    metadata[key] = val

        if metadata:
            data['metadata'] = metadata

        # Create the record instance
        return record_class(**data)  # type: ignore

    except Exception:
        # Return None for any parsing errors
        return None
