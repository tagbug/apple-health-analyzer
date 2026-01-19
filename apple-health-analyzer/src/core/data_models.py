"""Data models for Apple Health data using Pydantic.

Provides type-safe data structures for health records with validation and serialization.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import (
  BaseModel,
  ConfigDict,
  Field,
  computed_field,
  field_validator,
)


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


class BaseRecord(BaseModel, ABC):
  """Abstract base class for all health record types.

  Ensures consistent interface across all record types.
  """

  # Common attributes for all records
  source_name: str = Field(..., description="Data source device/app name")
  start_date: datetime = Field(..., description="Start time")
  end_date: datetime = Field(..., description="End time")
  metadata: dict[str, str | float | int] | None = Field(
    default_factory=dict, description="Additional metadata key-value pairs"
  )

  model_config = ConfigDict(
    validate_assignment=True,
    json_encoders={
      datetime: lambda v: v.isoformat(),
    },
  )

  @property
  @abstractmethod
  def record_type(self) -> str:
    """Return the record type identifier (must be implemented by subclasses)."""
    pass

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


class HealthRecord(BaseRecord):
  """Base class for all health records from Apple Health export."""

  # Health-specific attributes
  type: str = Field(..., description="Health data type identifier")
  source_version: str | None = Field(None, description="Source version")
  device: str | None = Field(None, description="Device identifier")
  unit: str | None = Field(None, description="Measurement unit")
  creation_date: datetime = Field(..., description="Record creation timestamp")

  model_config = ConfigDict(
    validate_assignment=True,
    json_encoders={
      datetime: lambda v: v.isoformat(),
    },
  )

  @field_validator("end_date")
  @classmethod
  def validate_end_date(cls, v, info):
    """Ensure end_date is not before start_date."""
    if info.data and "start_date" in info.data and v < info.data["start_date"]:
      raise ValueError("end_date cannot be before start_date")
    return v

  @property
  def record_type(self) -> str:
    """Return the health record type identifier."""
    return self.type


class QuantityRecord(HealthRecord):
  """Quantity-based health record (e.g., heart rate, steps)."""

  value: float = Field(..., description="Measured quantity value", gt=0)

  @field_validator("value")
  @classmethod
  def validate_value(cls, v):
    """Validate quantity value is positive."""
    if v <= 0:
      raise ValueError("Quantity value must be positive")
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
    if "type" not in data:
      data["type"] = "HKQuantityTypeIdentifierHeartRate"
    if "unit" not in data:
      data["unit"] = "count/min"
    super().__init__(**data)

  @field_validator("value")
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
      type="HKQuantityTypeIdentifierRestingHeartRate", unit="count/min", **data
    )


class HeartRateVariabilityRecord(QuantityRecord):
  """Heart rate variability measurement record."""

  def __init__(self, **data):
    super().__init__(
      type="HKQuantityTypeIdentifierHeartRateVariabilitySDNN", unit="ms", **data
    )


class WalkingHeartRateAverageRecord(QuantityRecord):
  """Walking heart rate average measurement record."""

  def __init__(self, **data):
    super().__init__(
      type="HKQuantityTypeIdentifierWalkingHeartRateAverage",
      unit="count/min",
      **data,
    )


class VO2MaxRecord(QuantityRecord):
  """VO2 Max (cardio fitness) measurement record."""

  def __init__(self, **data):
    super().__init__(
      type="HKQuantityTypeIdentifierVO2Max", unit="mL/minkg", **data
    )


class StepCountRecord(QuantityRecord):
  """Step count measurement record."""

  def __init__(self, **data):
    super().__init__(
      type="HKQuantityTypeIdentifierStepCount", unit="count", **data
    )


class DistanceRecord(QuantityRecord):
  """Walking/running distance measurement record."""

  def __init__(self, **data):
    super().__init__(
      type="HKQuantityTypeIdentifierDistanceWalkingRunning", unit="km", **data
    )


class SleepRecord(CategoryRecord):
  """Sleep analysis record."""

  def __init__(self, **data):
    # Set type if not provided
    if "type" not in data:
      data["type"] = "HKCategoryTypeIdentifierSleepAnalysis"
    super().__init__(**data)

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


class WorkoutRecord(BaseRecord):
  """Workout record from Apple Health export."""

  # Workout-specific attributes
  activity_type: str = Field(..., description="Workout activity type")
  workout_duration_seconds: float = Field(
    ..., description="Workout duration in seconds"
  )

  # Metrics
  calories: float | None = Field(None, description="Calories burned")
  distance_km: float | None = Field(None, description="Distance in kilometers")
  average_heart_rate: float | None = Field(
    None, description="Average heart rate"
  )

  @property
  def record_type(self) -> str:
    """Return workout record type identifier."""
    return f"Workout:{self.activity_type}"


class ActivitySummaryRecord(BaseRecord):
  """Daily activity summary record."""

  date: datetime = Field(..., description="Summary date")

  # Activity rings
  move_calories: float | None = Field(None, description="Move ring calories")
  exercise_minutes: float | None = Field(
    None, description="Exercise ring minutes"
  )
  stand_hours: float | None = Field(None, description="Stand ring hours")

  # Goals
  move_goal: float | None = Field(None, description="Move goal calories")
  exercise_goal: float | None = Field(None, description="Exercise goal minutes")
  stand_goal: float | None = Field(None, description="Stand goal hours")

  # Achievements
  move_achieved: bool = Field(default=False, description="Move goal achieved")
  exercise_achieved: bool = Field(
    default=False, description="Exercise goal achieved"
  )
  stand_achieved: bool = Field(default=False, description="Stand goal achieved")

  def __init__(self, **data):
    # For activity summaries, start_date and end_date are the same day
    if "date" in data and "start_date" not in data:
      data["start_date"] = data["date"]
      data["end_date"] = data["date"]

    # Calculate achievements automatically
    move_calories = data.get("move_calories")
    move_goal = data.get("move_goal")
    data["move_achieved"] = (
      move_calories is not None
      and move_goal is not None
      and isinstance(move_calories, (int, float))
      and isinstance(move_goal, (int, float))
      and move_calories >= move_goal
    )

    exercise_minutes = data.get("exercise_minutes")
    exercise_goal = data.get("exercise_goal")
    data["exercise_achieved"] = (
      exercise_minutes is not None
      and exercise_goal is not None
      and isinstance(exercise_minutes, (int, float))
      and isinstance(exercise_goal, (int, float))
      and exercise_minutes >= exercise_goal
    )

    stand_hours = data.get("stand_hours")
    stand_goal = data.get("stand_goal")
    data["stand_achieved"] = (
      stand_hours is not None
      and stand_goal is not None
      and isinstance(stand_hours, (int, float))
      and isinstance(stand_goal, (int, float))
      and stand_hours >= stand_goal
    )

    super().__init__(**data)

  @property
  def record_type(self) -> str:
    """Return activity summary record type identifier."""
    return "ActivitySummary"


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


def create_record_from_xml_element(
  element, use_defaults: bool = True
) -> tuple[AnyRecord | None, list[str]]:
  """Create appropriate record instance from XML element.

  Args:
      element: XML element from Apple Health export
      use_defaults: Whether to use default values for missing required fields

  Returns:
      Tuple of (record_instance, warnings_list)
      - record_instance: Parsed record or None if parsing completely fails
      - warnings_list: List of warning messages for missing/using default values
  """
  warnings = []

  try:
    record_type = element.get("type")
    if not record_type:
      return None, ["Missing required field: type"]

    # Get the appropriate record class
    record_class = RECORD_TYPE_MAPPING.get(record_type)
    if not record_class:
      # Fallback to generic record types
      if record_type.startswith("HKQuantityTypeIdentifier"):
        record_class = QuantityRecord
      elif record_type.startswith("HKCategoryTypeIdentifier"):
        record_class = CategoryRecord
      else:
        return None, [f"Unknown record type: {record_type}"]

    # Parse attributes with default value handling
    data: dict[
      str, str | float | int | datetime | dict[str, str | float | int] | None
    ] = {
      "type": record_type,
    }

    # Required fields - use defaults if missing
    source_name = element.get("sourceName")
    if not source_name:
      if use_defaults:
        warnings.append('Missing sourceName, using default "Unknown"')
        data["source_name"] = "Unknown"
      else:
        return None, ["Missing required field: sourceName"]
    else:
      data["source_name"] = source_name

    # Date fields - use current time as default
    current_time = datetime.now()
    for date_field, data_key in [
      ("creationDate", "creation_date"),
      ("startDate", "start_date"),
      ("endDate", "end_date"),
    ]:
      date_value = element.get(date_field)
      if not date_value:
        if use_defaults:
          warnings.append(f"Missing {date_field}, using current time")
          data[data_key] = current_time
        else:
          return None, [f"Missing required field: {date_field}"]
      else:
        try:
          data[data_key] = datetime.strptime(date_value, "%Y-%m-%d %H:%M:%S %z")
        except ValueError as e:
          if use_defaults:
            warnings.append(
              f"Invalid {date_field} format ({date_value}), using current time"
            )
            data[data_key] = current_time
          else:
            return None, [f"Invalid {date_field} format: {e}"]

    # Optional fields - no warnings for missing values
    data["source_version"] = element.get("sourceVersion")
    data["device"] = element.get("device")
    data["unit"] = element.get("unit")

    # Value field - required for quantity records, optional for category
    value = element.get("value")
    if value is not None:
      if record_class == QuantityRecord or issubclass(
        record_class, QuantityRecord
      ):
        try:
          data["value"] = float(value)
        except (ValueError, TypeError):
          if use_defaults:
            warnings.append(f"Invalid value format ({value}), using 0.0")
            data["value"] = 0.0
          else:
            return None, [f"Invalid value format: {value}"]
      else:
        data["value"] = value
    elif record_class == QuantityRecord or issubclass(
      record_class, QuantityRecord
    ):
      if use_defaults:
        warnings.append("Missing value for quantity record, using 0.0")
        data["value"] = 0.0
      else:
        return None, ["Missing required field: value (for quantity records)"]

    # Parse metadata
    metadata: dict[str, str | float | int] = {}
    for meta in element.findall(".//MetadataEntry"):
      key = meta.get("key")
      val = meta.get("value")
      if key and val is not None:
        # Try to convert to appropriate type
        try:
          # Try float first
          metadata[key] = float(val)
        except ValueError:
          # Fall back to string
          metadata[key] = val

    if metadata:
      data["metadata"] = metadata

    # Create the record instance
    try:
      record = record_class(**data)  # type: ignore
      return record, warnings
    except Exception as e:
      if use_defaults:
        warnings.append(
          f"Record validation failed: {e}, but proceeding with defaults"
        )
        return record_class(**data), warnings  # type: ignore
      else:
        return None, [f"Record validation failed: {e}"]

  except Exception as e:
    return None, [f"Unexpected parsing error: {str(e)}"]
