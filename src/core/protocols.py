"""Type protocols for health record interfaces.

Provides type-safe interfaces for different categories of health records,
enabling polymorphic behavior and better type checking.
"""

from datetime import datetime
from typing import Protocol, runtime_checkable


@runtime_checkable
class MeasurableRecord(Protocol):
  """Protocol for records that can be measured and analyzed.

  Any record type that implements this protocol can be used in
  statistical analysis, outlier detection, and validation.
  """

  @property
  def measurable_value(self) -> float:
    """Return the primary measurable value for this record.

    Returns:
        float: The primary metric value for analysis
    """
    ...

  @property
  def measurement_unit(self) -> str:
    """Return the unit of measurement.

    Returns:
        str: Unit description (e.g., "BPM", "steps", "kcal")
    """
    ...

  @property
  def record_type(self) -> str:
    """Return the record type identifier."""
    ...

  @property
  def start_date(self) -> datetime:
    """Return the start date of the record."""
    ...

  @property
  def source_name(self) -> str:
    """Return the data source name."""
    ...


@runtime_checkable
class CategoricalRecord(Protocol):
  """Protocol for records with categorical values.

  Used for records like sleep stages, menstrual flow, etc.
  """

  @property
  def category_value(self) -> str:
    """Return the categorical value."""
    ...

  @property
  def valid_categories(self) -> list[str]:
    """Return list of valid category values."""
    ...


@runtime_checkable
class TemporalRecord(Protocol):
  """Protocol for records with time-based data.

  All health records should implement this for temporal analysis.
  """

  @property
  def start_date(self) -> datetime:
    """Return the start date of the record."""
    ...

  @property
  def end_date(self) -> datetime:
    """Return the end date of the record."""
    ...

  @property
  def duration_seconds(self) -> float:
    """Return the duration in seconds."""
    ...
