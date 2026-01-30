"""Unit tests for protocol interfaces."""

from dataclasses import dataclass
from datetime import datetime, timedelta

from src.core.protocols import CategoricalRecord, MeasurableRecord, TemporalRecord


@dataclass
class SampleMeasurable:
  """Sample record implementing MeasurableRecord."""

  value: float
  unit: str
  type_name: str
  start: datetime
  source: str

  @property
  def measurable_value(self) -> float:
    return self.value

  @property
  def measurement_unit(self) -> str:
    return self.unit

  @property
  def record_type(self) -> str:
    return self.type_name

  @property
  def start_date(self) -> datetime:
    return self.start

  @property
  def source_name(self) -> str:
    return self.source


@dataclass
class SampleCategorical:
  """Sample record implementing CategoricalRecord."""

  value: str

  @property
  def category_value(self) -> str:
    return self.value

  @property
  def valid_categories(self) -> list[str]:
    return ["low", "medium", "high"]


@dataclass
class SampleTemporal:
  """Sample record implementing TemporalRecord."""

  start: datetime
  end: datetime

  @property
  def start_date(self) -> datetime:
    return self.start

  @property
  def end_date(self) -> datetime:
    return self.end

  @property
  def duration_seconds(self) -> float:
    return (self.end - self.start).total_seconds()


def test_measurable_record_runtime_checkable():
  """Verify MeasurableRecord runtime checks."""
  record = SampleMeasurable(
    value=72.5,
    unit="bpm",
    type_name="heart_rate",
    start=datetime(2024, 1, 1, 8, 0, 0),
    source="Apple Watch",
  )

  assert isinstance(record, MeasurableRecord)


def test_categorical_record_runtime_checkable():
  """Verify CategoricalRecord runtime checks."""
  record = SampleCategorical(value="medium")

  assert isinstance(record, CategoricalRecord)


def test_temporal_record_runtime_checkable():
  """Verify TemporalRecord runtime checks."""
  start = datetime(2024, 1, 1, 8, 0, 0)
  end = start + timedelta(minutes=10)
  record = SampleTemporal(start=start, end=end)

  assert isinstance(record, TemporalRecord)
