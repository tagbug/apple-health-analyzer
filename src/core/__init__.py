"""Core modules for Apple Health data processing"""

from .protocols import CategoricalRecord, MeasurableRecord, TemporalRecord

__all__ = [
  "MeasurableRecord",
  "CategoricalRecord",
  "TemporalRecord",
]
