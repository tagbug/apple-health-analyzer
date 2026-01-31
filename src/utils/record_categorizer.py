"""Utilities for categorizing Apple Health records by type."""

from collections.abc import Iterable
from typing import Any

HEART_RATE_TYPE = "HKQuantityTypeIdentifierHeartRate"
RESTING_HR_TYPE = "HKQuantityTypeIdentifierRestingHeartRate"
HRV_TYPE = "HKQuantityTypeIdentifierHeartRateVariabilitySDNN"
VO2_MAX_TYPE = "HKQuantityTypeIdentifierVO2Max"
SLEEP_TYPE = "HKCategoryTypeIdentifierSleepAnalysis"


def categorize_records(records: Iterable[Any]) -> dict[str, list[Any]]:
  """Categorize records by primary analysis groups."""
  categories: dict[str, list[Any]] = {
    "heart_rate": [],
    "resting_hr": [],
    "hrv": [],
    "vo2_max": [],
    "sleep": [],
  }

  for record in records:
    record_type = getattr(record, "type", "")
    if record_type == HEART_RATE_TYPE:
      categories["heart_rate"].append(record)
    elif record_type == RESTING_HR_TYPE:
      categories["resting_hr"].append(record)
    elif record_type == HRV_TYPE:
      categories["hrv"].append(record)
    elif record_type == VO2_MAX_TYPE:
      categories["vo2_max"].append(record)
    elif record_type == SLEEP_TYPE:
      categories["sleep"].append(record)

  return categories


def categorize_chart_records(records: Iterable[Any]) -> dict[str, list[Any]]:
  """Categorize records for visualization workflows."""
  categories = {
    "heart_rate": [],
    "resting_hr": [],
    "hrv": [],
    "sleep_records": [],
  }

  for record in records:
    record_type = getattr(record, "type", "")
    if record_type == HEART_RATE_TYPE:
      categories["heart_rate"].append(record)
    elif record_type == RESTING_HR_TYPE:
      categories["resting_hr"].append(record)
    elif record_type == HRV_TYPE:
      categories["hrv"].append(record)
    elif record_type == SLEEP_TYPE:
      categories["sleep_records"].append(record)

  return categories
