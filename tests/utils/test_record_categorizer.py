"""Tests for record categorization utilities."""

from types import SimpleNamespace

from src.utils.record_categorizer import (
  categorize_chart_records,
  categorize_records,
  HEART_RATE_TYPE,
  HRV_TYPE,
  RESTING_HR_TYPE,
  SLEEP_TYPE,
  VO2_MAX_TYPE,
)


def _record(record_type: str):
  return SimpleNamespace(type=record_type)


def test_categorize_records_groups_types():
  records = [
    _record(HEART_RATE_TYPE),
    _record(RESTING_HR_TYPE),
    _record(HRV_TYPE),
    _record(VO2_MAX_TYPE),
    _record(SLEEP_TYPE),
  ]

  categorized = categorize_records(records)

  assert categorized["heart_rate"]
  assert categorized["resting_hr"]
  assert categorized["hrv"]
  assert categorized["vo2_max"]
  assert categorized["sleep"]


def test_categorize_chart_records_groups_types():
  records = [
    _record(HEART_RATE_TYPE),
    _record(RESTING_HR_TYPE),
    _record(HRV_TYPE),
    _record(SLEEP_TYPE),
  ]

  categorized = categorize_chart_records(records)

  assert categorized["heart_rate"]
  assert categorized["resting_hr"]
  assert categorized["hrv"]
  assert categorized["sleep_records"]
