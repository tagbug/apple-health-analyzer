"""Data validation and quality assurance for health records.

Provides comprehensive validation including range checks, outlier detection,
data consistency validation, and quality scoring.
"""

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from ..core.data_models import (
  ActivitySummaryRecord,
  AnyRecord,
  CategoryRecord,
  HeartRateRecord,
  QuantityRecord,
  SleepRecord,
  WorkoutRecord,
)
from ..core.protocols import MeasurableRecord
from ..i18n import Translator, resolve_locale
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ValidationResult:
  """Result of data validation with detailed information."""

  def __init__(self, translator: Translator):
    self.translator = translator
    self.is_valid: bool = True
    self.errors: list[str] = []
    self.warnings: list[str] = []
    self.quality_score: float = 1.0
    self.issues_by_type: dict[str, list[str]] = defaultdict(list)
    self.outliers_detected: list[dict[str, Any]] = []
    self.consistency_checks: dict[str, bool] = {}

  def add_error(self, message: str, record_type: str = "general"):
    """Add a validation error."""
    self.errors.append(message)
    self.issues_by_type[record_type].append(
      self.translator.t("validation.issue.error", message=message)
    )
    self.is_valid = False
    self.quality_score = max(0.0, self.quality_score - 0.2)

  def add_warning(self, message: str, record_type: str = "general"):
    """Add a validation warning."""
    self.warnings.append(message)
    self.issues_by_type[record_type].append(
      self.translator.t("validation.issue.warning", message=message)
    )
    self.quality_score = max(0.0, self.quality_score - 0.05)

  def add_outlier(self, record: AnyRecord, reason: str, severity: str = "medium"):
    """Add an outlier detection result."""
    self.outliers_detected.append(
      {
        "record": record,
        "reason": reason,
        "severity": severity,
        "record_type": getattr(record, "record_type", "unknown"),
        "value": getattr(record, "value", None),
        "timestamp": getattr(record, "start_date", None),
      }
    )

    # Reduce quality score based on severity
    score_reduction = {"low": 0.01, "medium": 0.05, "high": 0.1}
    self.quality_score = max(
      0.0, self.quality_score - score_reduction.get(severity, 0.05)
    )

  def set_consistency_check(self, check_name: str, passed: bool):
    """Set result of a consistency check."""
    self.consistency_checks[check_name] = passed
    if not passed:
      self.quality_score = max(0.0, self.quality_score - 0.1)

  def get_summary(self) -> dict[str, Any]:
    """Get a summary of validation results."""
    return {
      "is_valid": self.is_valid,
      "quality_score": round(self.quality_score, 3),
      "total_errors": len(self.errors),
      "total_warnings": len(self.warnings),
      "outliers_count": len(self.outliers_detected),
      "consistency_checks_passed": sum(self.consistency_checks.values()),
      "consistency_checks_total": len(self.consistency_checks),
      "issues_by_type": dict(self.issues_by_type),
    }


class DataValidator:
  """Comprehensive data validator for health records.

  Performs multiple levels of validation:
  1. Range and format validation
  2. Outlier detection
  3. Data consistency checks
  4. Cross-record validation
  """

  def __init__(self, locale: str | None = None):
    self.translator = Translator(resolve_locale(locale))
    # Define validation ranges for different data types
    self.validation_ranges = {
      "HKQuantityTypeIdentifierHeartRate": {
        "min": 30,
        "max": 250,
        "unit": "count/min",
      },
      "HKQuantityTypeIdentifierRestingHeartRate": {
        "min": 30,
        "max": 120,
        "unit": "count/min",
      },
      "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": {
        "min": 1,
        "max": 200,
        "unit": "ms",
      },
      "HKQuantityTypeIdentifierWalkingHeartRateAverage": {
        "min": 40,
        "max": 180,
        "unit": "count/min",
      },
      "HKQuantityTypeIdentifierVO2Max": {
        "min": 10,
        "max": 90,
        "unit": "mL/min·kg",
      },
      "HKQuantityTypeIdentifierStepCount": {
        "min": 0,
        "max": 100000,
        "unit": "count",
      },
      "HKQuantityTypeIdentifierDistanceWalkingRunning": {
        "min": 0,
        "max": 200,
        "unit": "km",
      },
      "HKQuantityTypeIdentifierActiveEnergyBurned": {
        "min": 0,
        "max": 10000,
        "unit": "kcal",
      },
    }

    # Outlier detection parameters
    self.outlier_params = {
      "zscore_threshold": 3.0,  # Standard deviations
      "iqr_multiplier": 1.5,  # IQR multiplier for outlier detection
      "min_samples_for_stats": 10,  # Minimum samples needed for statistical analysis
    }

  def validate_records_comprehensive(
    self,
    records: Sequence[AnyRecord],
    enable_outlier_detection: bool = True,
    enable_consistency_checks: bool = True,
  ) -> ValidationResult:
    """Perform comprehensive validation on a list of records.

    Args:
        records: List of health records to validate
        enable_outlier_detection: Whether to perform outlier detection
        enable_consistency_checks: Whether to perform consistency checks

    Returns:
        ValidationResult with detailed validation information
    """
    result = ValidationResult(self.translator)

    if not records:
      result.add_warning(self.translator.t("validation.warning.no_records"))
      return result

    logger.info(
      self.translator.t(
        "log.validator.start",
        count=len(records),
      )
    )

    # Group records by type for efficient processing
    records_by_type = defaultdict(list)
    for record in records:
      record_type = getattr(record, "record_type", "unknown")
      records_by_type[record_type].append(record)

    # Validate each record individually
    for record in records:
      self._validate_single_record(record, result)

    # Perform outlier detection if enabled
    if enable_outlier_detection:
      for _record_type, type_records in records_by_type.items():
        self._detect_outliers(type_records, result)

    # Perform consistency checks if enabled
    if enable_consistency_checks:
      self._perform_consistency_checks(records, result)

    # Perform cross-record validation
    self._validate_cross_record_consistency(records, result)

    logger.info(
      self.translator.t(
        "log.validator.completed",
        score=result.quality_score,
        errors=len(result.errors),
        warnings=len(result.warnings),
        outliers=len(result.outliers_detected),
      )
    )

    return result

  def _validate_single_record(
    self, record: AnyRecord, result: ValidationResult
  ) -> None:
    """Validate a single record."""
    record_type = getattr(record, "record_type", "unknown")

    try:
      # Basic field validation
      self._validate_basic_fields(record, result)

      # Type-specific validation
      if isinstance(record, QuantityRecord):
        self._validate_quantity_record(record, result)
      elif isinstance(record, CategoryRecord):
        self._validate_category_record(record, result)
      elif isinstance(record, HeartRateRecord):
        self._validate_heart_rate_record(record, result)
      elif isinstance(record, SleepRecord):
        self._validate_sleep_record(record, result)
      elif isinstance(record, WorkoutRecord):
        self._validate_workout_record(record, result)
      elif isinstance(record, ActivitySummaryRecord):
        self._validate_activity_summary_record(record, result)

    except Exception as e:
      result.add_error(
        self.translator.t(
          "validation.error.record_failed",
          record_type=record_type,
          error=e,
        ),
        record_type,
      )

  def _validate_basic_fields(self, record: AnyRecord, result: ValidationResult) -> None:
    """Validate basic required fields for any record."""
    record_type = getattr(record, "record_type", "unknown")

    # Check required fields
    if not hasattr(record, "source_name") or not record.source_name:
      result.add_error(
        self.translator.t("validation.error.missing_source"),
        record_type,
      )

    if not hasattr(record, "start_date") or not record.start_date:
      result.add_error(
        self.translator.t("validation.error.invalid_start_date"),
        record_type,
      )

    if not hasattr(record, "end_date") or not record.end_date:
      result.add_error(
        self.translator.t("validation.error.invalid_end_date"),
        record_type,
      )

    # Validate date consistency
    if hasattr(record, "start_date") and hasattr(record, "end_date"):
      if record.start_date and record.end_date:
        if record.end_date < record.start_date:
          result.add_error(
            self.translator.t("validation.error.end_before_start"),
            record_type,
          )
        elif (record.end_date - record.start_date) > timedelta(days=1):
          result.add_warning(
            self.translator.t("validation.warning.span_over_24h"),
            record_type,
          )

    # Check for future dates
    now = datetime.now()
    if hasattr(record, "start_date") and record.start_date and record.start_date > now:
      result.add_warning(
        self.translator.t("validation.warning.start_future"),
        record_type,
      )

    if hasattr(record, "end_date") and record.end_date and record.end_date > now:
      result.add_warning(
        self.translator.t("validation.warning.end_future"),
        record_type,
      )

  def _validate_quantity_record(
    self, record: QuantityRecord, result: ValidationResult
  ) -> None:
    """Validate quantity-based records."""
    record_type = record.record_type

    # Check value range
    if record_type in self.validation_ranges:
      range_info = self.validation_ranges[record_type]
      min_val = range_info["min"]
      max_val = range_info["max"]
      if (
        isinstance(record.value, (int, float))
        and isinstance(min_val, (int, float))
        and isinstance(max_val, (int, float))
      ):
        if record.value < min_val:
          result.add_warning(
            self.translator.t(
              "validation.warning.value_below_min",
              value=record.value,
              min=min_val,
              unit=range_info["unit"],
            ),
            record_type,
          )
        elif record.value > max_val:
          result.add_warning(
            self.translator.t(
              "validation.warning.value_above_max",
              value=record.value,
              max=max_val,
              unit=range_info["unit"],
            ),
            record_type,
          )

    # Check for unrealistic precision
    if (
      isinstance(record.value, (int, float))
      and record.value != 0
      and abs(record.value) < 0.001
    ):
      result.add_warning(
        self.translator.t(
          "validation.warning.unrealistic_precision",
          value=record.value,
        ),
        record_type,
      )

  def _validate_category_record(
    self, record: CategoryRecord, result: ValidationResult
  ) -> None:
    """Validate category-based records."""
    record_type = record.record_type

    # Validate category values are reasonable strings
    if not isinstance(record.value, str):
      result.add_error(
        self.translator.t("validation.error.category_type"),
        record_type,
      )
    elif len(record.value.strip()) == 0:
      result.add_error(
        self.translator.t("validation.error.category_empty"),
        record_type,
      )

  def _validate_heart_rate_record(
    self, record: HeartRateRecord, result: ValidationResult
  ) -> None:
    """Validate heart rate specific rules."""
    # Additional heart rate specific validation
    value = record.value

    # Check for physiologically impossible values
    if value < 20:
      result.add_error(
        self.translator.t(
          "validation.error.hr_impossible",
          value=value,
        ),
        "heart_rate",
      )
    elif value > 300:
      result.add_error(
        self.translator.t(
          "validation.error.hr_danger",
          value=value,
        ),
        "heart_rate",
      )

    # Check for exercise vs resting context
    # This is a simplified check - in practice, you'd need more context
    if hasattr(record, "metadata") and record.metadata:
      # Look for exercise indicators in metadata
      exercise_indicators = ["workout", "exercise", "running", "cycling"]
      is_exercise = any(
        indicator in str(record.metadata).lower() for indicator in exercise_indicators
      )

      if is_exercise and value < 100:
        result.add_warning(
          self.translator.t("validation.warning.hr_low_exercise"),
          "heart_rate",
        )
      elif not is_exercise and value > 200:
        result.add_warning(
          self.translator.t("validation.warning.hr_high_rest"),
          "heart_rate",
        )

  def _validate_sleep_record(
    self, record: SleepRecord, result: ValidationResult
  ) -> None:
    """Validate sleep record specific rules."""
    # Validate sleep stage values
    valid_stages = {
      "HKCategoryValueSleepAnalysisInBed",
      "HKCategoryValueSleepAnalysisAsleepUnspecified",
      "HKCategoryValueSleepAnalysisAsleepCore",
      "HKCategoryValueSleepAnalysisAsleepDeep",
      "HKCategoryValueSleepAnalysisAsleepREM",
      "HKCategoryValueSleepAnalysisAwake",
    }

    if record.value not in valid_stages:
      result.add_warning(
        self.translator.t(
          "validation.warning.sleep_stage_unknown",
          value=record.value,
        ),
        "sleep",
      )

    # Check sleep duration
    duration_hours = record.duration_hours
    if duration_hours > 24:
      result.add_error(
        self.translator.t("validation.error.sleep_too_long"),
        "sleep",
      )
    elif duration_hours < 0.1:  # Less than 6 minutes
      result.add_warning(
        self.translator.t("validation.warning.sleep_too_short"),
        "sleep",
      )

  def _validate_workout_record(
    self, record: WorkoutRecord, result: ValidationResult
  ) -> None:
    """Validate workout record specific rules."""
    # Validate workout duration
    if record.workout_duration_seconds <= 0:
      result.add_error(
        self.translator.t("validation.error.workout_duration"),
        "workout",
      )
    elif record.workout_duration_seconds > 86400:  # 24 hours
      result.add_error(
        self.translator.t("validation.error.workout_too_long"),
        "workout",
      )

    # Validate calories if present
    if record.calories is not None:
      if record.calories < 0:
        result.add_error(
          self.translator.t("validation.error.workout_calories_negative"),
          "workout",
        )
      elif record.calories > 10000:  # Unrealistic for a single workout
        result.add_warning(
          self.translator.t("validation.warning.workout_calories_high"),
          "workout",
        )

    # Validate distance if present
    if record.distance_km is not None:
      if record.distance_km < 0:
        result.add_error(
          self.translator.t("validation.error.workout_distance_negative"),
          "workout",
        )
      elif record.distance_km > 1000:  # Unrealistic for a single workout
        result.add_warning(
          self.translator.t("validation.warning.workout_distance_long"),
          "workout",
        )

  def _validate_activity_summary_record(
    self, record: ActivitySummaryRecord, result: ValidationResult
  ) -> None:
    """Validate activity summary record specific rules."""
    # Validate ring values are non-negative
    for field in ["move_calories", "exercise_minutes", "stand_hours"]:
      value = getattr(record, field, None)
      if value is not None and value < 0:
        result.add_error(
          self.translator.t(
            "validation.error.activity_negative",
            field=field,
          ),
          "activity_summary",
        )

    # Validate goals are reasonable
    if record.move_goal and record.move_goal > 2000:
      result.add_warning(
        self.translator.t("validation.warning.move_goal_high"),
        "activity_summary",
      )

    if record.exercise_goal and record.exercise_goal > 200:
      result.add_warning(
        self.translator.t("validation.warning.exercise_goal_high"),
        "activity_summary",
      )

    if record.stand_goal and record.stand_goal > 24:
      result.add_warning(
        self.translator.t("validation.warning.stand_goal_high"),
        "activity_summary",
      )

  def _detect_outliers(
    self, records: Sequence[AnyRecord], result: ValidationResult
  ) -> None:
    """Detect outliers in a list of records of the same type."""
    if len(records) < self.outlier_params["min_samples_for_stats"]:
      return  # Not enough data for statistical analysis

    # Filter records that implement MeasurableRecord protocol
    measurable_records: list[MeasurableRecord] = []
    for record in records:
      if isinstance(record, MeasurableRecord):
        measurable_records.append(record)

    if len(measurable_records) < self.outlier_params["min_samples_for_stats"]:
      return

    # Extract numeric values using the protocol
    values = [record.measurable_value for record in measurable_records]
    values_array = np.array(values, dtype=np.float64)

    # Z-score based outlier detection
    try:
      # Calculate z-scores manually to avoid scipy type issues
      mean_val = np.mean(values_array)
      std_val = np.std(values_array)
      if std_val > 0:
        z_scores = np.abs((values_array - mean_val) / std_val)
        zscore_outliers = np.where(z_scores > self.outlier_params["zscore_threshold"])[
          0
        ]
      else:
        zscore_outliers = np.array([])
    except (ValueError, TypeError):
      zscore_outliers = np.array([])

    # IQR based outlier detection
    q1, q3 = np.percentile(values_array, [25, 75])
    iqr = q3 - q1
    iqr_lower = q1 - (self.outlier_params["iqr_multiplier"] * iqr)
    iqr_upper = q3 + (self.outlier_params["iqr_multiplier"] * iqr)
    iqr_outliers = np.where((values_array < iqr_lower) | (values_array > iqr_upper))[0]

    # Combine outlier detections
    all_outliers = set(zscore_outliers) | set(iqr_outliers)

    for idx in all_outliers:
      record = measurable_records[idx]
      value = values[idx]

      # Determine severity
      z_score = float(z_scores[idx]) if idx < len(z_scores) else 3.5
      if z_score > 5:
        severity = "high"
      elif z_score > 3:
        severity = "medium"
      else:
        severity = "low"

      # Type-safe call using MeasurableRecord protocol
      result.add_outlier(
        record,  # Type checker knows this is MeasurableRecord
        self.translator.t(
          "validation.warning.outlier",
          value=value,
          unit=record.measurement_unit,
          z_score=z_score,
        ),
        severity,
      )

  def _perform_consistency_checks(
    self, records: Sequence[AnyRecord], result: ValidationResult
  ) -> None:
    """Perform consistency checks across records."""
    # Check for duplicate timestamps within same source
    timestamp_source_combos = defaultdict(list)

    for record in records:
      if hasattr(record, "start_date") and hasattr(record, "source_name"):
        key = (record.start_date, record.source_name)
        timestamp_source_combos[key].append(record)

    duplicates_found = 0
    for _key, records_list in timestamp_source_combos.items():
      if len(records_list) > 1:
        duplicates_found += len(records_list) - 1

    result.set_consistency_check("no_duplicate_timestamps", duplicates_found == 0)
    if duplicates_found > 0:
      result.add_warning(
        self.translator.t(
          "validation.warning.duplicate_timestamps",
          count=duplicates_found,
        )
      )

    # Check data source consistency
    sources = set()
    for record in records:
      if hasattr(record, "source_name"):
        sources.add(record.source_name)

    result.set_consistency_check("reasonable_source_count", len(sources) <= 10)
    if len(sources) > 10:
      result.add_warning(
        self.translator.t(
          "validation.warning.source_count_high",
          count=len(sources),
        )
      )

    # Check date range consistency
    dates = []
    for record in records:
      if hasattr(record, "start_date") and record.start_date:
        dates.append(record.start_date)

    if dates:
      min_date = min(dates)
      max_date = max(dates)
      date_range_days = (max_date - min_date).days

      result.set_consistency_check(
        "reasonable_date_range", date_range_days <= 3650
      )  # 10 years
      if date_range_days > 3650:
        result.add_warning(
          self.translator.t(
            "validation.warning.date_range_long",
            days=date_range_days,
          )
        )

  def _validate_cross_record_consistency(
    self, records: Sequence[AnyRecord], result: ValidationResult
  ) -> None:
    """Validate consistency between different types of records."""
    # Group records by date for cross-validation
    records_by_date = defaultdict(lambda: defaultdict(list))

    for record in records:
      if hasattr(record, "start_date") and record.start_date:
        date_key = record.start_date.date()
        record_type = getattr(record, "record_type", "unknown")
        records_by_date[date_key][record_type].append(record)

    # Check for logical consistency between heart rate and workouts
    for _date, date_records in records_by_date.items():
      heart_rates = date_records.get("HKQuantityTypeIdentifierHeartRate", [])
      # Check for any workout records (they have "Workout:" prefix)
      workouts = []
      for record_type, records_list in date_records.items():
        if record_type.startswith("Workout:"):
          workouts.extend(records_list)

      if heart_rates and workouts:
        # Check if heart rate during workouts is reasonable
        hr_values = [
          getattr(r, "value", None)
          for r in heart_rates
          if hasattr(r, "value") and isinstance(getattr(r, "value", None), (int, float))
        ]
        if hr_values:
          # Filter out None values and ensure we have valid numbers
          valid_hr_values = [
            v for v in hr_values if v is not None and isinstance(v, (int, float))
          ]
          if valid_hr_values:
            max_hr = max(valid_hr_values)
            if max_hr > 0 and max_hr < 120:
              result.add_warning(
                self.translator.t(
                  "validation.warning.low_hr_workout_day",
                  value=max_hr,
                ),
                "cross_record",
              )


def validate_health_data(
  records: Sequence[AnyRecord],
  enable_outlier_detection: bool = True,
  enable_consistency_checks: bool = True,
  locale: str | None = None,
) -> ValidationResult:
  """Convenience function for validating health data.

  Args:
      records: List of health records to validate
      enable_outlier_detection: Whether to perform outlier detection
      enable_consistency_checks: Whether to perform consistency checks

  Returns:
      ValidationResult with detailed validation information
  """
  validator = DataValidator(locale=locale)
  return validator.validate_records_comprehensive(
    records, enable_outlier_detection, enable_consistency_checks
  )
