# Phase 2 Cleanup: Dead Code and Placeholder Removal

## Scope
- Removed placeholder logic in analyzers, processors, and visualization modules while preserving behavior.
- Reduced duplicate parsing paths in CLI and aligned tests with updated behavior.

## Changes
### CLI and Parsing
- Consolidated XML parsing in the `analyze` command to avoid multiple passes over the same file and ensure single-pass classification of heart rate and sleep records.
- Added explicit typing casts in `analyze` where analyzer interfaces expect narrower types.

### Analyzers
- Implemented contextual sleep/wake anomaly detection using sleep-hour vs wake-hour statistics instead of a no-op placeholder.
- Adjusted extended analysis metrics (sleep, activity, metabolic health, stress resilience) to compute from available data rather than fixed placeholder constants.

### Processors
- Implemented record merge logic for overlapping records in the data cleaner (merge time bounds, values, and metadata for quantity/category/health records).
- Removed unused enum in configuration (duplicate `DataSourcePriority`).
- Removed no-op `pass` in benchmark timeout exception.

### Visualization
- Implemented report chart generation for heart rate and sleep reports with data preparation and output handling.

### Data Models
- Replaced abstract `record_type` placeholder with explicit `NotImplementedError`.
- Simplified record creation typing to avoid overly-restrictive inference in the parser.

### Tests
- Updated anomaly tests to reflect sleep/wake detection being implemented.
- Removed config and data model tests that referenced the deleted `DataSourcePriority` enum.

## Files Updated
- `src/cli.py`
- `src/analyzers/anomaly.py`
- `src/analyzers/extended_analyzer.py`
- `src/processors/cleaner.py`
- `src/processors/benchmark.py`
- `src/visualization/charts.py`
- `src/config.py`
- `src/core/data_models.py`
- `tests/test_anomaly.py`
- `tests/test_config.py`
- `tests/test_data_models.py`

## Notes
- `src/analyzers/extended_analyzer.py`, `src/processors/optimized_processor.py`, and `src/utils/type_conversion.py` remain in place because they are covered by tests or referenced by other modules.
- Remaining language normalization and comment cleanup are deferred to Stage 3.
