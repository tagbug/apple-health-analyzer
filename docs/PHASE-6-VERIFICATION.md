# Phase 6 - Systemic Test Run and Coverage

## Overview
- Full test suite executed with coverage reporting.
- Coverage gaps in CLI helpers and report chart generation were addressed with targeted tests.

## Commands
```bash
uv run pytest --cov=src --cov-report=term-missing --cov-report=html
```

## Results
- Tests: 453 passed.
- Coverage: 79% total.
- HTML report: `htmlcov/index.html`.

## Coverage Improvements
- Added CLI helper coverage for serialization, display helpers, and error handling.
- Added chart report generation coverage for heart rate and sleep report charts.
- Added coverage for chart helpers used by report pipelines.

## Warnings Observed
- Pydantic V2 deprecation warnings for `json_encoders` and class-based config.
- Plotly/Kaleido deprecation warning for the `engine` argument.
- Heart rate validator warning on extreme values.

## Files Added
- `tests/test_cli_helpers.py`
- `tests/test_cli_coverage.py`
- `tests/test_cli_visualize_helpers.py`
- `tests/test_charts_report_generation.py`
- `tests/test_heart_rate_coverage.py`

## Notes
- Coverage is still below the README target (74% previously, now 79%).
- Lowest coverage remains in `src/cli.py`, `src/cli_visualize.py`, and selected processor/analyzer modules.
