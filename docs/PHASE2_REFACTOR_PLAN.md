# Phase 2 Refactor Plan

## Objectives
- Apply best-practice refactors in small, reversible slices.
- Keep behavior stable and run full tests after each slice.
- Avoid changes that reduce test coverage or weaken test intent.

## Guiding Principles
- Prefer small, isolated changes over sweeping edits.
- Preserve public interfaces unless explicitly approved.
- Add or adjust tests only when behavior is clarified or risk is high.
- Stop and ask for confirmation if behavior might change.

## Refactor Slices

### Slice 1: Priority Semantics Unification
Goal: Align priority rules across config, models, and cleaning.
Scope:
- `src/config.py` (source priority map)
- `src/core/data_models.py` (BaseRecord.source_priority)
- `src/processors/cleaner.py` (DataCleaner source_priority rules)
Approach:
- Define a single convention: higher numeric value = higher priority.
- Update DataCleaner logic to match the shared convention.
- Add a small test update to confirm consistent behavior.
Tests:
- `tests/processors/test_cleaner.py`
- `tests/core/test_data_models.py`
Risk: Medium (behavioral change); confirm if any CLI outputs depend on old rules.

### Slice 2: Parser Defaults vs Validation Consistency
Goal: Prevent parser defaults from violating model validation.
Scope:
- `src/core/data_models.py` (QuantityRecord validation or default strategy)
- `src/core/xml_parser.py` (create_record_from_xml_element defaults)
Approach:
- Replace default numeric value `0.0` with `None` and allow QuantityRecord optional value, or
- Keep `value > 0` but skip record creation when defaults are applied.
- Select the safest option with minimal behavior change, add tests for malformed inputs.
Tests:
- `tests/core/test_data_models.py`
- `tests/core/test_xml_parser.py`
Risk: Medium (validation/parse outcomes).

### Slice 3: CLI Decomposition
Goal: Reduce large CLI functions and improve testability.
Scope:
- `src/cli.py`
- `src/cli_visualize.py`
Approach:
- Extract record loading, categorization, and report generation into helper functions.
- Keep Click command signatures stable.
- Avoid changing output text in this slice.
Tests:
- `tests/cli/test_cli.py`
- `tests/cli/test_cli_commands.py`
Risk: Low if refactor-only.

### Slice 4: Record Categorization Utilities
Goal: Remove duplicated record type mapping logic.
Scope:
- New utility module under `src/utils/` or `src/core/`.
- Update `cli.py`, `cli_visualize.py`, and report generation to use it.
Approach:
- Implement one categorizer function with explicit record type lists.
- Add unit tests for categorization.
Tests:
- New tests under `tests/utils/` or `tests/core/`.
Risk: Low to Medium.

### Slice 5: Public Sleep Session API
Goal: Avoid private method usage and improve API clarity.
Scope:
- `src/processors/sleep.py`
- `src/cli_visualize.py`
Approach:
- Add a public `parse_sleep_sessions` wrapper in SleepAnalyzer.
- Replace calls to `_parse_sleep_sessions` with public method.
Tests:
- `tests/processors/test_sleep.py` (update or add coverage)
Risk: Low.

### Slice 6: Report Generation Decomposition
Goal: Separate template/layout from data assembly.
Scope:
- `src/visualization/reports.py`
Approach:
- Extract HTML template building into dedicated helper methods or template strings.
- Isolate data formatting functions.
- Avoid modifying report content in this slice.
Tests:
- `tests/visualization/test_reports.py` (update if needed)
Risk: Medium (large file, formatting risk).

## Test Strategy per Slice
- Run targeted tests for changed modules.
- Run full `pytest` after each slice.
- If any test is flaky or unclear, stop and request confirmation.

## Stop Conditions
- Ambiguous behavior changes.
- Conflicting tests or unclear intent.
- Repeated test failures without a safe fix.
