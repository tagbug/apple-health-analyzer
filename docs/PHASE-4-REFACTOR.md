## Phase 4 - Minimal Refactor and Optimization

### Goals
- Improve readability and reuse without changing outputs or public APIs.
- Reduce duplicated data conversion and numeric normalization logic.
- Keep refactors small, local, and test-safe.

### Changes
- Added a shared DataFrame builder for heart-rate-style record conversions.
- Consolidated numeric scalar normalization using a shared helper.
- Added a fast-path deduplication flow for large batches (priority/latest) using dict grouping.

### Files Updated
- src/visualization/data_converter.py
- src/analyzers/anomaly.py
- src/analyzers/statistical.py
- src/processors/heart_rate.py
- src/processors/sleep.py
- src/processors/cleaner.py

### Notes
- No UI strings or report output strings were modified.
- Outputs should remain consistent; refactor focuses on reuse and type safety.
- Fast-path deduplication reduces DataFrame overhead for large batches while preserving strategy behavior.
