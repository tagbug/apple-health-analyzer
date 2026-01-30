# Apple Health Analyzer: Planning Blueprint

## Scope and Working Rules
- This plan covers code review, dead-code removal, comment cleanup, minimal refactoring, test alignment, and full test validation.
- All documentation outputs live in `docs/` and are written in English.
- Each stage produces a dedicated report and requires user confirmation before moving to the next stage.
- If blockers or uncertain changes arise, pause and ask the user.

## Stage 0: Project Baseline and Planning Artifacts
Deliverables:
- `docs/AGENTS.md` with roles, responsibilities, and collaboration rules.
- `docs/TODO.md` with staged tasks and completion tracking.
- `docs/PLAN.md` (this file).
Notes:
- Map repository structure, entrypoints, and core data flow.

## Stage 1: Module Review and Phase Summary (src/)
Goal:
- Read every module under `src/` and summarize responsibilities, data flow, and risk areas.
Output:
- `docs/PHASE-1-REVIEW.md` with:
  - Module-by-module summary.
  - Identified debt (dead code, placeholders, inconsistent typing, duplicated logic).
  - High-risk areas for refactor and tests.

## Stage 2: Dead Code and Placeholder Removal
Goal:
- Remove dead code, placeholders, and no-op paths while preserving behavior.
Typical targets:
- Unused branches and TODOs.
- Placeholder chart/report generators.
- Duplicated or unused helpers.
Output:
- `docs/PHASE-2-CLEANUP.md` detailing removals and rationale.

## Stage 3: Comment Normalization (English-only)
Goal:
- Replace non-English comments with concise English.
- Remove redundant comments and keep only meaningful ones.
- Add detailed comments for key algorithms and non-obvious logic.
Focus areas:
- Deduplication strategies and time window logic.
- Sleep session parsing and stage mapping.
- Anomaly detection methods and thresholds.
- Statistical aggregation and trend analysis.
Output:
- `docs/PHASE-3-COMMENTS.md` summarizing comment changes.

## Stage 4: Minimal Refactor and Algorithmic Optimization
Goal:
- Apply smallest refactor that improves readability, testability, and performance.
- Reduce duplicated parsing passes and data conversions.
- Keep public APIs stable and outputs consistent.
Output:
- `docs/PHASE-4-REFACTOR.md` describing structural changes and optimizations.

## Stage 5: Test Alignment and Test Refactor
Goal:
- Update unit and integration tests to match changes.
- Refactor tests for clarity and consistent style; English comments only.
- Consolidate or split test files if it improves coverage and readability.
Output:
- `docs/PHASE-5-TESTS.md` with test updates and coverage notes.

## Stage 6: Systemic Test Run and Fixes
Goal:
- Run the full test suite and diagnose failures.
- Fix test or code issues uncovered by tests.
- Report any unresolved failures with reasoning and constraints.
Output:
- `docs/PHASE-6-VERIFICATION.md` including test commands and results summary.

## Stage 7: Final Report and README Update
Goal:
- Produce a final report of optimizations and remaining issues.
- Update `README.md` to reflect the refined structure and usage.
Output:
- `docs/FINAL-REPORT.md`.

## Confirmation Gates
- After each stage report, pause and request user confirmation before proceeding.
