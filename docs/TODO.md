# Work Breakdown and Status

## Stage 0: Planning Artifacts
- [x] Create `docs/PLAN.md`.
- [x] Create `docs/AGENTS.md`.
- [x] Create and maintain `docs/TODO.md`.

## Stage 1: Source Review (src/)
- [x] Review core modules (`src/core/*`, `src/config.py`).
- [x] Review processors (`src/processors/*`).
- [x] Review analyzers (`src/analyzers/*`).
- [x] Review visualization (`src/visualization/*`).
- [x] Review CLI entrypoints (`src/cli.py`, `src/cli_visualize.py`).
- [x] Produce `docs/PHASE-1-REVIEW.md`.

## Stage 2: Dead Code Removal
- [x] Identify placeholders and no-op logic.
- [x] Remove dead code with minimal behavior change.
- [x] Update tests for removed paths.
- [x] Produce `docs/PHASE-2-CLEANUP.md`.

## Stage 3: Comment Cleanup
- [ ] Replace non-English comments with English.
- [ ] Remove redundant comments.
- [ ] Add detailed algorithm notes where needed.
- [ ] Produce `docs/PHASE-3-COMMENTS.md`.

## Stage 4: Refactor and Optimization
- [ ] Minimize duplicate parsing passes.
- [ ] Improve data conversion hot paths.
- [ ] Refactor oversized functions safely.
- [ ] Produce `docs/PHASE-4-REFACTOR.md`.

## Stage 5: Tests Update
- [ ] Update unit tests for refactor changes.
- [ ] Refactor test structure and comments.
- [ ] Produce `docs/PHASE-5-TESTS.md`.

## Stage 6: Systemic Testing
- [ ] Run full test suite.
- [ ] Fix failing tests or report blockers.
- [ ] Produce `docs/PHASE-6-VERIFICATION.md`.

## Stage 7: Finalization
- [ ] Produce `docs/FINAL-REPORT.md`.
- [ ] Update `README.md` in English.
