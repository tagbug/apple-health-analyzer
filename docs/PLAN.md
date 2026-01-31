# Plan

## Scope and Constraints
- Planning only in this phase; no code changes.
- All documentation lives under docs/.
- Conversation in Chinese; documents in English.
- Build phase may run tests (pytest/pytest-cov).

## Repository Overview
- Source modules in src/: core, processors, analyzers, visualization, utils, cli.
- Tests in tests/: unit, integration, cli, visualization, processors, analyzers.
- Tooling: Ruff, MyPy, Pyright, Pytest, pytest-cov.

## Phase 1: Systematic Analysis (src + tests)
1. Map modules, responsibilities, and dependencies.
2. Review coding style consistency and identify code smells.
3. Identify architectural issues (tight coupling, duplication, unclear boundaries).
4. Produce Phase 1 analysis report with ranked refactor candidates.

Deliverables:
- docs/PHASE1_ANALYSIS.md
- docs/TODO.md updated with Phase 1 items marked.

## Phase 2: Incremental Refactor Plan
1. Break refactor work into small, reversible slices.
2. For each slice, define invariants and expected behavior.
3. Run full test suite after each slice in build phase.

Deliverables:
- docs/PHASE2_REFACTOR_PLAN.md
- docs/TODO.md updated with Phase 2 items marked.

## Phase 3: Test Validity and Coverage Strategy
1. Evaluate test intent and correctness; avoid tests written only for coverage.
2. Keep or improve current coverage baseline.
3. Define validation matrix for unit, integration, and coverage runs.

Deliverables:
- docs/PHASE3_TEST_STRATEGY.md
- docs/TODO.md updated with Phase 3 items marked.

## Phase 4: Build Handoff
1. Provide a step-by-step execution checklist to BUILD model.
2. Define stop conditions and escalation rules.

Deliverables:
- docs/BUILD_HANDOFF.md

## Phase 5: Final Report and README Update (post-build)
1. Summarize optimizations and remaining issues.
2. Update README with changes to behavior or usage.

Deliverables:
- docs/FINAL_REPORT.md
- README.md updated (post-build).

## Escalation Rules
- If a change is uncertain or impacts behavior ambiguously, ask the user.
- If repeated failures occur, stop and report with context.
