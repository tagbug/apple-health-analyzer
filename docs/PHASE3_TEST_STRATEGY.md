# Phase 3 Test Strategy

## Goals
- Ensure tests reflect real behavior, not coverage targets.
- Preserve or improve current coverage baseline.
- Define a clear, repeatable test execution matrix.

## Current Test Landscape (Observed)
- Unit tests: core models, processors, analyzers, utilities.
- CLI tests: command surface and helpers.
- Integration tests: end-to-end behavior.
- Visualization tests: data conversion and chart/report generation.

## Test Principles
- Test behavior, not implementation details.
- Avoid relying on private methods; prefer public APIs.
- Keep fixtures small and explicit.
- Favor deterministic tests (no randomness unless seeded).
- Prefer targeted tests for refactor slices, then full suite.

## Coverage Baseline Policy
- Maintain current coverage percentage or higher.
- Any reduction must be justified with clear rationale and approval.

## Test Execution Matrix

### During Refactor Slices
1. Targeted tests for the affected module(s).
2. Full test suite after each slice.

### Recommended Commands
- Unit + integration: `uv run pytest`
- Coverage report: `uv run pytest --cov=src --cov-report=html`

## Test Validity Checklist
- Does the test confirm a user-visible or domain-meaningful behavior?
- Is the test resilient to refactors (avoids internal/private calls)?
- Are assertions focused and non-redundant?
- Are error cases meaningful (not testing trivial failures)?

## Risk Controls
- If a refactor changes output format or text, add or adjust tests only if user-visible behavior is intended to change.
- If a test is brittle, refactor the test to validate the outcome rather than the exact internal path.

## Stop and Escalation
- Ambiguous behavior changes or conflicting test intent -> stop and ask.
- Repeated failures without a clear fix -> stop and report.
