# Project Plan

## Goals (unchanged)
1. Systematically analyze existing src + tests and produce staged summaries.
2. Add i18n support for logs and reports (Chinese + English).
3. Add/extend tests (unit/integration/coverage) with meaningful coverage.
4. Ask promptly when blocked; stop and report if stuck.
5. Maintain and update TODO.md after each phase.
6. Produce phase documentation and confirm before next phase.
7. Produce final report and update README at the end.

## Phase 1 - Codebase Analysis and Baseline
Scope:
- Review src/ and tests/ structure and responsibilities.
- Identify all user-facing strings in CLI, logs, reports, and charts.
Deliverables:
- Phase summary doc (docs/phase-1-analysis.md).
- Update TODO.md.

## Phase 2 - i18n Design and Integration
Scope:
- Add a lightweight i18n module (translation registry + formatter).
- Add locale selection (config/env/CLI option).
- Replace user-facing strings in CLI, logs, reports, charts with i18n keys.
- Ensure report outputs support English and Chinese.
Deliverables:
- i18n design note (docs/phase-2-i18n.md).
- Update TODO.md.

## Phase 3 - Tests and Coverage
Scope:
- Add tests for i18n translations (CLI strings and report outputs).
- Add/adjust unit and integration tests for localization.
- Ensure coverage is maintained and meaningful.
Deliverables:
- Test summary (docs/phase-3-tests.md).
- Update TODO.md.

## Phase 4 - Finalization
Scope:
- Final report (optimizations + remaining gaps).
- Update README.
Deliverables:
- Final report (docs/final-report.md).
- Updated README.
- Update TODO.md.

## Acceptance Criteria
- i18n supports zh and en for logs and reports.
- Tests pass with consistent or improved coverage.
- Each phase documented and TODO updated.
