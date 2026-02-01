# AGENTS

This repository uses a two-stage workflow:

- PLANNING model: Read-only analysis, produces docs/PLAN.md and docs/TODO.md.
- BUILD model: Implements changes strictly according to the plan and keeps docs/TODO.md updated after each phase.

## Responsibilities

### PLANNING model
- Analyze src/ and tests/ structure.
- Identify improvements for analyzers and visualization modules.
- Produce the plan and the initial TODO list.
- Ask for clarification only when blocked.

### BUILD model
- Implement improvements in analyzers and visualization.
- Add or update tests (unit, integration, coverage) with meaningful assertions.
- Update docs/TODO.md after each phase and mark completed tasks.
- Produce phase documentation and final report.
- Update README only after all phases complete.

## Constraints
- All markdown documents must live in docs/.
- Do not commit secrets or local artifacts.
- Keep changes aligned with existing style and i18n patterns.
