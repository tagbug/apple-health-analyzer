# Build Handoff Checklist

## Preconditions
- Confirm current branch and working tree status.
- Ensure dependencies are installed (`uv sync` or equivalent).

## Execution Steps (Per Refactor Slice)
1. Read the relevant slice section in `docs/PHASE2_REFACTOR_PLAN.md`.
2. Identify impacted files and tests.
3. Implement the smallest possible change set.
4. Run targeted tests for the slice.
5. Run full test suite: `uv run pytest`.
6. Record changes and update `docs/TODO.md` status.
7. Document slice outcomes in a phase note file if needed.

## Test Commands
- Targeted tests: `uv run pytest path/to/test_file.py`
- Full suite: `uv run pytest`
- Coverage: `uv run pytest --cov=src --cov-report=html`

## Stop Conditions
- Behavior change is ambiguous or user-facing output changes unexpectedly.
- Tests fail repeatedly without a safe fix.
- Changes require new external dependencies or credentials.

## Escalation
- Ask the user before any behavior change that affects outputs, formats, or public APIs.
- Report blockers with reproduction steps and observed error output.
