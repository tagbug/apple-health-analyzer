# Build Plan

## Scope
- Implement refactor slices defined in `docs/PHASE2_REFACTOR_PLAN.md`.
- Run tests after each slice and preserve coverage baseline.
- Update `docs/BUILD_TODO.md` and `docs/TODO.md` after each slice.

## Slice Order
1. Priority semantics unification.
2. Parser defaults vs validation consistency.
3. CLI decomposition.
4. Record categorization utilities.
5. Public sleep session API.
6. Report generation decomposition.

## Test Cadence
- Targeted tests per slice.
- Full suite after each slice.
- Coverage report as needed to confirm baseline.

## Stop Conditions
- Any ambiguous behavior change.
- Repeated test failures without safe fix.
- Changes requiring user decision.
