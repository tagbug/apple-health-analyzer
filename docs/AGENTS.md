# Agent Roles and Collaboration Guide

## Purpose
This document defines how agents should collaborate on this repository. It is the operational contract for BUILD work.

## Roles
- **Planner**: Produces staged plans and prioritizes work. No code changes.
- **Builder**: Implements code changes and updates tests according to the plan.
- **Reviewer**: Validates code changes, checks for regressions, and reviews documentation.

## Responsibilities
- Align changes with existing architecture and public APIs.
- Preserve behavior unless explicitly requested to change it.
- Keep documentation in English under `docs/`.
- Use English-only code comments and remove redundant comments.
- Avoid destructive operations and unapproved refactors.

## Collaboration Rules
- Read relevant modules before changing them.
- Group changes by stage; do not mix unrelated refactors.
- If uncertain about behavior or missing requirements, pause and ask the user.
- After each stage, produce the report and request confirmation.

## Code Standards
- Keep diffs minimal and consistent with current code style.
- Do not introduce non-ASCII text unless the file already uses it.
- Prefer small pure functions and clear naming.

## Test Standards
- Keep tests deterministic and focused.
- Update tests to match refactors and behavior changes.
- Use English-only comments in tests.

## Communication
- Provide short summaries and specific file references in responses.
- Propose next steps after completing each stage.
