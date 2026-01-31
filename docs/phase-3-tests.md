## Phase 3 - Tests and Coverage

### Scope
- Add/adjust tests for i18n translations in CLI and report outputs.
- Update config validation tests to match key-based error messages.
- Ensure test assertions align with localized output.

### What Changed
- CLI tests now assert translated output instead of English literals.
- Config validation tests assert i18n key errors from validators.
- Report generation tests force locale=zh to confirm localized output.

### Files Updated
- tests/cli/test_cli.py
- tests/cli/test_cli_commands.py
- tests/cli/test_cli_visualize_commands.py
- tests/config/test_config.py
- tests/visualization/test_reports.py

### Notes
- CLI help and error outputs are now asserted via Translator.
- Config validation errors are key-based; main.py translates them for CLI.

### Suggested Follow-up
- Run `uv run pytest` and `uv run pytest --cov=src --cov-report=html`.
