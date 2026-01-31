# Phase 1 - Codebase Analysis

## Scope
Reviewed repository structure, core modules, tests, and user-facing strings in CLI/logging/reports/charts to establish the baseline for i18n and testing work.

## Repository Structure
- Entry point: `main.py` calls `src/cli.py`.
- CLI: `src/cli.py` defines Click commands for parse/export/analyze/report/visualize/benchmark and prints rich output.
- Core processing: `src/processors/*` (heart_rate, sleep, exporter, validator, benchmark, optimized_processor).
- Analysis: `src/analyzers/*` (statistical, anomaly, highlights).
- Visualization: `src/visualization/*` (reports, charts, data_converter).
- Utilities: `src/utils/*` (logger, type_conversion, record_categorizer).
- Config: `src/config.py` loads env/.env configuration.
- Tests: `tests/cli`, `tests/processors`, `tests/utils`, `tests/integration`.

## Data Flow Overview
1. CLI loads config and parses XML (streaming parser) for selected record types.
2. Records are categorized (heart rate, sleep, HRV, etc.).
3. Processors run analysis (statistics, anomalies, reports) and generate outputs.
4. Visualization layer creates reports (HTML/Markdown) and charts (Plotly/Matplotlib).
5. Exporter writes CSV/JSON and manifests; validation emits quality reports.

## User-Facing Strings Inventory (high-level)
### CLI and Rich Output
- `src/cli.py`: command help strings, errors, warnings, tips, progress messages, and table headers.
- CLI output is a mix of English and Chinese; many messages are direct string literals.

### Reports
- `src/visualization/reports.py`:
  - Markdown and HTML report titles, section headers, labels, and narrative text.
  - Large blocks of Chinese text in HTML/Markdown output.

### Charts
- `src/visualization/charts.py`:
  - Titles, axis labels, annotations, legends, hover templates, and category labels.
  - Many Chinese strings embedded in Plotly/Matplotlib definitions.

### Logging
- `src/utils/logger.py` for formatting and lifecycle messages.
- Log messages across processors/analyzers are user-visible in console/file logs.
- Logging strings are mostly English, but some processors embed Chinese content.

## Tests Summary
- CLI tests: `tests/cli/*` validate CLI commands and helper behavior.
- Processor tests: `tests/processors/*` cover exporter, heart rate, sleep, validator, benchmark, cleaner.
- Utils tests: `tests/utils/*` cover logger and type conversion.
- Integration tests: `tests/integration/*` cover end-to-end parsing/analysis flows.
- No explicit i18n coverage yet.

## Risks and Observations
- i18n touches multiple layers: CLI, logs, reports, charts.
- HTML templates and Plotly hover text contain many Chinese literals.
- Some CLI outputs embed emojis and formatting tags that should be preserved with translation tokens.
- Both `README.md` and `docs/README.md` exist and are identical (possible duplication).

## Next Phase Entry Criteria
- Establish an i18n design with clear translation keys and a single formatting helper.
- Decide locale selection strategy (config/env/CLI) and default behavior.
- Plan incremental refactor to avoid breaking CLI output or report layout.
