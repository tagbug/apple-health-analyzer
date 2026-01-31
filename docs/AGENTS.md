# AGENTS

## Project Overview
Apple Health Analyzer is a CLI tool for parsing Apple Health XML exports and producing heart-rate and sleep analyses, reports, and charts.

## Architecture Map
- Entry: main.py -> src/cli.py (Click-based CLI)
- Processing: src/processors (exporter, validator, heart_rate, sleep, benchmark, optimized_processor)
- Analysis: src/analyzers (statistical, anomaly, highlights)
- Visualization: src/visualization (reports, charts, data_converter)
- Utilities: src/utils (logger, type_conversion, record_categorizer)
- Config: src/config.py (env/.env driven)

## Key Commands
- Run CLI: `uv run python main.py [command]`
- Tests: `uv run pytest`
- Coverage: `uv run pytest --cov=src --cov-report=html`

## Data and Outputs
- Input: Apple Health export.xml
- Outputs: CSV/JSON exports, analysis reports (HTML/Markdown/Text), charts

## Logging
- Loguru-based logging with console/file handlers in src/utils/logger.py

## Conventions
- Python 3.12, Ruff formatting, Pyright type checking
- Tests live in tests/ with pytest config in pyproject.toml

## Planned i18n
- Provide zh and en for user-facing strings in CLI, logs, reports, charts
- Locale configured by config/env/CLI
