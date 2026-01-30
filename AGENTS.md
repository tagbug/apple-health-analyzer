# Apple Health Analyzer - Developer Guide & Agent Rules

This document outlines the development workflows, code standards, and project structure for the `apple-health-analyzer` repository. AI agents and developers should follow these guidelines strictly.

## 1. Environment & Commands

This project uses `uv` for dependency management and task execution.

### Setup
- **Install dependencies:** `uv sync`
- **Activate environment:** `source .venv/bin/activate` (Linux/Mac) or `.venv\Scripts\activate` (Windows)

### Testing
Use `pytest` for all testing. Tests are located in the `tests/` directory.

- **Run all tests:**
  ```bash
  uv run pytest
  ```
- **Run a specific test file:**
  ```bash
  uv run pytest tests/test_xml_parser.py
  ```
- **Run a single test case (by name):**
  ```bash
  uv run pytest -k test_valid_record_creation
  ```
- **Run with coverage:**
  ```bash
  uv run pytest --cov=src --cov-report=term-missing
  ```
- **Debug tests (drop into PDB on failure):**
  ```bash
  uv run pytest --pdb
  ```

### Linting & Formatting
The project uses `ruff` for both linting and formatting, and `pyright` for static type checking.

- **Format code:**
  ```bash
  uv run ruff format .
  ```
- **Lint code (check only):**
  ```bash
  uv run ruff check .
  ```
- **Lint and fix auto-fixable issues:**
  ```bash
  uv run ruff check . --fix
  ```
- **Static Type Check:**
  ```bash
  uv run pyright
  ```
  *Note: `mypy` configuration is also present in `pyproject.toml`, but `pyright` is the primary tool referenced in documentation.*

### Running the Application
Entry point is `src/cli.py` (aliased as `main.py` in root).

- **Parse data:**
  ```bash
  uv run python main.py parse export_data/export.xml
  ```
- **Generate report:**
  ```bash
  uv run python main.py report export_data/export.xml --age 30 --gender male
  ```

## 2. Code Style & Conventions

### General
- **Python Version:** 3.12+
- **Line Length:** 88 characters (enforced by Ruff).
- **Quotes:** Double quotes `"` preferred over single quotes (enforced by Ruff/Black format).

### Imports
- **Sorting:** Imports are automatically sorted by `ruff` (isort rules).
- **Structure:**
  1. Standard library imports
  2. Third-party library imports
  3. Local application imports (`src`)
- **Absolute Imports:** Always use absolute imports for `src` modules (e.g., `from src.core.data_models import ...`) rather than relative imports where possible, though relative imports within sub-packages are acceptable if clear.

### Typing & Data Models
- **Strong Typing:** All function signatures must have type hints.
- **Pydantic:** Use `pydantic` (v2) for all data validation, configuration, and domain models.
  - Models should generally be immutable (`frozen=True`) where applicable.
  - Field validation should be done using `@field_validator` or `@model_validator`.
- **Runtime Checks:** Do not rely solely on static analysis; use Pydantic models to parse and validate external data (XML, JSON).

### Naming Conventions
- **Variables/Functions:** `snake_case`
- **Classes/Types:** `PascalCase`
- **Constants:** `UPPER_CASE`
- **Private members:** Prefix with `_` (e.g., `_internal_helper`).

### Error Handling
- **Custom Exceptions:** Use exceptions defined in `src/core/exceptions.py`.
- **Try/Except:** Catch specific exceptions, never bare `except:`.
- **Context:** When re-raising exceptions, use `raise ... from e` to preserve the stack trace.

### Logging
- **Library:** Use `loguru` for all logging. **Do not** use Python's built-in `logging` module.
- **Usage:**
  ```python
  from loguru import logger
  logger.info("Processing started")
  logger.error("Failed to parse file: {}", file_path)
  ```

## 3. Project Structure

- **`src/core/`**: Core domain logic, data models (`data_models.py`), and base parsers (`xml_parser.py`).
- **`src/processors/`**: Business logic for processing specific health data types (heart rate, sleep) and exporting data.
- **`src/analyzers/`**: Statistical analysis, anomaly detection, and generating insights.
- **`src/visualization/`**: Plotly/Matplotlib charting logic and report generation.
- **`src/utils/`**: Shared utilities (logger, type conversion).
- **`src/cli.py`**: Command-line interface entry point (using `click`).
- **`tests/`**: Mirror structure of `src/` for unit and integration tests.

## 4. Agent Operational Rules

1.  **Read First:** Always read `pyproject.toml` and related code files before editing to understand context.
2.  **Test Driven:** When fixing bugs, create a reproduction test case in `tests/` first.
3.  **No Hallucinations:** Do not invent import paths. Check `src/` structure using `ls -R` or similar if unsure.
4.  **Configuration:** Respect `pyproject.toml` settings. Do not modify tool configurations (ruff, pyright) unless explicitly requested.
5.  **Language:** Code comments and commit messages should be in English. User communication can be in the user's preferred language (Chinese/English).
