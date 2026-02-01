# Contributing to Apple Health Analyzer

Thank you for contributing to Apple Health Analyzer. Please read the following guidelines before submitting.

## Development Environment
```bash
git clone https://github.com/tagbug/apple-health-analyzer.git
cd apple-health-analyzer
uv sync
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

## Branching and Commit Strategy
- It is recommended to create branches from `master` or `dev`.
- Suggested branch naming: `feat/<topic>`, `fix/<topic>`, `docs/<topic>`.

## Commit Message Guidelines
Use the following format:
```
<type>: <summary>
```

Common types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.

Example:
```
docs: refresh README structure and usage
```

## PR Guidelines
PR descriptions are recommended to include the following information:
- Background and objectives of the change
- Main changes (1-3 points)
- Impact scope and risk assessment
- Test results (including commands)

## Quality Checks
It is recommended to run the following before submitting:
```bash
uv run ruff format .
uv run ruff check .
uv run pyright --level error
uv run pytest
```

## Data and Privacy
- Do not submit local data files such as `export_data`, `output`, `.env`.
- Avoid pasting real health data in Issues or PRs.