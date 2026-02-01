# Apple Health Analyzer

Analyze Apple Health exports with reliable heart-rate and sleep insights.

Language: English | [Chinese](/docs/README.zh.md)

## Highlights
- Streaming parser for Apple Health XML exports.
- Heart rate analytics: trends, anomalies, HRV, daily/diurnal metrics.
- Sleep analytics: quality score, latency, awakenings, stage summary.
- Export data to CSV or JSON.
- Reports and charts (interactive or static) with i18n support.

## Getting Started
### Install (uv recommended)
```bash
git clone https://github.com/tagbug/apple-health-analyzer.git
cd apple-health-analyzer
uv sync
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### Install (pip)
```bash
pip install -e .
```

### Export Apple Health Data
1. Open the Health app on iPhone.
2. Tap your profile photo and choose "Export All Health Data".
3. Copy `export.xml` into the repo (for example, `export_data`).

### First Run
```bash
uv run python main.py info export_data/export.xml
uv run python main.py analyze export_data/export.xml
uv run python main.py report export_data/export.xml --age 30 --gender male
```

### Generate Sample Data
```bash
python example/create_example_xml.py --count 2000
python example/create_example_xml.py --count 5000 --seed 12345
```

## Configuration
Create a `.env` file when you want defaults:
```bash
ENVIRONMENT=dev
DEBUG=true
EXPORT_XML_PATH=./export_data/export.xml
OUTPUT_DIR=./output
APPLE_WATCH_PRIORITY=3
XIAOMI_HEALTH_PRIORITY=2
IPHONE_PRIORITY=1
LOG_LEVEL=INFO
LOG_FILE=./logs/health_analyzer.log
BATCH_SIZE=1000
MEMORY_LIMIT_MB=500
LOCALE=en
```

## CLI Usage
```bash
uv run python main.py [OPTIONS] COMMAND [ARGS]...

Options:
  --config PATH    Configuration file path
  --verbose, -v    Enable verbose logging
  --locale [en|zh] Output locale
  --version        Show version
  --help           Show help

Commands:
  info       Show export metadata
  parse      Parse Apple Health export
  export     Export data to CSV/JSON
  analyze    Analyze heart rate and sleep data
  report     Generate comprehensive report
  visualize  Generate charts
  benchmark  Run performance benchmarks
```

## Common Tasks
### Parse
```bash
uv run python main.py parse export_data/export.xml
uv run python main.py parse export_data/export.xml --types HKQuantityTypeIdentifierHeartRate
uv run python main.py parse export_data/export.xml --preview
uv run python main.py parse export_data/export.xml --output ./my_output
```

### Export
```bash
uv run python main.py export export_data/export.xml --format csv
uv run python main.py export export_data/export.xml --format json
uv run python main.py export export_data/export.xml --format both
```

### Analyze
```bash
uv run python main.py analyze export_data/export.xml
uv run python main.py analyze export_data/export.xml --output ./analysis_results
```

### Report
```bash
uv run python main.py report export_data/export.xml --age 30 --gender male
uv run python main.py report export_data/export.xml --format markdown --age 30 --gender male
uv run python main.py report export_data/export.xml --format both --age 30 --gender male
uv run python main.py report export_data/export.xml --format html --age 30 --gender male --locale zh
```

### Visualize
```bash
uv run python main.py visualize export_data/export.xml -c all --interactive
uv run python main.py visualize export_data/export.xml -c heart_rate_timeseries -c sleep_quality_trend --interactive
uv run python main.py visualize export_data/export.xml --static
```

### Benchmark
```bash
uv run python main.py benchmark export_data/export.xml
uv run python main.py benchmark export_data/export.xml --output ./benchmark_results
uv run python main.py benchmark export_data/export.xml --timeout 60
```

## Locale and i18n
Set locale globally via `.env` or per-command with `--locale`.
```bash
# .env
LOCALE=en

# CLI override
uv run python main.py --locale zh info export_data/export.xml
```

## Development
### VS Code Setup
Recommended extensions:
- Python (Microsoft)
- Pylance (Microsoft)
- Ruff (Charlie Marsh)
- Python Debugger (Microsoft)

### Tests
```bash
uv run pytest
uv run pytest --cov=src --cov-report=html
uv run pytest tests/test_xml_parser.py
uv run pytest --pdb tests/test_data_models.py::TestHealthRecord::test_valid_record_creation
```

Coverage reports are written to `htmlcov/index.html`.

### Code Quality
```bash
uv run ruff format .
uv run ruff check . --fix
uv run pyright --level error
```

### Debugging
```bash
uv run python -m pdb src/cli.py info ./export_data/export.xml
```

### Contributing
Contributions are welcome! Please review the [CONTRIBUTING](/docs/CONTRIBUTING.en.md) for details.

## FAQ
### Is my data safe?
Your Apple Health export stays local. Do not commit `export_data` or `.env` to Git.

### The export is huge. Can I limit memory?
Set `BATCH_SIZE` and `MEMORY_LIMIT_MB` in `.env`, then rerun the CLI.
