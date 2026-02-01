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
### Info
```bash
uv run python main.py info export_data/export.xml
```

**Example output:**
```
Analyzing file: export_data/export.xml
               File Information               
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Property          ┃ Value                  ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ File Path         │ export_data/export.xml │
│ File Size         │ 0.00 MB                │
│ Estimated Records │ 13                     │
│ Last Modified     │ 1769943888.4648178     │
└───────────────────┴────────────────────────┘

Data date range (sample): 2024-01-01 to 2024-01-02

Record types in sample:
  HKCategoryTypeIdentifierSleepAnalysis: 5
  HKQuantityTypeIdentifierHeartRate: 4
  HKQuantityTypeIdentifierStepCount: 2
  HKQuantityTypeIdentifierHeartRateVariabilitySDNN: 1
  HKQuantityTypeIdentifierRestingHeartRate: 1
```

### Parse
```bash
uv run python main.py parse export_data/export.xml
uv run python main.py parse export_data/export.xml --types HKQuantityTypeIdentifierHeartRate
uv run python main.py parse export_data/export.xml --preview
uv run python main.py parse export_data/export.xml --output ./my_output
```

**Example output (--preview):**
```
Parsing Results               
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric        ┃                    Value ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Total Records │                       13 │
│ Processed     │                       13 │
│ Skipped       │                        0 │
│ Invalid       │                        0 │
│ Success Rate  │                   100.0% │
│ Date Range    │ 2024-01-01 to 2024-01-02 │
└───────────────┴──────────────────────────┘

Top Record Types:
   1. HKCategoryTypeIdentifierSleepAnalysis: 5
   2. HKQuantityTypeIdentifierHeartRate: 4
   3. HKQuantityTypeIdentifierStepCount: 2

✓ Parsing completed successfully!
Processed 13 records with 100.0% success rate
```

### Export
```bash
uv run python main.py export export_data/export.xml --format csv
uv run python main.py export export_data/export.xml --format json
uv run python main.py export export_data/export.xml --format both
```

**Files generated (CSV format):**
```
output/
├── HeartRate.csv
├── SleepAnalysis.csv
├── StepCount.csv
├── HeartRateVariabilitySDNN.csv
├── RestingHeartRate.csv
└── manifest.json
```

### Analyze
```bash
uv run python main.py analyze export_data/export.xml --age 30 --gender male
uv run python main.py analyze export_data/export.xml --output ./analysis_results
```

**Example output:**
```
🎯 Analysis Results

❤️ Heart Rate Analysis
  Resting HR: 62.0 bpm
  Trend: stable
  Health Rating: excellent
  HRV (SDNN): 45.0 ms
  Stress Level: moderate
  Recovery Status: good
  Data Quality: 100.0%
  Total Records: 4

😴 Sleep Analysis
  Average Duration: 3.5 hours
  Average Efficiency: 85.4%
  Consistency Score: 44.1%
  Data Quality: 92.9%
  Total Records: 5

💡 Health Insights

Key Insights:
  1. Insufficient sleep duration
     Average sleep duration is only 3.5 hours; aim for 7-9 hours
  2. Irregular sleep schedule
     Sleep timing is irregular; maintain a consistent schedule
  3. Excellent heart rate health
     Resting heart rate is 62 bpm, an excellent level

Recommendations:
  1. Get 7-9 hours of sleep each night and avoid staying up late
  2. Maintain a regular sleep schedule, including weekends
  3. Keep consistent wake and bedtime, even on weekends

✓ Analysis completed! Results saved to: output
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

**Example output:**
```
✅ Chart generation completed!
Files generated: 1
Output directory: output/charts

Generated files:
  • heart_rate_timeseries.png (0.05 MB)
  • sleep_quality_trend.png (0.04 MB)

Chart index: output/charts/index.md
```

### Benchmark
```bash
uv run python main.py benchmark export_data/export.xml
uv run python main.py benchmark export_data/export.xml --output ./benchmark_results
uv run python main.py benchmark export_data/export.xml --timeout 60
```

**Example output:**
```
🔍 Module Performance                                          
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Module               ┃ Status ┃   Time (s) ┃    Records ┃ Throughput (records/s) ┃ Memory delta (MB) ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ XML parsing          │   ✅   │       0.00 │         13 │                  5,233 │             +0.00 │
│ Data cleaning        │   ✅   │       0.02 │         13 │                    793 │             +2.72 │
│ Statistical analysis │   ✅   │       0.02 │         13 │                    553 │             +1.29 │
│ Report generation    │   ✅   │       0.00 │         13 │                 13,000 │             +0.00 │
│ Data export          │   ✅   │       0.00 │         13 │                  2,790 │             +0.54 │
└──────────────────────┴────────┴────────────┴────────────┴────────────────────────┴───────────────────┘

💡 Bottleneck analysis:
  ⚠️  Statistical analysis is slowest (0.02s)
  ⚠️  Statistical analysis has lowest throughput (553 records/s)

✅ Completed at: 2026-02-01 11:07:46
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
