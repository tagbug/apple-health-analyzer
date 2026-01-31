# Apple Health Analyzer

用于心率与睡眠洞察的 Apple Health 数据分析工具。

语言: 中文 | `docs/README.en.md` (English)

## 核心特性
- **流式解析器**，支持 Apple Health XML 导出文件。
- **心率分析**，包含趋势、异常信号与 HRV 评估。
- **睡眠分析**，包含质量评分与睡眠阶段汇总。
- **数据导出**，支持 CSV 与 JSON。
- **可视化报告**，支持交互式或静态图表。
- **i18n 输出**，CLI/日志/报告/图表支持中英双语。

## 安装
### 使用 uv（推荐）
```bash
git clone https://github.com/tagbug/apple-health-analyzer.git
cd apple-health-analyzer
uv sync
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### 使用 pip
```bash
pip install -e .
```

## 快速开始
### 导出 Apple Health 数据
1. 在 iPhone 上打开“健康”App。
2. 点击头像，选择“导出所有健康数据”。
3. 将 `export.xml` 拷贝到仓库中（例如 `export_data`）。

### 生成示例数据
```bash
python example/create_example_xml.py --count 2000
python example/create_example_xml.py --count 5000 --seed 12345
```

### 第一次运行
```bash
uv run python main.py info example/example.xml
uv run python main.py report example/example.xml --age 30 --gender male
uv run python main.py visualize example/example.xml -c all --interactive
```

## 语言与 i18n
可以通过 `.env` 全局设置，或使用命令行 `--locale` 覆盖。

```bash
# .env
LOCALE=zh

# CLI 覆盖
uv run python main.py --locale en info example/example.xml
```

## CLI 用法
```bash
uv run python main.py [OPTIONS] COMMAND [ARGS]...

Options:
  --config PATH    配置文件路径
  --verbose, -v    启用详细日志
  --locale [en|zh] 输出语言
  --version        显示版本
  --help           显示帮助

Commands:
  info       查看导出元数据
  parse      解析 Apple Health 导出文件
  export     导出 CSV/JSON
  analyze    分析心率与睡眠数据
  report     生成综合报告
  visualize  生成图表
  benchmark  运行性能基准测试
```

## 常用任务
### 解析数据
```bash
uv run python main.py parse export_data/export.xml
uv run python main.py parse export_data/export.xml --types HKQuantityTypeIdentifierHeartRate
uv run python main.py parse export_data/export.xml --preview
uv run python main.py parse export_data/export.xml --output ./my_output
```

### 导出数据
```bash
uv run python main.py export export_data/export.xml --format csv
uv run python main.py export export_data/export.xml --format json
uv run python main.py export export_data/export.xml --format both
```

### 分析数据
```bash
uv run python main.py analyze export_data/export.xml
uv run python main.py analyze export_data/export.xml --output ./analysis_results
```

### 生成报告
```bash
uv run python main.py report export_data/export.xml --age 30 --gender male
uv run python main.py report export_data/export.xml --format markdown --age 30 --gender male
uv run python main.py report export_data/export.xml --format both --age 30 --gender male
uv run python main.py report export_data/export.xml --format html --age 30 --gender male --locale zh
```

### 生成图表
```bash
uv run python main.py visualize export_data/export.xml -c all --interactive
uv run python main.py visualize export_data/export.xml -c heart_rate_timeseries -c sleep_quality_trend --interactive
uv run python main.py visualize export_data/export.xml --static
```

### 性能基准测试
```bash
uv run python main.py benchmark export_data/export.xml
uv run python main.py benchmark export_data/export.xml --output ./benchmark_results
uv run python main.py benchmark export_data/export.xml --timeout 60
```

## 配置
创建 `.env` 文件：
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
LOCALE=zh
```

## 开发
### VS Code 配置
推荐扩展：
- Python (Microsoft)
- Pylance (Microsoft)
- Ruff (Charlie Marsh)
- Python Debugger (Microsoft)

### 测试
```bash
uv run pytest
uv run pytest --cov=src --cov-report=html
uv run pytest tests/test_xml_parser.py
uv run pytest --pdb tests/test_data_models.py::TestHealthRecord::test_valid_record_creation
```

### 代码质量
```bash
uv run ruff format .
uv run ruff check . --fix
uv run pyright --level error
```

### 调试
```bash
uv run python -m pdb src/cli.py info ./export_data/export.xml
```
