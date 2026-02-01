# Apple Health Analyzer

面向心率与睡眠洞察的 Apple Health 数据分析工具。

语言: 中文 | [English](docs/README.en.md)

## 亮点
- 流式解析 Apple Health XML 导出文件。
- 心率分析：趋势、异常信号、HRV、日/昼夜指标。
- 睡眠分析：质量评分、入睡延迟、觉醒与阶段汇总。
- 数据导出：CSV 与 JSON。
- 报告与图表（交互或静态），支持中英文输出。

## 快速开始
### 安装（推荐 uv）
```bash
git clone https://github.com/tagbug/apple-health-analyzer.git
cd apple-health-analyzer
uv sync
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### 安装（pip）
```bash
pip install -e .
```

### 导出 Apple Health 数据
1. 在 iPhone 上打开“健康”App。
2. 点击头像，选择“导出所有健康数据”。
3. 将 `export.xml` 放入仓库（例如 `export_data`）。

### 第一次运行
```bash
uv run python main.py info export_data/export.xml
uv run python main.py analyze export_data/export.xml
uv run python main.py report export_data/export.xml --age 30 --gender male
```

### 生成示例数据
```bash
python example/create_example_xml.py --count 2000
python example/create_example_xml.py --count 5000 --seed 12345
```

## 配置
需要默认参数时创建 `.env`：
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
### 解析
```bash
uv run python main.py parse export_data/export.xml
uv run python main.py parse export_data/export.xml --types HKQuantityTypeIdentifierHeartRate
uv run python main.py parse export_data/export.xml --preview
uv run python main.py parse export_data/export.xml --output ./my_output
```

### 导出
```bash
uv run python main.py export export_data/export.xml --format csv
uv run python main.py export export_data/export.xml --format json
uv run python main.py export export_data/export.xml --format both
```

### 分析
```bash
uv run python main.py analyze export_data/export.xml
uv run python main.py analyze export_data/export.xml --output ./analysis_results
```

### 报告
```bash
uv run python main.py report export_data/export.xml --age 30 --gender male
uv run python main.py report export_data/export.xml --format markdown --age 30 --gender male
uv run python main.py report export_data/export.xml --format both --age 30 --gender male
uv run python main.py report export_data/export.xml --format html --age 30 --gender male --locale zh
```

### 图表
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

## 语言与 i18n
可通过 `.env` 全局设置，或使用命令行 `--locale` 覆盖：
```bash
# .env
LOCALE=zh

# CLI 覆盖
uv run python main.py --locale en info export_data/export.xml
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

覆盖率报告输出至 `htmlcov/index.html`。

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

### 贡献
欢迎贡献！请参阅 [CONTRIBUTING.zh.md](CONTRIBUTING.zh.md) 了解详情。

## 常见问题
### 数据安全吗？
Apple Health 导出数据仅在本地处理，请勿提交 `export_data` 或 `.env`。

### 导出很大，怎么控制内存？
在 `.env` 设置 `BATCH_SIZE` 与 `MEMORY_LIMIT_MB` 后重试。
