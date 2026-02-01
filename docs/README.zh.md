# Apple Health Analyzer

面向心率与睡眠洞察的 Apple Health 数据分析工具。

语言: 中文 | [English](/docs/README.en.md)

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
### 信息
```bash
uv run python main.py info export_data/export.xml
```

**输出示例：**
```
正在分析文件: export_data/export.xml
                文件信息                 
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 属性         ┃ 值                     ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 文件路径     │ export_data/export.xml │
│ 文件大小     │ 0.00 MB                │
│ 预估记录数   │ 13                     │
│ 最后修改时间 │ 1769943888.4648178     │
└──────────────┴────────────────────────┘

数据日期范围 (样本): 2024-01-01 至 2024-01-02

样本中的记录类型:
  HKCategoryTypeIdentifierSleepAnalysis: 5
  HKQuantityTypeIdentifierHeartRate: 4
  HKQuantityTypeIdentifierStepCount: 2
  HKQuantityTypeIdentifierHeartRateVariabilitySDNN: 1
  HKQuantityTypeIdentifierRestingHeartRate: 1
```

### 解析
```bash
uv run python main.py parse export_data/export.xml
uv run python main.py parse export_data/export.xml --types HKQuantityTypeIdentifierHeartRate
uv run python main.py parse export_data/export.xml --preview
uv run python main.py parse export_data/export.xml --output ./my_output
```

**输出示例（--preview）：**
```
解析结果                
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 指标     ┃                     数值 ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 记录总数 │                       13 │
│ 已处理   │                       13 │
│ 已跳过   │                        0 │
│ 无效记录 │                        0 │
│ 成功率   │                   100.0% │
│ 日期范围 │ 2024-01-01 至 2024-01-02 │
└──────────┴──────────────────────────┘

记录类型 Top:
   1. HKCategoryTypeIdentifierSleepAnalysis: 5
   2. HKQuantityTypeIdentifierHeartRate: 4
   3. HKQuantityTypeIdentifierStepCount: 2

✓ 解析完成成功!
处理了 13 条记录，成功率 100.0%
```

### 导出
```bash
uv run python main.py export export_data/export.xml --format csv
uv run python main.py export export_data/export.xml --format json
uv run python main.py export export_data/export.xml --format both
```

**生成的文件（CSV 格式）：**
```
output/
├── HeartRate.csv
├── SleepAnalysis.csv
├── StepCount.csv
├── HeartRateVariabilitySDNN.csv
├── RestingHeartRate.csv
└── manifest.json
```

### 分析
```bash
uv run python main.py analyze export_data/export.xml --age 30 --gender male
uv run python main.py analyze export_data/export.xml --output ./analysis_results
```

**输出示例：**
```
🎯 分析结果

❤️ 心率分析
  静息心率: 62.0 bpm
  趋势: stable
  健康评级: excellent
  HRV (SDNN): 45.0 ms
  压力水平: moderate
  恢复状态: good
  数据质量: 100.0%
  记录总数: 4

😴 睡眠分析
  平均时长: 3.5 hours
  平均效率: 85.4%
  规律性评分: 44.1%
  数据质量: 92.9%
  记录总数: 5

💡 健康洞察

关键洞察:
  1. 睡眠时长不足
     平均睡眠时长仅3.5小时，建议保证7-9小时睡眠
  2. 睡眠规律性差
     睡眠时间不规律，建议保持固定的作息时间
  3. 心率健康优秀
     静息心率为62 bpm，处于优秀水平

健康建议:
  1. 保证每晚7-9小时的睡眠时间，避免熬夜
  2. 建立规律的作息时间表，包括周末
  3. 保持固定的起床和睡觉时间，即使在周末

✓ 分析完成! 结果已保存至: output
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

**输出示例：**
```
✅ 图表生成完成!
生成文件数: 1
输出目录: output/charts

生成的文件:
  • heart_rate_timeseries.png (0.05 MB)
  • sleep_quality_trend.png (0.04 MB)

图表索引: output/charts/index.md
```

### 性能基准测试
```bash
uv run python main.py benchmark export_data/export.xml
uv run python main.py benchmark export_data/export.xml --output ./benchmark_results
uv run python main.py benchmark export_data/export.xml --timeout 60
```

**输出示例：**
```
🔍 模块性能                                           
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ 模块                 ┃  状态  ┃   耗时 (s) ┃     记录数 ┃       吞吐 (条/秒) ┃  内存变化 (MB) ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ XML parsing          │   ✅   │       0.00 │         13 │              5,111 │          +0.00 │
│ Data cleaning        │   ✅   │       0.02 │         13 │                789 │          +2.58 │
│ Statistical analysis │   ✅   │       0.02 │         13 │                544 │          +1.12 │
│ Report generation    │   ✅   │       0.00 │         13 │             13,000 │          +0.00 │
│ Data export          │   ✅   │       0.01 │         13 │              1,375 │          +0.59 │
└──────────────────────┴────────┴────────────┴────────────┴────────────────────┴────────────────┘

💡 瓶颈分析:
  ⚠️  Statistical analysis 最慢 (0.02s)
  ⚠️  Statistical analysis 吞吐最低 (544 条/秒)

✅ 完成时间: 2026-02-01 11:06:01
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
欢迎贡献！请参阅 [贡献指南](/docs/CONTRIBUTING.zh.md) 了解详情。

## 常见问题
### 数据安全吗？
Apple Health 导出数据仅在本地处理，请勿提交 `export_data` 或 `.env`。

### 导出很大，怎么控制内存？
在 `.env` 设置 `BATCH_SIZE` 与 `MEMORY_LIMIT_MB` 后重试。
