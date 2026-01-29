# Apple Health Analyzer

Apple Health 数据分析工具

## 核心功能

- **数据解析**: 流式解析 Apple Health XML 导出文件，支持多种健康数据类型
- **心率分析**: 心率趋势分析、异常检测、心率变异性评估
- **睡眠分析**: 睡眠质量评分、睡眠阶段解析、睡眠规律性分析
- **数据导出**: 支持 CSV、JSON 格式导出
- **可视化报告**: 生成交互式图表和健康洞察报告

## 安装

### 使用uv（推荐）

```bash
# 克隆项目
git clone https://github.com/tagbug/apple-health-analyzer.git
cd apple-health-analyzer

# 安装依赖
uv sync

# 激活虚拟环境
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate     # Windows
```

### 使用pip

```bash
# 安装依赖
pip install -e .
```

## 快速开始

### 获取 Apple Health 数据

1. **打开健康 App**: 在 iPhone 上打开「健康」应用
2. **访问个人资料**: 点击右上角头像 → 「导出健康数据」
3. **选择数据范围**: 建议选择「所有数据」（可能需要几分钟）
4. **等待导出**: 系统会生成包含所有健康数据的 ZIP 文件
5. **传输文件**: 将导出的 `export.xml` 文件复制到项目目录 (如 `export_data`)

### 快速上手

```bash
# 1. 克隆并安装
git clone https://github.com/tagbug/apple-health-analyzer.git
cd apple-health-analyzer
uv sync

# 2. 激活环境
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate     # Windows

# 3. 查看数据概览
uv run python main.py info export_data/export.xml

# 4. 生成完整分析报告
uv run python main.py report export_data/export.xml --age 30 --gender male

# 5. 生成可视化图表
uv run python main.py visualize export_data/export.xml -c all --interactive
```

### 典型使用场景

#### 心率趋势分析
```bash
# 生成心率分析报告
uv run python main.py report export_data/export.xml --age 30 --gender male --format html
```

#### 睡眠质量评估
```bash
# 生成睡眠专项图表
uv run python main.py visualize export_data/export.xml -c sleep_quality_trend -c sleep_stages_distribution --interactive
```

#### 数据导出分析
```bash
# 导出为 CSV 用于 Excel 分析
uv run python main.py export export_data/export.xml --format csv

# 导出为 JSON 用于程序处理
uv run python main.py export export_data/export.xml --format json
```

## 使用方法

### 命令行接口

```bash
uv run python main.py [OPTIONS] COMMAND [ARGS]...

Options:
  --config PATH    配置文件路径
  --verbose, -v    启用详细日志
  --version        显示版本信息
  --help           显示帮助信息

Commands:
  info       获取导出文件信息
  parse      解析Apple Health导出文件
  export     导出数据到各种格式
  analyze    分析心率和睡眠数据
  report     生成综合健康分析报告
  visualize  生成可视化图表
  benchmark  运行性能基准测试
```

### 解析数据

```bash
# 解析所有数据类型
uv run python main.py parse export_data/export.xml

# 只解析心率数据
uv run python main.py parse export_data/export.xml --types HKQuantityTypeIdentifierHeartRate

# 预览解析结果
uv run python main.py parse export_data/export.xml --preview

# 指定输出目录
uv run python main.py parse export_data/export.xml --output ./my_output
```

### 数据导出

```bash
# 导出为CSV（默认）
uv run python main.py export export_data/export.xml

# 导出为JSON
uv run python main.py export export_data/export.xml --format json

# 导出为CSV和JSON（同时）
uv run python main.py export export_data/export.xml --format both
```

### 数据分析

```bash
# 分析心率和睡眠数据
uv run python main.py analyze export_data/export.xml

# 指定输出目录
uv run python main.py analyze export_data/export.xml --output ./analysis_results
```

### 生成报告

```bash
# 生成完整分析报告（推荐）
uv run python main.py report export_data/export.xml --age 30 --gender male

# 生成Markdown格式报告
uv run python main.py report export_data/export.xml --format markdown --age 30 --gender male

# 生成HTML和Markdown格式报告
uv run python main.py report export_data/export.xml --format both --age 30 --gender male
```

### 生成可视化图表

```bash
# 生成所有类型的图表
uv run python main.py visualize export_data/export.xml -c all --interactive

# 生成特定图表
uv run python main.py visualize export_data/export.xml -c heart_rate_timeseries -c sleep_quality_trend --interactive

# 生成静态PNG图表
uv run python main.py visualize export_data/export.xml --static
```

### 性能基准测试

```bash
# 运行完整性能基准测试
uv run python main.py benchmark export_data/export.xml

# 指定输出目录保存测试结果
uv run python main.py benchmark export_data/export.xml --output ./benchmark_results
```

## 配置

创建`.env`文件进行配置：

```bash
# 环境设置
ENVIRONMENT=dev
DEBUG=true

# 路径配置
EXPORT_XML_PATH=../export_data/export.xml
OUTPUT_DIR=./output

# 数据源优先级（数字越大优先级越高）
APPLE_WATCH_PRIORITY=3
XIAOMI_HEALTH_PRIORITY=2
IPHONE_PRIORITY=1

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=./logs/health_analyzer.log

# 性能设置
BATCH_SIZE=1000
MEMORY_LIMIT_MB=500
```

## 开发

### VS Code 配置

项目已配置完整的VS Code开发环境支持：

1. **Pylance 类型检查**: 严格的类型检查和智能提示
2. **Ruff 代码质量**: 自动格式化和代码检查
3. **Pytest 测试**: 集成测试运行和调试
4. **调试配置**: 预配置的调试启动配置

**推荐扩展**:
- Python (Microsoft)
- Pylance (Microsoft)
- Ruff (Charlie Marsh)
- Python Debugger (Microsoft)

### 运行测试

```bash
# 运行所有测试
uv run pytest

# 运行带覆盖率的测试
uv run pytest --cov=src --cov-report=html

# 运行特定测试
uv run pytest tests/test_xml_parser.py

# 调试模式运行测试
uv run pytest --pdb tests/test_data_models.py::TestHealthRecord::test_valid_record_creation
```

### 代码质量

```bash
# 代码格式化
uv run ruff format .

# 代码检查和自动修复
uv run ruff check . --fix

# 类型检查
uv run pyright --level error
```

### 调试

使用VS Code的调试面板或命令行：

```bash
# 命令行调试
uv run python -m pdb src/cli.py info ../export_data/export.xml
```
