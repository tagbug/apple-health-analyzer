# Apple Health Analyzer

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

苹果健康数据分析工具 - Advanced Apple Health data analysis with focus on heart rate and sleep patterns.

## 功能特性

- 🚀 **流式XML解析**: 高效处理大型Apple Health导出文件（300MB+），内存占用低
- 📊 **数据分类**: 自动按数据类型分类（Activity、Heart Rate、Steps、Sleep等）
- 💾 **多格式导出**: 支持CSV、JSON等多种数据导出格式
- ❤️ **心率分析**: 深度分析心率数据，包括异常检测和趋势分析
- 😴 **睡眠分析**: 全面睡眠数据分析和可视化
- 🔄 **数据优先级**: 支持数据源优先级配置（Apple Watch > 小米运动健康 > iPhone）
- 📈 **统计聚合**: 按小时/天/周/月/季度/年等时间区间计算统计值
- 📋 **异常检测**: 自动识别和报告异常数据
- 📊 **图表生成**: 自动生成各类分析图表
- 🎯 **Highlights生成**: 基于分析结果生成健康洞察

## 安装

### 使用uv（推荐）

```bash
# 克隆项目
git clone <repository-url>
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

1. **准备数据**: 从Apple Health导出数据到`export_data/export.xml`

2. **配置环境**:
   ```bash
   cp .env.example .env
   # 编辑.env文件，设置正确的路径
   ```

3. **基本使用**:
   ```bash
   # 查看文件信息
   health-analyzer info export_data/export.xml

   # 解析数据
   health-analyzer parse export_data/export.xml --preview

   # 导出为CSV
   health-analyzer export export_data/export.xml --format csv
   ```

## 使用方法

### 命令行接口

```bash
health-analyzer [OPTIONS] COMMAND [ARGS]...

Options:
  --config PATH    配置文件路径
  --verbose, -v    启用详细日志
  --version        显示版本信息
  --help           显示帮助信息

Commands:
  info     获取导出文件信息
  parse    解析Apple Health导出文件
  export   导出数据到各种格式
  analyze  分析心率和睡眠数据
```

### 解析数据

```bash
# 解析所有数据类型
health-analyzer parse export_data/export.xml

# 只解析心率数据
health-analyzer parse export_data/export.xml --types HKQuantityTypeIdentifierHeartRate

# 预览解析结果
health-analyzer parse export_data/export.xml --preview

# 指定输出目录
health-analyzer parse export_data/export.xml --output ./my_output
```

### 数据导出

```bash
# 导出为CSV（默认）
health-analyzer export export_data/export.xml

# 导出为JSON
health-analyzer export export_data/export.xml --format json

# 导出为Parquet
health-analyzer export export_data/export.xml --format parquet
```

### 数据分析

```bash
# 分析心率和睡眠数据
health-analyzer analyze export_data/export.xml

# 指定输出目录
health-analyzer analyze export_data/export.xml --output ./analysis_results
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

## 项目结构

```
apple-health-analyzer/
├── src/
│   ├── __init__.py
│   ├── cli.py                 # 命令行接口
│   ├── config.py              # 配置管理
│   ├── core/
│   │   ├── __init__.py
│   │   ├── data_models.py     # 数据模型定义
│   │   ├── exceptions.py      # 自定义异常
│   │   └── xml_parser.py      # 流式XML解析器
│   └── utils/
│       ├── __init__.py
│       └── logger.py          # 日志系统
├── tests/                     # 单元测试
├── pyproject.toml             # 项目配置
├── README.md                  # 项目文档
├── .env.example               # 环境配置示例
└── .gitignore                 # Git忽略文件
```

## 开发

### VS Code 配置

项目已配置完整的VS Code开发环境支持：

1. **Pylance 类型检查**: 严格的类型检查和智能提示
2. **Ruff 代码质量**: 自动格式化和代码检查
3. **Pytest 测试**: 集成测试运行和调试
4. **调试配置**: 预配置的调试启动配置

**推荐扩展** (会自动提示安装):
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

# 类型检查 (通过Pylance)
# 在VS Code中自动运行，或手动运行:
uv run pyright --level error
```

### 调试

使用VS Code的调试面板或命令行：

```bash
# 命令行调试
uv run python -m pdb src/cli.py info ../export_data/export.xml

# 或使用VS Code调试配置
# F5 -> 选择调试配置
```

### 构建文档

```bash
# 安装文档依赖
uv sync --extra docs

# 构建文档
uv run mkdocs build

# 本地预览文档
uv run mkdocs serve
```

## 数据类型支持

### 心率相关
- `HKQuantityTypeIdentifierHeartRate` - 心率
- `HKQuantityTypeIdentifierRestingHeartRate` - 静息心率
- `HKQuantityTypeIdentifierHeartRateVariabilitySDNN` - 心率变异性
- `HKQuantityTypeIdentifierWalkingHeartRateAverage` - 步行平均心率
- `HKQuantityTypeIdentifierVO2Max` - 最大摄氧量

### 睡眠相关
- `HKCategoryTypeIdentifierSleepAnalysis` - 睡眠分析

### 活动相关
- `HKQuantityTypeIdentifierStepCount` - 步数
- `HKQuantityTypeIdentifierDistanceWalkingRunning` - 步行/跑步距离
- `HKQuantityTypeIdentifierActiveEnergyBurned` - 活动能量消耗

## 性能优化

- **流式解析**: 使用iterparse处理大型XML文件，避免内存溢出
- **批处理**: 支持配置批处理大小，平衡内存使用和性能
- **内存管理**: 及时清理已处理的XML元素，减少内存占用
- **多线程**: 在合适场景下使用并行处理

## 贡献

欢迎提交Issue和Pull Request！

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 致谢

- 感谢Apple提供Health数据导出功能
- 参考了开源项目[applehealth](https://github.com/tdda/applehealth)的实现思路
- 使用了优秀的开源库：pandas、pydantic、loguru等
