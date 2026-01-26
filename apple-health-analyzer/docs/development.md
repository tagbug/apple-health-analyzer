# 开发文档

## 项目概述

Apple Health Analyzer 是一个专业级的 Apple Health 数据分析工具，专注于深度分析心率、睡眠等健康数据，生成个性化健康洞察报告。

## 架构设计

### 核心架构

```
apple-health-analyzer/
├── src/                          # 源代码目录
│   ├── cli.py                    # 主命令行接口
│   ├── cli_visualize.py          # 可视化命令接口
│   ├── config.py                 # 配置管理
│   ├── analyzers/                # 分析器模块
│   │   ├── anomaly.py            # 异常检测分析器
│   │   ├── highlights.py         # 健康洞察生成器
│   │   └── statistical.py        # 统计分析器
│   ├── core/                     # 核心模块
│   │   ├── data_models.py        # Pydantic 数据模型
│   │   ├── exceptions.py         # 自定义异常
│   │   └── xml_parser.py         # 流式 XML 解析器
│   ├── processors/               # 数据处理器
│   │   ├── cleaner.py            # 数据清洗器
│   │   ├── exporter.py           # 数据导出器
│   │   ├── heart_rate.py         # 心率处理器
│   │   └── sleep.py              # 睡眠处理器
│   ├── utils/                    # 工具模块
│   │   └── logger.py             # 日志系统
│   └── visualization/            # 可视化模块
│       ├── charts.py             # 图表生成器
│       ├── data_converter.py     # 数据转换器
│       └── reports.py            # 报告生成器
└── tests/                        # 测试套件
```

### 设计原则

1. **模块化设计**: 各组件独立，职责单一
2. **流式处理**: 大文件处理时内存使用恒定
3. **类型安全**: 广泛使用类型注解
4. **错误隔离**: 单条数据错误不影响整体处理
5. **可扩展性**: 易于添加新的数据类型和分析功能

## 核心组件详解

### XML 解析器 (xml_parser.py)

**功能**: 流式解析 Apple Health XML 导出文件

**关键特性**:
- 使用 `iterparse` 实现内存高效的流式解析
- 支持多种数据类型自动识别和分类
- 异常数据自动跳过，不影响整体解析
- 进度回调支持大文件处理状态监控

**使用示例**:
```python
from src.core.xml_parser import AppleHealthXMLParser

parser = AppleHealthXMLParser()
records = parser.parse_file("export.xml")

for record in records:
    print(f"Type: {record.type}, Value: {record.value}")
```

### 数据模型 (data_models.py)

**功能**: 定义健康数据的结构化表示

**核心模型**:
- `HealthRecord`: 基础健康记录
- `HeartRateRecord`: 心率数据
- `SleepRecord`: 睡眠数据
- `WorkoutRecord`: 运动数据

**特性**:
- 使用 Pydantic 进行数据验证
- 自动类型转换和验证
- 支持 JSON 序列化

### 分析器模块

#### 异常检测分析器 (anomaly.py)

**算法**:
- Z-score 方法: 基于标准差的异常检测
- IQR 方法: 基于四分位距的鲁棒异常检测
- 滑动窗口分析: 时间序列异常检测

**使用场景**:
- 识别异常心率值
- 检测睡眠模式异常
- 运动数据异常识别

#### 统计分析器 (statistical.py)

**功能**:
- 时间序列聚合分析
- 趋势分析和周期性检测
- 相关性分析
- 分布统计

#### 健康洞察生成器 (highlights.py)

**功能**:
- 基于规则的健康建议生成
- 个性化洞察分析
- 健康风险评估

### 数据处理器

#### 数据清洗器 (cleaner.py)

**功能**:
- 重复数据检测和合并
- 数据源优先级处理
- 时间窗口去重
- 数据完整性验证

#### 心率处理器 (heart_rate.py)

**分析功能**:
- 心率区间分析 (脂肪燃烧、有氧、无氧)
- 心率变异性 (HRV) 计算
- 运动心率分析
- 静息心率趋势

#### 睡眠处理器 (sleep.py)

**分析功能**:
- 睡眠阶段解析
- 睡眠质量评分
- 睡眠规律性分析
- 睡眠效率计算

### 可视化模块

#### 图表生成器 (charts.py)

**支持图表类型**:
- 时序图: 心率趋势、睡眠模式
- 分布图: 心率分布、睡眠阶段分布
- 相关性图: 变量关系分析
- 仪表盘: 综合健康指标展示

**特性**:
- Plotly 交互式图表
- 响应式设计
- 专业医疗配色方案

## 开发环境设置

### 环境要求

- Python 3.12+
- uv 包管理器
- VS Code (推荐)

### 开发依赖安装

```bash
# 克隆项目
git clone <repository-url>
cd apple-health-analyzer

# 安装所有依赖（包括开发依赖）
uv sync --dev

# 激活虚拟环境
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows
```

### VS Code 配置

**推荐扩展**:
- Python (Microsoft)
- Pylance (Microsoft)
- Ruff (Charlie Marsh)
- Python Debugger (Microsoft)

**工作区设置**:
```json
{
  "python.defaultInterpreterPath": "./.venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "ruff",
  "python.testing.pytestEnabled": true
}
```

## 开发工作流

### 1. 功能开发

```bash
# 创建特性分支
git checkout -b feature/new-feature

# 开发代码
# ... 编写代码 ...

# 运行测试
uv run pytest tests/ -v

# 代码格式化
uv run ruff format .

# 代码检查
uv run ruff check . --fix

# 类型检查
uv run pyright
```

### 2. 测试驱动开发

**测试策略**:
- 单元测试: 核心功能覆盖
- 集成测试: 端到端流程验证
- 性能测试: 大规模数据处理验证
- 边缘情况测试: 边界条件和错误处理

**测试覆盖目标**:
- 核心业务逻辑: 80-90%
- CLI 界面: 50%+
- 工具函数: 70%+

### 3. 代码质量保证

**代码规范**:
- PEP 8 风格指南
- Google 风格文档字符串
- 类型注解覆盖率 > 90%

**自动化检查**:
```bash
# 格式化检查
uv run ruff format --check .

# 代码质量检查
uv run ruff check .

# 类型检查
uv run pyright --level error
```

## 性能优化指南

### 内存优化

1. **流式处理**: 大文件使用流式解析
2. **分批处理**: 数据分块处理，避免大对象
3. **及时释放**: 处理完及时释放内存
4. **对象复用**: 复用频繁创建的对象

### 速度优化

1. **向量化计算**: 使用 NumPy/Pandas
2. **并发处理**: I/O 密集操作使用异步
3. **缓存策略**: 缓存重复计算结果
4. **算法优化**: 选择合适的数据结构和算法

### 监控和分析

```python
# 性能监控示例
import time
from src.utils.logger import logger

start_time = time.time()
# ... 执行操作 ...
duration = time.time() - start_time
logger.info(f"操作耗时: {duration:.2f}秒")
```

## 扩展开发指南

### 添加新的数据类型

1. **定义数据模型**:
```python
# 在 data_models.py 中添加
class NewHealthRecord(HealthRecord):
    new_field: Optional[float] = None
```

2. **实现解析器**:
```python
# 在 xml_parser.py 中添加解析逻辑
elif record_type == "HKQuantityTypeIdentifierNewType":
    record = NewHealthRecord(...)
```

3. **添加处理器**:
```python
# 在 processors/ 中创建 new_processor.py
class NewProcessor:
    def process(self, records: List[NewHealthRecord]) -> Dict[str, Any]:
        # 处理逻辑
        pass
```

4. **添加测试**:
```python
# 在 tests/ 中创建 test_new_processor.py
def test_new_processor():
    # 测试用例
    pass
```

### 添加新的分析功能

1. **创建分析器**:
```python
# 在 analyzers/ 中创建 new_analyzer.py
class NewAnalyzer:
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        # 分析逻辑
        pass
```

2. **集成到主流程**:
```python
# 在 cli.py 中添加命令
@app.command()
def new_analysis(file_path: str):
    analyzer = NewAnalyzer()
    results = analyzer.analyze(data)
    # 输出结果
```

### 添加新的可视化

1. **扩展图表生成器**:
```python
# 在 charts.py 中添加方法
def create_new_chart(self, data: pd.DataFrame) -> go.Figure:
    # 图表创建逻辑
    pass
```

2. **更新 CLI 命令**:
```python
# 在 cli_visualize.py 中添加选项
@click.option('--new-chart', is_flag=True, help='生成新类型图表')
def visualize(new_chart: bool, ...):
    if new_chart:
        chart = charts.create_new_chart(data)
        # 保存图表
```

## 调试技巧

### 常见调试场景

1. **XML 解析问题**:
```python
# 启用调试日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 解析小文件测试
parser = AppleHealthXMLParser()
records = list(parser.parse_file("small_test.xml"))
```

2. **内存问题**:
```python
# 使用 tracemalloc 进行内存分析
import tracemalloc

tracemalloc.start()
# ... 执行操作 ...
current, peak = tracemalloc.get_traced_memory()
print(f"当前内存使用: {current / 1024 / 1024:.1f} MB")
print(f"峰值内存使用: {peak / 1024 / 1024:.1f} MB")
tracemalloc.stop()
```

3. **性能瓶颈**:
```python
# 使用 cProfile
import cProfile
cProfile.run('main()', 'profile_output.prof')

# 分析结果
import pstats
p = pstats.Stats('profile_output.prof')
p.sort_stats('cumulative').print_stats(10)
```

### 日志调试

```python
from src.utils.logger import logger

# 不同级别日志
logger.debug("调试信息")
logger.info("一般信息")
logger.warning("警告信息")
logger.error("错误信息")
```

## 部署和发布

### 本地部署

```bash
# 构建分发包
uv build

# 安装到系统
pip install dist/*.whl
```

### Docker 部署

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY . .
RUN pip install -e .

CMD ["health-analyzer", "--help"]
```

### 云端部署

**推荐配置**:
- CPU: 2 核心
- 内存: 4GB RAM
- 存储: 20GB SSD

**部署脚本**:
```bash
# 使用 uv 部署
uv sync --no-dev
uv run health-analyzer --version
```

## 故障排除

### 常见问题

1. **依赖安装失败**:
   - 检查 Python 版本 (需要 3.12+)
   - 更新 pip: `pip install --upgrade pip`
   - 使用 uv: `uv sync`

2. **内存不足**:
   - 增加系统内存
   - 使用 `--batch-size` 参数减小批处理大小
   - 分批处理大文件

3. **解析失败**:
   - 检查 XML 文件完整性
   - 查看错误日志
   - 使用小文件测试

4. **图表生成失败**:
   - 安装系统依赖: `apt-get install libgconf-2-4`
   - 更新 plotly: `pip install --upgrade plotly`

### 获取帮助

- 📖 **文档**: [README.md](../README.md)
- 🐛 **问题**: [GitHub Issues](https://github.com/your-repo/apple-health-analyzer/issues)
- 💬 **讨论**: [GitHub Discussions](https://github.com/your-repo/apple-health-analyzer/discussions)

## 贡献规范

### 代码提交规范

使用 [Conventional Commits](https://conventionalcommits.org/) 规范：

```
type(scope): description

[optional body]

[optional footer]
```

**类型**:
- `feat`: 新功能
- `fix`: 修复 bug
- `docs`: 文档更改
- `style`: 代码格式调整
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建过程或工具配置

### Pull Request 要求

1. **清晰描述**: 详细说明更改内容和原因
2. **测试覆盖**: 所有新功能包含测试
3. **代码审查**: 通过至少一位维护者审查
4. **文档更新**: 更新相关文档

### 代码审查清单

- [ ] 代码符合项目规范
- [ ] 包含必要的测试
- [ ] 更新了相关文档
- [ ] 通过了所有自动化检查
- [ ] 性能表现良好
- [ ] 安全性考虑充分

---

## 更新日志

### v1.0.0 (2024-01-XX)
- ✨ 初始版本发布
- 🚀 支持心率和睡眠数据分析
- 📊 提供交互式图表和报告
- 🧪 完整测试套件 (197 个测试用例)
- 📚 详细文档和使用指南

### 近期规划
- 🔄 增量数据处理优化
- 🎨 更多图表类型支持
- 📱 Web 界面开发
- 🔧 配置管理系统改进
- 📈 性能监控和分析工具