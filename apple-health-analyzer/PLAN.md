## prompt

我希望能写一个python脚本来对我从Apple Health导出的Data进行分析，具体来说要求（功能点）如下，每个要求（功能点）可以再做拆分：

1. 读取并解析export.xml的数据结构
2. 将export.xml拆分为不同的子项数据集，按数据类型进行分类，比如Activity、Heart Rate、Steps、Sleep等
3. 将分类数据导出到csv、json格式文件
4. 针对Heart Rate、Sleep数据进行分析
  4.1 清洗并合并重复/叠加数据，数据来源优先级为 Apple Watch (🐙Watch) > 小米运动健康 > iPhone (🐙Phone)
  4.1 计算并保存以Hour/Day/Week/Month/6Month/Year不同区间的数据最大值/最小值/平均值
  4.2 对异常数据进行分析，报告分析结果
  4.3 生成相关图表
  4.4 生成Apple Health中的Heart Rate、Resting Heart Rate、Heart Rate Variability、Walking Heart Rate Average、Cardio Fitness、Cardio Recovery、Sleep、Heart Rate: Sleep、信息
  4.5 针对上面的信息进行分析并汇总出Highlights
5. 项目使用uv进行管理，项目结构及代码遵循现代软件工程最佳实践，避免不必要的注释，保留足够的调试信息，允许切换dev环境和prod环境，模块功能间高耦合低内聚，对过大的函数和模块及时进行拆分和重构

注意：
- 原始xml文件非常大（300MB+），不要尝试直接读取
- applehealth 子文件夹是我找到的开源实现，你可以参考其中关于读取export.xml数据结构的实现，但它没有满足我的需求，applehealth/health_out 是我之前使用这个脚本对Health数据进行分析导出的结果
- export_data 子文件夹是直接通过Apple Health导出的数据

## 项目概览

你的export.xml文件包含**784,390条记录**（362MB），包含丰富的健康数据类型。现有的`applehealth`实现提供了基础的XML解析功能，但不满足你的深度分析需求。

## 核心数据类型分析

根据health_types_report.json，你的数据包含：

**心率相关数据**（你的重点分析对象）：
- HeartRate: 194,076条记录
- RestingHeartRate: 563条记录  
- HeartRateVariabilitySDNN: 4,923条记录
- WalkingHeartRateAverage: 534条记录
- HeartRateRecoveryOneMinute: 6条记录
- VO2Max (心肺适能): 51条记录

**睡眠数据**：
- SleepAnalysis: 18,878条记录（包含InBed, Asleep等不同阶段）

**数据来源优先级**（根据你的要求）：
1. Apple Watch (🐙Watch) - 主要数据源
2. 小米运动健康 (Xiaomi Home)
3. iPhone (🐙Phone)

## 技术架构设计

### 1. 项目结构
```
apple_health_analyzer/
├── pyproject.toml           # uv项目配置
├── README.md
├── .env.example
├── src/
│   ├── __init__.py
│   ├── config.py           # 环境配置管理
│   ├── core/
│   │   ├── __init__.py
│   │   ├── xml_parser.py   # 流式XML解析（处理大文件）
│   │   ├── data_models.py  # Pydantic数据模型
│   │   └── storage.py      # 数据存储抽象层
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── base.py         # 基础处理器
│   │   ├── heart_rate.py   # 心率数据处理
│   │   └── sleep.py        # 睡眠数据处理
│   ├── analyzers/
│   │   ├── __init__.py
│   │   ├── statistics.py   # 统计分析
│   │   ├── anomaly.py      # 异常检测
│   │   └── trends.py       # 趋势分析
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── charts.py       # 图表生成
│   │   └── reports.py      # 报告生成
│   └── cli.py              # CLI入口
├── tests/
│   └── ...
└── output/                 # 输出目录
    ├── raw/               # 原始CSV/JSON
    ├── processed/         # 处理后的数据
    ├── charts/            # 图表
    └── reports/           # 分析报告
```

### 2. 核心技术选型

**XML解析**：使用`xml.etree.ElementTree.iterparse()`进行流式解析，避免一次性加载整个文件到内存

**数据处理**：
- `pandas`: 数据处理和分析
- `polars`: 可选，处理超大数据集时性能更好
- `pydantic`: 数据验证和类型安全

**可视化**：
- `plotly`: 交互式图表
- `matplotlib/seaborn`: 静态图表
- `kaleido`: 导出静态图片

**统计分析**：
- `scipy`: 统计检验
- `numpy`: 数值计算
- `scikit-learn`: 异常检测

**项目管理**：
- `uv`: 快速依赖管理
- `ruff`: 代码格式化和linting
- `pytest`: 测试框架

### 3. 功能模块详细设计

#### 模块1: XML解析与数据提取
```python
# 流式解析，分批处理
class StreamingXMLParser:
    def parse_records(self, xml_path, record_type=None):
        # 使用iterparse逐条读取
        # 支持按类型过滤
        # 返回生成器避免内存溢出
        
    def extract_by_category(self):
        # 提取所有数据类型
        # 分类存储到不同文件
```

#### 模块2: 数据清洗与合并
```python
class DataCleaner:
    def deduplicate(self, df, source_priority):
        # 按时间窗口检测重复
        # 根据数据源优先级保留数据
        
    def merge_overlapping(self, df):
        # 合并叠加的时间段（如睡眠）
        # 处理数据冲突
```

#### 模块3: 统计分析
```python
class TimeSeriesAnalyzer:
    def aggregate_by_interval(self, df, intervals):
        # intervals: ['hour', 'day', 'week', 'month', '6month', 'year']
        # 计算 max/min/mean/median/std
        
    def detect_anomalies(self, df, method='zscore'):
        # 统计方法检测异常
        # 标记异常数据点和原因
```

#### 模块4: 心率专项分析
```python
class HeartRateAnalyzer:
    def analyze_resting_hr(self):
        # 静息心率趋势
        # 与年龄正常范围对比
        
    def analyze_hrv(self):
        # HRV趋势分析
        # 压力和恢复状态评估
        
    def analyze_cardio_fitness(self):
        # VO2Max趋势
        # 心肺适能评级
        
    def generate_highlights(self):
        # 提取关键发现
        # 生成可读性强的总结
```

#### 模块5: 睡眠专项分析
```python
class SleepAnalyzer:
    def parse_sleep_stages(self):
        # 解析睡眠阶段
        # 计算各阶段占比
        
    def analyze_sleep_quality(self):
        # 睡眠效率
        # 入睡时间、觉醒次数
        
    def correlate_with_hr(self):
        # 睡眠期间心率分析
        # 睡眠质量与心率关联
```

#### 模块6: 可视化
```python
class Visualizer:
    def plot_timeseries(self, data, title):
        # 时序图（可交互）
        
    def plot_heatmap(self, data):
        # 热力图（如心率日历视图）
        
    def plot_distribution(self, data):
        # 分布直方图
        
    def generate_dashboard(self):
        # HTML仪表板整合所有图表
```

### 4. 数据输出格式

**CSV格式**（便于Excel分析）：
```
output/raw/
├── heart_rate_raw.csv
├── resting_heart_rate_raw.csv
├── hrv_raw.csv
├── sleep_raw.csv
└── ...

output/processed/
├── heart_rate_daily_stats.csv
├── heart_rate_weekly_stats.csv
├── sleep_quality_metrics.csv
└── anomalies_report.csv
```

**JSON格式**（便于程序处理）：
```json
{
  "analysis_date": "2026-01-19",
  "data_range": {"start": "2023-11-08", "end": "2026-01-19"},
  "heart_rate": {
    "summary": {...},
    "intervals": {...},
    "anomalies": [...]
  },
  "highlights": [...]
}
```

### 5. 开发与运行环境

**开发模式**：
- 详细日志输出
- 保留中间结果
- 性能分析
- 数据验证检查

**生产模式**：
- 精简日志
- 只保留最终结果
- 优化性能
- 错误优雅处理

### 6. 实施步骤

**Phase 1: 基础设施（Week 1）**
1. 项目脚手架搭建（uv init）
2. 流式XML解析器实现
3. 数据模型定义
4. 基础测试框架

**Phase 2: 核心功能（Week 2-3）**
1. 数据提取与分类
2. CSV/JSON导出
3. 数据清洗与去重
4. 统计分析模块

**Phase 3: 专项分析（Week 4-5）**
1. 心率深度分析
2. 睡眠深度分析
3. 异常检测
4. 关联分析

**Phase 4: 可视化与报告（Week 6）**
1. 图表生成
2. 交互式仪表板
3. Highlights生成
4. 完整报告导出

**Phase 5: 优化与文档（Week 7）**
1. 性能优化
2. 代码重构
3. 完善文档
4. 使用示例

## 关键技术挑战与解决方案

**挑战1: 300MB+ XML文件**
- 解决：流式解析 + 分批处理 + 增量写入

**挑战2: 数据源优先级与去重**
- 解决：时间窗口 + 数据源权重 + 相似度算法

**挑战3: 睡眠阶段重叠**
- 解决：时间段合并算法 + 优先级规则

**挑战4: 异常检测准确性**
- 解决：多种方法组合（Z-score + IQR + 移动平均）

**挑战5: 可扩展性**
- 解决：插件式处理器架构 + 抽象基类

## 代码质量保证

- **类型提示**: 全面使用类型注解
- **文档字符串**: 遵循Google风格
- **单元测试**: >80%覆盖率
- **集成测试**: 端到端测试关键流程
- **性能测试**: 确保处理速度合理
- **代码审查**: Ruff自动检查

## 下一步

如果你同意这个方案，请**切换到Act模式**，我将开始实现：

1. 首先创建项目结构和配置文件
2. 实现流式XML解析器（这是处理大文件的关键）
3. 逐步实现各个功能模块
4. 在每个关键步骤后测试验证

有任何疑问或需要调整的地方吗？