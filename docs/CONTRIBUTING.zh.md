# 贡献指南

感谢参与 Apple Health Analyzer 的改进。请在提交前阅读以下约定。

## 开发环境
```bash
git clone https://github.com/tagbug/apple-health-analyzer.git
cd apple-health-analyzer
uv sync
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

## 分支与提交策略
- 推荐从 `master` 或 `dev` 创建分支。
- 分支命名建议：`feat/<topic>`、`fix/<topic>`、`docs/<topic>`。

## 提交信息规范
使用以下格式：
```
<type>: <summary>
```

常见 type：`feat`、`fix`、`docs`、`refactor`、`test`、`chore`。

示例：
```
docs: refresh README structure and usage
```

## PR 规范
PR 描述建议包含以下信息：
- 变更背景与目标
- 主要改动点（1-3 条）
- 影响范围与风险评估
- 测试结果（包含命令）

## 质量检查
提交前建议执行：
```bash
uv run ruff format .
uv run ruff check .
uv run pyright --level error
uv run pytest
```

## 数据与隐私
- 不要提交 `export_data`、`output`、`.env` 等本地数据文件。
- 避免在 Issue 或 PR 中粘贴真实健康数据。
