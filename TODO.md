# 项目重构计划 (Project Refactoring Plan)

## 1. 代码清理与优化 (Code Cleanup & Optimization)
- [ ] **移除 Dead Code**: 删除所有标记为 `pass` 的占位符和未使用的代码。
- [ ] **优化注释**: 将所有注释转换为英文，移除冗余注释，确保关键算法有详细文档。
- [ ] **代码格式化**: 使用 `ruff` 进行代码格式化和 import 排序。
- [ ] **类型注解**: 确保所有函数和类都有完整的类型注解，通过 `pyright` 检查。

## 2. 模块重构 (Module Refactoring)
- [ ] **`src/core`**: 优化 `data_models.py` 和 `xml_parser.py`，确保类型安全和解析效率。
- [ ] **`src/processors`**: 重构 `cleaner.py` 和 `sleep.py`，提取公共逻辑，优化数据处理流程。
- [ ] **`src/analyzers`**: 完善 `anomaly.py` 和 `statistical.py`，实现未完成的异常检测算法。
- [ ] **`src/utils`**: 统一日志记录和类型转换工具。

## 3. 算法优化 (Algorithm Optimization)
- [ ] **XML 解析**: 优化流式 XML 解析算法，减少内存占用。
- [ ] **数据去重**: 优化 `DeduplicationResult` 和去重逻辑，使用向量化操作加速 pandas 处理。
- [ ] **统计分析**: 使用 numpy/scipy 加速统计计算。

## 4. 测试改进 (Test Improvement)
- [ ] **单元测试**: 补充缺失的单元测试，提高覆盖率。
- [ ] **集成测试**: 编写端到端集成测试，验证完整流程。
- [ ] **测试重构**: 整理测试代码，统一风格，使用 fixture 复用测试数据。

## 5. 文档更新 (Documentation Update)
- [ ] **README.md**: 更新项目说明、安装指南和使用文档。
- [ ] **AGENTS.md**: 更新 Agent 指引，反映最新的代码结构和规范。

## 6. 最终验证 (Final Verification)
- [ ] **构建检查**: 运行 `uv build` 确保构建成功。
- [ ] **Lint 检查**: 运行 `ruff check .` 确保无 lint 错误。
- [ ] **类型检查**: 运行 `pyright` 确保类型安全。
- [ ] **全量测试**: 运行 `pytest` 确保所有测试通过。
