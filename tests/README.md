# 测试说明

## 快速开始

直接使用 pytest 命令运行测试：

```bash
# 运行所有测试
pytest

# 运行单元测试
pytest tests/unit/

# 运行集成测试
pytest tests/integration/

# 详细输出
pytest -v

# 带覆盖率报告
pytest --cov=ampilot.tools.sequence

# 遇到失败就停止
pytest -x
```

## 测试结构

```
tests/
├── unit/                           # 单元测试
│   └── tools/
│       └── sequence/
│           ├── test_ingest_normalize.py
│           ├── test_homology_search.py
│           └── test_clustering_derep.py
├── integration/                    # 集成测试
│   └── tools/
│       └── sequence/
│           ├── test_workflow_integration.py
│           └── test_mcp_integration.py
├── fixtures/                       # 测试数据和工具
│   └── sequence_fixtures.py
└── run_tests.py                    # 简单的使用说明
```

## 具体模块测试

```bash
# 测试序列摄取模块
pytest tests/unit/tools/sequence/test_ingest_normalize.py

# 测试同源搜索模块
pytest tests/unit/tools/sequence/test_homology_search.py

# 测试聚类去重模块
pytest tests/unit/tools/sequence/test_clustering_derep.py

# 测试工作流集成
pytest tests/integration/tools/sequence/test_workflow_integration.py

# 测试MCP集成
pytest tests/integration/tools/sequence/test_mcp_integration.py
```

## 测试标记

使用标记来选择特定类型的测试：

```bash
# 只运行单元测试
pytest -m unit

# 只运行集成测试  
pytest -m integration

# 只运行MCP测试
pytest -m mcp

# 跳过慢速测试
pytest -m "not slow"
```

## 覆盖率报告

```bash
# 生成覆盖率报告
pytest --cov=ampilot.tools.sequence --cov-report=html

# 查看HTML覆盖率报告
open htmlcov/index.html
```

就这么简单！
