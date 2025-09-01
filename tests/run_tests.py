"""Simple test runner script for AMPilot sequence tools.

Just use pytest directly! Examples:

# Run all tests
pytest

# Run unit tests only  
pytest tests/unit/

# Run integration tests only
pytest tests/integration/

# Run with coverage
pytest --cov=ampilot.tools.sequence

# Run specific module tests
pytest tests/unit/tools/sequence/test_ingest_normalize.py

# Run verbose
pytest -v

# Run and stop on first failure
pytest -x
"""

if __name__ == "__main__":
    print("直接使用 pytest 命令就可以了！")
    print("\n常用命令:")
    print("  pytest                    # 运行所有测试")
    print("  pytest tests/unit/        # 只运行单元测试")
    print("  pytest tests/integration/ # 只运行集成测试")
    print("  pytest -v                 # 详细输出")
    print("  pytest --cov=ampilot      # 带覆盖率报告")
    print("  pytest -x                 # 遇到失败就停止")
    print("\n具体模块测试:")
    print("  pytest tests/unit/tools/sequence/test_ingest_normalize.py")
    print("  pytest tests/unit/tools/sequence/test_homology_search.py") 
    print("  pytest tests/unit/tools/sequence/test_clustering_derep.py")
