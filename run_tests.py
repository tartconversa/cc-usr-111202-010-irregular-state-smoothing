#!/usr/bin/env python
"""隔离模式（python -I）下运行全部单元测试的薄启动脚本。

-I 忽略 cwd 与 PYTHONPATH，所以先显式插入项目根目录；
fixtures 路径只作为参数被测试代码读取，解释器不会从输入目录加载代码。

    python -I run_tests.py
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
