#!/usr/bin/env python
"""隔离模式（python -I）下的薄启动脚本。

-I 会忽略 cwd 与 PYTHONPATH，因此这里在导入项目包之前显式把本脚本
所在的项目根目录插入 sys.path。fixtures 等输入路径只作为字符串参数
传入，解释器不会从输入目录加载任何代码。

运行方式（在项目根目录）：

    python -I run_estimation.py fixtures/observations.csv fixtures/state.json
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.run_estimation import main  # noqa: E402

if __name__ == "__main__":
    main()
