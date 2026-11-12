# 异步仪器观测的滤波平滑与不确定性

本项目保存虚构科学工程分析的合成输入与 Python 状态估计组件。所有仪器、实验、研究与专家身份均为虚构；组件只读取给定合成文件，不连接真实仪器服务，不调用任何外部模型。

## 解决的问题

两台仪器不同步时，“先把缺测补成 0 再平滑”会把停采期间的无信息误判成物理突降。本组件改为在**真实时间间隔**下融合观测，并把两种口径严格分开：

- **当时可得的滤波估计（实时口径）**：只使用 `known_at` 截止前已经 `received_at` 的观测，按 `event_at` 重放；
- **事后 RTS 平滑（回看口径）**：使用该截止内的全部未来观测，单独存放，不得冒充实时预测效果。

## 模型口径

状态 `x = [位置, 速度]`，真实间隔 dt（内部统一秒）：

```
F = [[1, dt], [0, 1]]
Q = q * [[dt^3/3, dt^2/2], [dt^2/2, dt]]     # 连续白加速度噪声离散化
H = [1, 0]                                    # 每台仪器只观测位置
```

- 仪器各自有已知偏置（进入创新前扣除）和严格正的观测方差；
- 缺测就是没有记录：不补零、不产生更新；长空档只按真实 dt 传播 `F`、`Q`，没有固定步长；
- 同一 `event_at` 的独立仪器在同一先验上做联合更新，dt 只传播一次；
- 更新采用 Joseph 形式，每步检查协方差对称半正定；显著负特征值直接报错，**不通过裁剪特征值掩盖错误**；
- 初始协方差允许半正定（速度方差为 0）；RTS 遇到奇异预测协方差时用带显式截断计数的对称伪逆，行为可审计；
- 每个观测组记录创新、创新协方差、标准化残差与高斯创新对数似然。

## 异步与版本口径

- `event_at` 是测量发生时刻，`received_at` 是获知时刻；先按 `known_at` 筛观测，再按 `event_at` 重放；
- `observation_id` 是唯一键：完全一致的重送去重，任一内容字段冲突则拒绝（`ObservationConflictError`）；
- 迟到观测只生成截止更晚的新版本，已算出的旧版本结果对象不回写；数值上验证“迟到点之前的共享前缀逐元素一致”；
- 离线重放缓存按 `(known_at, q)` 记忆化，命中返回同一对象；原始观测批次变化时丢弃整个缓存、从文件确定性重算，而不是增量修补。

## q 的选择与冻结

`q` 由配置给定。需要比较多个 q 时，只允许用训练时段（`received_at <= training_end`）的创新似然之和选择；似然并列取较小 q。验证段及所有 `known_at` 版本冻结所选 q，不重新拟合。

## 目录

| 路径 | 职责 |
| --- | --- |
| `src/state_reader.py` | 只解析 CSV/JSONL/JSON 文件，无设备连接 |
| `src/linalg.py` | 小矩阵运算、Jacobi 特征分解、Joseph 更新、PSD 校验与对称伪逆 |
| `src/timeutils.py` | ISO-8601/数值时间解析、时间单位与 q 的量纲换算 |
| `src/config.py` | `state.json` 解析与校验 |
| `src/observations.py` | 观测解析、去重、冲突拒绝、缺测不补零 |
| `src/model.py` | F、Q、H、R 构造 |
| `src/estimator.py` | 变步长前向滤波、预报、RTS 平滑 |
| `src/replay.py` | 版本化重放缓存、训练段选 q |
| `src/reporting.py` | JSON 序列化与 Markdown 报告 |
| `src/run_estimation.py` | 应用编排与 CLI 逻辑 |
| `run_estimation.py` / `run_tests.py` | 隔离模式（`python -I`）薄启动脚本 |
| `tests/` | 52 个单元/端到端测试 |

## 本地运行

Python 3.11 及以上，仅标准库，无第三方依赖。fixtures 是不可信数据，运行解释器使用 `-I`（忽略 cwd 与环境变量），输入路径只作为参数传入：

    python -I run_estimation.py fixtures/observations.csv fixtures/state.json \
        --output-dir .workspace-output/run

输出（不覆盖任何输入）：

- `summary.json`：q 选择得分、版本清单、跨版本不可变性检查；
- `version_0.json` / `version_1.json`：每个 `known_at` 版本的滤波轨迹、创新/标准化残差、RTS 平滑轨迹与截止处无观测预报；
- `report.md`：滤波与平滑的中文对照报告。

编译检查：

    python -m compileall -q src tests

全部测试（含原有输入契约测试）：

    python -I run_tests.py

## 固定容器

    docker build -t irregular-state-smoothing:local .
    docker run --rm irregular-state-smoothing:local

容器按完整镜像摘要固定，默认运行全部测试，不访问外部实验系统。
