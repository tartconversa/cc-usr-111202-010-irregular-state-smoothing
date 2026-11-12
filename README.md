# 异步仪器观测的滤波平滑与不确定性

本项目保存虚构科学工程分析的合成输入与Python读取工具。当前工具仅解析输入并显示类型和条数，不执行参数拟合、统计分析、方程求解、优化或模型训练。所有仪器、实验、研究与专家身份均为虚构。

## 数据口径

event_at是仪器测量时刻，received_at是获知时刻；sensor偏置和观测方差来自state配置。状态初值可半正定，时间单位明确为秒，q为连续白加速度谱强度。

fixtures保存原始输入，src/state_reader.py是当前读取器，material-index.json为文件内容摘要。研究计算输出可写到独立.workspace-output目录，输入不应被覆盖。

## 本地运行

Python 3.11及以上；已有读取工具仅依赖标准库，无第三方安装步骤。进入项目根目录执行：

    python -m src.state_reader fixtures/observations.csv

编译检查：

    python -m compileall -q src tests

输入契约验证：

    python -m unittest discover -s tests -v

## 固定容器

    docker build -t irregular-state-smoothing:local .
    docker run --rm irregular-state-smoothing:local

容器按完整镜像摘要固定，默认运行读取测试，不访问外部实验系统。
