"""时间解析、单位换算与过程噪声谱强度的量纲换算。

内部统一使用“秒”作为连续时间单位：
- ISO-8601 带偏移量的字符串解析为 epoch 秒；
- 数值时间戳按配置的 time_unit 解释为相对秒数；
- q 是连续白加速度的谱强度（连续时间单位为秒），当输入以其他
  时间单位给定时按物理量纲换算，保证同一物理过程的估计结果不变。
"""

from datetime import datetime, timezone

from .errors import StateEstimationError

# 1 个输入时间单位等于多少秒
UNIT_TO_SECONDS = {
    "seconds": 1.0,
    "second": 1.0,
    "s": 1.0,
    "milliseconds": 1e-3,
    "millisecond": 1e-3,
    "ms": 1e-3,
    "minutes": 60.0,
    "minute": 60.0,
    "min": 60.0,
    "hours": 3600.0,
    "hour": 3600.0,
    "h": 3600.0,
}


def unit_factor(time_unit):
    try:
        return UNIT_TO_SECONDS[str(time_unit).strip().lower()]
    except KeyError as exc:
        raise StateEstimationError(
            f"未知时间单位 {time_unit!r}，支持：{sorted(set(UNIT_TO_SECONDS))}"
        ) from exc


def parse_iso(text):
    """解析带时区的 ISO-8601 字符串为 epoch 秒；朴素时间戳按契约拒绝。"""
    if not isinstance(text, str) or not text.strip():
        raise StateEstimationError(f"时间戳必须是非空字符串或数值，收到 {text!r}")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise StateEstimationError(f"无法解析 ISO-8601 时间戳 {text!r}: {exc}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise StateEstimationError(
            f"时间戳 {text!r} 缺少时区偏移量，无法无歧义地比较发生与获知时刻"
        )
    return parsed.timestamp()


def parse_time(value, time_unit="seconds"):
    """字符串走 ISO-8601；数值按 time_unit 换算成秒。"""
    if isinstance(value, str):
        return parse_iso(value)
    if isinstance(value, (int, float)):
        return float(value) * unit_factor(time_unit)
    raise StateEstimationError(f"不支持的时间戳类型 {type(value).__name__}: {value!r}")


def to_iso(epoch_seconds):
    """epoch 秒转回 UTC ISO-8601 字符串，仅用于报告展示。"""
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat()


def q_to_seconds(q, time_unit):
    """把以输入时间单位为连续时间基准的 q 换算到秒基准。

    过程噪声谱强度 q 的量纲是 [状态量]^2/[时间]^3（加速度白噪声）。
    设 1 输入单位 = f 秒，则同一物理过程在秒基准下 q_s = q / f。
    数值时间轴经 dt_s = f * dt_u 缩放后，Q_s 与直接用秒表达完全一致。
    """
    factor = unit_factor(time_unit)
    return q / factor
