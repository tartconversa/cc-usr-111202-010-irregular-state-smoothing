"""观测记录的解析、校验、去重与冲突拒绝。

契约要点：
- observation_id 是唯一键；同键重送且内容完全一致时去重；
- 同键但任一内容字段（sensor/event_at/received_at/value）不一致时拒绝；
- 缺测就是没有记录，解析层绝不把空值补成 0；
- event_at 是测量发生时刻，received_at 是获知时刻，后者不得早于前者。
"""

from dataclasses import dataclass

from .errors import ObservationConflictError, StateEstimationError
from .timeutils import parse_time


@dataclass(frozen=True)
class Observation:
    observation_id: str
    sensor: str
    event_at: float  # 秒
    received_at: float  # 秒
    value: float

    @property
    def signature(self):
        return (self.sensor, self.event_at, self.received_at, self.value)


def _parse_rows(rows, source_label):
    if not isinstance(rows, list):
        raise StateEstimationError(f"{source_label} 的观测必须是记录数组")
    required = ("observation_id", "sensor", "event_at", "received_at", "value")
    parsed = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise StateEstimationError(f"{source_label} 第 {index} 行不是对象")
        missing = [field for field in required if field not in row]
        if missing:
            raise StateEstimationError(
                f"{source_label} 第 {index} 行缺少字段 {missing}，缺测应表现为无记录而非空列"
            )
        parsed.append((index, row))
    return parsed


def parse_observations(rows, config, source_label="observations"):
    """把 CSV/JSONL 读出的原始字典列表转成去重后的 Observation 列表。"""
    records = {}
    result = []
    for index, row in _parse_rows(rows, source_label):
        obs_id = row["observation_id"]
        if not isinstance(obs_id, str) or not obs_id.strip():
            raise StateEstimationError(f"{source_label} 第 {index} 行 observation_id 非法")
        sensor = row["sensor"]
        if sensor not in config.sensors:
            raise StateEstimationError(
                f"{source_label} 第 {index} 行引用了未配置的传感器 {sensor!r}"
            )
        # 空字符串是缺测的文本形态，按契约报错而不是补 0。
        raw_value = row["value"]
        if isinstance(raw_value, str) and not raw_value.strip():
            raise StateEstimationError(
                f"{source_label} 第 {index} 行（{obs_id}）观测值为空；"
                "缺测不得补零，应删除该记录"
            )
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise StateEstimationError(
                f"{source_label} 第 {index} 行（{obs_id}）观测值无法解析为数值：{raw_value!r}"
            ) from exc
        event_at = parse_time(row["event_at"], config.time_unit)
        received_at = parse_time(row["received_at"], config.time_unit)
        if received_at < event_at:
            raise StateEstimationError(
                f"{source_label} 第 {index} 行（{obs_id}）received_at 早于 event_at，"
                "不可能在测量发生前获知该观测"
            )
        observation = Observation(
            observation_id=obs_id,
            sensor=sensor,
            event_at=event_at,
            received_at=received_at,
            value=value,
        )
        existing = records.get(obs_id)
        if existing is None:
            records[obs_id] = observation
            result.append(observation)
        elif existing.signature != observation.signature:
            raise ObservationConflictError(
                f"观测 {obs_id} 出现内容冲突的重送："
                f"首次 {existing.signature}，重送 {observation.signature}"
            )
        # 完全一致的重送：静默去重。
    return result


def filter_known_at(observations, known_at):
    """按获知截止时刻筛出“当时可得”的观测（received_at <= known_at）。"""
    return [obs for obs in observations if obs.received_at <= known_at]
