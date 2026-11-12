"""state.json 配置的解析与校验。"""

from dataclasses import dataclass

from .errors import StateEstimationError
from .timeutils import parse_time, q_to_seconds, unit_factor


@dataclass(frozen=True)
class SensorSpec:
    name: str
    bias: float
    variance: float


@dataclass(frozen=True)
class EstimationConfig:
    time_unit: str
    initial_time: float  # epoch 秒（ISO）或相对秒（数值时间轴）
    initial_mean: tuple[float, float]
    initial_covariance: tuple[tuple[float, float], tuple[float, float]]
    q: float  # 已统一到秒基准的连续白加速度谱强度
    sensors: dict[str, SensorSpec]
    known_at_cutoffs: tuple[float, ...]
    training_end: float | None
    q_grid: tuple[float, ...]
    hand_check: dict | None
    numeric_time_axis: bool

    @property
    def unit_factor(self):
        return unit_factor(self.time_unit)


def _require_number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StateEstimationError(f"配置字段 {field} 必须是数值，收到 {value!r}")
    return float(value)


def parse_config(raw, numeric_time_axis=None):
    """把 state.json 的原始字典转成 EstimationConfig。

    numeric_time_axis 为 True 时，时间戳按数值时间轴处理（按 time_unit
    换算）；为 None 时自动根据 initial_time 是否为字符串判断。
    """
    if not isinstance(raw, dict):
        raise StateEstimationError("state 配置必须是 JSON 对象")

    time_unit = raw.get("time_unit", "seconds")
    factor = unit_factor(time_unit)

    initial_time_raw = raw.get("initial_time")
    if numeric_time_axis is None:
        numeric_time_axis = not isinstance(initial_time_raw, str)
    initial_time = parse_time(initial_time_raw, time_unit)

    mean_raw = raw.get("initial_mean")
    if not isinstance(mean_raw, list) or len(mean_raw) != 2:
        raise StateEstimationError("initial_mean 必须是长度为 2 的数组 [位置, 速度]")
    initial_mean = (_require_number(mean_raw[0], "initial_mean[0]"),
                    _require_number(mean_raw[1], "initial_mean[1]"))

    cov_raw = raw.get("initial_covariance")
    if (not isinstance(cov_raw, list) or len(cov_raw) != 2
            or any(not isinstance(row, list) or len(row) != 2 for row in cov_raw)):
        raise StateEstimationError("initial_covariance 必须是 2x2 数组")
    initial_covariance = tuple(
        tuple(_require_number(cov_raw[i][j], f"initial_covariance[{i}][{j}]")
              for j in range(2))
        for i in range(2)
    )
    if abs(initial_covariance[0][1] - initial_covariance[1][0]) > 1e-12:
        raise StateEstimationError("initial_covariance 必须对称")
    # 允许半正定（速度方差可以为 0），只拒绝显著负特征值与负对角元。
    if any(value < -1e-12 for row in initial_covariance for value in row):
        raise StateEstimationError("initial_covariance 存在负元素，不是半正定矩阵")
    det = (initial_covariance[0][0] * initial_covariance[1][1]
           - initial_covariance[0][1] * initial_covariance[1][0])
    if det < -1e-12:
        raise StateEstimationError("initial_covariance 行列式为负，不是半正定矩阵")

    q = _require_number(raw.get("q", 0.0), "q")
    if q < 0:
        raise StateEstimationError("过程噪声谱强度 q 必须非负")
    q_seconds = q / factor if numeric_time_axis else q

    sensors_raw = raw.get("sensors")
    if not isinstance(sensors_raw, dict) or not sensors_raw:
        raise StateEstimationError("sensors 必须是非空对象")
    sensors = {}
    for name, spec in sensors_raw.items():
        if not isinstance(spec, dict):
            raise StateEstimationError(f"传感器 {name} 的配置必须是对象")
        bias = _require_number(spec.get("bias", 0.0), f"sensors.{name}.bias")
        variance = _require_number(spec.get("variance"), f"sensors.{name}.variance")
        if variance <= 0:
            raise StateEstimationError(
                f"传感器 {name} 的观测方差必须为正数，收到 {variance}"
            )
        sensors[name] = SensorSpec(name=name, bias=bias, variance=variance)

    cutoffs_raw = raw.get("known_at_cutoffs", [])
    if not isinstance(cutoffs_raw, list):
        raise StateEstimationError("known_at_cutoffs 必须是数组")
    cutoffs = tuple(parse_time(value, time_unit) for value in cutoffs_raw)
    if list(cutoffs) != sorted(cutoffs):
        raise StateEstimationError("known_at_cutoffs 必须按时间升序排列")

    training_end = None
    if raw.get("training_end") is not None:
        training_end = parse_time(raw["training_end"], time_unit)

    q_grid_raw = raw.get("q_grid", [])
    if not isinstance(q_grid_raw, list) or not q_grid_raw:
        raise StateEstimationError("q_grid 必须是非空数组")
    q_grid = tuple(_require_number(value, "q_grid") for value in q_grid_raw)
    if any(value < 0 for value in q_grid):
        raise StateEstimationError("q_grid 中的候选 q 必须非负")
    q_grid_seconds = tuple(
        q_to_seconds(value, time_unit) if numeric_time_axis else float(value)
        for value in q_grid
    )

    return EstimationConfig(
        time_unit=time_unit,
        initial_time=initial_time,
        initial_mean=initial_mean,
        initial_covariance=initial_covariance,
        q=q_seconds,
        sensors=sensors,
        known_at_cutoffs=cutoffs,
        training_end=training_end,
        q_grid=q_grid_seconds,
        hand_check=raw.get("hand_check"),
        numeric_time_axis=numeric_time_axis,
    )
